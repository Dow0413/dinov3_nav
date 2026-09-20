# dinov3_nav/bev.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BEVConfig:
    resolution: float = 0.05

    # base_link:
    # x = forward
    # y = left
    x_min: float = 0.0
    x_max: float = 4.0
    y_min: float = -2.0
    y_max: float = 2.0

    min_depth: float = 0.20
    max_depth: float = 6.0

    # 每隔几个图像像素取一个点，降低计算量
    pixel_stride: int = 2

    min_observed_points: int = 1
    min_obstacle_points: int = 2


@dataclass
class BEVGrid:
    traversability: np.ndarray  # float32 [0, 1]
    observed: np.ndarray        # bool
    obstacle: np.ndarray        # bool

    cfg: BEVConfig

    @property
    def shape(self):
        return self.traversability.shape

    def xy_to_ij(self, x: float, y: float):
        """
        base_link:
            x -> forward
            y -> left

        BEV:
            i -> forward
            j -> lateral
        """
        if (
            x < self.cfg.x_min
            or x >= self.cfg.x_max
            or y < self.cfg.y_min
            or y >= self.cfg.y_max
        ):
            return None

        i = int((x - self.cfg.x_min) / self.cfg.resolution)
        j = int((y - self.cfg.y_min) / self.cfg.resolution)

        if i < 0 or j < 0:
            return None
        if i >= self.shape[0] or j >= self.shape[1]:
            return None

        return i, j


def transform_to_matrix(transform) -> np.ndarray:
    """
    geometry_msgs/Transform -> 4x4 homogeneous matrix.
    """

    t = transform.translation
    q = transform.rotation

    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    # Quaternion -> rotation matrix
    R = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array(
        [float(t.x), float(t.y), float(t.z)],
        dtype=np.float32,
    )

    return T


def build_local_bev(
    traversable_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_base_from_camera: np.ndarray,
    cfg: BEVConfig,
) -> BEVGrid:
    """
    将图像中的 traversability / obstacle 投影到 base_link BEV。

    注意：
    深度反投影使用 optical camera convention：

        X = image right
        Y = image down
        Z = forward

    所以 T_base_from_camera 必须对应真正的 optical frame。
    """

    h, w = depth.shape

    nx = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.resolution))
    ny = int(np.ceil((cfg.y_max - cfg.y_min) / cfg.resolution))

    observed_count = np.zeros((nx, ny), dtype=np.int32)
    traversable_count = np.zeros((nx, ny), dtype=np.int32)
    obstacle_count = np.zeros((nx, ny), dtype=np.int32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("invalid camera intrinsics")

    stride = max(1, int(cfg.pixel_stride))

    vv = np.arange(0, h, stride, dtype=np.int32)
    uu = np.arange(0, w, stride, dtype=np.int32)

    U, V = np.meshgrid(uu, vv)

    Z = depth[V, U]

    valid = (
        np.isfinite(Z)
        & (Z >= cfg.min_depth)
        & (Z <= cfg.max_depth)
    )

    if not valid.any():
        return BEVGrid(
            traversability=np.zeros((nx, ny), np.float32),
            observed=np.zeros((nx, ny), bool),
            obstacle=np.zeros((nx, ny), bool),
            cfg=cfg,
        )

    u = U[valid].astype(np.float32)
    v = V[valid].astype(np.float32)
    z = Z[valid].astype(np.float32)

    # optical camera frame
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    points_camera = np.stack(
        [x_cam, y_cam, z_cam],
        axis=1,
    )

    R = T_base_from_camera[:3, :3]
    t = T_base_from_camera[:3, 3]

    # Nx3
    points_base = points_camera @ R.T + t

    x = points_base[:, 0]
    y = points_base[:, 1]

    inside = (
        (x >= cfg.x_min)
        & (x < cfg.x_max)
        & (y >= cfg.y_min)
        & (y < cfg.y_max)
    )

    if not inside.any():
        return BEVGrid(
            traversability=np.zeros((nx, ny), np.float32),
            observed=np.zeros((nx, ny), bool),
            obstacle=np.zeros((nx, ny), bool),
            cfg=cfg,
        )

    x = x[inside]
    y = y[inside]

    source_trav = (
        traversable_mask[V[valid], U[valid]] > 0
    )[inside]

    source_obstacle = (
        obstacle_mask[V[valid], U[valid]] > 0
    )[inside]

    ii = ((x - cfg.x_min) / cfg.resolution).astype(np.int32)
    jj = ((y - cfg.y_min) / cfg.resolution).astype(np.int32)

    good = (
        (ii >= 0)
        & (ii < nx)
        & (jj >= 0)
        & (jj < ny)
    )

    ii = ii[good]
    jj = jj[good]

    source_trav = source_trav[good]
    source_obstacle = source_obstacle[good]

    np.add.at(observed_count, (ii, jj), 1)
    np.add.at(
        traversable_count,
        (ii, jj),
        source_trav.astype(np.int32),
    )
    np.add.at(
        obstacle_count,
        (ii, jj),
        source_obstacle.astype(np.int32),
    )

    observed = observed_count >= cfg.min_observed_points

    traversability = np.zeros((nx, ny), dtype=np.float32)

    np.divide(
        traversable_count,
        np.maximum(observed_count, 1),
        out=traversability,
        where=observed_count > 0,
    )

    obstacle = obstacle_count >= cfg.min_obstacle_points

    return BEVGrid(
        traversability=traversability,
        observed=observed,
        obstacle=obstacle,
        cfg=cfg,
    )
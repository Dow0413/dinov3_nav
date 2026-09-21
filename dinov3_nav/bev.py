# -*- coding: utf-8 -*-
"""Local RGB-D BEV construction for DINOv3 traversability navigation.

Coordinate convention of the BEV/planner frame follows ROS REP-103:
    x: forward
    y: left
    z: up

Depth back-projection always starts in the camera *optical* convention:
    X: image right
    Y: image down
    Z: camera forward

The Gazebo zed_x camera stamps images with frame_id ``zed_camera_link``
(via ``<gz_frame_id>``) while the TF ``base_link -> zed_camera_link`` carries
an identity rotation, i.e. the *named* frame is body-convention but the pixel
data is optical-convention. Therefore the caller must compose

    T_planner_from_camera = T_planner_from_bodyframe @ T_BODY_FROM_OPTICAL

when ``camera.projection_convention`` is "optical" (the default, verified
against gazebo_sim_ws_3 zed_x_camera.xacro). With convention "body" the TF
itself already describes the optical axes and is used directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isnan
from typing import Optional, Tuple

import cv2
import numpy as np

# x_body =  z_opt,  y_body = -x_opt,  z_body = -y_opt
T_BODY_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class BEVConfig:
    resolution: float = 0.05
    x_min: float = -0.80
    x_max: float = 4.00
    y_min: float = -2.20
    y_max: float = 2.20
    min_depth: float = 0.20
    max_depth: float = 6.00
    pixel_stride: int = 2
    min_observed_points: int = 1
    min_obstacle_points: int = 2
    # Small closing of the observed-ground support: fills RGB-D raster holes
    # without freely growing the traversable boundary into unknown space.
    close_radius_m: float = 0.08
    support_traversability: float = 0.70
    # Height-based obstacle evidence, relative to the estimated ground plane.
    # A point higher than ground_z + obstacle_height is obstacle evidence no
    # matter what the DINO mask says; a point far below is a drop/hole.
    obstacle_height: float = 0.15
    drop_height: float = 0.25
    min_ground_z_points: int = 200
    ground_z_source_rows: float = 0.40  # fallback: bottom fraction of image
    # Ground-confirmed depth rays make the raw grid a visibility map rather
    # than sparse isolated point hits.  Rays ending on obstacles are not used
    # to invent free ground; cells behind an endpoint stay unknown.
    visibility_raycast: bool = True
    visibility_raycast_stride_px: int = 8


@dataclass
class BEVGrid:
    traversability: np.ndarray   # float32 [nx, ny], 0..1
    observed: np.ndarray         # bool [nx, ny]
    obstacle: np.ndarray         # bool [nx, ny], hard geometric evidence
    observed_count: np.ndarray   # int32 [nx, ny]
    ground_count: np.ndarray     # int32 [nx, ny]
    obstacle_count: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    cfg: BEVConfig = field(default_factory=BEVConfig)
    ground_z: float = float("nan")            # estimated ground height in planner frame
    obstacle_pixels: Optional[np.ndarray] = None  # bool (H, W) image-space evidence

    @property
    def shape(self) -> Tuple[int, int]:
        return self.traversability.shape

    def xy_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not (self.cfg.x_min <= x < self.cfg.x_max):
            return None
        if not (self.cfg.y_min <= y < self.cfg.y_max):
            return None
        i = int((x - self.cfg.x_min) / self.cfg.resolution)
        j = int((y - self.cfg.y_min) / self.cfg.resolution)
        if i < 0 or j < 0 or i >= self.shape[0] or j >= self.shape[1]:
            return None
        return i, j

    def ij_to_xy(self, i: int, j: int) -> Tuple[float, float]:
        x = self.cfg.x_min + (float(i) + 0.5) * self.cfg.resolution
        y = self.cfg.y_min + (float(j) + 0.5) * self.cfg.resolution
        return x, y

    def nearest_ground_x(self) -> Optional[float]:
        """Forward distance of the closest observed traversable cell.

        A forward-facing camera mounted ~0.45 m above ground cannot observe
        the ground right in front of the robot; everything closer than this
        distance is genuinely unknown (not obstacle, not free). The planner
        uses this value to relax surface checks inside the blind band.
        """
        sel = np.argwhere(self.observed & (self.traversability >= 0.5))
        if sel.size == 0:
            return None
        i0 = int(sel[:, 0].min())
        return self.cfg.x_min + (float(i0) + 0.5) * self.cfg.resolution


def transform_to_matrix(transform) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4 homogeneous matrix."""
    t = transform.translation
    q = transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = max((x * x + y * y + z * z + w * w) ** 0.5, 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
    return T


def _ellipse_kernel(radius_cells: int) -> np.ndarray:
    radius_cells = max(0, int(radius_cells))
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = 2 * radius_cells + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def estimate_ground_z(
    z_pts: np.ndarray,
    ground_sel: np.ndarray,
    v_pix: np.ndarray,
    image_height: int,
    cfg: BEVConfig,
) -> float:
    """Robust ground-plane height (planner-frame z) from projected points."""
    if ground_sel.any() and int(ground_sel.sum()) >= cfg.min_ground_z_points:
        return float(np.median(z_pts[ground_sel]))
    # Fallback: the bottom of a forward-facing image is dominated by ground.
    bottom = v_pix >= (1.0 - cfg.ground_z_source_rows) * image_height
    if bottom.any():
        return float(np.median(z_pts[bottom]))
    return float("nan")


def _mark_ground_visibility_rays(
    support: np.ndarray,
    endpoints_i: np.ndarray,
    endpoints_j: np.ndarray,
    origin_ij: Optional[Tuple[int, int]],
    sample_stride: int,
) -> None:
    """Mark cells visible along sparse, ground-confirmed depth rays.

    This is deliberately not an aggressive occupancy-ray model: only rays
    whose endpoint was classified as geometric + visual ground are used.  An
    obstacle ray therefore does not hallucinate traversable floor, and cells
    beyond every endpoint remain UNKNOWN.
    """
    if origin_ij is None or endpoints_i.size == 0:
        return
    oi, oj = origin_ij
    stride = max(1, int(sample_stride))
    for i, j in zip(endpoints_i[::stride], endpoints_j[::stride]):
        cv2.line(support, (int(oj), int(oi)), (int(j), int(i)), 1, 1)


def build_local_bev(
    traversable_mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_planner_from_camera: np.ndarray,
    cfg: BEVConfig,
    obstacle_mask: Optional[np.ndarray] = None,
) -> BEVGrid:
    """Project registered depth + traversability mask into a metric BEV.

    Obstacle evidence is primarily *geometric*: any 3-D point that sits more
    than ``cfg.obstacle_height`` above the estimated ground plane is an
    obstacle, regardless of the DINO mask (a wall bottom that visually looks
    like floor still blocks the robot). DINO non-ground pixels only lower a
    cell's traversability; they become hard obstacles later via cluster
    promotion in :mod:`dinov3_nav.footprint`.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    h, w = depth.shape
    if traversable_mask.shape[:2] != (h, w):
        raise ValueError("traversable mask must match depth shape")
    if obstacle_mask is not None and obstacle_mask.shape[:2] != (h, w):
        raise ValueError("obstacle mask must match depth shape")

    nx = int(ceil((cfg.x_max - cfg.x_min) / cfg.resolution))
    ny = int(ceil((cfg.y_max - cfg.y_min) / cfg.resolution))
    observed_count = np.zeros((nx, ny), np.int32)
    ground_count = np.zeros((nx, ny), np.int32)
    obstacle_count = np.zeros((nx, ny), np.int32)
    obstacle_pixels = np.zeros((h, w), bool)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("invalid camera intrinsics")

    stride = max(1, int(cfg.pixel_stride))
    vv = np.arange(0, h, stride, dtype=np.int32)
    uu = np.arange(0, w, stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)
    Z = depth[V, U].astype(np.float32)
    valid = np.isfinite(Z) & (Z >= cfg.min_depth) & (Z <= cfg.max_depth)

    if valid.any():
        u = U[valid]              # int32 pixel column, reused for mask indexing
        v = V[valid]              # int32 pixel row
        z = Z[valid]
        pts_cam = np.stack(
            [(u.astype(np.float32) - cx) * z / fx,
             (v.astype(np.float32) - cy) * z / fy, z],
            axis=1,
        )
        R = T_planner_from_camera[:3, :3]
        t = T_planner_from_camera[:3, 3]
        pts = pts_cam @ R.T + t
        x, y, zp = pts[:, 0], pts[:, 1], pts[:, 2]

        inb = (
            (x >= cfg.x_min) & (x < cfg.x_max)
            & (y >= cfg.y_min) & (y < cfg.y_max)
        )
        u, v, zp = u[inb], v[inb], zp[inb]
        x, y = x[inb], y[inb]

        m_ground = traversable_mask[v, u] > 0
        m_obst_img = (
            obstacle_mask[v, u] > 0 if obstacle_mask is not None else np.zeros_like(m_ground)
        )

        ground_z = estimate_ground_z(zp, m_ground, v, h, cfg)
        if isnan(ground_z):
            height_obst = np.zeros_like(m_ground)
            drop = np.zeros_like(m_ground)
        else:
            height_obst = zp > (ground_z + cfg.obstacle_height)
            drop = zp < (ground_z - cfg.drop_height)
        pix_obst = m_obst_img | height_obst

        ii = ((x - cfg.x_min) / cfg.resolution).astype(np.int32)
        jj = ((y - cfg.y_min) / cfg.resolution).astype(np.int32)
        good = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        ii, jj = ii[good], jj[good]
        ground_evidence = (m_ground & ~height_obst & ~drop)[good]
        pix_obst = pix_obst[good]

        np.add.at(observed_count, (ii, jj), 1)
        np.add.at(ground_count, (ii, jj), ground_evidence.astype(np.int32))
        np.add.at(obstacle_count, (ii, jj), pix_obst.astype(np.int32))
        obstacle_pixels[v[good], u[good]] = pix_obst
    else:
        ground_z = float("nan")

    observed_raw = observed_count >= max(1, int(cfg.min_observed_points))
    traversability = np.zeros((nx, ny), np.float32)
    np.divide(
        ground_count,
        np.maximum(observed_count, 1),
        out=traversability,
        where=observed_count > 0,
    )
    obstacle = obstacle_count >= max(1, int(cfg.min_obstacle_points))

    if cfg.visibility_raycast and valid.any():
        # Camera origin projected onto the local x-y grid.  It is only a ray
        # source; marking starts at this cell and never overrides hard hits.
        ox, oy = float(T_planner_from_camera[0, 3]), float(T_planner_from_camera[1, 3])
        oi = int((ox - cfg.x_min) / cfg.resolution)
        oj = int((oy - cfg.y_min) / cfg.resolution)
        origin = (oi, oj) if 0 <= oi < nx and 0 <= oj < ny else None
        visible_ground = np.zeros((nx, ny), np.uint8)
        # Original point stride is already applied above; this second stride
        # bounds ray work independently of image resolution.
        sample_stride = max(1, int(round(cfg.visibility_raycast_stride_px / max(cfg.pixel_stride, 1))))
        _mark_ground_visibility_rays(
            visible_ground, ii[ground_evidence], jj[ground_evidence], origin, sample_stride
        )
        ray_only = (visible_ground > 0) & ~observed_raw & ~obstacle
        observed_raw |= visible_ground.astype(bool)
        traversability[ray_only] = float(cfg.support_traversability)

    # RGB-D points do not land on every 5 cm cell. Closing the observed ground
    # support prevents one empty raster cell from invalidating a whole path.
    close_cells = int(round(cfg.close_radius_m / max(cfg.resolution, 1e-6)))
    if close_cells > 0:
        kernel = _ellipse_kernel(close_cells)
        observed_closed = cv2.morphologyEx(
            observed_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ) > 0
        ground_seed = (observed_raw & (traversability >= 0.5)).astype(np.uint8)
        ground_closed = cv2.morphologyEx(ground_seed, cv2.MORPH_CLOSE, kernel) > 0
        new_support = ground_closed & ~observed_raw
        observed = observed_closed
        traversability[new_support] = np.maximum(
            traversability[new_support], float(cfg.support_traversability)
        )
    else:
        observed = observed_raw

    # Explicit obstacle evidence must never be filled back as ground support.
    traversability[obstacle] = 0.0
    observed[obstacle] = True

    return BEVGrid(
        traversability=traversability,
        observed=observed,
        obstacle=obstacle,
        observed_count=observed_count,
        ground_count=ground_count,
        obstacle_count=obstacle_count,
        cfg=cfg,
        ground_z=ground_z,
        obstacle_pixels=obstacle_pixels,
    )

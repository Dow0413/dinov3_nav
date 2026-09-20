# dinov3_nav/local_planner.py

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi

import cv2
import numpy as np

from .bev import BEVGrid
from .footprint import FootprintResult


def wrap_angle(a: float) -> float:
    return (a + pi) % (2.0 * pi) - pi


@dataclass
class PlannerConfig:
    horizon: float = 2.0
    dt: float = 0.1

    v_samples: tuple = (0.15, 0.30, 0.45)
    w_samples: tuple = (
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    )

    # 相机通常看不到狗身体正下方。
    # 前 start_ignore_distance 米不要求 observed，
    # 但依旧检查明显障碍。
    start_ignore_distance: float = 0.25

    clearance_cap: float = 0.80

    progress_weight: float = 2.0
    traversability_weight: float = 1.0
    clearance_weight: float = 1.2
    heading_weight: float = 0.8
    turn_weight: float = 0.25
    smooth_weight: float = 0.20

    max_angular: float = 0.8

    turn_in_place_angle: float = 0.61  # about 35 deg
    turn_gain: float = 1.2


@dataclass
class Trajectory:
    v: float
    w: float
    poses: np.ndarray     # Nx3: x, y, yaw
    score: float


class LocalTrajectoryPlanner:

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg

    def _rollout(
        self,
        v: float,
        w: float,
    ) -> np.ndarray:

        n = max(
            1,
            int(round(self.cfg.horizon / self.cfg.dt)),
        )

        poses = np.zeros((n + 1, 3), dtype=np.float32)

        x = 0.0
        y = 0.0
        yaw = 0.0

        for k in range(1, n + 1):

            # 当前朝向积分。
            x += v * cos(yaw) * self.cfg.dt
            y += v * np.sin(yaw) * self.cfg.dt
            yaw = wrap_angle(yaw + w * self.cfg.dt)

            poses[k] = (x, y, yaw)

        return poses

    def plan(
        self,
        bev: BEVGrid,
        footprint: FootprintResult,
        goal_xy,
        previous_w: float = 0.0,
    ) -> Trajectory | None:

        gx, gy = goal_xy

        d0 = hypot(gx, gy)

        if d0 < 1e-3:
            return None

        goal_bearing = atan2(gy, gx)

        # 目标大角度位于侧面时，先原地调整。
        #
        # 由于 footprint 用的是机器狗外接圆，
        # 原地旋转不会超出这个圆，因此只要当前位置本身安全即可。
        if abs(goal_bearing) >= self.cfg.turn_in_place_angle:

            w = float(
                np.clip(
                    self.cfg.turn_gain * goal_bearing,
                    -self.cfg.max_angular,
                    self.cfg.max_angular,
                )
            )

            return Trajectory(
                v=0.0,
                w=w,
                poses=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                score=0.0,
            )

        # 到膨胀后障碍的净空。
        free_for_distance = (
            ~footprint.inflated_blocked
        ).astype(np.uint8)

        clearance_map = cv2.distanceTransform(
            free_for_distance,
            cv2.DIST_L2,
            5,
        ).astype(np.float32)

        clearance_map *= bev.cfg.resolution

        max_forward = max(
            max(self.cfg.v_samples) * self.cfg.horizon,
            0.1,
        )

        max_w = max(
            max(abs(float(w)) for w in self.cfg.w_samples),
            0.1,
        )

        best = None

        for v in self.cfg.v_samples:
            for w in self.cfg.w_samples:

                v = float(v)
                w = float(w)

                poses = self._rollout(v, w)

                valid = True

                trav_values = []
                clear_values = []

                for x, y, _ in poses[1:]:

                    cell = bev.xy_to_ij(
                        float(x),
                        float(y),
                    )

                    if cell is None:
                        valid = False
                        break

                    i, j = cell

                    # 机身 footprint 硬约束。
                    if footprint.inflated_blocked[i, j]:
                        valid = False
                        break

                    # 相机脚下存在天然盲区。
                    if x >= self.cfg.start_ignore_distance:
                        if not footprint.safe[i, j]:
                            valid = False
                            break

                        trav_values.append(
                            float(bev.traversability[i, j])
                        )

                    clear_values.append(
                        min(
                            float(clearance_map[i, j])
                            / max(self.cfg.clearance_cap, 1e-3),
                            1.0,
                        )
                    )

                if not valid:
                    continue

                xn = float(poses[-1, 0])
                yn = float(poses[-1, 1])
                yaw_n = float(poses[-1, 2])

                dn = hypot(
                    gx - xn,
                    gy - yn,
                )

                # 真正的 global goal progress。
                progress = (d0 - dn) / max_forward

                if trav_values:
                    traversability = float(np.mean(trav_values))
                else:
                    traversability = 1.0

                if clear_values:
                    clearance = float(np.mean(clear_values))
                else:
                    clearance = 0.0

                desired_heading_end = atan2(
                    gy - yn,
                    gx - xn,
                )

                heading = cos(
                    wrap_angle(
                        desired_heading_end - yaw_n
                    )
                )

                turn_cost = abs(w) / max_w

                smooth_cost = (
                    abs(w - previous_w)
                    / (2.0 * max_w)
                )

                score = (
                    self.cfg.progress_weight * progress
                    + self.cfg.traversability_weight * traversability
                    + self.cfg.clearance_weight * clearance
                    + self.cfg.heading_weight * heading
                    - self.cfg.turn_weight * turn_cost
                    - self.cfg.smooth_weight * smooth_cost
                )

                candidate = Trajectory(
                    v=v,
                    w=w,
                    poses=poses,
                    score=float(score),
                )

                if best is None or candidate.score > best.score:
                    best = candidate

        return best
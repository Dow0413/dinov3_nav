# -*- coding: utf-8 -*-
"""Footprint-aware feasibility checks on the local BEV."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, sin
from typing import Tuple

import cv2
import numpy as np

from .bev import BEVGrid


@dataclass
class FootprintConfig:
    length: float = 0.65
    width: float = 0.40
    safety_margin: float = 0.08
    # Body center expressed in planner frame at robot pose (0,0,0).
    # If planner_frame is base_link these are normally 0.
    center_x: float = 0.0
    center_y: float = 0.0
    sample_step: float = 0.05
    traversability_threshold: float = 0.50
    hard_nontrav_threshold: float = 0.15
    hard_nontrav_min_cells: int = 3
    max_unknown_fraction: float = 0.40
    min_ground_fraction: float = 0.55
    min_mean_traversability: float = 0.45


@dataclass
class FootprintLayers:
    hard_blocked: np.ndarray
    inflated_hard_blocked: np.ndarray
    clearance_m: np.ndarray


@dataclass
class FootprintCheck:
    valid: bool
    reason: str
    unknown_fraction: float
    ground_fraction: float
    mean_traversability: float


def _remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(1, int(min_cells)):
            out[labels == label] = True
    return out


def _inflate(mask: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    cells = int(ceil(max(0.0, radius_m) / max(resolution, 1e-6)))
    if cells <= 0:
        return mask.copy()
    size = 2 * cells + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def build_footprint_layers(bev: BEVGrid, cfg: FootprintConfig) -> FootprintLayers:
    """Build hard collision layers.

    DINO non-traversable cells are not blindly inflated one-by-one. Only small-
    threshold *clusters* are promoted to hard obstacles, which prevents a sparse
    segmentation hole from sealing an otherwise open corridor.
    """
    strong_nontrav = (
        bev.observed
        & (bev.traversability <= float(cfg.hard_nontrav_threshold))
        & (bev.observed_count > 0)
    )
    strong_nontrav = _remove_small_components(strong_nontrav, cfg.hard_nontrav_min_cells)
    hard = bev.obstacle | strong_nontrav
    inflated = _inflate(hard, cfg.safety_margin, bev.cfg.resolution)

    free = (~inflated).astype(np.uint8)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)
    clearance *= bev.cfg.resolution
    # Open maps have no zero cell and OpenCV returns FLT_MAX. Clearance is a
    # metric costmap field, so cap it at the local-map diagonal.
    clearance = np.minimum(clearance, hypot(*bev.shape) * bev.cfg.resolution)
    return FootprintLayers(
        hard_blocked=hard,
        inflated_hard_blocked=inflated,
        clearance_m=clearance,
    )


class FootprintChecker:
    """Checks a rectangular robot footprint at arbitrary local SE(2) poses."""

    def __init__(self, bev: BEVGrid, cfg: FootprintConfig, layers: FootprintLayers):
        self.bev = bev
        self.cfg = cfg
        self.layers = layers
        half_l = max(0.01, cfg.length * 0.5)
        half_w = max(0.01, cfg.width * 0.5)
        step = max(0.025, min(float(cfg.sample_step), bev.cfg.resolution))
        xs = np.arange(-half_l, half_l + 0.5 * step, step, dtype=np.float32)
        ys = np.arange(-half_w, half_w + 0.5 * step, step, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        self._body_points = np.stack(
            [X.reshape(-1) + cfg.center_x, Y.reshape(-1) + cfg.center_y], axis=1
        )

    def _footprint_indices(self, x: float, y: float, yaw: float):
        c, s = cos(yaw), sin(yaw)
        bx = self._body_points[:, 0]
        by = self._body_points[:, 1]
        wx = x + c * bx - s * by
        wy = y + s * bx + c * by
        ii = ((wx - self.bev.cfg.x_min) / self.bev.cfg.resolution).astype(np.int32)
        jj = ((wy - self.bev.cfg.y_min) / self.bev.cfg.resolution).astype(np.int32)
        inside = (
            (ii >= 0) & (ii < self.bev.shape[0])
            & (jj >= 0) & (jj < self.bev.shape[1])
        )
        return ii, jj, inside

    def inflated_cells_under(
        self, x: float, y: float, yaw: float
    ) -> "np.ndarray | None":
        """Linear indices of inflated cells covered by the footprint.

        None when the footprint leaves the BEV. Used for margin containment:
        once the robot stands inside the inflation ring, poses may keep the
        cells they already cover but must not acquire new ones.
        """
        ii, jj, inside = self._footprint_indices(x, y, yaw)
        if not inside.all():
            return None
        mask = self.layers.inflated_hard_blocked[ii, jj]
        cols = int(self.bev.shape[1])
        return (
            ii[mask].astype(np.int64) * cols + jj[mask].astype(np.int64)
        )

    def check_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        relax_surface: bool = False,
        ignore_inflation: bool = False,
        unknown_is_soft: bool = False,
    ) -> FootprintCheck:
        """Pose feasibility for the rectangular footprint.

        Hard collision is always against raw obstacle evidence. The inflated
        layer (safety margin) is a planning comfort distance, not physics: a
        robot that is already standing inside the inflation ring must still be
        allowed to rotate/back away, so recovery code passes
        ``ignore_inflation=True`` and keeps only the hard constraint.
        """
        ii, jj, inside = self._footprint_indices(x, y, yaw)
        if not inside.all():
            return FootprintCheck(False, "out_of_bounds", 1.0, 0.0, 0.0)

        blocked = self.layers.hard_blocked if ignore_inflation \
            else self.layers.inflated_hard_blocked
        if blocked[ii, jj].any():
            return FootprintCheck(False, "collision", 0.0, 0.0, 0.0)

        observed = self.bev.observed[ii, jj]
        unknown_fraction = float((~observed).mean())
        if relax_surface:
            # Around the current pose, the front camera naturally cannot observe
            # all cells under/behind the robot. Collision remains a hard check.
            return FootprintCheck(True, "ok", unknown_fraction, 1.0, 1.0)

        if not unknown_is_soft and unknown_fraction > self.cfg.max_unknown_fraction:
            return FootprintCheck(False, "unknown", unknown_fraction, 0.0, 0.0)

        observed_idx = observed
        if not observed_idx.any():
            return FootprintCheck(False, "unknown", 1.0, 0.0, 0.0)

        trav = self.bev.traversability[ii[observed_idx], jj[observed_idx]]
        ground_fraction = float((trav >= self.cfg.traversability_threshold).mean())
        mean_trav = float(trav.mean()) if trav.size else 0.0
        if ground_fraction < self.cfg.min_ground_fraction:
            return FootprintCheck(False, "not_traversable", unknown_fraction, ground_fraction, mean_trav)
        if mean_trav < self.cfg.min_mean_traversability:
            return FootprintCheck(False, "not_traversable", unknown_fraction, ground_fraction, mean_trav)
        return FootprintCheck(True, "ok", unknown_fraction, ground_fraction, mean_trav)

    def clearance_at(self, x: float, y: float) -> float:
        cell = self.bev.xy_to_ij(x, y)
        if cell is None:
            return 0.0
        return float(self.layers.clearance_m[cell])

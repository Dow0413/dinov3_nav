# -*- coding: utf-8 -*-
"""Stable local planning BEV built from raw RGB-D observations.

``build_local_bev`` intentionally remains a per-frame sensor product.  This
module turns it into a planning product: old evidence is motion-compensated
into the current ``base_link`` frame, decayed, combined with the raw frame,
then converted to FREE / OBSTACLE / UNKNOWN plus an inflated costmap.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional

import cv2
import numpy as np

from .bev import BEVGrid
from .footprint import FootprintConfig, FootprintLayers, build_footprint_layers


@dataclass
class TemporalBEVConfig:
    enabled: bool = True
    evidence_half_life_s: float = 0.70
    max_evidence: float = 6.0
    observation_hit: float = 1.0
    obstacle_hit: float = 2.0
    obstacle_clear: float = 1.5
    min_observed_evidence: float = 0.50
    obstacle_evidence_threshold: float = 1.25
    cleanup_min_obstacle_cells: int = 2
    ego_clear_radius_m: float = 0.30
    ego_clear_traversability: float = 0.80


@dataclass
class PlanningCostConfig:
    unknown_cost: float = 3.0
    nontraversable_cost: float = 6.0
    clearance_target_m: float = 0.45
    clearance_cost_weight: float = 4.0


@dataclass
class PlanningBEV:
    """Stable map and fields which a future A*/GDF can consume directly."""
    grid: BEVGrid
    free: np.ndarray
    unknown: np.ndarray
    hard_obstacle: np.ndarray
    inflated_obstacle: np.ndarray
    clearance_m: np.ndarray
    planning_cost: np.ndarray
    layers: FootprintLayers


def _remove_tiny_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_cells:
            out[labels == label] = True
    return out


class TemporalBEVFusion:
    """Evidence fusion in a robot-centred grid with SE(2) reprojection."""

    def __init__(self, cfg: TemporalBEVConfig):
        self.cfg = cfg
        self._observed_weight: Optional[np.ndarray] = None
        self._traversability_sum: Optional[np.ndarray] = None
        self._obstacle_evidence: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._observed_weight = None
        self._traversability_sum = None
        self._obstacle_evidence = None

    @staticmethod
    def _warp_to_current(values: np.ndarray, grid: BEVGrid,
                         T_current_from_previous: np.ndarray) -> np.ndarray:
        """Reproject a previous base_link grid into the current base_link."""
        nx, ny = grid.shape
        xs = grid.cfg.x_min + (np.arange(nx, dtype=np.float32) + 0.5) * grid.cfg.resolution
        ys = grid.cfg.y_min + (np.arange(ny, dtype=np.float32) + 0.5) * grid.cfg.resolution
        x_cur, y_cur = np.meshgrid(xs, ys, indexing="ij")
        T_prev_from_current = np.linalg.inv(T_current_from_previous)
        x_prev = T_prev_from_current[0, 0] * x_cur + T_prev_from_current[0, 1] * y_cur + T_prev_from_current[0, 2]
        y_prev = T_prev_from_current[1, 0] * x_cur + T_prev_from_current[1, 1] * y_cur + T_prev_from_current[1, 2]
        # cv2.remap coordinates are (source column=j, source row=i).
        map_x = (y_prev - grid.cfg.y_min) / grid.cfg.resolution - 0.5
        map_y = (x_prev - grid.cfg.x_min) / grid.cfg.resolution - 0.5
        return cv2.remap(
            values.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def update(
        self,
        raw: BEVGrid,
        dt_s: float,
        T_current_from_previous: Optional[np.ndarray],
    ) -> BEVGrid:
        """Fuse a raw frame. Missing motion compensation safely resets state."""
        shape = raw.shape
        if (not self.cfg.enabled or self._observed_weight is None
                or self._observed_weight.shape != shape
                or T_current_from_previous is None):
            old_obs = np.zeros(shape, np.float32)
            old_trav = np.zeros(shape, np.float32)
            old_obstacle = np.zeros(shape, np.float32)
        else:
            half_life = max(self.cfg.evidence_half_life_s, 1e-3)
            decay = exp(-max(0.0, dt_s) * np.log(2.0) / half_life)
            old_obs = self._warp_to_current(self._observed_weight, raw, T_current_from_previous) * decay
            old_trav = self._warp_to_current(self._traversability_sum, raw, T_current_from_previous) * decay
            old_obstacle = self._warp_to_current(self._obstacle_evidence, raw, T_current_from_previous) * decay

        raw_observed = raw.observed.astype(np.float32)
        raw_free = raw.observed & (raw.traversability >= 0.5) & ~raw.obstacle
        observed_weight = np.minimum(
            self.cfg.max_evidence, old_obs + self.cfg.observation_hit * raw_observed
        )
        traversability_sum = np.minimum(
            self.cfg.max_evidence, old_trav + self.cfg.observation_hit * raw_observed * raw.traversability
        )
        obstacle_evidence = old_obstacle.copy()
        obstacle_evidence[raw_free] = np.maximum(
            0.0, obstacle_evidence[raw_free] - self.cfg.obstacle_clear
        )
        obstacle_evidence[raw.obstacle] = np.minimum(
            self.cfg.max_evidence,
            obstacle_evidence[raw.obstacle] + self.cfg.obstacle_hit,
        )
        obstacle = obstacle_evidence >= self.cfg.obstacle_evidence_threshold
        # Remove only weak singleton fragments. A current geometric obstacle
        # hit is already strong evidence and must never disappear merely
        # because it occupies one 5-cm cell.
        weak = obstacle & (obstacle_evidence < 1.5 * self.cfg.obstacle_evidence_threshold)
        obstacle = (obstacle & ~weak) | _remove_tiny_components(
            weak, self.cfg.cleanup_min_obstacle_cells
        )
        obstacle_evidence[~obstacle] = 0.0

        observed = observed_weight >= self.cfg.min_observed_evidence
        traversability = np.zeros(shape, np.float32)
        np.divide(traversability_sum, np.maximum(observed_weight, 1e-6),
                  out=traversability, where=observed_weight > 0.0)

        # The forward camera cannot see below the robot.  Supply a compact
        # start island only for previously UNKNOWN cells; never erase actual
        # obstacle evidence or create a long artificial corridor.
        nx, ny = shape
        xs = raw.cfg.x_min + (np.arange(nx, dtype=np.float32) + 0.5) * raw.cfg.resolution
        ys = raw.cfg.y_min + (np.arange(ny, dtype=np.float32) + 0.5) * raw.cfg.resolution
        x, y = np.meshgrid(xs, ys, indexing="ij")
        ego = (x * x + y * y) <= self.cfg.ego_clear_radius_m ** 2
        ego_fill = ego & ~observed & ~obstacle
        observed[ego_fill] = True
        traversability[ego_fill] = self.cfg.ego_clear_traversability
        observed[obstacle] = True
        traversability[obstacle] = 0.0

        self._observed_weight = observed_weight.astype(np.float32)
        self._traversability_sum = traversability_sum.astype(np.float32)
        self._obstacle_evidence = obstacle_evidence.astype(np.float32)
        counts = np.rint(observed_weight).astype(np.int32)
        ground_counts = np.rint(traversability * observed_weight).astype(np.int32)
        obstacle_counts = np.rint(obstacle_evidence).astype(np.int32)
        return BEVGrid(
            traversability=traversability,
            observed=observed,
            obstacle=obstacle,
            observed_count=counts,
            ground_count=ground_counts,
            obstacle_count=obstacle_counts,
            cfg=raw.cfg,
            ground_z=raw.ground_z,
            obstacle_pixels=raw.obstacle_pixels,
        )


def build_planning_bev(
    stable: BEVGrid,
    footprint_cfg: FootprintConfig,
    cost_cfg: PlanningCostConfig,
) -> PlanningBEV:
    """Build footprint-inflated, clearance-weighted planning fields."""
    layers = build_footprint_layers(stable, footprint_cfg)
    unknown = ~stable.observed
    hard = layers.hard_blocked
    inflated = layers.inflated_hard_blocked
    free = stable.observed & (stable.traversability >= footprint_cfg.traversability_threshold) & ~inflated
    clearance = layers.clearance_m
    cost = np.full(stable.shape, 1.0, np.float32)
    cost[unknown] = float(cost_cfg.unknown_cost)
    nontrav = stable.observed & ~hard & (stable.traversability < footprint_cfg.traversability_threshold)
    cost[nontrav] = float(cost_cfg.nontraversable_cost)
    target = max(float(cost_cfg.clearance_target_m), 1e-3)
    clearance_penalty = float(cost_cfg.clearance_cost_weight) * np.clip(
        (target - clearance) / target, 0.0, 1.0
    )
    cost += clearance_penalty.astype(np.float32)
    cost[inflated] = np.inf
    return PlanningBEV(
        grid=stable,
        free=free,
        unknown=unknown,
        hard_obstacle=hard,
        inflated_obstacle=inflated,
        clearance_m=clearance,
        planning_cost=cost,
        layers=layers,
    )

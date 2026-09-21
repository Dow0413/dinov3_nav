# -*- coding: utf-8 -*-
"""Grid-distance-field local planner.

This module deliberately contains no trajectory rollout.  GDF answers which
free-space branch reaches the local goal; SDF only supplies obstacle clearance
and a local repulsive correction to that GDF direction.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bev import BEVGrid
from .footprint import FootprintLayers


def wrap_angle(a: float) -> float:
    return (a + pi) % (2.0 * pi) - pi


@dataclass
class GDFPlannerConfig:
    traversability_threshold: float = 0.50
    unknown_cost: float = 2.0
    goal_search_radius_m: float = 1.0
    lookahead_m: float = 0.70
    clearance_target_m: float = 0.45
    clearance_emergency_m: float = 0.18
    clearance_gain: float = 1.25
    cruise_linear: float = 0.35
    max_angular: float = 1.2
    angular_gain: float = 1.5
    turn_in_place_angle: float = 0.75
    side_lock_s: float = 1.5
    side_hysteresis_m: float = 0.15
    avoid_heading_delta: float = 0.20
    # GDF has already verified a free detour branch in AVOID. Permit a
    # slow forward arc for ordinary detour angles; otherwise a 45-degree
    # branch is a permanent turn-in-place fixed point.
    avoid_turn_in_place_angle: float = 1.30
    avoid_min_linear: float = 0.08
    corridor_lookahead_m: float = 1.20
    corridor_half_width_m: float = 0.40
    corridor_emergency_m: float = 0.30
    avoid_exit_clear_frames: int = 6


@dataclass
class GDFPlanResult:
    command: Tuple[float, float]
    mode: str
    desired_heading: Optional[float]
    gdf_heading: Optional[float]
    clearance_m: float
    gdf: np.ndarray
    sdf: np.ndarray
    free: np.ndarray
    seed_ij: Optional[Tuple[int, int]]
    start_ij: Optional[Tuple[int, int]]
    path_xy: List[Tuple[float, float]] = field(default_factory=list)
    side: int = 0                    # +1 left, -1 right
    diagnostics: Dict[str, float] = field(default_factory=dict)
    corridor_blocked: bool = False
    goal_corridor_clear: bool = False
    exit_clear_frames: int = 0
    forward_corridor_xy: List[Tuple[float, float]] = field(default_factory=list)
    goal_corridor_xy: List[Tuple[float, float]] = field(default_factory=list)


_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


class GDFPlanner:
    """GDF branch selection, SDF clearance correction and simple control."""

    def __init__(self, cfg: GDFPlannerConfig):
        self.cfg = cfg
        self._side = 0
        self._side_until = 0.0
        self._mode = "GO_TO_GOAL"
        self._exit_clear_frames = 0

    def _free_map(self, bev: BEVGrid, layers: FootprintLayers) -> np.ndarray:
        """Unknown is traversable with a cost, never an automatic obstacle.

        Observed low-traversability cells are blocked unless they are already
        covered by the hard/inflated layer.  This retains DINO evidence while
        avoiding a raster of unknown depth holes sealing the local map.
        """
        observed_free = bev.observed & (bev.traversability >= self.cfg.traversability_threshold)
        unknown = ~bev.observed
        return (~layers.inflated_hard_blocked) & (observed_free | unknown)

    def _corridor_mask(self, bev: BEVGrid, heading: float) -> np.ndarray:
        """Cells swept by a straight rectangular body corridor.

        The corridor is in the robot's current local frame.  Rotating its
        longitudinal axis by ``heading`` is used only for the AVOID exit
        condition, to ensure the global-goal direction is genuinely open.
        """
        nx, ny = bev.shape
        xs = bev.cfg.x_min + (np.arange(nx, dtype=np.float32) + 0.5) * bev.cfg.resolution
        ys = bev.cfg.y_min + (np.arange(ny, dtype=np.float32) + 0.5) * bev.cfg.resolution
        x, y = np.meshgrid(xs, ys, indexing="ij")
        c, s = cos(heading), sin(heading)
        forward = c * x + s * y
        lateral = -s * x + c * y
        return ((forward >= 0.0) & (forward <= self.cfg.corridor_lookahead_m)
                & (np.abs(lateral) <= self.cfg.corridor_half_width_m))

    def _corridor_polygon(self, heading: float) -> List[Tuple[float, float]]:
        """Rectangle corners in base_link for debug rendering."""
        c, s = cos(heading), sin(heading)
        half = self.cfg.corridor_half_width_m
        out = []
        for forward, lateral in ((0.0, -half), (self.cfg.corridor_lookahead_m, -half),
                                 (self.cfg.corridor_lookahead_m, half), (0.0, half)):
            out.append((c * forward - s * lateral, s * forward + c * lateral))
        return out

    def _corridor_is_blocked(self, bev: BEVGrid, layers: FootprintLayers,
                             heading: float) -> Tuple[bool, bool]:
        """Return (blocked, emergency) for a footprint-width forward corridor."""
        corridor = self._corridor_mask(bev, heading)
        blocked = bool((corridor & layers.inflated_hard_blocked).any())
        # Emergency is intentionally raw obstacle evidence close to the body;
        # it remains a fail-safe independent of planner state transitions.
        nx, ny = bev.shape
        xs = bev.cfg.x_min + (np.arange(nx, dtype=np.float32) + 0.5) * bev.cfg.resolution
        ys = bev.cfg.y_min + (np.arange(ny, dtype=np.float32) + 0.5) * bev.cfg.resolution
        x, y = np.meshgrid(xs, ys, indexing="ij")
        c, s = cos(heading), sin(heading)
        forward = c * x + s * y
        lateral = -s * x + c * y
        emergency_zone = ((forward >= 0.0) & (forward <= self.cfg.corridor_emergency_m)
                          & (np.abs(lateral) <= self.cfg.corridor_half_width_m))
        emergency = bool((emergency_zone & layers.hard_blocked).any())
        return blocked, emergency

    def _command_for_heading(self, heading: float, clearance: float,
                             emergency: bool, allow_avoid_arc: bool = False) -> Tuple[float, float]:
        """Heading controller with a forward-arc mode for verified GDF paths."""
        angular = float(np.clip(self.cfg.angular_gain * heading,
                                -self.cfg.max_angular, self.cfg.max_angular))
        heading_scale = max(0.0, cos(heading))
        clearance_scale = float(np.clip(
            clearance / max(self.cfg.clearance_target_m, 1e-3), 0.0, 1.0
        ))
        linear = self.cfg.cruise_linear * heading_scale * clearance_scale
        turn_in_place = (self.cfg.avoid_turn_in_place_angle if allow_avoid_arc
                         else self.cfg.turn_in_place_angle)
        if allow_avoid_arc and not emergency and abs(heading) < turn_in_place:
            # A GDF descent is a valid free-space branch in the inflated map.
            # Keep a small, bounded arc speed even for a strong but feasible
            # turn, so the robot can make geometric progress around the wall.
            linear = max(linear, self.cfg.avoid_min_linear * clearance_scale)
        if emergency or abs(heading) >= turn_in_place:
            linear = 0.0
        return float(linear), angular

    @staticmethod
    def _sdf(free: np.ndarray, resolution: float) -> np.ndarray:
        sdf = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * resolution
        # Open maps have no zero pixel, and OpenCV then returns FLT_MAX.
        # SDF is a metric diagnostic/control input, so cap it at map diagonal.
        cap = hypot(*free.shape) * resolution
        return np.minimum(sdf, cap).astype(np.float32)

    def _nearest_free(self, bev: BEVGrid, free: np.ndarray, xy: Tuple[float, float],
                      max_radius: Optional[float] = None) -> Optional[Tuple[int, int]]:
        ii, jj = np.nonzero(free)
        if ii.size == 0:
            return None
        x = bev.cfg.x_min + (ii.astype(np.float32) + 0.5) * bev.cfg.resolution
        y = bev.cfg.y_min + (jj.astype(np.float32) + 0.5) * bev.cfg.resolution
        d2 = (x - xy[0]) ** 2 + (y - xy[1]) ** 2
        k = int(np.argmin(d2))
        if max_radius is not None and float(np.sqrt(d2[k])) > max_radius:
            return None
        return int(ii[k]), int(jj[k])

    def _edge_seed(self, bev: BEVGrid, free: np.ndarray,
                   goal_xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Choose a free BEV-edge seed closest to the global-goal ray."""
        gx, gy = goal_xy
        norm = hypot(gx, gy)
        if norm < 1e-6:
            return self._nearest_free(bev, free, goal_xy)
        # Intersect the ray with BEV bounds, then use the nearest free edge
        # cell. This keeps an out-of-map global goal as a directional seed.
        ts = []
        if gx > 1e-6:
            ts.append(bev.cfg.x_max / gx)
        elif gx < -1e-6:
            ts.append(bev.cfg.x_min / gx)
        if gy > 1e-6:
            ts.append(bev.cfg.y_max / gy)
        elif gy < -1e-6:
            ts.append(bev.cfg.y_min / gy)
        ts = [t for t in ts if t > 0.0]
        if not ts:
            return None
        target = (min(ts) * gx, min(ts) * gy)
        edge = np.zeros_like(free, bool)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        return self._nearest_free(bev, free & edge, target) or self._nearest_free(bev, free, target)

    def _seed(self, bev: BEVGrid, free: np.ndarray,
              goal_xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        if bev.xy_to_ij(*goal_xy) is not None:
            return self._nearest_free(bev, free, goal_xy, self.cfg.goal_search_radius_m)
        return self._edge_seed(bev, free, goal_xy)

    def _gdf(self, bev: BEVGrid, free: np.ndarray,
             seed: Tuple[int, int]) -> np.ndarray:
        """Dijkstra cost-to-go. Unknown is permitted but has an uncertainty cost."""
        shape = free.shape
        dist = np.full(shape, np.inf, np.float32)
        dist[seed] = 0.0
        heap = [(0.0, seed[0], seed[1])]
        res = bev.cfg.resolution
        while heap:
            cost, i, j = heapq.heappop(heap)
            if cost > float(dist[i, j]) + 1e-6:
                continue
            for di, dj in _NEIGHBORS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < shape[0] and 0 <= nj < shape[1]) or not free[ni, nj]:
                    continue
                step = res * (1.41421356 if di and dj else 1.0)
                uncertainty = self.cfg.unknown_cost if not bev.observed[ni, nj] else 1.0
                nc = cost + step * uncertainty
                if nc < float(dist[ni, nj]):
                    dist[ni, nj] = nc
                    heapq.heappush(heap, (nc, ni, nj))
        return dist

    def _descent_neighbors(self, gdf: np.ndarray, cell: Tuple[int, int]):
        i, j = cell
        cur = float(gdf[i, j])
        out = []
        for di, dj in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if 0 <= ni < gdf.shape[0] and 0 <= nj < gdf.shape[1] and float(gdf[ni, nj]) < cur - 1e-6:
                out.append((float(gdf[ni, nj]), ni, nj))
        return out

    def _trace(self, bev: BEVGrid, gdf: np.ndarray, start: Tuple[int, int],
               goal_heading: float, now: float, forced_side: int = 0) -> Tuple[List[Tuple[int, int]], int]:
        path = [start]
        current = start
        first_side = 0
        # Once the safety corridor has triggered, trace at least as far as
        # that corridor.  A shorter GDF trace can otherwise look straight
        # ahead at an obstacle that is already inside the trigger range and
        # postpone the first lateral decision until unnecessarily late.
        trace_m = max(self.cfg.lookahead_m, self.cfg.corridor_lookahead_m)
        max_steps = max(2, int(trace_m / bev.cfg.resolution))
        for step_idx in range(max_steps):
            choices = self._descent_neighbors(gdf, current)
            if not choices:
                break
            choices.sort()
            selected = choices[0]
            if step_idx == 0:
                left = [x for x in choices if x[2] > current[1]]
                right = [x for x in choices if x[2] < current[1]]
                raw_side = int(np.sign(selected[2] - current[1]))
                # In an AVOID state the selected side is a state-machine
                # commitment, not a soft per-frame score. Only pick another
                # branch when the committed side has literally no descent.
                if forced_side:
                    same = left if forced_side > 0 else right
                    if same:
                        selected = same[0]
                        raw_side = forced_side
                elif self._side != 0 and now < self._side_until and raw_side != self._side:
                    same = left if self._side > 0 else right
                    if same and float(same[0][0]) <= float(selected[0]) + self.cfg.side_hysteresis_m:
                        selected = same[0]
                        raw_side = self._side
                first_side = raw_side
            current = (selected[1], selected[2])
            path.append(current)
        # A committed AVOID side is an invariant of the state machine.  In
        # particular, do not clear it merely because a noisy/local GDF trace
        # briefly points at the global goal; only ``plan`` may clear it after
        # the multi-frame corridor exit condition has succeeded.
        if forced_side:
            self._side = forced_side
            self._side_until = now + self.cfg.side_lock_s
            return path, self._side

        # Before a side is committed, retain a detour only when GDF clearly
        # disagrees with the direct goal direction.
        if len(path) >= 2:
            x0, y0 = bev.ij_to_xy(*path[0])
            x1, y1 = bev.ij_to_xy(*path[-1])
            heading = atan2(y1 - y0, x1 - x0)
            path_side = int(np.sign(y1 - y0))
            if path_side and abs(wrap_angle(heading - goal_heading)) >= self.cfg.avoid_heading_delta:
                self._side = forced_side or first_side or path_side
                self._side_until = now + self.cfg.side_lock_s
            elif abs(wrap_angle(heading - goal_heading)) < 0.5 * self.cfg.avoid_heading_delta:
                self._side = 0
                self._side_until = now
        return path, self._side

    def plan(self, bev: BEVGrid, layers: FootprintLayers,
             goal_xy: Tuple[float, float], now: float) -> GDFPlanResult:
        """Stateful goal tracking with trigger-only GDF/SDF avoidance."""
        free = self._free_map(bev, layers)
        sdf = self._sdf(free, bev.cfg.resolution)
        start = self._nearest_free(bev, free, (0.0, 0.0), self.cfg.goal_search_radius_m)
        empty = np.full(bev.shape, np.inf, np.float32)
        goal_heading = atan2(goal_xy[1], goal_xy[0])
        forward_poly = self._corridor_polygon(0.0)
        goal_poly = self._corridor_polygon(goal_heading)
        forward_blocked, emergency = self._corridor_is_blocked(bev, layers, 0.0)
        goal_blocked, _ = self._corridor_is_blocked(bev, layers, goal_heading)
        if start is None:
            return GDFPlanResult((0.0, 0.0), "NO_FREE_START", None, None, 0.0,
                                 empty, sdf, free, None, None, corridor_blocked=forward_blocked,
                                 goal_corridor_clear=not goal_blocked,
                                 forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)
        clearance = float(sdf[start])

        # Normal navigation is direct global-goal tracking. GDF and SDF do
        # not influence steering here; SDF only reduces speed / stops on an
        # emergency obstacle.
        if self._mode == "GO_TO_GOAL" and not forward_blocked:
            self._side = 0
            self._exit_clear_frames = 0
            command = self._command_for_heading(goal_heading, clearance, emergency)
            return GDFPlanResult(command, "GO_TO_GOAL", goal_heading, None, clearance,
                                 empty, sdf, free, None, start, diagnostics={"gdf_active": 0.0},
                                 corridor_blocked=False, goal_corridor_clear=not goal_blocked,
                                 forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)

        if self._mode == "GO_TO_GOAL":
            self._mode = "AVOID"
            self._exit_clear_frames = 0
        elif not forward_blocked and not goal_blocked:
            self._exit_clear_frames += 1
            if self._exit_clear_frames >= max(1, int(self.cfg.avoid_exit_clear_frames)):
                self._mode = "GO_TO_GOAL"
                self._side = 0
                command = self._command_for_heading(goal_heading, clearance, emergency)
                return GDFPlanResult(command, "GO_TO_GOAL", goal_heading, None, clearance,
                                     empty, sdf, free, None, start, diagnostics={"gdf_active": 0.0},
                                     corridor_blocked=False, goal_corridor_clear=True,
                                     exit_clear_frames=self._exit_clear_frames,
                                     forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)
        else:
            self._exit_clear_frames = 0

        # AVOID: GDF selects the committed detour branch; SDF only bends that
        # branch away from nearby obstacles. The state remains AVOID until the
        # stable multi-frame exit condition above succeeds.
        seed = self._seed(bev, free, goal_xy)
        if seed is None:
            return GDFPlanResult((0.0, 0.0), "NO_FREE_SEED", None, None, clearance,
                                 empty, sdf, free, None, start, side=self._side,
                                 corridor_blocked=forward_blocked, goal_corridor_clear=not goal_blocked,
                                 exit_clear_frames=self._exit_clear_frames,
                                 forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)
        gdf = self._gdf(bev, free, seed)
        if not np.isfinite(gdf[start]):
            return GDFPlanResult((0.0, 0.0), "NO_GDF_PATH", None, None, clearance,
                                 gdf, sdf, free, seed, start, side=self._side,
                                 corridor_blocked=forward_blocked, goal_corridor_clear=not goal_blocked,
                                 exit_clear_frames=self._exit_clear_frames,
                                 forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)
        cells, side = self._trace(bev, gdf, start, goal_heading, now, forced_side=self._side)
        if len(cells) < 2:
            return GDFPlanResult((0.0, 0.0), "NO_GDF_DESCENT", None, None, clearance,
                                 gdf, sdf, free, seed, start, side=self._side,
                                 corridor_blocked=forward_blocked, goal_corridor_clear=not goal_blocked,
                                 exit_clear_frames=self._exit_clear_frames,
                                 forward_corridor_xy=forward_poly, goal_corridor_xy=goal_poly)
        x0, y0 = bev.ij_to_xy(*cells[0])
        x1, y1 = bev.ij_to_xy(*cells[-1])
        gdf_heading = atan2(y1 - y0, x1 - x0)
        if self._side == 0:
            self._side = int(np.sign(y1 - y0)) or (1 if goal_heading >= 0.0 else -1)
        gi = slice(max(0, start[0] - 1), min(bev.shape[0], start[0] + 2))
        gj = slice(max(0, start[1] - 1), min(bev.shape[1], start[1] + 2))
        dx, dy = np.gradient(sdf, bev.cfg.resolution)
        rep = np.array([float(dx[gi, gj].mean()), float(dy[gi, gj].mean())])
        risk = float(np.clip((self.cfg.clearance_target_m - clearance)
                             / max(self.cfg.clearance_target_m - self.cfg.clearance_emergency_m, 1e-3), 0.0, 1.0))
        direction = np.array([cos(gdf_heading), sin(gdf_heading)])
        rep_norm = float(np.linalg.norm(rep))
        if rep_norm > 1e-5:
            direction += risk * self.cfg.clearance_gain * rep / rep_norm
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
        desired = atan2(float(direction[1]), float(direction[0]))
        command = self._command_for_heading(
            desired, clearance, emergency, allow_avoid_arc=True
        )
        mode = "AVOID_LEFT" if self._side > 0 else "AVOID_RIGHT"
        return GDFPlanResult(command, mode, desired, gdf_heading, clearance,
                             gdf, sdf, free, seed, start,
                             [bev.ij_to_xy(i, j) for i, j in cells], self._side,
                             {"gdf_start": float(gdf[start]), "risk": risk, "gdf_active": 1.0},
                             forward_blocked, not goal_blocked, self._exit_clear_frames,
                             forward_poly, goal_poly)

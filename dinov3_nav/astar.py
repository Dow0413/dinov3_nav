"""Deterministic local-grid A* and collision-preserving path smoothing."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import hypot
from typing import List, Optional, Tuple

import numpy as np

from .planning_bev import PlanningBEV


@dataclass
class LocalPath:
    points: List[Tuple[float, float]]
    raw_points: List[Tuple[float, float]]
    cost: float
    reason: str = "ok"


class AStarLocalPlanner:
    """A* chooses the route homotopy; MPPI only tracks its lookahead."""

    _NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1))

    @staticmethod
    def _nearest_free(valid: np.ndarray, target: Tuple[int, int], radius: int = 12):
        ti, tj = target
        if 0 <= ti < valid.shape[0] and 0 <= tj < valid.shape[1] and valid[ti, tj]:
            return target
        best = None
        for i in range(max(0, ti-radius), min(valid.shape[0], ti+radius+1)):
            for j in range(max(0, tj-radius), min(valid.shape[1], tj+radius+1)):
                if valid[i, j]:
                    d = (i-ti)**2 + (j-tj)**2
                    if best is None or d < best[0]: best = (d, (i, j))
        return None if best is None else best[1]

    def plan(self, planning: PlanningBEV, goal_xy: Tuple[float, float]) -> LocalPath:
        grid = planning.grid
        valid = np.isfinite(planning.planning_cost) & ~planning.inflated_obstacle
        start0 = grid.xy_to_ij(0.0, 0.0)
        goal0 = grid.xy_to_ij(*goal_xy)
        if start0 is None or goal0 is None:
            return LocalPath([], [], float("inf"), "goal_out_of_map")
        start, goal = self._nearest_free(valid, start0), self._nearest_free(valid, goal0)
        if start is None or goal is None:
            return LocalPath([], [], float("inf"), "no_free_start_or_goal")
        queue = [(0.0, 0.0, start)]
        parent = {start: None}
        g = {start: 0.0}
        while queue:
            _, current_cost, cur = heapq.heappop(queue)
            if current_cost != g.get(cur): continue
            if cur == goal: break
            for di, dj in self._NEIGHBORS:
                ni, nj = cur[0]+di, cur[1]+dj
                if not (0 <= ni < valid.shape[0] and 0 <= nj < valid.shape[1]) or not valid[ni, nj]:
                    continue
                step = hypot(di, dj) * grid.cfg.resolution
                candidate = current_cost + step * float(planning.planning_cost[ni, nj])
                nxt = (ni, nj)
                if candidate < g.get(nxt, float("inf")):
                    g[nxt] = candidate; parent[nxt] = cur
                    heuristic = hypot(ni-goal[0], nj-goal[1]) * grid.cfg.resolution
                    heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
        if goal not in parent:
            return LocalPath([], [], float("inf"), "no_path")
        cells = []
        cur = goal
        while cur is not None: cells.append(cur); cur = parent[cur]
        cells.reverse()
        raw = [grid.ij_to_xy(*cell) for cell in cells]
        smooth = self._smooth(raw, planning)
        return LocalPath(smooth, raw, g[goal])

    @staticmethod
    def _segment_free(a: Tuple[float, float], b: Tuple[float, float], planning: PlanningBEV) -> bool:
        d = hypot(b[0]-a[0], b[1]-a[1]); step = max(.02, planning.grid.cfg.resolution*.5)
        for q in np.linspace(0.0, 1.0, max(2, int(d/step)+1)):
            cell = planning.grid.xy_to_ij(a[0]+q*(b[0]-a[0]), a[1]+q*(b[1]-a[1]))
            if cell is None or not np.isfinite(planning.planning_cost[cell]): return False
        return True

    def _smooth(self, raw: List[Tuple[float, float]], planning: PlanningBEV) -> List[Tuple[float, float]]:
        if len(raw) < 3: return raw
        out, i = [raw[0]], 0
        while i < len(raw)-1:
            j = len(raw)-1
            while j > i+1 and not self._segment_free(raw[i], raw[j], planning): j -= 1
            out.append(raw[j]); i = j
        return out

    @staticmethod
    def lookahead(path: LocalPath, distance: float) -> Optional[Tuple[float, float]]:
        if not path.points: return None
        remaining = max(0.0, distance)
        for a, b in zip(path.points, path.points[1:]):
            d = hypot(b[0]-a[0], b[1]-a[1])
            if d >= remaining:
                t = remaining / max(d, 1e-6)
                return a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])
            remaining -= d
        return path.points[-1]

# -*- coding: utf-8 -*-
"""Debug rendering for BEV and local planner.

Color code (see docs/acceptance spec):
    gray    unknown / unobserved
    darker gray  observed, not ground
    green   traversable (intensity ~ traversability)
    red     hard obstacle
    orange  safety-margin inflation around hard obstacles
    blue    candidate trajectories (dark blue = rejected)
    white   selected trajectory
    yellow  global goal direction arrow
    cyan    robot footprint outline + recovery rotation arrow
Canvas orientation: forward is up, +left is left, robot at bottom center.
"""
from __future__ import annotations

from math import atan2, cos, sin
from typing import Optional, Tuple

import cv2
import numpy as np

from .bev import BEVGrid
from .footprint import FootprintLayers
from .local_planner import MODE_ROTATE_RECOVERY, PlanResult
from .gdf_planner import GDFPlanResult
from .planning_bev import PlanningBEV


def _cell_to_canvas(bev: BEVGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
    cell = bev.xy_to_ij(x, y)
    if cell is None:
        return None
    i, j = cell
    # array: i forward, j left. Canvas: forward is up, left is left.
    row = bev.shape[0] - 1 - i
    col = bev.shape[1] - 1 - j
    return col, row


def render_bev_debug(
    bev: BEVGrid,
    layers: FootprintLayers,
    plan: Optional[PlanResult],
    goal_xy: Optional[Tuple[float, float]],
    scale: int = 5,
    footprint: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """RGB debug image of the BEV, trajectories, goal and robot footprint.

    footprint: optional (length, width) to outline the robot rectangle.
    """
    nx, ny = bev.shape
    img = np.zeros((nx, ny, 3), np.uint8)
    img[:] = (45, 45, 45)  # RGB unknown

    observed = bev.observed
    img[observed] = (80, 80, 80)
    trav = np.clip(bev.traversability, 0.0, 1.0)
    good = observed & (trav >= 0.5)
    img[good, 0] = (40 + 40 * (1.0 - trav[good])).astype(np.uint8)
    img[good, 1] = (100 + 155 * trav[good]).astype(np.uint8)
    img[good, 2] = 40
    img[layers.hard_blocked] = (245, 45, 45)
    inflated_only = layers.inflated_hard_blocked & ~layers.hard_blocked
    img[inflated_only] = (240, 140, 35)

    # BEV array axes already map to [forward, left]. Flip both axes so
    # forward is visually up and +left is visually left.
    canvas = np.flipud(np.fliplr(img))
    canvas = cv2.resize(canvas, (ny * scale, nx * scale), interpolation=cv2.INTER_NEAREST)

    def draw_path(poses: np.ndarray, color, thickness=1):
        pts = []
        for x, y, _ in poses:
            p = _cell_to_canvas(bev, float(x), float(y))
            if p is not None:
                pts.append((p[0] * scale + scale // 2, p[1] * scale + scale // 2))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, np.int32)], False, color, thickness,
                          lineType=cv2.LINE_AA)

    if plan is not None:
        for c in plan.candidates:
            if c.v <= 0.0:
                continue
            draw_path(c.poses, (90, 90, 220) if not c.valid else (120, 180, 255), 1)
        if plan.trajectory is not None and plan.trajectory.v > 0.0:
            draw_path(plan.trajectory.poses, (255, 255, 255), 2)

    robot = _cell_to_canvas(bev, 0.0, 0.0)
    if robot is not None:
        rx, ry = robot[0] * scale + scale // 2, robot[1] * scale + scale // 2

        if footprint is not None:
            fl, fw = footprint
            res = bev.cfg.resolution
            half_l = int(round(0.5 * fl / res)) * scale
            half_w = int(round(0.5 * fw / res)) * scale
            cv2.rectangle(canvas, (rx - half_w, ry - half_l),
                          (rx + half_w, ry + half_l), (255, 170, 30), 1, cv2.LINE_AA)

        cv2.circle(canvas, (rx, ry), max(3, scale), (70, 180, 255), -1)
        # Robot-centric coordinate axes: +X (forward) is red, +Y (left) green.
        axis = max(4 * scale, int(0.45 / bev.cfg.resolution * scale))
        cv2.arrowedLine(canvas, (rx, ry), (rx, ry - axis), (255, 60, 60), 2,
                        cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(canvas, (rx, ry), (rx - axis, ry), (60, 255, 60), 2,
                        cv2.LINE_AA, tipLength=0.18)

        if goal_xy is not None:
            gx, gy = goal_xy
            a = atan2(gy, gx)
            length = int(0.8 / bev.cfg.resolution * scale)
            end = (int(rx - sin(a) * length), int(ry - cos(a) * length))
            cv2.arrowedLine(canvas, (rx, ry), end, (255, 230, 40), 2,
                            line_type=cv2.LINE_AA, tipLength=0.15)

        if plan is not None and plan.mode == MODE_ROTATE_RECOVERY \
                and plan.escape_heading is not None:
            a = plan.escape_heading
            r = int(0.7 / bev.cfg.resolution * scale)
            p0 = (int(rx + sin(a) * 0.35 * r), int(ry - cos(a) * 0.35 * r))
            p1 = (int(rx + sin(a) * r), int(ry - cos(a) * r))
            cv2.arrowedLine(canvas, p0, p1, (255, 120, 255), 2,
                            line_type=cv2.LINE_AA, tipLength=0.25)

    return canvas


def render_bev_layer(bev: BEVGrid, which: str) -> np.ndarray:
    """Mono8 image of a single BEV layer, oriented like render_bev_debug
    (forward up, left left) so the three topics compare visually."""
    if which == "traversability":
        layer = np.clip(bev.traversability, 0.0, 1.0)
        arr = (layer * 255.0).astype(np.uint8)
    elif which == "obstacle":
        arr = bev.obstacle.astype(np.uint8) * 255
    elif which == "observed":
        arr = bev.observed.astype(np.uint8) * 255
    else:
        raise ValueError(f"unknown layer {which!r}")
    return np.flipud(np.fliplr(arr)).copy()


def render_raw_bev(raw: BEVGrid, scale: int = 5) -> np.ndarray:
    """Text-free raw sensor BEV: unknown gray, support green, obstacle red."""
    nx, ny = raw.shape
    img = np.full((nx, ny, 3), (45, 45, 45), np.uint8)
    img[raw.observed] = (78, 78, 78)
    free = raw.observed & (raw.traversability >= 0.5) & ~raw.obstacle
    img[free] = (35, 170, 45)
    img[raw.obstacle] = (245, 45, 45)
    canvas = np.flipud(np.fliplr(img))
    return cv2.resize(canvas, (ny * scale, nx * scale), interpolation=cv2.INTER_NEAREST)


def render_planning_bev(planning: PlanningBEV, scale: int = 5) -> np.ndarray:
    """Text-free costmap summary with robot, hard/inflated obstacles and cost."""
    bev = planning.grid
    nx, ny = bev.shape
    img = np.full((nx, ny, 3), (45, 45, 45), np.uint8)
    known = ~planning.unknown
    img[known] = (76, 76, 76)
    img[planning.free] = (35, 165, 45)
    inflated_only = planning.inflated_obstacle & ~planning.hard_obstacle
    img[inflated_only] = (235, 140, 35)
    img[planning.hard_obstacle] = (245, 45, 45)
    canvas = np.flipud(np.fliplr(img))
    canvas = cv2.resize(canvas, (ny * scale, nx * scale), interpolation=cv2.INTER_NEAREST)
    robot = _cell_to_canvas(bev, 0.0, 0.0)
    if robot is not None:
        p = (robot[0] * scale + scale // 2, robot[1] * scale + scale // 2)
        cv2.circle(canvas, p, max(3, scale), (70, 180, 255), -1)
    return canvas


def render_planning_layer(planning: PlanningBEV, which: str) -> np.ndarray:
    """Mono8 planning field, oriented forward-up / left-left for ROS debug."""
    if which == "free":
        arr = planning.free.astype(np.uint8) * 255
    elif which == "unknown":
        arr = planning.unknown.astype(np.uint8) * 255
    elif which == "hard_obstacle":
        arr = planning.hard_obstacle.astype(np.uint8) * 255
    elif which == "inflated_obstacle":
        arr = planning.inflated_obstacle.astype(np.uint8) * 255
    elif which == "clearance":
        finite = planning.clearance_m[np.isfinite(planning.clearance_m)]
        hi = max(float(np.percentile(finite, 95)) if finite.size else 0.0, 1e-3)
        arr = (255.0 * np.clip(planning.clearance_m / hi, 0.0, 1.0)).astype(np.uint8)
    elif which == "cost":
        finite = np.isfinite(planning.planning_cost)
        arr = np.zeros(planning.grid.shape, np.uint8)
        if finite.any():
            lo = float(np.min(planning.planning_cost[finite]))
            hi = max(float(np.percentile(planning.planning_cost[finite], 95)), lo + 1e-3)
            arr[finite] = (255.0 * np.clip(
                (planning.planning_cost[finite] - lo) / (hi - lo), 0.0, 1.0
            )).astype(np.uint8)
        arr[~finite] = 255
    else:
        raise ValueError(f"unknown planning layer {which!r}")
    return np.flipud(np.fliplr(arr)).copy()


def render_gdf_debug(
    bev: BEVGrid,
    layers: FootprintLayers,
    plan: Optional[GDFPlanResult],
    goal_xy: Optional[Tuple[float, float]],
    scale: int = 5,
) -> np.ndarray:
    """BEV debug for the GDF/SDF planner; no rollout trajectories are drawn."""
    nx, ny = bev.shape
    img = np.full((nx, ny, 3), (45, 45, 45), np.uint8)
    img[bev.observed] = (82, 82, 82)
    good = bev.observed & (bev.traversability >= 0.5)
    img[good] = (35, 155, 45)
    inflated_only = layers.inflated_hard_blocked & ~layers.hard_blocked
    img[inflated_only] = (235, 140, 35)
    img[layers.hard_blocked] = (245, 45, 45)
    if plan is not None:
        finite = np.isfinite(plan.gdf) & plan.free
        if finite.any():
            values = plan.gdf[finite]
            hi = max(float(np.percentile(values, 95)), 1e-3)
            heat = cv2.applyColorMap(
                (255.0 * np.clip(plan.gdf / hi, 0.0, 1.0)).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )[:, :, ::-1]
            img[finite] = (0.45 * img[finite] + 0.55 * heat[finite]).astype(np.uint8)
        if plan.path_xy:
            points = []
            for x, y in plan.path_xy:
                p = _cell_to_canvas(bev, x, y)
                if p is not None:
                    points.append(p)
            # Array canvas coordinates before final flip/scale.
            if len(points) >= 2:
                cv2.polylines(img, [np.asarray([(ny - 1 - x, nx - 1 - y)
                                                 for x, y in points], np.int32)],
                              False, (40, 255, 255), 1, cv2.LINE_AA)

    canvas = np.flipud(np.fliplr(img))
    canvas = cv2.resize(canvas, (ny * scale, nx * scale), interpolation=cv2.INTER_NEAREST)

    def draw_polygon(points, color, thickness=2):
        cells = [_cell_to_canvas(bev, x, y) for x, y in points]
        cells = [p for p in cells if p is not None]
        if len(cells) >= 3:
            pixels = np.asarray(
                [(x * scale + scale // 2, y * scale + scale // 2) for x, y in cells],
                np.int32,
            )
            cv2.polylines(canvas, [pixels], True, color, thickness, cv2.LINE_AA)

    if plan is not None:
        # Green is the normal straight-ahead trigger corridor; red means it
        # intersects the inflated obstacle layer. Blue is the global-goal
        # corridor used to debounce the AVOID -> GO_TO_GOAL transition.
        draw_polygon(plan.forward_corridor_xy,
                     (255, 70, 70) if plan.corridor_blocked else (70, 255, 120))
        draw_polygon(plan.goal_corridor_xy, (80, 160, 255), 1)

    robot = _cell_to_canvas(bev, 0.0, 0.0)
    if robot is not None:
        rx, ry = robot[0] * scale + scale // 2, robot[1] * scale + scale // 2
        cv2.circle(canvas, (rx, ry), max(3, scale), (70, 180, 255), -1)
        if goal_xy is not None:
            a = atan2(goal_xy[1], goal_xy[0])
            end = (int(rx - sin(a) * 0.8 / bev.cfg.resolution * scale),
                   int(ry - cos(a) * 0.8 / bev.cfg.resolution * scale))
            cv2.arrowedLine(canvas, (rx, ry), end, (255, 230, 40), 2, cv2.LINE_AA, 0, 0.15)
        if plan is not None and plan.gdf_heading is not None:
            a = plan.gdf_heading
            end = (int(rx - sin(a) * 0.72 / bev.cfg.resolution * scale),
                   int(ry - cos(a) * 0.72 / bev.cfg.resolution * scale))
            cv2.arrowedLine(canvas, (rx, ry), end, (40, 255, 255), 2, cv2.LINE_AA, 0, 0.15)
        if plan is not None and plan.desired_heading is not None:
            a = plan.desired_heading
            end = (int(rx - sin(a) * 0.65 / bev.cfg.resolution * scale),
                   int(ry - cos(a) * 0.65 / bev.cfg.resolution * scale))
            cv2.arrowedLine(canvas, (rx, ry), end, (255, 80, 255), 2, cv2.LINE_AA, 0, 0.15)
    return canvas

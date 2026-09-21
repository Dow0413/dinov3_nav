# -*- coding: utf-8 -*-
"""ROS-free acceptance tests for trigger-only GDF/SDF avoidance."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dinov3_nav.bev import BEVConfig, BEVGrid
from dinov3_nav.debug_viz import render_gdf_debug
from dinov3_nav.footprint import FootprintConfig, build_footprint_layers
from dinov3_nav.gdf_planner import GDFPlanner, GDFPlannerConfig


CFG = BEVConfig(resolution=0.10, x_min=-0.50, x_max=4.0, y_min=-2.0, y_max=2.0)


def bev_from_rects(rects, unknown=()):
    nx = int(np.ceil((CFG.x_max - CFG.x_min) / CFG.resolution))
    ny = int(np.ceil((CFG.y_max - CFG.y_min) / CFG.resolution))
    xs = CFG.x_min + (np.arange(nx) + 0.5) * CFG.resolution
    ys = CFG.y_min + (np.arange(ny) + 0.5) * CFG.resolution
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    obstacle = np.zeros((nx, ny), bool)
    observed = np.ones((nx, ny), bool)
    for x0, y0, x1, y1 in rects:
        obstacle |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    for x0, y0, x1, y1 in unknown:
        observed[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = False
    traversability = np.where(obstacle, 0.0, 1.0).astype(np.float32)
    observed |= obstacle
    count = observed.astype(np.int32)
    return BEVGrid(traversability, observed, obstacle, count, count, cfg=CFG)


def plan(bev, planner=None, now=0.0):
    layers = build_footprint_layers(bev, FootprintConfig(safety_margin=0.20))
    planner = planner or GDFPlanner(GDFPlannerConfig(
        lookahead_m=0.8,
        corridor_lookahead_m=1.2,
        corridor_half_width_m=0.4,
        avoid_exit_clear_frames=3,
    ))
    return planner, planner.plan(bev, layers, (8.0, 0.0), now)


def test_a_open_space():
    bev = bev_from_rects([])
    _, result = plan(bev)
    assert result.command[0] > 0.0, result
    assert result.desired_heading is not None and abs(result.desired_heading) < 0.08, result
    assert result.mode == "GO_TO_GOAL", result.mode
    assert not result.corridor_blocked and result.gdf_heading is None, result
    print("A open: OK", result.mode, result.command)


def test_b_wall_ahead_commits_side_before_collision():
    bev = bev_from_rects([(1.0, -0.55, 2.1, 0.55)])
    planner, first = plan(bev)
    assert first.corridor_blocked, first
    assert first.mode in ("AVOID_LEFT", "AVOID_RIGHT"), first.mode
    assert first.side != 0 and first.desired_heading is not None and first.gdf_heading is not None, first
    assert abs(first.desired_heading) > 0.20, first
    _, second = plan(bev, planner, now=0.1)
    assert second.mode == first.mode, (first.mode, second.mode)
    assert second.side == first.side, (first.side, second.side)
    print("B central wall: OK", first.mode, first.command, "side", first.side)


def test_c_left_closed_choose_right():
    bev = bev_from_rects([(1.0, -0.55, 2.1, 0.55), (0.8, 0.50, 3.8, 2.0)])
    _, result = plan(bev)
    assert result.mode == "AVOID_RIGHT", result.mode
    assert result.side < 0, result
    assert result.desired_heading is not None and result.desired_heading < -0.20, result
    print("C left closed: OK", result.mode, result.command)


def test_unknown_is_costly_not_a_wall():
    _, result = plan(bev_from_rects([], unknown=[(0.8, -0.2, 1.2, 0.2)]))
    assert result.start_ij is not None and result.mode == "GO_TO_GOAL", result
    print("unknown policy: OK")


def test_exit_requires_stable_clear_frames():
    planner, blocked = plan(bev_from_rects([(1.0, -0.55, 2.1, 0.55)]))
    assert blocked.mode.startswith("AVOID_"), blocked
    clear = bev_from_rects([])
    _, first_clear = plan(clear, planner, now=0.1)
    _, second_clear = plan(clear, planner, now=0.2)
    _, third_clear = plan(clear, planner, now=0.3)
    assert first_clear.mode.startswith("AVOID_") and first_clear.exit_clear_frames == 1, first_clear
    assert second_clear.mode.startswith("AVOID_") and second_clear.exit_clear_frames == 2, second_clear
    assert third_clear.mode == "GO_TO_GOAL" and third_clear.exit_clear_frames == 3, third_clear
    print("stable exit: OK")


def test_avoid_arc_moves_instead_of_spinning_at_45_degrees():
    """A body-safe inflated detour at ~45 deg must not be a spin fixed point."""
    bev = bev_from_rects([(1.0, -0.55, 2.1, 0.55)])
    layers = build_footprint_layers(bev, FootprintConfig(width=0.60, safety_margin=0.40))
    planner = GDFPlanner(GDFPlannerConfig(
        corridor_lookahead_m=1.2,
        corridor_half_width_m=0.4,
        lookahead_m=0.7,
        avoid_turn_in_place_angle=1.30,
        avoid_min_linear=0.08,
    ))
    result = planner.plan(bev, layers, (8.0, 0.0), now=0.0)
    assert result.mode.startswith("AVOID_") and result.desired_heading is not None, result
    assert 0.70 <= abs(result.desired_heading) < 1.30, result.desired_heading
    assert result.command[0] > 0.0, result.command
    print("avoid arc at 45 deg: OK", result.command)


def save_debug_outputs(output_dir: Path):
    """Save text-free OpenCV BEV renders for the acceptance scenarios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = (
        ("a_open", bev_from_rects([])),
        ("b_wall_ahead", bev_from_rects([(1.0, -0.55, 2.1, 0.55)])),
        ("c_left_closed", bev_from_rects([
            (1.0, -0.55, 2.1, 0.55), (0.8, 0.50, 3.8, 2.0),
        ])),
    )
    for name, bev in scenarios:
        planner, result = plan(bev)
        layers = build_footprint_layers(bev, FootprintConfig(safety_margin=0.20))
        image = render_gdf_debug(bev, layers, result, (8.0, 0.0), scale=5)
        path = output_dir / f"{name}_bev.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to save {path}")
    print(f"saved text-free debug images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    test_a_open_space()
    test_b_wall_ahead_commits_side_before_collision()
    test_c_left_closed_choose_right()
    test_unknown_is_costly_not_a_wall()
    test_exit_requires_stable_clear_frames()
    test_avoid_arc_moves_instead_of_spinning_at_45_degrees()
    if args.output_dir is not None:
        save_debug_outputs(args.output_dir)
    print("all GDF/SDF synthetic tests passed")


if __name__ == "__main__":
    main()

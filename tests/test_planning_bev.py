# -*- coding: utf-8 -*-
"""ROS-free checks for temporal fusion and planning BEV semantics."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dinov3_nav.bev import BEVConfig, BEVGrid
from dinov3_nav.debug_viz import (
    render_bev_debug,
    render_planning_bev,
    render_planning_layer,
    render_raw_bev,
)
from dinov3_nav.footprint import FootprintConfig
from dinov3_nav.planning_bev import (
    PlanningCostConfig,
    TemporalBEVConfig,
    TemporalBEVFusion,
    build_planning_bev,
)


CFG = BEVConfig(resolution=0.10, x_min=-0.5, x_max=3.0, y_min=-1.5, y_max=1.5)


def raw_grid(obstacles=(), observed=True):
    nx = int(np.ceil((CFG.x_max - CFG.x_min) / CFG.resolution))
    ny = int(np.ceil((CFG.y_max - CFG.y_min) / CFG.resolution))
    obs = np.full((nx, ny), observed, bool)
    trav = np.where(obs, 1.0, 0.0).astype(np.float32)
    obstacle = np.zeros((nx, ny), bool)
    for x, y in obstacles:
        cell = BEVGrid(trav, obs, obstacle, np.zeros((nx, ny), np.int32),
                       np.zeros((nx, ny), np.int32), cfg=CFG).xy_to_ij(x, y)
        assert cell is not None
        obstacle[cell] = True
        trav[cell] = 0.0
    count = obs.astype(np.int32)
    return BEVGrid(trav, obs, obstacle, count, (trav * count).astype(np.int32), cfg=CFG)


def fusion():
    return TemporalBEVFusion(TemporalBEVConfig(
        evidence_half_life_s=2.0,
        cleanup_min_obstacle_cells=2,
        ego_clear_radius_m=0.20,
    ))


def test_temporal_obstacle_persists_and_reprojects():
    fuser = fusion()
    first = fuser.update(raw_grid([(1.0, 0.0)]), 0.0, None)
    assert first.obstacle[first.xy_to_ij(1.0, 0.0)]
    # Robot advances 0.2 m: a static obstacle appears 0.2 m closer in the
    # new base_link frame. The next frame contributes no observation.
    T_current_from_previous = np.array([[1.0, 0.0, -0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    second = fuser.update(raw_grid(observed=False), 0.1, T_current_from_previous)
    shifted = second.xy_to_ij(0.8, 0.0)
    assert shifted is not None and second.obstacle[shifted], "obstacle was not motion-compensated"
    print("temporal persistence/reprojection: OK")


def test_unknown_and_ego_clearing_semantics():
    stable = fusion().update(raw_grid(observed=False), 0.0, None)
    start = stable.xy_to_ij(0.0, 0.0)
    far = stable.xy_to_ij(2.0, 1.0)
    assert start is not None and far is not None
    assert stable.observed[start] and stable.traversability[start] >= 0.5
    assert not stable.observed[far], "unseen area must remain UNKNOWN"
    planning = build_planning_bev(
        stable, FootprintConfig(safety_margin=0.25), PlanningCostConfig(unknown_cost=3.0)
    )
    assert planning.free[start] and planning.unknown[far]
    assert planning.planning_cost[far] > planning.planning_cost[start]
    print("unknown + bounded ego clearing: OK")


def test_inflation_clearance_and_cost():
    stable = fusion().update(raw_grid([(1.0, 0.0)]), 0.0, None)
    planning = build_planning_bev(
        stable, FootprintConfig(safety_margin=0.25),
        PlanningCostConfig(clearance_target_m=0.60, clearance_cost_weight=5.0),
    )
    obstacle = stable.xy_to_ij(1.0, 0.0)
    near = stable.xy_to_ij(0.8, 0.0)
    far = stable.xy_to_ij(2.0, 1.0)
    assert obstacle is not None and near is not None and far is not None
    assert planning.hard_obstacle[obstacle] and np.isinf(planning.planning_cost[obstacle])
    assert planning.inflated_obstacle[near] and np.isinf(planning.planning_cost[near])
    assert planning.clearance_m[far] > planning.clearance_m[near]
    print("inflation + clearance cost: OK")


def test_debug_bev_renders_before_a_goal_exists():
    """Perception/debug must be usable while navigation authority is idle."""
    stable = fusion().update(raw_grid([(1.0, 0.0)]), 0.0, None)
    planning = build_planning_bev(stable, FootprintConfig(), PlanningCostConfig())
    image = render_bev_debug(planning.grid, planning.layers, None, None, scale=2)
    assert image.ndim == 3 and image.shape[2] == 3 and image.size > 0


def save_debug_outputs(output_dir: Path):
    """Write raw/stable planning layers for visual, text-free inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_grid([(1.0, 0.0)])
    fuser = fusion()
    stable = fuser.update(raw, 0.0, None)
    planning = build_planning_bev(
        stable, FootprintConfig(safety_margin=0.25), PlanningCostConfig()
    )
    rgb = {
        "raw_bev.png": render_raw_bev(raw, scale=5),
        "planning_bev.png": render_planning_bev(planning, scale=5),
    }
    for name, image in rgb.items():
        if not cv2.imwrite(str(output_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"cannot write {name}")
    for name in ("free", "unknown", "hard_obstacle", "inflated_obstacle", "clearance", "cost"):
        if not cv2.imwrite(str(output_dir / f"{name}.png"), render_planning_layer(planning, name)):
            raise RuntimeError(f"cannot write {name}")
    print(f"saved planning BEV debug images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    test_temporal_obstacle_persists_and_reprojects()
    test_unknown_and_ego_clearing_semantics()
    test_inflation_clearance_and_cost()
    if args.output_dir is not None:
        save_debug_outputs(args.output_dir)
    print("all planning BEV tests passed")


if __name__ == "__main__":
    main()

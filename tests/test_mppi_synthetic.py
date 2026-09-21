"""ROS-free behavioural tests for the costmap MPPI local planner."""
from __future__ import annotations

import numpy as np

from dinov3_nav.bev import BEVConfig, BEVGrid
from dinov3_nav.footprint import FootprintChecker, FootprintConfig
from dinov3_nav.mppi import MPPIConfig, MPPIPlanner
from dinov3_nav.planning_bev import PlanningCostConfig, build_planning_bev


CFG = BEVConfig(resolution=0.10, x_min=-0.60, x_max=4.0, y_min=-2.0, y_max=2.0)
FOOTPRINT = FootprintConfig(length=0.50, width=0.35, safety_margin=0.10)


def make_map(rects=(), unknown=()):
    nx = int(np.ceil((CFG.x_max - CFG.x_min) / CFG.resolution))
    ny = int(np.ceil((CFG.y_max - CFG.y_min) / CFG.resolution))
    xs = CFG.x_min + (np.arange(nx) + .5) * CFG.resolution
    ys = CFG.y_min + (np.arange(ny) + .5) * CFG.resolution
    x, y = np.meshgrid(xs, ys, indexing="ij")
    obstacle = np.zeros((nx, ny), bool)
    for x0, y0, x1, y1 in rects:
        obstacle |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    observed = np.ones((nx, ny), bool)
    for x0, y0, x1, y1 in unknown:
        observed[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = False
    traversability = (~obstacle).astype(np.float32)
    grid = BEVGrid(traversability, observed, obstacle, observed.astype(np.int32),
                   observed.astype(np.int32), cfg=CFG)
    planning = build_planning_bev(grid, FOOTPRINT, PlanningCostConfig())
    return planning, FootprintChecker(planning.grid, FOOTPRINT, planning.layers)


def planner():
    return MPPIPlanner(MPPIConfig(samples=320, horizon_steps=22, seed=12,
                                  max_vx=.40, max_accel_vx=2.0, max_accel_wz=4.0))


def test_open_space_moves_to_goal():
    p, c = make_map()
    result = planner().plan(p, c, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert result.trajectory and result.trajectory.valid
    assert result.command[0] > .05
    assert abs(result.command[2]) < .45


def test_wall_ahead_rollout_turns_without_collision():
    p, c = make_map([(1.0, -.45, 2.0, .45)])
    result = planner().plan(p, c, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert result.trajectory and result.trajectory.valid
    assert abs(result.command[2]) > .05, result.command
    for x, y, yaw in result.trajectory.poses[1:]:
        assert c.check_pose(float(x), float(y), float(yaw), relax_surface=False).valid


def test_camera_blind_band_is_not_a_false_collision():
    """Unknown near ground must not invalidate every forward rollout."""
    p, c = make_map(unknown=[(-.1, -.8, .85, .8)])
    result = planner().plan(p, c, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert result.trajectory and result.trajectory.valid, result.diagnostics


def test_warm_start_suppresses_left_right_flipping():
    p, c = make_map([(1.0, -.40, 1.8, .40)])
    local = planner()
    signs = []
    velocity = (0.0, 0.0, 0.0)
    for _ in range(5):
        result = local.plan(p, c, (3.0, 0.0), velocity)
        velocity = result.command
        signs.append(np.sign(result.command[2]))
    nonzero = [s for s in signs if s]
    assert nonzero and len(set(nonzero)) == 1, signs


def test_safe_cached_trajectory_is_reused_but_new_wall_triggers_replan():
    open_map, open_checker = make_map()
    local = planner()
    first = local.plan(open_map, open_checker, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert first.trajectory and first.trajectory.valid

    reused = local.reuse_if_safe(open_map, open_checker, (3.0, 0.0), first.command)
    assert reused and reused.valid

    blocked_map, blocked_checker = make_map([(0.30, -0.8, 2.0, 0.8)])
    assert local.reuse_if_safe(
        blocked_map, blocked_checker, (3.0, 0.0), first.command
    ) is None


def test_reuse_advances_by_elapsed_control_time_not_one_image_frame():
    planning, checker = make_map()
    local = planner()
    first = local.plan(planning, checker, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert first.trajectory and first.trajectory.valid
    original = first.trajectory.controls.copy()
    reused = local.reuse_if_safe(
        planning, checker, (3.0, 0.0), first.command, elapsed_steps=5
    )
    assert reused and reused.valid
    np.testing.assert_allclose(reused.controls[0], original[5])

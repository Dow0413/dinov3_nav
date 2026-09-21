"""Sampling-based MPPI controller for the robot-centred planning costmap.

The planner deliberately consumes a costmap, not pixels or segmentation
directions.  Every update rolls out complete ``vx, vy, wz`` sequences and
scores their future footprint-safe trajectories.  The warm-started control
sequence is retained between frames, which is the main temporal continuity
mechanism missing from a per-image steering controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Dict, Optional, Tuple

import numpy as np

from .footprint import FootprintChecker
from .planning_bev import PlanningBEV


@dataclass
class MPPIConfig:
    horizon_steps: int = 24
    dt: float = 0.10
    samples: int = 384
    temperature: float = 1.0
    noise_vx: float = 0.16
    noise_vy: float = 0.08
    noise_wz: float = 0.55
    max_vx: float = 0.40
    min_vx: float = -0.10
    max_vy: float = 0.0
    max_wz: float = 1.20
    max_accel_vx: float = 0.60
    max_accel_vy: float = 0.50
    max_accel_wz: float = 1.80
    # Forward RGB-D cameras cannot observe the floor under the robot.  Within
    # this distance, preserve hard collision/inflation checks but do not
    # reject a footprint solely because its support is unknown.
    start_relax_distance: float = 0.90
    goal_weight: float = 5.0
    terminal_goal_weight: float = 10.0
    heading_weight: float = 0.8
    costmap_weight: float = 1.0
    control_weight: float = 0.08
    smooth_weight: float = 0.35
    unknown_footprint_weight: float = 2.0
    exploration_fraction: float = 0.35
    valid_fraction_slow: float = 0.12
    clearance_slow_m: float = 0.35
    slow_speed_scale: float = 0.45
    turn_trigger_clearance_m: float = 0.45
    min_detour_wz: float = 0.05
    side_commit_weight: float = 2.0
    side_commit_steps: int = 8
    collision_cost: float = 1.0e6
    unknown_cost_scale: float = 1.0
    seed: int = 7


@dataclass
class MPPITrace:
    controls: np.ndarray
    poses: np.ndarray
    cost: float
    valid: bool
    reason: str = "ok"

    @property
    def v(self) -> float:
        return float(self.controls[0, 0])


@dataclass
class MPPIResult:
    command: Tuple[float, float, float]
    trajectory: Optional[MPPITrace]
    mode: str = "MPPI"
    candidates: list[MPPITrace] = field(default_factory=list)
    effective_samples: int = 0
    best_cost: float = float("inf")
    min_clearance_m: float = 0.0
    diagnostics: Dict[str, float] = field(default_factory=dict)


class MPPIPlanner:
    def __init__(self, cfg: MPPIConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._u: Optional[np.ndarray] = None
        self._turn_side = 0
        self._turn_side_remaining = 0

    def reset(self) -> None:
        self._u = None
        self._turn_side = 0
        self._turn_side_remaining = 0

    def _warm_start(self) -> np.ndarray:
        n = max(2, int(self.cfg.horizon_steps))
        if self._u is None or self._u.shape != (n, 3):
            self._u = np.zeros((n, 3), np.float32)
        else:
            self._u[:-1] = self._u[1:]
            self._u[-1] = self._u[-2]
        return self._u.copy()

    def _limit(self, controls: np.ndarray, initial_velocity: np.ndarray) -> np.ndarray:
        c = controls.copy()
        c[:, 0] = np.clip(c[:, 0], self.cfg.min_vx, self.cfg.max_vx)
        c[:, 1] = np.clip(c[:, 1], -self.cfg.max_vy, self.cfg.max_vy)
        c[:, 2] = np.clip(c[:, 2], -self.cfg.max_wz, self.cfg.max_wz)
        previous = initial_velocity.astype(np.float32).copy()
        limits = np.array([self.cfg.max_accel_vx, self.cfg.max_accel_vy,
                           self.cfg.max_accel_wz], np.float32) * self.cfg.dt
        for k in range(len(c)):
            c[k] = np.clip(c[k], previous - limits, previous + limits)
            previous = c[k]
        return c

    def _rollout_cost(self, controls: np.ndarray, initial_velocity: np.ndarray,
                      planning: PlanningBEV, checker: FootprintChecker,
                      goal_xy: Tuple[float, float]) -> MPPITrace:
        x = y = yaw = 0.0
        poses = np.zeros((len(controls) + 1, 3), np.float32)
        cost = 0.0
        previous = initial_velocity
        gx, gy = goal_xy
        for k, u in enumerate(controls):
            vx, vy, wz = map(float, u)
            yaw_mid = yaw + 0.5 * wz * self.cfg.dt
            x += (vx * cos(yaw_mid) - vy * sin(yaw_mid)) * self.cfg.dt
            y += (vx * sin(yaw_mid) + vy * cos(yaw_mid)) * self.cfg.dt
            yaw += wz * self.cfg.dt
            poses[k + 1] = (x, y, yaw)
            # Surface checks are relaxed only while the footprint overlaps the
            # camera blind island; collision/inflation are never relaxed.
            relax = x < self.cfg.start_relax_distance and abs(y) < 0.75
            # Unknown is epistemic risk, not a physical obstacle.  It stays
            # costly below, while geometric/inflated collision remains hard.
            chk = checker.check_pose(
                x, y, yaw, relax_surface=relax, unknown_is_soft=True
            )
            if not chk.valid:
                return MPPITrace(controls, poses, self.cfg.collision_cost, False, chk.reason)
            cell = planning.grid.xy_to_ij(x, y)
            if cell is None or not np.isfinite(planning.planning_cost[cell]):
                return MPPITrace(controls, poses, self.cfg.collision_cost, False, "costmap")
            dist = float(np.hypot(gx - x, gy - y))
            cost += self.cfg.goal_weight * dist * self.cfg.dt
            cost += self.cfg.costmap_weight * float(planning.planning_cost[cell]) * self.cfg.dt
            cost += self.cfg.unknown_footprint_weight * chk.unknown_fraction * self.cfg.dt
            cost += self.cfg.control_weight * float(np.dot(u, u)) * self.cfg.dt
            cost += self.cfg.smooth_weight * float(np.dot(u - previous, u - previous))
            if self._turn_side_remaining > 0 and self._turn_side and u[2] * self._turn_side < 0.0:
                # A short commitment is a trajectory-level regulariser, not
                # a direction selector: an unsafe branch remains rejected.
                cost += self.cfg.side_commit_weight * abs(float(u[2])) * self.cfg.dt
            previous = u
        desired = float(np.arctan2(gy - y, gx - x))
        heading_error = (desired - yaw + np.pi) % (2.0 * np.pi) - np.pi
        cost += self.cfg.terminal_goal_weight * float(np.hypot(gx - x, gy - y))
        cost += self.cfg.heading_weight * heading_error * heading_error
        return MPPITrace(controls, poses, cost, True)

    def plan(self, planning: PlanningBEV, checker: FootprintChecker,
             goal_xy: Tuple[float, float], current_velocity: Tuple[float, float, float]) -> MPPIResult:
        velocity = np.asarray(current_velocity, np.float32)
        nominal = self._warm_start()
        # Add a gentle forward prior; obstacle avoidance is still entirely
        # trajectory-cost driven, not a left/right image-space decision.
        nominal[:, 0] = np.maximum(nominal[:, 0], min(0.20, self.cfg.max_vx))
        n, t = max(1, self.cfg.samples), nominal.shape[0]
        noise = self._rng.normal(0.0, [self.cfg.noise_vx, self.cfg.noise_vy, self.cfg.noise_wz],
                                 size=(n, t, 3)).astype(np.float32)
        # Deterministic fan trajectories ensure both detour sides are always
        # represented even when Gaussian samples cluster around a warm start.
        fan = min(n, max(0, int(round(n * self.cfg.exploration_fraction))))
        if fan:
            headings = np.linspace(-self.cfg.max_wz, self.cfg.max_wz, fan, dtype=np.float32)
            fan_start = n - fan
            noise[fan_start:] = 0.0
            for i, wz in enumerate(headings):
                idx = fan_start + i
                noise[idx, :, 0] = min(self.cfg.max_vx, 0.65 * self.cfg.max_vx) - nominal[:, 0]
                noise[idx, :, 2] = wz - nominal[:, 2]
        sequences = np.empty_like(noise)
        traces: list[MPPITrace] = []
        costs = np.full(n, self.cfg.collision_cost, np.float64)
        for i in range(n):
            sequences[i] = self._limit(nominal + noise[i], velocity)
            trace = self._rollout_cost(sequences[i], velocity, planning, checker, goal_xy)
            traces.append(trace)
            costs[i] = trace.cost
        valid = np.array([x.valid for x in traces], bool)
        if not valid.any():
            self._u = np.zeros_like(nominal)
            reasons: Dict[str, float] = {"invalid": float(n)}
            for trace in traces:
                reasons[trace.reason] = reasons.get(trace.reason, 0.0) + 1.0
            return MPPIResult((0.0, 0.0, 0.0), None, "BLOCKED", traces, 0,
                              diagnostics=reasons)
        minimum = float(costs[valid].min())
        weights = np.zeros(n, np.float64)
        weights[valid] = np.exp(-(costs[valid] - minimum) / max(self.cfg.temperature, 1e-4))
        weights /= max(weights.sum(), 1e-12)
        self._u = np.tensordot(weights, sequences, axes=(0, 0)).astype(np.float32)
        self._u = self._limit(self._u, velocity)
        forward_clearance = checker.clearance_at(0.75, 0.0)
        if (forward_clearance < self.cfg.turn_trigger_clearance_m
                and abs(float(self._u[0, 2])) < self.cfg.min_detour_wz):
            detours = [trace for trace in traces if trace.valid
                       and abs(float(trace.controls[0, 2])) >= self.cfg.min_detour_wz]
            if self._turn_side:
                same_side = [trace for trace in detours
                             if trace.controls[0, 2] * self._turn_side > 0.0]
                detours = same_side or detours
            if detours:
                # MPPI's weighted mean must not average two distinct
                # left/right homotopies into a straight path at a wall.
                self._u = min(detours, key=lambda trace: trace.cost).controls.copy()
        if self._turn_side_remaining > 0 and self._turn_side and self._u[0, 2] * self._turn_side < 0.0:
            # Do not reverse yaw in one control tick because a nearly
            # symmetric costmap fluctuated. Zero is safe; later MPPI updates
            # may reverse after the short commitment has expired.
            self._u[0, 2] = 0.0
        selected = self._rollout_cost(self._u, velocity, planning, checker, goal_xy)
        if not selected.valid:  # numerical averaging can cross a narrow obstacle gap
            selected = min((x for x in traces if x.valid), key=lambda x: x.cost)
            self._u = selected.controls.copy()
        turn = float(self._u[0, 2])
        if abs(turn) > 0.03:
            side = 1 if turn > 0.0 else -1
            if self._turn_side == 0 or side == self._turn_side:
                self._turn_side = side
                self._turn_side_remaining = max(0, int(self.cfg.side_commit_steps))
        if self._turn_side_remaining > 0:
            self._turn_side_remaining -= 1
        else:
            self._turn_side = 0
        min_clearance = min(
            (checker.clearance_at(float(x), float(y)) for x, y, _ in selected.poses),
            default=0.0,
        )
        command = self._u[0].copy()
        valid_fraction = float(valid.mean())
        if (valid_fraction < self.cfg.valid_fraction_slow
                or min_clearance < self.cfg.clearance_slow_m):
            command[:2] *= self.cfg.slow_speed_scale
        return MPPIResult(tuple(map(float, command)), selected, "MPPI", traces,
                          int(valid.sum()), float(selected.cost), float(min_clearance),
                          {"min_sample_cost": minimum, "valid_fraction": valid_fraction,
                           "forward_clearance": forward_clearance})

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
    horizon_steps: int = 30
    dt: float = 0.10
    samples: int = 384
    temperature: float = 1.0
    noise_vx: float = 0.16
    noise_vy: float = 0.08
    noise_wz: float = 0.55
    max_vx: float = 0.40
    min_vx: float = -0.10
    nominal_vx: float = 0.20
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
    # Suppress tiny stochastic yaw commands in an otherwise clear corridor.
    # Without this, a small random MPPI bias can accumulate into a visible
    # circle even when the goal is straight ahead.
    angular_deadband: float = 0.04
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

    def reuse_if_safe(
        self,
        planning: PlanningBEV,
        checker: FootprintChecker,
        goal_xy: Tuple[float, float],
        current_velocity: Tuple[float, float, float],
        elapsed_steps: int = 1,
    ) -> Optional[MPPITrace]:
        """Advance the previous control sequence when it remains safe.

        This is the inexpensive receding-horizon path: a new BEV still
        validates the complete remaining trajectory, but no random rollout
        batch is generated unless that validation fails.  It therefore reacts
        immediately when a newly visible obstacle intersects the cached path
        while avoiding redundant MPPI sampling in open space.
        """
        if self._u is None:
            return None
        steps = max(1, int(elapsed_steps))
        if steps >= len(self._u):
            return None
        controls = self._u.copy()
        controls[:-steps] = controls[steps:]
        controls[-steps:] = controls[-steps - 1]
        velocity = np.asarray(current_velocity, np.float32)
        controls = self._limit(controls, velocity)
        trace = self._rollout_cost(controls, velocity, planning, checker, goal_xy)
        if not trace.valid:
            return None
        self._u = controls
        return trace

    def _warm_start(self, elapsed_steps: int = 1) -> np.ndarray:
        n = max(2, int(self.cfg.horizon_steps))
        if self._u is None or self._u.shape != (n, 3):
            self._u = np.zeros((n, 3), np.float32)
        else:
            steps = max(1, int(elapsed_steps))
            if steps >= n:
                self._u.fill(0.0)
            else:
                self._u[:-steps] = self._u[steps:]
                self._u[-steps:] = self._u[-steps - 1]
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

    def _limit_batch(self, controls: np.ndarray, initial_velocity: np.ndarray) -> np.ndarray:
        """Apply velocity and acceleration limits to ``[samples, horizon, 3]``.

        The former implementation called :meth:`_limit` once per sampled
        trajectory.  That was correct but made Python execute thousands of
        tiny array operations for every camera frame.  Limits at each horizon
        step are independent across trajectories, so they can be applied in
        one batched operation.
        """
        c = controls.copy()
        c[..., 0] = np.clip(c[..., 0], self.cfg.min_vx, self.cfg.max_vx)
        c[..., 1] = np.clip(c[..., 1], -self.cfg.max_vy, self.cfg.max_vy)
        c[..., 2] = np.clip(c[..., 2], -self.cfg.max_wz, self.cfg.max_wz)
        previous = np.broadcast_to(initial_velocity.astype(np.float32), (len(c), 3)).copy()
        limits = np.array(
            [self.cfg.max_accel_vx, self.cfg.max_accel_vy, self.cfg.max_accel_wz],
            np.float32,
        ) * self.cfg.dt
        for k in range(c.shape[1]):
            c[:, k] = np.clip(c[:, k], previous - limits, previous + limits)
            previous = c[:, k]
        return c

    def _rollout_batch(
        self,
        controls: np.ndarray,
        initial_velocity: np.ndarray,
        planning: PlanningBEV,
        checker: FootprintChecker,
        goal_xy: Tuple[float, float],
    ) -> list[MPPITrace]:
        """Score all sampled trajectories with vectorized footprint checks.

        Collision, unknown-surface, and costmap rules are identical to
        :meth:`_rollout_cost`.  Only the execution layout changes: one NumPy
        operation handles all rollouts at a horizon step instead of entering
        Python once per pose.  This keeps the safety model intact while making
        the controller practical on CPU-only robots.
        """
        samples, steps, _ = controls.shape
        poses = np.zeros((samples, steps + 1, 3), np.float32)
        costs = np.zeros(samples, np.float64)
        alive = np.ones(samples, bool)
        reasons = np.full(samples, "ok", dtype=object)
        x = np.zeros(samples, np.float32)
        y = np.zeros(samples, np.float32)
        yaw = np.zeros(samples, np.float32)
        previous = np.broadcast_to(initial_velocity.astype(np.float32), (samples, 3))

        body = checker._body_points
        bx, by = body[:, 0], body[:, 1]
        nx, ny = planning.grid.shape
        resolution = planning.grid.cfg.resolution
        x_min, y_min = planning.grid.cfg.x_min, planning.grid.cfg.y_min
        blocked = checker.layers.inflated_hard_blocked
        observed = planning.grid.observed
        traversability = planning.grid.traversability
        costmap = planning.planning_cost
        cfg = checker.cfg
        gx, gy = goal_xy

        def reject(mask: np.ndarray, reason: str) -> None:
            newly_invalid = alive & mask
            reasons[newly_invalid] = reason
            alive[newly_invalid] = False

        for k in range(steps):
            u = controls[:, k]
            yaw_mid = yaw + 0.5 * u[:, 2] * self.cfg.dt
            c, s = np.cos(yaw_mid), np.sin(yaw_mid)
            x += (u[:, 0] * c - u[:, 1] * s) * self.cfg.dt
            y += (u[:, 0] * s + u[:, 1] * c) * self.cfg.dt
            yaw += u[:, 2] * self.cfg.dt
            poses[:, k + 1] = np.stack((x, y, yaw), axis=1)

            # One [samples, footprint_points] index computation replaces a
            # FootprintChecker.check_pose call for every rollout pose.
            wx = x[:, None] + c[:, None] * bx - s[:, None] * by
            wy = y[:, None] + s[:, None] * bx + c[:, None] * by
            ii = ((wx - x_min) / resolution).astype(np.int32)
            jj = ((wy - y_min) / resolution).astype(np.int32)
            inside = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            reject(~inside.all(axis=1), "out_of_bounds")

            # Clip only for safe array gathering; rows that were outside are
            # already invalid and never contribute cost below.
            ii = np.clip(ii, 0, nx - 1)
            jj = np.clip(jj, 0, ny - 1)
            reject(blocked[ii, jj].any(axis=1), "collision")

            relax = (x < self.cfg.start_relax_distance) & (np.abs(y) < 0.75)
            need_surface = alive & ~relax
            footprint_observed = observed[ii, jj]
            unknown_fraction = (~footprint_observed).mean(axis=1)
            reject(need_surface & (unknown_fraction > cfg.max_unknown_fraction), "unknown")

            active_surface = alive & ~relax
            observed_count = footprint_observed.sum(axis=1)
            reject(active_surface & (observed_count == 0), "unknown")
            # Aggregate over observed cells without ragged indexing.
            observed_float = footprint_observed.astype(np.float32)
            trav = traversability[ii, jj]
            denom = np.maximum(observed_count, 1)
            ground_fraction = ((trav >= cfg.traversability_threshold) * observed_float).sum(axis=1) / denom
            mean_trav = (trav * observed_float).sum(axis=1) / denom
            reject(
                active_surface
                & ((ground_fraction < cfg.min_ground_fraction) | (mean_trav < cfg.min_mean_traversability)),
                "not_traversable",
            )

            ci = ((x - x_min) / resolution).astype(np.int32)
            cj = ((y - y_min) / resolution).astype(np.int32)
            cell_inside = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
            reject(~cell_inside, "costmap")
            ci = np.clip(ci, 0, nx - 1)
            cj = np.clip(cj, 0, ny - 1)
            cell_cost = costmap[ci, cj]
            reject(~np.isfinite(cell_cost), "costmap")

            valid_now = alive
            distance = np.hypot(gx - x, gy - y)
            smooth = np.sum((u - previous) ** 2, axis=1)
            step_cost = (
                self.cfg.goal_weight * distance * self.cfg.dt
                + self.cfg.costmap_weight * cell_cost * self.cfg.dt
                + self.cfg.control_weight * np.sum(u * u, axis=1) * self.cfg.dt
                + self.cfg.smooth_weight * smooth
            )
            if self._turn_side_remaining > 0 and self._turn_side:
                step_cost += (
                    self.cfg.side_commit_weight
                    * np.maximum(0.0, -u[:, 2] * self._turn_side)
                    * self.cfg.dt
                )
            costs[valid_now] += step_cost[valid_now]
            previous = u

        terminal_distance = np.hypot(gx - x, gy - y)
        desired = np.arctan2(gy - y, gx - x)
        heading_error = (desired - yaw + np.pi) % (2.0 * np.pi) - np.pi
        costs[alive] += (
            self.cfg.terminal_goal_weight * terminal_distance[alive]
            + self.cfg.heading_weight * heading_error[alive] ** 2
        )
        costs[~alive] = self.cfg.collision_cost
        return [
            MPPITrace(controls[i], poses[i], float(costs[i]), bool(alive[i]), str(reasons[i]))
            for i in range(samples)
        ]

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
            chk = checker.check_pose(x, y, yaw, relax_surface=relax)
            if not chk.valid:
                return MPPITrace(controls, poses, self.cfg.collision_cost, False, chk.reason)
            cell = planning.grid.xy_to_ij(x, y)
            if cell is None or not np.isfinite(planning.planning_cost[cell]):
                return MPPITrace(controls, poses, self.cfg.collision_cost, False, "costmap")
            dist = float(np.hypot(gx - x, gy - y))
            cost += self.cfg.goal_weight * dist * self.cfg.dt
            cost += self.cfg.costmap_weight * float(planning.planning_cost[cell]) * self.cfg.dt
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

    def plan(
        self,
        planning: PlanningBEV,
        checker: FootprintChecker,
        goal_xy: Tuple[float, float],
        current_velocity: Tuple[float, float, float],
        elapsed_steps: int = 1,
    ) -> MPPIResult:
        velocity = np.asarray(current_velocity, np.float32)
        nominal = self._warm_start(elapsed_steps)
        # Add a gentle forward prior; obstacle avoidance is still entirely
        # trajectory-cost driven, not a left/right image-space decision.
        nominal[:, 0] = np.maximum(
            nominal[:, 0], min(self.cfg.nominal_vx, self.cfg.max_vx)
        )
        n, t = max(1, self.cfg.samples), nominal.shape[0]
        noise = self._rng.normal(0.0, [self.cfg.noise_vx, self.cfg.noise_vy, self.cfg.noise_wz],
                                 size=(n, t, 3)).astype(np.float32)
        sequences = self._limit_batch(nominal[None, :, :] + noise, velocity)
        traces = self._rollout_batch(sequences, velocity, planning, checker, goal_xy)
        costs = np.asarray([trace.cost for trace in traces], np.float64)
        valid = np.array([x.valid for x in traces], bool)
        if not valid.any():
            self._u = np.zeros_like(nominal)
            reasons: Dict[str, float] = {"invalid": float(n)}
            for trace in traces:
                reasons[trace.reason] = reasons.get(trace.reason, 0.0) + 1.0
            return MPPIResult((0.0, 0.0, 0.0), None, "NO_VALID_TRAJECTORY", traces, 0,
                              diagnostics=reasons)
        minimum = float(costs[valid].min())
        weights = np.zeros(n, np.float64)
        weights[valid] = np.exp(-(costs[valid] - minimum) / max(self.cfg.temperature, 1e-4))
        weights /= max(weights.sum(), 1e-12)
        self._u = np.tensordot(weights, sequences, axes=(0, 0)).astype(np.float32)
        self._u = self._limit(self._u, velocity)
        deadband = max(0.0, float(self.cfg.angular_deadband))
        if deadband and abs(float(self._u[0, 2])) < deadband:
            # A tiny random yaw component is only suppressed when the
            # resulting straightened *whole trajectory* remains valid.  A
            # mild initial turn can be essential to clear a nearby obstacle,
            # so deadbanding a command without this collision check is unsafe.
            straightened = self._u.copy()
            straightened[np.abs(straightened[:, 2]) < deadband, 2] = 0.0
            if self._rollout_cost(
                straightened, velocity, planning, checker, goal_xy
            ).valid:
                self._u = straightened
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
        return MPPIResult(tuple(map(float, self._u[0])), selected, "MPPI", traces,
                          int(valid.sum()), float(selected.cost),
                          {"min_sample_cost": minimum, "valid_fraction": float(valid.mean())})

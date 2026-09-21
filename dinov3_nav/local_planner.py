# -*- coding: utf-8 -*-
"""Goal-directed, footprint-aware local trajectory planner with recovery.

Planner states (simple state machine, updated every cycle):
    GO_TO_GOAL        drive along the best scored forward trajectory
    AVOID_LEFT/RIGHT  forward swerve around an obstacle, side commitment locked
    ROTATE_TO_GOAL    goal far to the side -> turn in place first
    ROTATE_RECOVERY   every forward trajectory invalid -> rotate toward the
                      best free opening, replan next frame (never dead-lock)
    GOAL_REACHED      handled by the caller (goal tolerance)

Hard constraints (immediate reject):
    * rectangular footprint intersects an inflated hard-obstacle cell
    * footprint leaves the local BEV
    * forward candidate's straight extension to obstacle_trigger_distance
      hits an inflated obstacle (early-turn trigger)
Soft constraints (scored):
    * unknown-cell ratio inside the footprint (capped, then penalised)
    * mean traversability along the trajectory
    * clearance (distance transform), progress toward the global goal,
      end heading, turn magnitude, smoothness vs. previous command
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, ceil, cos, hypot, pi, sin
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bev import BEVGrid
from .footprint import FootprintChecker

MODE_GO_TO_GOAL = "GO_TO_GOAL"
MODE_AVOID_LEFT = "AVOID_LEFT"
MODE_AVOID_RIGHT = "AVOID_RIGHT"
MODE_ROTATE_TO_GOAL = "ROTATE_TO_GOAL"
MODE_ROTATE_RECOVERY = "ROTATE_RECOVERY"
MODE_BACK_UP = "BACK_UP"
MODE_GOAL_REACHED = "GOAL_REACHED"
MODE_NO_VALID = "NO_VALID_TRAJECTORY"


def wrap_angle(a: float) -> float:
    return (a + pi) % (2.0 * pi) - pi


@dataclass
class PlannerConfig:
    horizon: float = 2.0
    dt: float = 0.10
    v_samples: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40)
    # A quadruped pivots nearly in place: w up to 1.2 rad/s gives a 0.25 m
    # turning radius at v = 0.3, tight enough to swerve around a near wall.
    w_samples: Tuple[float, ...] = (
        -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
    )
    # Poses closer than this to the robot only get the collision check: the
    # forward camera physically cannot observe ground under/at the robot.
    # The effective relax distance auto-adapts to the nearest observed ground.
    start_relax_distance: float = 0.90
    clearance_cap: float = 0.80
    # scoring weights
    progress_weight: float = 2.2
    traversability_weight: float = 0.9
    clearance_weight: float = 1.2
    heading_weight: float = 0.8
    turn_weight: float = 0.20
    smooth_weight: float = 0.25
    side_lock_weight: float = 0.35
    # mode classification / state machine
    # Forward candidates must respect a minimum turning radius: tighter curls
    # are pivoting, which belongs to recovery rotation, not forward arcs.
    min_turn_radius: float = 0.15
    # Early-turn radius (path length): each forward candidate is probed past
    # its rollout by a straight continuation along the final heading until
    # this total path length; a probe that hits an inflated obstacle rejects
    # the candidate ("front_blocked"). Avoidance therefore starts when a
    # wall enters this radius, not when it reaches the horizon * v collision
    # lookahead (~0.8 m at v = 0.4). 0 disables the probe.
    obstacle_trigger_distance: float = 1.50
    avoid_w_threshold: float = 0.30
    side_lock_s: float = 2.00
    max_angular: float = 1.2
    turn_gain: float = 1.4
    turn_in_place_angle: float = 0.70          # rad, goal bearing threshold
    # recovery scan (rotate toward the best free opening)
    escape_max_angle: float = 1.40
    escape_angle_step: float = 0.17
    escape_ray_length: float = 1.40
    escape_min_distance: float = 0.45
    escape_goal_weight: float = 0.60
    escape_turn_weight: float = 0.12
    recovery_rotate_speed: float = 0.50
    recovery_min_rotate: float = 0.25
    # last resort when boxed in (cannot advance, cannot rotate): straight
    # reverse; the area behind the robot was just traversed
    backup_speed: float = 0.10


@dataclass
class CandidateTrace:
    v: float
    w: float
    poses: np.ndarray
    valid: bool
    reason: str
    score: float = float("-inf")

    @property
    def command(self) -> Tuple[float, float]:
        return float(self.v), float(self.w)


@dataclass
class PlanResult:
    trajectory: Optional[CandidateTrace]
    mode: str
    diagnostics: Dict[str, int] = field(default_factory=dict)
    candidates: List[CandidateTrace] = field(default_factory=list)
    escape_heading: Optional[float] = None
    avoid_side: int = 0            # -1 right, +1 left while locked
    relax_distance: float = 0.0
    near_ground_x: Optional[float] = None

    @property
    def command(self) -> Tuple[float, float]:
        if self.trajectory is None:
            return 0.0, 0.0
        return self.trajectory.command


class LocalTrajectoryPlanner:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self._side: int = 0          # committed avoidance side while locked
        self._side_until: float = 0.0  # absolute seconds

    # ---------------------------------------------------------------- rollout
    def _rollout(self, v: float, w: float) -> np.ndarray:
        n = max(1, int(round(self.cfg.horizon / self.cfg.dt)))
        poses = np.zeros((n + 1, 3), np.float32)
        x = y = yaw = 0.0
        for k in range(1, n + 1):
            # midpoint integration is noticeably better for tight arcs
            yaw_mid = yaw + 0.5 * w * self.cfg.dt
            x += v * cos(yaw_mid) * self.cfg.dt
            y += v * sin(yaw_mid) * self.cfg.dt
            yaw = wrap_angle(yaw + w * self.cfg.dt)
            poses[k] = (x, y, yaw)
        return poses

    # ------------------------------------------------------------- utilities
    def _rotation_is_safe(
        self,
        checker: FootprintChecker,
        target_angle: float,
        guard: Optional[np.ndarray] = None,
    ) -> bool:
        """In-place rotation sweep. Checked against RAW hard obstacles: the
        safety-margin inflation is planning comfort, and a robot already
        inside the inflation ring must still be allowed to turn its corners
        (raw collision stays a hard constraint). With a margin ``guard`` (the
        inflated cells under the current pose) the sweep additionally must
        not acquire any NEW inflated cell, i.e. it may not close in further."""
        steps = max(2, int(abs(target_angle) / 0.10) + 1)
        for yaw in np.linspace(0.0, target_angle, steps):
            chk = checker.check_pose(
                0.0, 0.0, float(yaw), relax_surface=True, ignore_inflation=True
            )
            if not chk.valid:
                return False
            if guard is not None:
                cells = checker.inflated_cells_under(0.0, 0.0, float(yaw))
                if cells is None or not np.isin(cells, guard).all():
                    return False
        return True

    def _front_extension_blocked(
        self,
        checker: FootprintChecker,
        poses: np.ndarray,
        v: float,
        guard: Optional[np.ndarray] = None,
    ) -> bool:
        """Early-turn probe: continue the candidate straight along its final
        heading until the total path length reaches
        ``obstacle_trigger_distance``. Only a collision with the obstacle
        layer blocks the probe: unknown or out-of-BEV extension must not
        (unknown != obstacle). With a margin ``guard`` the probe follows the
        same raw-obstacle + no-new-inflated-cell rule as the rollout, so
        wall-hugging recovery keeps working."""
        trigger = self.cfg.obstacle_trigger_distance
        if trigger <= 0.0:
            return False
        remaining = trigger - abs(v) * self.cfg.horizon
        if remaining <= 0.0:
            return False  # the rollout already covers the trigger distance
        x, y, yaw = map(float, poses[-1])
        step = max(checker.bev.cfg.resolution, 0.05)
        c, s = cos(yaw), sin(yaw)
        n = max(1, int(ceil(remaining / step)))
        for k in range(1, n + 1):
            d = min(remaining, k * step)
            chk = checker.check_pose(
                x + c * d, y + s * d, yaw,
                relax_surface=True, ignore_inflation=(guard is not None),
            )
            if not chk.valid and chk.reason == "collision":
                return True
            if guard is not None:
                cells = checker.inflated_cells_under(x + c * d, y + s * d, yaw)
                if cells is not None and not np.isin(cells, guard).all():
                    return True
        return False

    def _set_side(self, side: int, now: float):
        self._side = side
        self._side_until = now + max(0.0, self.cfg.side_lock_s)

    def _backup_trajectory(
        self, checker: FootprintChecker, guard: Optional[np.ndarray] = None
    ) -> Optional[CandidateTrace]:
        """Straight reverse, collision-checked only (the ground behind the
        robot is outside the camera FOV; it was traversed to get here)."""
        v = -max(0.05, self.cfg.backup_speed)
        n = max(1, int(round(self.cfg.horizon / self.cfg.dt)))
        poses = np.zeros((n + 1, 3), np.float32)
        for k in range(1, n + 1):
            poses[k, 0] = v * k * self.cfg.dt
        for x, y, yaw in poses[1:]:
            # Raw obstacles + no NEW inflated cells: when boxed in close to a
            # wall the inflated ring already covers the current pose;
            # retreating along a raw-clear path is exactly the escape we want.
            if not checker.check_pose(float(x), float(y), float(yaw),
                                      relax_surface=True,
                                      ignore_inflation=True).valid:
                return None
            if guard is not None:
                cells = checker.inflated_cells_under(float(x), float(y), float(yaw))
                if cells is not None and not np.isin(cells, guard).all():
                    return None
        return CandidateTrace(v=v, w=0.0, poses=poses, valid=True,
                              reason="backup", score=0.0)

    # -------------------------------------------------------- recovery scan
    def _escape_heading(
        self,
        checker: FootprintChecker,
        goal_xy: Tuple[float, float],
        now: float,
        guard: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], float]:
        """Scan left/right rays for the most promising opening.

        Considers free length, clearance to obstacles, goal alignment, turn
        cost and the currently committed side. Never decides purely by
        "goal is left -> turn left": the more open side wins.
        """
        gx, gy = goal_xy
        goal_heading = atan2(gy, gx)
        max_a = max(0.1, self.cfg.escape_max_angle)
        step = max(0.05, self.cfg.escape_angle_step)
        all_angles = np.arange(-max_a, max_a + 0.5 * step, step)
        side_locked = self._side != 0 and now < self._side_until
        # While a side commitment is active, only scan that side; if it has
        # no opening at all, fall back to the full scan (auto-unlock).
        angle_sets = [all_angles[np.sign(all_angles) == self._side], all_angles] \
            if side_locked else [all_angles]

        best_angle: Optional[float] = None
        best_score = float("-inf")
        ray_step = max(checker.bev.cfg.resolution * 1.5, 0.07)

        for angles in angle_sets:
            for a in angles:
                a = float(a)
                # avoid selecting almost straight ahead after all forward
                # primitives failed
                if abs(a) < 0.5 * step:
                    continue
                if not self._rotation_is_safe(checker, a, guard):
                    continue
                reached = 0.0
                r = max(0.10, self.cfg.start_relax_distance)
                while r <= self.cfg.escape_ray_length + 1e-6:
                    x, y = r * cos(a), r * sin(a)
                    chk = checker.check_pose(
                        x, y, a, relax_surface=(r <= self.cfg.start_relax_distance)
                    )
                    if not chk.valid:
                        break
                    reached = r
                    r += ray_step
                if reached < self.cfg.escape_min_distance:
                    continue
                goal_align = cos(wrap_angle(a - goal_heading))
                turn_cost = abs(a) / max_a
                clearance = min(
                    checker.clearance_at(reached * cos(a), reached * sin(a))
                    / max(self.cfg.clearance_cap, 1e-3), 1.0
                )
                score = (
                    reached / max(self.cfg.escape_ray_length, 1e-3)
                    + self.cfg.escape_goal_weight * goal_align
                    + 0.25 * clearance
                    - self.cfg.escape_turn_weight * turn_cost
                )
                if score > best_score:
                    best_score = score
                    best_angle = a
            if best_angle is not None:
                break
        return best_angle, best_score

    # ------------------------------------------------------------------ plan
    def plan(
        self,
        bev: BEVGrid,
        checker: FootprintChecker,
        goal_xy: Tuple[float, float],
        previous_w: float = 0.0,
        now: float = 0.0,
    ) -> PlanResult:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        d0 = hypot(gx, gy)
        diagnostics: Dict[str, int] = {
            "total": 0,
            "valid": 0,
            "collision": 0,
            "unknown": 0,
            "not_traversable": 0,
            "out_of_bounds": 0,
            "front_blocked": 0,
        }
        if d0 < 1e-4:
            return PlanResult(None, MODE_GOAL_REACHED, diagnostics)

        near_ground_x = bev.nearest_ground_x()
        relax = float(self.cfg.start_relax_distance)
        if near_ground_x is not None:
            relax = max(relax, near_ground_x + 0.5 * bev.cfg.resolution)

        goal_bearing = atan2(gy, gx)
        side_locked = self._side != 0 and now < self._side_until
        # Margin guard: if the robot already stands inside the inflation
        # ring (e.g. sliding along a wall closer than the margin), the margin
        # can no longer be enforced absolutely. Trajectory poses then check
        # RAW obstacles (hard) plus "no NEW inflated cell": parallel sliding
        # keeps its current cells and stays legal, closing in further is
        # rejected. Cell-centre sampling alone would otherwise let a corner
        # poke ~one cell into an obstacle unnoticed.
        origin_cells = checker.inflated_cells_under(0.0, 0.0, 0.0)
        margin_guard = origin_cells if origin_cells is not None and origin_cells.size \
            else None
        # If the requested goal itself is far to the side, turning first is
        # the physically meaningful action; the footprint stays in place.
        # Suppressed while an avoidance side is committed: re-centering on
        # the goal bearing mid-detour caused rotate-left / rotate-right
        # limit cycles in front of walls.
        if (
            abs(goal_bearing) >= self.cfg.turn_in_place_angle
            and not side_locked
            and self._rotation_is_safe(checker, goal_bearing, margin_guard)
        ):
            w = float(np.clip(self.cfg.turn_gain * goal_bearing,
                              -self.cfg.max_angular, self.cfg.max_angular))
            traj = CandidateTrace(
                v=0.0, w=w, poses=np.array([[0.0, 0.0, 0.0]], np.float32),
                valid=True, reason="goal_turn", score=0.0,
            )
            return PlanResult(traj, MODE_ROTATE_TO_GOAL, diagnostics, [traj],
                              None, 0, relax, near_ground_x)

        candidates: List[CandidateTrace] = []
        best: Optional[CandidateTrace] = None          # best forward (v > 0)
        best_rotate: Optional[CandidateTrace] = None   # best in-place rotation
        max_forward = max(max(self.cfg.v_samples, default=0.1) * self.cfg.horizon, 0.1)
        max_w = max(max((abs(v) for v in self.cfg.w_samples), default=0.1), 0.1)

        for v0 in self.cfg.v_samples:
            for w0 in self.cfg.w_samples:
                v, w = float(v0), float(w0)
                if v < 0.0:
                    continue
                if v == 0.0 and w == 0.0:
                    continue  # full stop is the fallback, not a candidate
                if v > 0.0 and abs(w) > 1e-6 \
                        and v / abs(w) < self.cfg.min_turn_radius:
                    continue  # near-pivot curl -> recovery territory
                if side_locked and w != 0.0 and np.sign(w) != self._side:
                    # Side commitment is decisive while active: counter-side
                    # swerves caused goal-bearing flip-flop in front of walls.
                    # The lock auto-expires (side_lock_s) and the escape scan
                    # unlocks early if the committed side seals up.
                    continue
                poses = self._rollout(v, w)
                reason = "ok"
                valid = True
                trav_scores: List[float] = []
                clear_scores: List[float] = []

                for x, y, yaw in poses[1:]:
                    if v == 0.0:
                        # In-place rotation: judge by the raw-obstacle sweep
                        # (same rule as the recovery paths) so a wall inside
                        # the inflation ring cannot forbid turning away.
                        if not self._rotation_is_safe(checker, float(yaw), margin_guard):
                            valid = False
                            reason = "collision"
                            diagnostics[reason] = diagnostics.get(reason, 0) + 1
                            break
                        continue
                    dist = hypot(float(x), float(y))
                    chk = checker.check_pose(
                        float(x), float(y), float(yaw),
                        relax_surface=(dist <= relax),
                        ignore_inflation=(margin_guard is not None),
                    )
                    if not chk.valid:
                        valid = False
                        reason = chk.reason
                        diagnostics[reason] = diagnostics.get(reason, 0) + 1
                        break
                    if margin_guard is not None:
                        cells = checker.inflated_cells_under(
                            float(x), float(y), float(yaw))
                        if cells is not None and not np.isin(cells, margin_guard).all():
                            valid = False
                            reason = "collision"
                            diagnostics[reason] = diagnostics.get(reason, 0) + 1
                            break
                    if dist > relax:
                        trav_scores.append(chk.mean_traversability)
                    clear_scores.append(
                        min(checker.clearance_at(float(x), float(y))
                            / max(self.cfg.clearance_cap, 1e-3), 1.0)
                    )

                if valid and v > 0.0 and self._front_extension_blocked(
                    checker, poses, v, margin_guard
                ):
                    # Obstacle inside the early-turn radius ahead: reject the
                    # candidates that keep pointing at it now, instead of
                    # waiting for the horizon * v lookahead to reach it.
                    valid = False
                    reason = "front_blocked"
                    diagnostics[reason] = diagnostics.get(reason, 0) + 1

                trace = CandidateTrace(v=v, w=w, poses=poses, valid=valid, reason=reason)
                diagnostics["total"] += 1
                if not valid:
                    candidates.append(trace)
                    continue

                diagnostics["valid"] += 1
                if v == 0.0:
                    # pure rotation: progress is heading alignment toward goal
                    yaw_n = float(poses[-1, 2])
                    progress = cos(wrap_angle(goal_bearing - yaw_n))
                    trav_score = 1.0
                else:
                    xn, yn, yaw_n = map(float, poses[-1])
                    dn = hypot(gx - xn, gy - yn)
                    progress = (d0 - dn) / max_forward
                    trav_score = float(np.mean(trav_scores)) if trav_scores else 1.0
                clearance = float(np.mean(clear_scores)) if clear_scores else 0.0
                desired_end = atan2(gy - poses[-1, 1], gx - poses[-1, 0])
                heading = cos(wrap_angle(float(desired_end) - float(poses[-1, 2])))
                turn_cost = abs(w) / max_w
                smooth_cost = abs(w - previous_w) / (2.0 * max_w)
                side_bias = 0.0
                if side_locked and w != 0.0:
                    side_bias = self.cfg.side_lock_weight * (
                        1.0 if np.sign(w) == self._side else -1.0
                    )
                score = (
                    self.cfg.progress_weight * progress
                    + self.cfg.traversability_weight * trav_score
                    + self.cfg.clearance_weight * clearance
                    + self.cfg.heading_weight * heading
                    - self.cfg.turn_weight * turn_cost
                    - self.cfg.smooth_weight * smooth_cost
                    + side_bias
                )
                trace.score = float(score)
                candidates.append(trace)
                if v == 0.0:
                    if best_rotate is None or trace.score > best_rotate.score:
                        best_rotate = trace
                elif best is None or trace.score > best.score:
                    best = trace

        if best is not None:
            if abs(best.w) >= self.cfg.avoid_w_threshold:
                side = 1 if best.w > 0 else -1
                self._set_side(side, now)
                mode = MODE_AVOID_LEFT if side > 0 else MODE_AVOID_RIGHT
            else:
                mode = MODE_GO_TO_GOAL
            return PlanResult(best, mode, diagnostics, candidates,
                              None, self._side, relax, near_ground_x)

        # No forward primitive is feasible. Scan for a safe opening and rotate
        # in place toward it; the next RGB-D frame re-evaluates forward arcs.
        # This is the anti-deadlock recovery: cmd=(0,0) only when even
        # rotating is unsafe or nothing opens within the scan range.
        escape_heading, escape_score = self._escape_heading(
            checker, (gx, gy), now, margin_guard)
        if escape_heading is not None:
            side = 1 if escape_heading > 0 else -1
            self._set_side(side, now)
            mag = max(self.cfg.recovery_min_rotate,
                      min(self.cfg.recovery_rotate_speed, self.cfg.max_angular))
            traj = CandidateTrace(
                v=0.0, w=float(side * mag),
                poses=np.array([[0.0, 0.0, 0.0]], np.float32),
                valid=True, reason="escape_turn", score=float(escape_score),
            )
            candidates.append(traj)
            return PlanResult(traj, MODE_ROTATE_RECOVERY, diagnostics,
                              candidates, escape_heading, side, relax, near_ground_x)

        # Nothing opens within the scan range; if goal-aligned in-place
        # rotation is at least safe, keep turning toward the goal.
        if best_rotate is not None:
            self._set_side(1 if best_rotate.w > 0 else -1, now)
            return PlanResult(best_rotate, MODE_ROTATE_TO_GOAL, diagnostics,
                              candidates, None, self._side, relax, near_ground_x)

        # Fully boxed in (even rotating collides): straight reverse.
        backup = self._backup_trajectory(checker, margin_guard)
        if backup is not None:
            self._set_side(0, now)
            candidates.append(backup)
            return PlanResult(backup, MODE_BACK_UP, diagnostics,
                              candidates, None, 0, relax, near_ground_x)

        self._set_side(0, now)
        return PlanResult(None, MODE_NO_VALID, diagnostics, candidates,
                          None, 0, relax, near_ground_x)

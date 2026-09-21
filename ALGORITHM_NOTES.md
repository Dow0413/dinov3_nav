# DINOv3 quadruped visual navigation — BEV planner version

ROS 2 Jazzy package (`ament_python`). It keeps the existing `ground_seg`
perception API untouched and replaces image-space waypoint chasing with:

`DINOv3 traversability + registered depth -> metric local BEV (base_link)
-> rectangular footprint checks -> goal-directed trajectory rollout
-> progress/traversability/clearance/heading/turn/smooth score -> cmd_vel`.

## Frames (verified, not guessed)

- Planner frame: `base_link` (`planner.frame`). The global goal is transformed
  `odom -> base_link` every cycle; planning never happens in camera frames.
- Camera: the Gazebo sim stamps images `zed_camera_link` with an identity
  rotation vs `base_link`, but pixel data is optical convention. The node
  composes `T_body_from_optical` itself (`camera.projection_convention`).
  `zed_left_camera_optical_frame` does not exist in this sim and is not used.
- Sanity gate: center pixel must land x>0, |y|~0; image-left -> +y. A failed
  projection sanity check independently forces cmd_vel = 0.

## Pipeline per synced RGB+depth frame

1. `ground_seg` ground mask (SAM2 refinement preserved verbatim).
2. Back-projection (stride `bev.pixel_stride`) into the local BEV:
   0.05 m cells, x -0.8..4.0, y ±2.2. Height evidence classifies obstacles
   (z > ground_z + 0.15) and drops (z < ground_z - 0.25); morphological
   closing fills raster sparsity of the ground support only — free space is
   never grown from nothing, and unknown != obstacle != free.
3. Footprint layers: strong non-traversable clusters promoted to hard,
   inflated by `robot.safety_margin`, distance transform for clearance.
4. Trajectory rollout (`planner.v/w_samples`, horizon 2 s): hard reject on
   raw/inflated obstacle intersection; soft-scored unknown ratio,
   traversability, clearance, progress, heading, turn, smoothness.
   Forward arcs respect `planner.min_turn_radius` — tighter curls are
   pivoting, i.e. recovery territory. Each forward arc is additionally
   probed straight along its end heading out to
   `planner.obstacle_trigger_distance` (1.5 m path length): a blocked probe
   rejects the candidate (`front_blocked`), so the robot commits to a
   swerve while the wall is still outside the horizon*v lookahead (~0.8 m).
5. Anti-deadlock ladder: forward arcs -> side-locked escape rotation ->
   goal-aligned rotation -> straight reverse. `cmd=(0,0)` only when the
   footprint truly cannot move or rotate.

## Side commitment and margin semantics

- `AVOID_LEFT/RIGHT` locks the avoidance side for `recovery.side_lock_s`;
  while locked, counter-side forward candidates are not even sampled (no
  left/right flip-flop in front of walls), and the decisive goal turn is
  suppressed mid-detour. The lock auto-expires and the escape scan unlocks
  early if the committed side seals.
- Inflation (`robot.safety_margin`) is planning comfort, not physics. If the
  robot already stands inside the inflation ring, trajectory poses fall back
  to the RAW hard layer plus a containment rule: no NEW inflated cells may be
  acquired. Parallel sliding along a wall stays legal; closing in further is
  rejected. This keeps recovery possible without letting cell-centre sampling
  aliasing hide a corner poking into an obstacle.

## Build

```bash
cd /home/dow/DOW/dinov3/dinov3_nav_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select dinov3_nav --symlink-install
source install/setup.bash
ros2 launch dinov3_nav dinov3_nav.launch.py
```

## Debug topics

- `/dinov3_nav/bev_debug`: gray unknown, dark-gray observed, green
  traversable, red hard obstacle, orange inflated, blue candidate arcs
  (dark = rejected), white selected trajectory, yellow goal arrow, robot at
  bottom center.
- `/dinov3_nav/bev_traversability|bev_obstacle|bev_observed` (mono8 layers).
- `/dinov3_nav/status` (String): `MODE goal(base)=(x,y) cand=N valid=N
  coll=N front=N unk=N lowtrav=N oob=N sel=(v,w) score=S cmd=(v,w)` per
  cycle (`front` = candidates rejected by the early-turn probe).

## Safety sequence

1. Keep `control.enabled: false`.
2. Confirm `CameraInfo` and `base_link <- zed_camera_link` TF.
3. Confirm BEV: forward objects appear upward, image-left -> +y (left).
4. Measure and set `robot.length/width/center_*`.
5. Confirm obstacles render red/orange and open ground stays green.
6. Only then set `control.enabled: true`, starting with low `planner.v_samples`.

Watch out: an external `/cmd_vel` publisher on the same domain fights this
node — run tests on an isolated `ROS_DOMAIN_ID`.

## Tests

```bash
cd dinov3_nav_ws/src/dinov3_nav
PYTHONPATH=. /home/dow/DOW/dinov3/.venv/bin/python tests/test_planner_synthetic.py
```

Covers: RGB-D back-projection axis sanity (D), open ground (A), blocked
dead-ahead with side commitment (B), swerve-vs-recovery at two wall distances
(B2), sparse unknown holes (C), and the closed-loop acceptance run around a
wall toward a 19 m goal (E): rounds the wall, re-converges, zero deadlock
commands, corner clearance never below the safety margin.

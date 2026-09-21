# DINOv3 RGB-D local navigation — MPPI version

This ROS 2 package leaves the existing `ground_seg` DINOv3/SAM2 inference path intact. DINOv3 answers only **where the current image is traversable**; it does not select left/right steering.

```
RGB + registered depth + CameraInfo
  -> DINOv3/SAM2 image-space traversability
  -> RGB-D back-projection and base_link BEV
  -> odom-motion-compensated local costmap
  -> hard-obstacle promotion + footprint inflation + clearance field
  -> MPPI future (vx, vy, wz) rollouts
  -> Twist on cmd_vel
```

## Frames, TF and odometry

- The local BEV, footprint and trajectories are in `planner_frame` (default `base_link`): x forward and y left.
- `PoseStamped` goals are transformed from their header frame (default `odom`) into `base_link` every camera update via TF.
- Temporal BEV fusion obtains `odom <- base_link` from TF, reprojects prior evidence into the current robot-centred grid, and decays it. Missing TF resets temporal evidence safely.
- `/odom` supplies measured body-frame `twist.twist.linear.{x,y}` and `angular.z`, the initial velocity state for every MPPI rollout.
- The camera projection composes optical-to-body axes when configured as `optical`; a projection sanity check blocks non-zero commands if image centre is not forward or image-left does not map to base_link +y.

## Costmap semantics

Registered depth is back-projected with CameraInfo intrinsics and camera TF. Points above robust ground height become obstacles; drops and DINO non-traversable cells lower traversability. Ground-confirmed rays fill only observed support, so unknown is never fabricated as free. The map retains three states: free, unknown (finite but costly), and hard obstacle (forbidden). Hard obstacles are inflated by body half-width plus safety margin; a distance transform adds clearance cost.

## MPPI behaviour

`mppi.py` warm-starts the prior best control sequence, adds sampled Gaussian control noise, limits velocity and acceleration, then rolls every sequence over the horizon. Each trajectory is rejected if its rectangular footprint leaves BEV or intersects an inflated obstacle. Valid trajectories accumulate goal-progress, terminal-heading, costmap, effort and smoothness costs. The path-integral weighted sequence produces the next command; only its first `(vx, vy, wz)` is published.

A short yaw-side commitment regulariser prevents immediate sign reversal when nearly symmetric visual geometry fluctuates. It never accepts collision: every sequence remains footprint checked, and a reverse is only delayed with zero yaw until commitment expires.

`mppi.max_vy` defaults to zero for a differential-drive base. Set it nonzero only when the downstream controller genuinely supports `Twist.linear.y`.

## Topics

Inputs: RGB image, registered depth, CameraInfo, `/goal_pose`, `/odom`, and TF. Outputs include `/cmd_vel`, image-space ground/traversability/obstacle masks, costmap debug layers, `/dinov3_nav/bev_debug`, and `/dinov3_nav/planner_status`. Status reports valid rollout count, selected `(vx, vy, wz)`, and cost.

## Build and test

```bash
cd /home/dow/DOW/dinov3/dinov3_nav_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select dinov3_nav --symlink-install
source install/setup.bash
ros2 launch dinov3_nav dinov3_nav.launch.py

cd src/dinov3_nav
PYTHONPATH=. python3 -m pytest -q tests/test_mppi_synthetic.py tests/test_planning_bev.py
```

Keep `control.enabled: false` while validating TF, BEV orientation, intrinsics and measured footprint. Then enable it at a low `mppi.max_vx`.

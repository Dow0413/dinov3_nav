# -*- coding: utf-8 -*-
"""DINOv3 RGB-D traversability navigation with metric local planning.

Pipeline:
    RGB -> ground_seg(DINOv3 + optional SAM2) -> traversability mask
    registered depth + CameraInfo -> optical back-projection
      -> optical->body rotation -> TF -> base_link
    BEV (x forward, y left, base_link) with height-based hard obstacles
    global PoseStamped (odom) -> TF -> local goal in base_link
    BEV free/obstacle -> obstacle inflation -> local costmap -> MPPI rollout
    MPPI (vx, vy, wz sequences) -> /cmd_vel

Frames (verified against gazebo_sim_ws_3 zed_x_camera.xacro):
    planner/global frames   base_link / odom
    image+depth frame_id    zed_camera_link  (body orientation, identity rot
                            vs base_link), but pixel data is optical
                            convention -> composed here explicitly. No
                            zed_left_camera_optical_frame is required.
"""
from __future__ import annotations

import time
import threading
import os
from array import array
from math import hypot
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformException, TransformListener

from ground_seg import GroundSegmenter, SAM2MaskRefiner
from ground_seg.features import DEFAULT_CKPT
from ground_seg.sam2_refiner import DEFAULT_SAM2_CKPT

from .bev import BEVConfig, T_BODY_FROM_OPTICAL, build_local_bev, transform_to_matrix
from .debug_viz import (
    render_bev_layer,
    render_bev_debug,
    render_planning_bev,
    render_planning_layer,
    render_raw_bev,
)
from .footprint import (
    FootprintConfig,
    FootprintChecker,
)
from .mppi import MPPIConfig, MPPIPlanner, MPPIResult
from .planning_bev import (
    PlanningCostConfig,
    PlanningBEV,
    TemporalBEVConfig,
    TemporalBEVFusion,
    build_planning_bev,
)


_RGB_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    channels = _RGB_CHANNELS.get(enc)
    if channels is None:
        raise ValueError(
            f"unsupported encoding {msg.encoding!r}; use rgb8/bgr8/rgba8/bgra8/mono8"
        )
    if msg.is_bigendian:
        raise ValueError("big-endian images are not supported")
    h, w = msg.height, msg.width
    row = np.frombuffer(msg.data, dtype=np.uint8, count=h * msg.step)
    arr = row.reshape(h, msg.step)[:, : w * channels].reshape(h, w, channels)
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "rgba8":
        arr = arr[:, :, :3]
    elif enc == "bgra8":
        arr = arr[:, :, 2::-1]
    elif enc == "mono8":
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    # ``msg.data`` is exposed as a read-only buffer.  ``ascontiguousarray``
    # keeps that read-only flag when the source is already contiguous, then
    # torchvision warns because ``torch.from_numpy`` could write into it.
    # Make one owned C-order RGB copy at the ROS boundary instead.
    return np.array(arr, dtype=np.uint8, order="C", copy=True)


def depth_to_meters(msg: Image, scale: float = 1.0) -> np.ndarray:
    """Depth image -> float32 meters.

    32FC1 is assumed meters, 16UC1 millimeters (Gazebo plugin), then the
    user-supplied depth.scale multiplies the result (default 1.0).
    """
    enc = msg.encoding.lower()
    if enc == "32fc1":
        dtype, base = np.float32, 1.0
    elif enc == "16uc1":
        dtype, base = np.uint16, 0.001
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}; use 32FC1 or 16UC1")
    h, w = msg.height, msg.width
    item = np.dtype(dtype).itemsize
    if msg.step % item:
        raise ValueError(f"depth step {msg.step} is not divisible by item size {item}")
    row = np.frombuffer(msg.data, dtype=dtype, count=h * (msg.step // item))
    return row.reshape(h, msg.step // item)[:, :w].astype(np.float32) * (base * float(scale))


def gray_to_image_msg(mask: np.ndarray, header) -> Image:
    arr = np.ascontiguousarray(mask.astype(np.uint8))
    out = Image()
    out.header = header
    out.height, out.width = arr.shape[:2]
    out.encoding = "mono8"
    out.is_bigendian = 0
    out.step = out.width
    out.data = array("B", arr.tobytes())
    return out


def rgb_to_image_msg(rgb: np.ndarray, header) -> Image:
    arr = np.ascontiguousarray(rgb.astype(np.uint8))
    out = Image()
    out.header = header
    out.height, out.width = arr.shape[:2]
    out.encoding = "rgb8"
    out.is_bigendian = 0
    out.step = out.width * 3
    out.data = array("B", arr.tobytes())
    return out


class Dinov3NavNode(Node):
    def __init__(self):
        super().__init__("dinov3_nav")
        p = self._declare_params()

        # ----------------------------- perception
        self._gseg = GroundSegmenter(
            resolution=p["resolution"],
            ckpt=p["ckpt"],
            device=p["device"],
            seed_rows_frac=p["seed_rows_frac"],
            seed_cols_frac=p["seed_cols_frac"],
            threshold=p["threshold"],
        )
        # Depth remains mandatory for BEV geometry.  Its optional use inside
        # the DINO segmentation head is separately configurable, matching the
        # fast RGB-only path in ground_seg/main.py when disabled.
        self._gseg_depth_fusion = bool(p["segmentation.use_depth_fusion"])
        self._sam2: Optional[SAM2MaskRefiner] = None
        self._sam2_max_expand_px = max(0, int(p["sam2.max_expand_px"]))
        if p["sam2.enable"]:
            try:
                self._sam2 = SAM2MaskRefiner(
                    checkpoint=p["sam2.ckpt"], device=p["device"]
                )
            except Exception as exc:
                self.get_logger().warning(f"SAM2 unavailable; using DINO mask: {exc}")
        if p["perception.preload"]:
            self._warmup_perception()

        self._stairs = None
        if p["stairs.enable"]:
            try:
                from ground_seg import DINOStairsDetector

                self._stairs = DINOStairsDetector(
                    resolution=p["resolution"],
                    threshold=p["stairs.threshold"],
                    min_area_frac=p["stairs.min_area"],
                    max_area_frac=p["stairs.max_area"],
                    min_risers=p["stairs.min_risers"],
                    device=p["device"],
                    extractor=self._gseg._extractor,
                )
            except Exception as exc:
                self.get_logger().warning(f"stairs detector unavailable: {exc}")
        self._stairs_traversable = bool(p["stairs.treat_traversable"])

        # ----------------------------- depth / frames
        self._depth_enabled = bool(p["depth.enable"])
        self._depth_scale = float(p["depth.scale"])
        self._safe_distance = float(p["depth.safe_distance"])
        self._bottom_exempt = float(p["depth.bottom_exempt_frac"])
        self._min_depth = float(p["depth.min_valid"])
        self._planner_frame = str(p["planner_frame"]).strip()
        self._camera_frame_param = str(p["camera.frame"]).strip()
        self._projection_convention = str(p["camera.projection_convention"]).strip().lower()
        if self._projection_convention not in ("optical", "body"):
            raise ValueError(
                "camera.projection_convention must be 'optical' or 'body', got "
                f"{self._projection_convention!r}"
            )
        self._default_goal_frame = str(p["global.default_goal_frame"]).strip()
        self._camera_info: Optional[CameraInfo] = None
        self._cached_camera_key: Optional[Tuple[str, str]] = None
        self._cached_camera_T: Optional[np.ndarray] = None
        self._sanity_ok: Optional[bool] = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # Local planning is useful only for the newest camera observation.
        # Keep one best-effort image per topic so a slow inference/planning
        # cycle cannot build a seconds-old RGB-D backlog.
        self._image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ----------------------------- BEV / footprint / MPPI planner
        self._bev_cfg = BEVConfig(
            resolution=float(p["bev.resolution"]),
            x_min=float(p["bev.x_min"]),
            x_max=float(p["bev.x_max"]),
            y_min=float(p["bev.y_min"]),
            y_max=float(p["bev.y_max"]),
            min_depth=self._min_depth,
            max_depth=float(p["bev.max_depth"]),
            pixel_stride=int(p["bev.pixel_stride"]),
            min_observed_points=int(p["bev.min_observed_points"]),
            min_obstacle_points=int(p["bev.min_obstacle_points"]),
            close_radius_m=float(p["bev.close_radius_m"]),
            support_traversability=float(p["bev.support_traversability"]),
            obstacle_height=float(p["bev.obstacle_height"]),
            drop_height=float(p["bev.drop_height"]),
            min_ground_z_points=int(p["bev.min_ground_z_points"]),
            visibility_raycast=bool(p["bev.visibility_raycast"]),
            visibility_raycast_stride_px=int(p["bev.visibility_raycast_stride_px"]),
        )
        self._min_observed_fraction = float(p["bev.min_observed_fraction"])
        # MPPI checks the full rectangular footprint at every rollout pose.
        # Therefore this inflation is only the requested extra clearance;
        # adding half-width here would double-count the body geometry.
        self._foot_cfg = FootprintConfig(
            length=float(p["robot.length"]),
            width=float(p["robot.width"]),
            safety_margin=max(0.0, float(p["robot.safety_margin"])),
            center_x=float(p["robot.center_x"]),
            center_y=float(p["robot.center_y"]),
            sample_step=float(p["robot.footprint_sample_step"]),
            traversability_threshold=float(p["bev.traversability_threshold"]),
            hard_nontrav_threshold=float(p["bev.hard_nontrav_threshold"]),
            hard_nontrav_min_cells=int(p["bev.hard_nontrav_min_cells"]),
        )
        self._bev_fusion = TemporalBEVFusion(TemporalBEVConfig(
            enabled=bool(p["bev.temporal_enable"]),
            evidence_half_life_s=float(p["bev.temporal_half_life_s"]),
            max_evidence=float(p["bev.temporal_max_evidence"]),
            observation_hit=float(p["bev.temporal_observation_hit"]),
            obstacle_hit=float(p["bev.temporal_obstacle_hit"]),
            obstacle_clear=float(p["bev.temporal_obstacle_clear"]),
            min_observed_evidence=float(p["bev.temporal_min_observed_evidence"]),
            obstacle_evidence_threshold=float(p["bev.temporal_obstacle_threshold"]),
            cleanup_min_obstacle_cells=int(p["bev.cleanup_min_obstacle_cells"]),
            ego_clear_radius_m=float(p["bev.ego_clear_radius_m"]),
            ego_clear_traversability=float(p["bev.ego_clear_traversability"]),
        ))
        self._planning_cost_cfg = PlanningCostConfig(
            unknown_cost=float(p["planning.unknown_cost"]),
            nontraversable_cost=float(p["planning.nontraversable_cost"]),
            clearance_target_m=float(p["planning.clearance_target_m"]),
            clearance_cost_weight=float(p["planning.clearance_cost_weight"]),
        )
        self._fusion_frame = str(p["bev.temporal_frame"]).strip()
        self._prev_fusion_pose: Optional[np.ndarray] = None
        self._prev_fusion_time: Optional[float] = None
        self._planner = MPPIPlanner(MPPIConfig(
            horizon_steps=int(p["mppi.horizon_steps"]), dt=float(p["mppi.dt"]),
            samples=int(p["mppi.samples"]), temperature=float(p["mppi.temperature"]),
            noise_vx=float(p["mppi.noise_vx"]), noise_vy=float(p["mppi.noise_vy"]), noise_wz=float(p["mppi.noise_wz"]),
            min_vx=float(p["mppi.min_vx"]), max_vx=float(p["mppi.max_vx"]),
            nominal_vx=float(p["mppi.nominal_vx"]),
            max_vy=float(p["mppi.max_vy"]), max_wz=float(p["mppi.max_wz"]),
            max_accel_vx=float(p["mppi.max_accel_vx"]), max_accel_vy=float(p["mppi.max_accel_vy"]), max_accel_wz=float(p["mppi.max_accel_wz"]),
            start_relax_distance=float(p["mppi.start_relax_distance"]),
            goal_weight=float(p["mppi.goal_weight"]), terminal_goal_weight=float(p["mppi.terminal_goal_weight"]),
            heading_weight=float(p["mppi.heading_weight"]), costmap_weight=float(p["mppi.costmap_weight"]),
            control_weight=float(p["mppi.control_weight"]), smooth_weight=float(p["mppi.smooth_weight"]),
            angular_deadband=float(p["mppi.angular_deadband"]),
            side_commit_steps=int(p["mppi.side_commit_steps"]),
            seed=int(p["mppi.seed"]),
        ))
        self._mppi_dt = self._planner.cfg.dt
        # ----------------------------- control / global goal
        self._control_enabled = bool(p["control.enabled"])
        self._goal_enabled = bool(p["global.enable"])
        self._goal_tolerance = float(p["control.goal_tolerance"])
        # This is deliberately based on the last BEV collision validation,
        # not on the start/end of an image callback.  DINO inference is
        # allowed to take longer than one control period, so publishing zero
        # throughout a valid inference interval causes a walk-stop-walk gait.
        self._max_bev_age = float(p["control.max_bev_age"])
        # Image inference can take substantially longer than a ROS callback
        # budget.  The goal subscription therefore has its own callback group
        # (see below), and this lock makes the hand-off to the image callback
        # explicit and race-free.
        self._goal_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._goal_callback_group = MutuallyExclusiveCallbackGroup()
        self._cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # A plan is a time sequence, not one command to repeat.  The timer
        # below executes these entries at MPPI's dt while DINO prepares the
        # next BEV, so an old angular command cannot spin the robot in place.
        self._active_controls: Optional[np.ndarray] = None
        self._active_control_started: Optional[float] = None
        self._odom_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._goal: Optional[PoseStamped] = None
        self._goal_revision = 0
        self._goal_reached = False
        self._last_plan: Optional[MPPIResult] = None
        self._last_mppi_plan_time: Optional[float] = None
        self._last_mppi_goal_revision = -1
        self._last_safe_bev_time: Optional[float] = None
        # <= 0 disables time-driven MPPI resampling.  Cached trajectories are
        # still validated against every fresh BEV before control is reused.
        self._mppi_replan_max_age = float(p["mppi.replan_max_age"])
        self._last_mode: str = ""
        self._last_goal_local: Optional[Tuple[float, float]] = None
        self._last_process: Optional[float] = None
        self._process_period = float(p["process_period"])
        self._frame_count = 0
        self._debug_show_window = bool(p["debug.show_window"])
        self._debug_window_name = str(p["debug.window_name"])
        self._debug_window_scale = float(p["debug.window_scale"])
        self._debug_window_rate_hz = max(1.0, float(p["debug.window_rate_hz"]))
        self._debug_window_backend = str(p["debug.window_backend"]).strip().lower()
        self._profile_timing = bool(p["debug.profile_timing"])
        if self._debug_window_backend not in ("auto", "opencv", "tk"):
            raise ValueError("debug.window_backend must be auto, opencv, or tk")
        self._debug_window_active = self._debug_show_window
        self._debug_window_running = threading.Event()
        self._debug_window_lock = threading.Lock()
        self._debug_window_image: Optional[np.ndarray] = None
        self._debug_window_thread: Optional[threading.Thread] = None
        if self._debug_show_window:
            self._debug_window_running.set()
            self._debug_window_thread = threading.Thread(
                target=self._debug_loop, name="dinov3_nav_debug", daemon=True
            )
            self._debug_window_thread.start()

        # ----------------------------- output
        self._out_dir: Optional[Path] = None
        self._save_every = max(1, int(p["output.save_every"]))
        if p["output.enable"]:
            base = (
                Path(p["output.dir"])
                if p["output.dir"]
                else Path(__file__).resolve().parents[1] / "output"
            )
            self._out_dir = base / time.strftime("%Y%m%d_%H%M%S")
            self._out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------- publishers
        self._ground_pub = self.create_publisher(
            Image, p["ground_mask_topic"], qos_profile_sensor_data
        )
        self._trav_pub = self.create_publisher(
            Image, p["traversable_mask_topic"], qos_profile_sensor_data
        )
        self._safe_pub = self.create_publisher(
            Image, p["safe_mask_topic"], qos_profile_sensor_data
        )
        self._obst_pub = self.create_publisher(
            Image, p["obstacle_mask_topic"], qos_profile_sensor_data
        )
        self._coverage_pub = self.create_publisher(Float32, p["coverage_topic"], 10)
        self._status_pub = self.create_publisher(String, p["planner_status_topic"], 10)
        self._overlay_pub = (
            self.create_publisher(Image, p["overlay_topic"], qos_profile_sensor_data)
            if p["publish_overlay"]
            else None
        )
        self._bev_debug_pub = (
            self.create_publisher(Image, p["bev_debug_topic"], qos_profile_sensor_data)
            if p["publish_bev_debug"]
            else None
        )
        self._raw_bev_pub = (
            self.create_publisher(Image, p["raw_bev_topic"], qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._planning_bev_pub = (
            self.create_publisher(Image, p["planning_bev_topic"], qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._bev_trav_pub = (
            self.create_publisher(
                Image, "/dinov3_nav/bev_traversability", qos_profile_sensor_data
            )
            if p["publish_bev_debug"]
            else None
        )
        self._bev_obst_pub = (
            self.create_publisher(
                Image, "/dinov3_nav/bev_obstacle", qos_profile_sensor_data
            )
            if p["publish_bev_debug"]
            else None
        )
        self._bev_obs_pub = (
            self.create_publisher(
                Image, "/dinov3_nav/bev_observed", qos_profile_sensor_data
            )
            if p["publish_bev_debug"]
            else None
        )
        self._bev_free_pub = (
            self.create_publisher(Image, "/dinov3_nav/bev_free", qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._bev_unknown_pub = (
            self.create_publisher(Image, "/dinov3_nav/bev_unknown", qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._bev_inflated_pub = (
            self.create_publisher(Image, "/dinov3_nav/bev_inflated_obstacle", qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._bev_clearance_pub = (
            self.create_publisher(Image, "/dinov3_nav/bev_clearance", qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._bev_cost_pub = (
            self.create_publisher(Image, "/dinov3_nav/bev_cost", qos_profile_sensor_data)
            if p["publish_bev_debug"] else None
        )
        self._cmd_pub = (
            self.create_publisher(Twist, p["cmd_vel_topic"], 10)
            if self._control_enabled
            else None
        )

        # ----------------------------- subscriptions
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            p["camera_info_topic"],
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        if self._depth_enabled:
            sub_rgb = Subscriber(
                self, Image, p["image_topic"], qos_profile=self._image_qos
            )
            sub_depth = Subscriber(
                self, Image, p["depth_topic"], qos_profile=self._image_qos
            )
            self._sync = ApproximateTimeSynchronizer(
                [sub_rgb, sub_depth], queue_size=1, slop=float(p["depth.sync_slop"])
            )
            self._sync.registerCallback(self._on_images)
        else:
            self._image_sub = self.create_subscription(
                Image, p["image_topic"], self._on_image, self._image_qos
            )
        if self._goal_enabled:
            # Do not use the default callback group: it is occupied by the
            # synchronous RGB-D inference callback while a frame is planned.
            # A dedicated group lets a MultiThreadedExecutor accept a newly
            # published goal immediately instead of waiting for that frame.
            self._goal_sub = self.create_subscription(
                PoseStamped,
                p["goal_pose_topic"],
                self._on_goal,
                10,
                callback_group=self._goal_callback_group,
            )
        self._odom_sub = self.create_subscription(
            Odometry,
            p["odom_topic"],
            self._on_odom,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        if self._cmd_pub is not None:
            self.create_timer(
                1.0 / float(p["control.cmd_rate"]),
                self._on_cmd_timer,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        self.get_logger().info(
            "dinov3_nav MPPI planner ready: "
            f"DINOv3_device='{self._gseg.device}', "
            f"SAM2_device='{self._sam2.device if self._sam2 is not None else 'disabled'}', "
            f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}', "
            f"planner_frame='{self._planner_frame}', goal_frame="
            f"'{self._default_goal_frame}', convention="
            f"{self._projection_convention}, "
            f"BEV={self._bev_cfg.x_min:.1f}..{self._bev_cfg.x_max:.1f}m x "
            f"{self._bev_cfg.y_min:.1f}..{self._bev_cfg.y_max:.1f}m @ "
            f"{self._bev_cfg.resolution:.2f}m, footprint="
            f"{self._foot_cfg.length:.2f}x{self._foot_cfg.width:.2f}m, "
            f"control={self._control_enabled}"
        )

    def _declare_params(self) -> dict:
        defaults = {
            # topics
            "image_topic": "/zed/zed_node/rgb/image_rect_color",
            "depth_topic": "/zed/zed_node/depth/depth_registered",
            "camera_info_topic": "/zed/zed_node/rgb/camera_info",
            "cmd_vel_topic": "/cmd_vel",
            "goal_pose_topic": "/goal_pose",
            "odom_topic": "/odom",
            "ground_mask_topic": "/dinov3_nav/ground_mask",
            "traversable_mask_topic": "/dinov3_nav/traversable_mask",
            "safe_mask_topic": "/dinov3_nav/safe_mask",
            "obstacle_mask_topic": "/dinov3_nav/obstacle_mask",
            "overlay_topic": "/dinov3_nav/overlay",
            "bev_debug_topic": "/dinov3_nav/bev_debug",
            "raw_bev_topic": "/dinov3_nav/raw_bev",
            "planning_bev_topic": "/dinov3_nav/planning_bev",
            "coverage_topic": "/dinov3_nav/coverage",
            "planner_status_topic": "/dinov3_nav/planner_status",
            "publish_overlay": True,
            "publish_bev_debug": True,
            # frames
            "planner_frame": "base_link",
            # "" -> use CameraInfo.header.frame_id (zed_camera_link in the sim)
            "camera.frame": "",
            # Pixel data convention: "optical" (x right, y down, z forward;
            # the Gazebo zed_x case) or "body" (TF frame already optical).
            "camera.projection_convention": "optical",
            "global.default_goal_frame": "odom",
            "segmentation.use_depth_fusion": False,
            # Load weights and run a synthetic CUDA warmup before subscribing
            # to camera topics. This prevents the first real image from
            # absorbing model-load and kernel-initialization latency.
            "perception.preload": True,
            # robot geometry
            "robot.length": 0.65,
            "robot.width": 0.40,
            "robot.safety_margin": 0.18,
            "robot.center_x": 0.0,
            "robot.center_y": 0.0,
            "robot.footprint_sample_step": 0.05,
            # depth
            "depth.enable": True,
            "depth.scale": 1.0,
            "depth.safe_distance": 1.0,
            "depth.bottom_exempt_frac": 0.35,
            "depth.min_valid": 0.20,
            "depth.sync_slop": 0.05,
            # model
            "ckpt": DEFAULT_CKPT,
            "resolution": 448,
            "device": "auto",
            "seed_rows_frac": 0.10,
            "seed_cols_frac": 0.60,
            "threshold": "otsu",
            "sam2.enable": True,
            "sam2.ckpt": str(DEFAULT_SAM2_CKPT),
            "sam2.max_expand_px": 12,
            "stairs.enable": False,
            "stairs.threshold": 0.45,
            "stairs.min_area": 0.008,
            "stairs.max_area": 0.35,
            "stairs.min_risers": 3,
            "stairs.treat_traversable": True,
            # BEV
            "bev.resolution": 0.05,
            "bev.x_min": -0.80,
            "bev.x_max": 4.00,
            "bev.y_min": -2.20,
            "bev.y_max": 2.20,
            "bev.max_depth": 6.0,
            "bev.pixel_stride": 2,
            "bev.min_observed_points": 1,
            "bev.min_obstacle_points": 2,
            "bev.close_radius_m": 0.08,
            "bev.support_traversability": 0.70,
            "bev.obstacle_height": 0.15,
            "bev.drop_height": 0.25,
            "bev.min_ground_z_points": 200,
            "bev.min_observed_fraction": 0.05,
            "bev.visibility_raycast": False,
            "bev.visibility_raycast_stride_px": 8,
            "bev.traversability_threshold": 0.50,
            "bev.hard_nontrav_threshold": 0.15,
            "bev.hard_nontrav_min_cells": 3,
            # Robot-centred temporal map.  Stored evidence is motion-
            # compensated through odom before it is fused with a raw frame.
            "bev.temporal_enable": True,
            "bev.temporal_frame": "odom",
            "bev.temporal_half_life_s": 0.70,
            "bev.temporal_max_evidence": 6.0,
            "bev.temporal_observation_hit": 1.0,
            "bev.temporal_obstacle_hit": 2.0,
            "bev.temporal_obstacle_clear": 1.5,
            "bev.temporal_min_observed_evidence": 0.50,
            "bev.temporal_obstacle_threshold": 1.25,
            "bev.cleanup_min_obstacle_cells": 2,
            "bev.ego_clear_radius_m": 0.0,
            "bev.ego_clear_traversability": 0.80,
            # UNKNOWN stays traversable at a penalty;
            # hard/inflated cells are +inf and clearance raises local cost.
            "planning.unknown_cost": 3.0,
            "planning.nontraversable_cost": 6.0,
            "planning.clearance_target_m": 0.45,
            "planning.clearance_cost_weight": 4.0,
            # MPPI samples complete vx/vy/wz sequences over this horizon.
            # max_vy=0 is deliberate for a differential-drive base; set it
            # nonzero only for a robot/controller that accepts lateral twist.
            "mppi.horizon_steps": 30,
            "mppi.dt": 0.10,
            "mppi.samples": 384,
            "mppi.temperature": 1.0,
            "mppi.noise_vx": 0.16,
            "mppi.noise_vy": 0.08,
            "mppi.noise_wz": 0.55,
            "mppi.min_vx": -0.10,
            "mppi.max_vx": 0.80,
            "mppi.nominal_vx": 0.40,
            "mppi.max_vy": 0.0,
            "mppi.max_wz": 1.20,
            "mppi.max_accel_vx": 0.60,
            "mppi.max_accel_vy": 0.50,
            "mppi.max_accel_wz": 1.80,
            "mppi.start_relax_distance": 0.90,
            "mppi.goal_weight": 5.0,
            "mppi.terminal_goal_weight": 10.0,
            "mppi.heading_weight": 0.8,
            "mppi.costmap_weight": 1.0,
            "mppi.control_weight": 0.08,
            "mppi.smooth_weight": 0.35,
            "mppi.angular_deadband": 0.04,
            "mppi.side_commit_steps": 8,
            "mppi.seed": 7,
            "mppi.replan_max_age": 0.0,
            # control/global
            "control.enabled": False,
            "control.max_angular": 1.2,
            "control.cruise_linear": 0.35,
            "control.angular_gain": 1.5,
            "control.turn_in_place_angle_deg": 43.0,
            "control.cmd_rate": 10.0,
            # Keep a BEV-validated command alive across normal DINO frame
            # time.  The 4 m BEV and 0.8 m/s cap give a conservative
            # perception horizon; lower this if the camera pipeline becomes
            # substantially slower or the speed cap is increased.
            "control.max_bev_age": 3.5,
            "control.goal_tolerance": 0.35,
            "global.enable": True,
            # output/run
            "output.enable": True,
            "output.dir": "",
            "output.save_every": 5,
            "process_period": 0.10,
            "debug.show_window": False,
            "debug.window_name": "DINOv3 MPPI BEV",
            "debug.window_scale": 1.0,
            "debug.window_rate_hz": 20.0,
            "debug.window_backend": "auto",
            # Print a per-frame timing breakdown for live performance
            # diagnosis.  Disabled by default because it is intentionally
            # verbose on a camera stream.
            "debug.profile_timing": False,
        }
        values = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            values[name] = self.get_parameter(name).value
        thr = str(values["threshold"]).strip()
        values["threshold"] = "otsu" if thr == "otsu" else float(thr)
        return values

    # ----------------------------- callbacks
    def _on_camera_info(self, msg: CameraInfo):
        self._camera_info = msg

    def _on_goal(self, msg: PoseStamped):
        # Keep this callback deliberately small.  Transforming the target is
        # deferred to the image/planning callback, where the current TF is
        # needed anyway.
        with self._goal_lock:
            self._goal = msg
            self._goal_revision += 1
            self._goal_reached = False
            self._last_goal_local = None
        # A goal replacement must never continue the old trajectory while
        # the next RGB-D frame is being processed.
        self._clear_active_control()
        self.get_logger().info(
            f"new goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
            f"frame='{msg.header.frame_id or self._default_goal_frame}'"
        )

    def _on_odom(self, msg: Odometry):
        """Keep measured body-frame velocity as the MPPI rollout state.

        REP-147 odometry twist is conventionally expressed in child_frame_id
        (base_link).  If a source uses another convention it must publish a
        proper odometry transform; this node does not silently reinterpret it.
        """
        tw = msg.twist.twist
        self._odom_velocity = (float(tw.linear.x), float(tw.linear.y), float(tw.angular.z))

    def _on_images(self, rgb_msg: Image, depth_msg: Image):
        self._process(rgb_msg, depth_msg)

    def _on_image(self, rgb_msg: Image):
        self._process(rgb_msg, None)

    def _warmup_perception(self) -> None:
        """Preload DINO/SAM2 and initialize CUDA before camera subscriptions.

        Both upstream wrappers intentionally load lazily.  In a ROS camera
        pipeline that deferred work makes the first received image stale by
        tens of seconds.  A small 16:9 dummy image triggers the same model
        loading and first CUDA kernels while no sensor messages are queued.
        """
        self.get_logger().info("preloading DINOv3/SAM2 and warming CUDA...")
        started = time.perf_counter()
        # The input has the same 16:9 aspect ratio as the configured ZED feed;
        # DINO subsequently applies its normal 448-pixel short-side resize.
        warmup_rgb = np.zeros((448, 800, 3), dtype=np.uint8)
        try:
            dino_started = time.perf_counter()
            prepared = self._gseg.prepare(warmup_rgb)
            ground = self._gseg.segment_prepared(prepared, depth=None)
            self._synchronize_perception_device()
            dino_s = time.perf_counter() - dino_started

            sam2_s = 0.0
            if self._sam2 is not None:
                sam2_started = time.perf_counter()
                # Explicitly load even if the synthetic DINO proposal happens
                # to be empty and refine_prepared would otherwise return early.
                self._sam2._ensure_model()
                self._sam2.refine_prepared(prepared, ground.mask)
                self._synchronize_perception_device()
                sam2_s = time.perf_counter() - sam2_started
            self.get_logger().info(
                "perception preload complete: "
                f"dino={dino_s:.2f}s sam2={sam2_s:.2f}s "
                f"total={time.perf_counter() - started:.2f}s"
            )
        except Exception as exc:
            # A later real-frame callback retains its normal exception handling
            # and can still run, while startup reports why preloading failed.
            self.get_logger().warning(f"perception preload failed: {exc}")

    def _synchronize_perception_device(self) -> None:
        """Make one-off CUDA initialization timings truthful in diagnostics."""
        if self._gseg.device.type == "cuda":
            torch.cuda.synchronize(self._gseg.device)

    # ----------------------------- geometry helpers
    def _camera_matrix(self, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        info = self._camera_info
        if info is None:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        h, w = image_shape
        if info.width > 0 and info.height > 0:
            sx, sy = w / float(info.width), h / float(info.height)
            fx, cx = fx * sx, cx * sx
            fy, cy = fy * sy, cy * sy
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float32)

    def _camera_to_planner(self) -> Optional[np.ndarray]:
        """T mapping optical-convention camera points into planner frame.

        base_link <- zed_camera_link carries an identity rotation (body
        axes), so with projection_convention=optical the fixed optical->body
        rotation is composed before the TF. No optical TF frame is needed.
        """
        info = self._camera_info
        camera_frame = self._camera_frame_param or (
            info.header.frame_id if info is not None else ""
        )
        camera_frame = camera_frame.strip()
        if not camera_frame:
            self.get_logger().warning(
                "camera frame unknown: set camera.frame or wait for CameraInfo",
                throttle_duration_sec=10.0,
            )
            return None
        key = (camera_frame, self._projection_convention)
        if self._cached_camera_T is not None and self._cached_camera_key == key:
            return self._cached_camera_T
        try:
            tf = self._tf_buffer.lookup_transform(
                self._planner_frame,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.15),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"TF unavailable: {self._planner_frame} <- {camera_frame}: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        T = transform_to_matrix(tf.transform)
        if self._projection_convention == "optical":
            T = (T @ T_BODY_FROM_OPTICAL).astype(np.float32)
        self._cached_camera_key = key
        self._cached_camera_T = T
        return T

    def _projection_sanity(self, K: np.ndarray, T: np.ndarray) -> bool:
        """Acceptance check before any non-zero command may be published.

        Image center at depth Z must land in front of the robot with y~0;
        an image-left pixel must land at y>0 (base left), image-right y<0.
        """
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        Z = 2.0
        du = max(4.0, 0.25 * fx)

        def project(u: float, v: float) -> Tuple[float, float]:
            p = T @ np.array(
                [(u - cx) * Z / fx, (v - cy) * Z / fy, Z, 1.0], np.float32
            )
            return float(p[0]), float(p[1])

        x_c, y_c = project(cx, cy)
        y_l = project(cx - du, cy)[1]
        y_r = project(cx + du, cy)[1]
        ok = (
            0.5 * Z < x_c < 1.5 * Z + 1.0
            and abs(y_c) < 0.10 * Z
            and y_l > 0.25
            and y_r < -0.25
        )
        detail = (
            f"center=({x_c:.2f},{y_c:.2f}) left_y={y_l:.2f} right_y={y_r:.2f}"
        )
        if ok:
            self.get_logger().info(f"projection sanity OK: {detail}")
        else:
            self.get_logger().error(
                f"projection sanity FAILED ({detail}); check "
                "camera.projection_convention and the "
                f"{self._planner_frame}<-{self._camera_frame_param or 'camera'} TF. "
                "Non-zero cmd_vel is suppressed."
            )
        return ok

    def _goal_in_planner(
        self,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], int]:
        if not self._goal_enabled:
            # no global goal: walk forward (waypoint at BEV front edge)
            return (self._bev_cfg.x_max - 0.5, 0.0), float("inf"), -1
        # Copy the reference under the lock, then do the potentially blocking
        # TF lookup without holding up a newly received goal callback.
        with self._goal_lock:
            goal = self._goal
            goal_revision = self._goal_revision
        if goal is None:
            return None, None, goal_revision
        goal_frame = goal.header.frame_id or self._default_goal_frame
        p = np.array(
            [
                goal.pose.position.x,
                goal.pose.position.y,
                goal.pose.position.z,
                1.0,
            ],
            np.float32,
        )
        if goal_frame == self._planner_frame:
            local = p
        else:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._planner_frame,
                    goal_frame,
                    Time(),
                    timeout=Duration(seconds=0.15),
                )
            except TransformException as exc:
                self.get_logger().warning(
                    f"goal TF unavailable: {self._planner_frame} <- {goal_frame}: {exc}",
                    throttle_duration_sec=5.0,
                )
                return None, None, goal_revision
            local = transform_to_matrix(tf.transform) @ p
        gx, gy = float(local[0]), float(local[1])
        return (gx, gy), hypot(gx, gy), goal_revision

    def _fuse_raw_bev(self, raw_bev):
        """Motion-compensate prior evidence, then fuse this raw sensor frame."""
        now = time.monotonic()
        T_current_from_previous = None
        current_pose = None
        if self._fusion_frame:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._fusion_frame, self._planner_frame, Time(),
                    timeout=Duration(seconds=0.15),
                )
                current_pose = transform_to_matrix(tf.transform)
                if self._prev_fusion_pose is not None:
                    relative = np.linalg.inv(current_pose) @ self._prev_fusion_pose
                    T_current_from_previous = relative[:3, :3]
                    T_current_from_previous = np.array(
                        [[relative[0, 0], relative[0, 1], relative[0, 3]],
                         [relative[1, 0], relative[1, 1], relative[1, 3]],
                         [0.0, 0.0, 1.0]], np.float32,
                    )
            except TransformException as exc:
                self._bev_fusion.reset()
                self._prev_fusion_pose = None
                self._prev_fusion_time = None
                self.get_logger().warning(
                    f"temporal BEV TF unavailable ({self._fusion_frame} <- "
                    f"{self._planner_frame}); using raw frame: {exc}",
                    throttle_duration_sec=5.0,
                )
        dt = 0.0 if self._prev_fusion_time is None else now - self._prev_fusion_time
        stable = self._bev_fusion.update(raw_bev, dt, T_current_from_previous)
        self._prev_fusion_pose = current_pose
        self._prev_fusion_time = now
        return stable

    def _depth_obstacle(self, depth: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Image-space near-depth obstacle (debug overlay only; the BEV uses
        height-based geometric evidence from build_local_bev)."""
        h, w = shape
        out = np.zeros((h, w), bool)
        if depth is None:
            return out
        valid = np.isfinite(depth) & (depth > self._min_depth)
        out = valid & (depth < self._safe_distance)
        out[int((1.0 - self._bottom_exempt) * h):, :] = False
        return out

    # ----------------------------- realtime MPPI debug window
    def _set_debug_image(self, image_rgb: np.ndarray, status: str) -> None:
        """Hand off the latest rendered BEV without blocking ROS callbacks."""
        if not self._debug_window_active:
            return
        canvas = image_rgb.copy()
        # Give the standalone window the same immediate observability as the
        # status topic: selected control, valid samples and rejection reasons.
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (18, 18, 18), -1)
        text = status[:180]
        cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "X forward  ^    Y left  <    blue=rollouts  white=selected",
                    (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 230, 255), 1,
                    cv2.LINE_AA)
        with self._debug_window_lock:
            self._debug_window_image = canvas

    def _debug_loop(self) -> None:
        """Run a real-time window without ever blocking ROS callbacks."""
        backend = self._debug_window_backend
        if backend in ("auto", "opencv") and self._debug_loop_opencv():
            return
        if backend == "opencv":
            self._disable_debug_window("OpenCV HighGUI is unavailable")
            return
        self._debug_loop_tk()

    def _latest_debug_image(self) -> Optional[np.ndarray]:
        with self._debug_window_lock:
            return (None if self._debug_window_image is None
                    else self._debug_window_image.copy())

    def _disable_debug_window(self, reason: str) -> None:
        self._debug_window_active = False
        self._debug_window_running.clear()
        self.get_logger().warning(
            f"cannot create MPPI debug window ({reason}); keep ROS image topics enabled"
        )

    def _debug_loop_opencv(self) -> bool:
        """Return false when this cv2 build lacks a HighGUI backend."""
        delay_ms = max(1, int(round(1000.0 / self._debug_window_rate_hz)))
        try:
            cv2.namedWindow(self._debug_window_name, cv2.WINDOW_NORMAL)
            while self._debug_window_running.is_set():
                image = self._latest_debug_image()
                if image is not None:
                    if self._debug_window_scale != 1.0:
                        image = cv2.resize(
                            image, None, fx=self._debug_window_scale,
                            fy=self._debug_window_scale, interpolation=cv2.INTER_NEAREST,
                        )
                    # renderers produce RGB for ROS; HighGUI expects BGR.
                    cv2.imshow(self._debug_window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    self._debug_window_running.clear()
                    self._debug_window_active = False
                    self.get_logger().info("MPPI debug window closed by user")
            return True
        except cv2.error as exc:
            self.get_logger().info(f"OpenCV HighGUI unavailable, falling back to Tk: {exc}")
            return False
        finally:
            try:
                cv2.destroyWindow(self._debug_window_name)
            except cv2.error:
                pass

    def _debug_loop_tk(self) -> None:
        """Tk/Pillow fallback for headless OpenCV wheels in the DINO venv."""
        try:
            import tkinter as tk
            from PIL import Image as PILImage
            from PIL import ImageTk

            root = tk.Tk()
            root.title(self._debug_window_name)
            label = tk.Label(root)
            label.pack()

            def close() -> None:
                self._debug_window_running.clear()

            root.protocol("WM_DELETE_WINDOW", close)
            root.bind("<Escape>", lambda _event: close())
            root.bind("q", lambda _event: close())
            period = 1.0 / self._debug_window_rate_hz
            while self._debug_window_running.is_set():
                image = self._latest_debug_image()
                if image is not None:
                    pil = PILImage.fromarray(image, mode="RGB")
                    if self._debug_window_scale != 1.0:
                        width = max(1, round(pil.width * self._debug_window_scale))
                        height = max(1, round(pil.height * self._debug_window_scale))
                        pil = pil.resize((width, height), PILImage.Resampling.NEAREST)
                    photo = ImageTk.PhotoImage(pil)
                    label.configure(image=photo)
                    label.image = photo  # retain the Tcl image reference
                root.update_idletasks()
                root.update()
                time.sleep(period)
            root.destroy()
            self._debug_window_active = False
            self.get_logger().info("MPPI debug window closed by user")
        except Exception as exc:
            self._disable_debug_window(f"Tk fallback failed: {exc}")

    # ----------------------------- processing
    def _process(self, rgb_msg: Image, depth_msg: Optional[Image]):
        now = time.monotonic()
        if self._last_process is not None and now - self._last_process < self._process_period:
            return
        self._last_process = now

        try:
            rgb = image_to_rgb(rgb_msg)
        except ValueError as exc:
            self.get_logger().warning(f"RGB conversion failed: {exc}", throttle_duration_sec=10.0)
            return

        depth = None
        if depth_msg is not None:
            try:
                depth = depth_to_meters(depth_msg, self._depth_scale)
                if depth.shape[:2] != rgb.shape[:2]:
                    self.get_logger().warning(
                        f"depth {depth.shape} != RGB {rgb.shape[:2]}; dropping depth",
                        throttle_duration_sec=10.0,
                    )
                    depth = None
            except ValueError as exc:
                self.get_logger().warning(f"depth conversion failed: {exc}", throttle_duration_sec=10.0)

        t0 = time.perf_counter()
        perception_s = dino_s = sam2_s = stairs_s = 0.0
        dino_load_s = dino_forward_s = sam2_load_s = sam2_forward_s = 0.0
        goal_tf_s = bev_project_s = bev_fusion_s = 0.0
        costmap_s = mppi_s = debug_render_s = 0.0
        try:
            # GroundSegmenter and SAM2MaskRefiner intentionally load lazily.
            # Split that one-off cost from actual image inference so a slow
            # cold start is not mistaken for low steady-state throughput.
            if self._gseg._extractor._model is None:
                dino_load_t0 = time.perf_counter()
                self._gseg._ensure_model()
                self._synchronize_perception_device()
                dino_load_s = time.perf_counter() - dino_load_t0
            dino_t0 = time.perf_counter()
            prepared = self._gseg.prepare(rgb)
            ground = self._gseg.segment_prepared(
                prepared, depth=depth if self._gseg_depth_fusion else None
            )
            dino_s = time.perf_counter() - dino_t0
            dino_forward_s = dino_s
            if self._sam2 is not None:
                try:
                    if self._sam2._predictor is None:
                        sam2_load_t0 = time.perf_counter()
                        self._sam2._ensure_model()
                        self._synchronize_perception_device()
                        sam2_load_s = time.perf_counter() - sam2_load_t0
                    sam2_t0 = time.perf_counter()
                    coarse_mask = ground.mask.copy()
                    refined = self._sam2.refine_prepared(prepared, coarse_mask)
                    sam2_s = time.perf_counter() - sam2_t0
                    sam2_forward_s = sam2_s
                    if refined.iou_with_coarse >= 0.35:
                        # ground_seg has already used depth geometry to suppress
                        # above-ground regions. Limit SAM2 expansion around that
                        # coarse support so RGB refinement cannot freely re-add
                        # a distant wall/object that geometry already rejected.
                        if self._sam2_max_expand_px > 0:
                            k = 2 * self._sam2_max_expand_px + 1
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                            allowed = cv2.dilate((coarse_mask > 0).astype(np.uint8), kernel) > 0
                            ground.mask = np.where(allowed, refined.mask, 0).astype(np.uint8)
                        else:
                            ground.mask = refined.mask
                        ground.coverage = float((ground.mask > 0).mean())
                    else:
                        self.get_logger().warning(
                            f"SAM2 coarse IoU={refined.iou_with_coarse:.2f}; keep DINO mask",
                            throttle_duration_sec=10.0,
                        )
                except Exception as exc:
                    self.get_logger().warning(
                        f"SAM2 refinement failed: {exc}", throttle_duration_sec=10.0
                    )
            stairs_t0 = time.perf_counter()
            sres = self._stairs.detect_prepared(prepared) if self._stairs is not None else None
            stairs_s = time.perf_counter() - stairs_t0
            traversable = ground.mask.copy()
            if sres is not None and self._stairs_traversable and sres.mask.any():
                traversable = cv2.bitwise_or(traversable, sres.mask)
        except Exception as exc:
            self.get_logger().error(f"perception failed: {exc}", throttle_duration_sec=10.0)
            self._clear_active_control()
            return
        perception_s = time.perf_counter() - t0

        obstacle_pixels = None
        goal_tf_t0 = time.perf_counter()
        local_goal, goal_distance, frame_goal_revision = self._goal_in_planner()
        goal_tf_s = time.perf_counter() - goal_tf_t0
        self._last_goal_local = local_goal
        with self._goal_lock:
            if frame_goal_revision == self._goal_revision:
                self._goal_reached = (
                    goal_distance is not None and goal_distance <= self._goal_tolerance
                )

        plan: Optional[MPPIResult] = None
        bev_debug = None
        bev = None
        raw_bev = None
        planning: Optional[PlanningBEV] = None
        near_obstacle = self._depth_obstacle(depth, rgb.shape[:2])
        if self._goal_reached_now():
            self._clear_active_control()
        elif depth is None:
            self._clear_active_control()
            self.get_logger().warning("BEV planner requires depth", throttle_duration_sec=5.0)
        else:
            K = self._camera_matrix(depth.shape)
            T = self._camera_to_planner()
            if K is None:
                self._clear_active_control()
                self.get_logger().warning("waiting for valid CameraInfo", throttle_duration_sec=5.0)
            elif T is None:
                self._clear_active_control()
            else:
                if self._sanity_ok is None:
                    self._sanity_ok = self._projection_sanity(K, T)
                try:
                    bev_t0 = time.perf_counter()
                    raw_bev = build_local_bev(
                        traversable_mask=traversable,
                        depth=depth,
                        K=K,
                        T_planner_from_camera=T,
                        cfg=self._bev_cfg,
                    )
                    bev_project_s = time.perf_counter() - bev_t0
                    obstacle_pixels = raw_bev.obstacle_pixels
                    fusion_t0 = time.perf_counter()
                    bev = self._fuse_raw_bev(raw_bev)
                    bev_fusion_s = time.perf_counter() - fusion_t0
                    observed_fraction = float(bev.observed.mean())
                    if observed_fraction < self._min_observed_fraction:
                        self._clear_active_control()
                        self.get_logger().warning(
                            f"BEV observed only {observed_fraction:.1%} of cells; "
                            "stopping (depth/CameraInfo problem?)",
                            throttle_duration_sec=5.0,
                        )
                    elif not self._sanity_ok:
                        self._clear_active_control()
                    else:
                        costmap_t0 = time.perf_counter()
                        planning = build_planning_bev(
                            bev, self._foot_cfg, self._planning_cost_cfg
                        )
                        costmap_s = time.perf_counter() - costmap_t0
                        # Do not let a long perception callback resume a
                        # trajectory for a goal that arrived while this frame
                        # was being segmented.
                        with self._goal_lock:
                            goal_is_current = frame_goal_revision == self._goal_revision
                        if not goal_is_current:
                            local_goal = None
                        self._last_goal_local = local_goal
                        if local_goal is None:
                            # Map production is intentionally independent of
                            # navigation authority. This lets raw/planning BEV
                            # be inspected before a global goal is issued.
                            self._clear_active_control()
                        else:
                            checker = FootprintChecker(planning.grid, self._foot_cfg, planning.layers)
                            mppi_t0 = time.perf_counter()
                            now_mppi = time.monotonic()
                            elapsed_steps = self._elapsed_control_steps(now_mppi)
                            goal_revision = frame_goal_revision
                            replan_due = (
                                self._last_mppi_plan_time is None
                                or goal_revision != self._last_mppi_goal_revision
                                or (
                                    self._mppi_replan_max_age > 0.0
                                    and now_mppi - self._last_mppi_plan_time
                                    >= self._mppi_replan_max_age
                                )
                            )
                            reused = None if replan_due else self._planner.reuse_if_safe(
                                planning, checker, local_goal, self._odom_velocity,
                                elapsed_steps=elapsed_steps,
                            )
                            if reused is None:
                                plan = self._planner.plan(
                                    planning, checker, local_goal, self._odom_velocity,
                                    elapsed_steps=elapsed_steps,
                                )
                                self._last_mppi_plan_time = now_mppi
                                self._last_mppi_goal_revision = goal_revision
                            else:
                                plan = MPPIResult(
                                    command=tuple(map(float, reused.controls[0])),
                                    trajectory=reused,
                                    mode="MPPI_REUSE",
                                    candidates=[reused],
                                    effective_samples=1,
                                    best_cost=float(reused.cost),
                                    diagnostics={"reused": 1.0},
                                )
                            mppi_s = time.perf_counter() - mppi_t0
                            with self._goal_lock:
                                command_goal_is_current = (
                                    frame_goal_revision == self._goal_revision
                            )
                            if command_goal_is_current:
                                self._activate_plan(plan)
                            else:
                                # A newer goal won the race while MPPI was
                                # sampling.  The dedicated goal callback has
                                # invalidated the old command; do not revive it.
                                self._clear_active_control()
                        if (self._bev_debug_pub is not None or self._out_dir is not None
                                or self._debug_window_active):
                            render_t0 = time.perf_counter()
                            bev_debug = render_bev_debug(
                                planning.grid, planning.layers, plan, local_goal, scale=5,
                                footprint=(self._foot_cfg.length, self._foot_cfg.width)
                            )
                            debug_render_s = time.perf_counter() - render_t0
                except Exception as exc:
                    self.get_logger().error(
                        f"BEV/planner failed: {exc}", throttle_duration_sec=5.0
                    )
                    self._clear_active_control()

        self._last_plan = plan
        elapsed = time.perf_counter() - t0

        # ----------------------------- publish diagnostics
        header = rgb_msg.header
        self._ground_pub.publish(gray_to_image_msg(ground.mask, header))
        self._trav_pub.publish(gray_to_image_msg(traversable, header))
        image_safe = (traversable > 0) & ~(obstacle_pixels if obstacle_pixels is not None else near_obstacle)
        self._safe_pub.publish(gray_to_image_msg((image_safe * 255).astype(np.uint8), header))
        self._obst_pub.publish(
            gray_to_image_msg(
                ((obstacle_pixels if obstacle_pixels is not None else near_obstacle) * 255).astype(np.uint8),
                header,
            )
        )
        self._coverage_pub.publish(Float32(data=float(ground.coverage)))

        status = self._status_text(plan, local_goal, goal_distance)
        self._status_pub.publish(String(data=status))
        if bev_debug is not None:
            self._set_debug_image(bev_debug, status)
        if plan is not None and plan.mode != self._last_mode:
            self.get_logger().info(f"planner mode: {self._last_mode or '-'} -> {plan.mode}")
            self._last_mode = plan.mode

        overlay = None
        if self._overlay_pub is not None or self._out_dir is not None:
            overlay = self._draw_overlay(
                rgb, traversable,
                obstacle_pixels if obstacle_pixels is not None else near_obstacle,
                plan, local_goal,
            )
            if self._overlay_pub is not None:
                self._overlay_pub.publish(rgb_to_image_msg(overlay, header))
        if self._bev_debug_pub is not None and bev_debug is not None:
            msg = rgb_to_image_msg(bev_debug, header)
            msg.header.frame_id = self._planner_frame
            self._bev_debug_pub.publish(msg)
        if raw_bev is not None and planning is not None:
            for pub, image in (
                (self._raw_bev_pub, render_raw_bev(raw_bev, scale=5)),
                (self._planning_bev_pub, render_planning_bev(planning, scale=5)),
            ):
                if pub is not None:
                    msg = rgb_to_image_msg(image, header)
                    msg.header.frame_id = self._planner_frame
                    pub.publish(msg)
        if planning is not None:
            for pub, name in (
                (self._bev_trav_pub, "traversability"),
                (self._bev_obst_pub, "obstacle"),
                (self._bev_obs_pub, "observed"),
                (self._bev_free_pub, "free"),
                (self._bev_unknown_pub, "unknown"),
                (self._bev_inflated_pub, "inflated_obstacle"),
                (self._bev_clearance_pub, "clearance"),
                (self._bev_cost_pub, "cost"),
            ):
                if pub is not None:
                    layer = (render_bev_layer(planning.grid, name) if name in {
                        "traversability", "obstacle", "observed"
                    } else render_planning_layer(planning, name))
                    layer_msg = gray_to_image_msg(layer, header)
                    layer_msg.header.frame_id = self._planner_frame
                    pub.publish(layer_msg)

        self._frame_count += 1
        if self._out_dir is not None and self._frame_count % self._save_every == 0:
            try:
                if overlay is not None:
                    cv2.imwrite(
                        str(self._out_dir / f"frame_{self._frame_count:06d}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
                if bev_debug is not None:
                    cv2.imwrite(
                        str(self._out_dir / f"frame_{self._frame_count:06d}_bev.png"),
                        cv2.cvtColor(bev_debug, cv2.COLOR_RGB2BGR),
                    )
                if raw_bev is not None:
                    raw_image = render_raw_bev(raw_bev, scale=5)
                    cv2.imwrite(
                        str(self._out_dir / f"frame_{self._frame_count:06d}_raw_bev.png"),
                        cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR),
                    )
                if planning is not None:
                    planning_image = render_planning_bev(planning, scale=5)
                    cv2.imwrite(
                        str(self._out_dir / f"frame_{self._frame_count:06d}_planning_bev.png"),
                        cv2.cvtColor(planning_image, cv2.COLOR_RGB2BGR),
                    )
                    for name in ("free", "unknown", "hard_obstacle", "inflated_obstacle",
                                 "clearance", "cost"):
                        cv2.imwrite(
                            str(self._out_dir / f"frame_{self._frame_count:06d}_{name}.png"),
                            render_planning_layer(planning, name),
                        )
            except Exception as exc:
                self.get_logger().warning(f"debug image save failed: {exc}")

        if self._frame_count == 1 or self._frame_count % 20 == 0:
            self.get_logger().info(
                f"frame={self._frame_count} cov={ground.coverage:.2f} "
                f"{status} t={elapsed:.2f}s"
            )
        if self._profile_timing:
            end_to_end_s = time.perf_counter() - t0
            self.get_logger().info(
                "timing[s]: "
                f"perception={perception_s:.2f} dino_load={dino_load_s:.2f} "
                f"dino_forward={dino_forward_s:.2f} sam2_load={sam2_load_s:.2f} "
                f"sam2_forward={sam2_forward_s:.2f} stairs={stairs_s:.2f} "
                f"goal_tf={goal_tf_s:.2f} "
                f"bev={bev_project_s:.2f} fusion={bev_fusion_s:.2f} "
                f"costmap={costmap_s:.2f} mppi={mppi_s:.2f} "
                f"debug_render={debug_render_s:.2f} "
                f"core={elapsed:.2f} end_to_end={end_to_end_s:.2f}"
            )

    def _status_text(
        self,
        plan: Optional[MPPIResult],
        local_goal: Optional[Tuple[float, float]],
        goal_distance: Optional[float],
    ) -> str:
        if self._goal_reached_now():
            return f"GOAL_REACHED d={goal_distance:.2f}m cmd=(0,0)"
        goal_txt = "goal=none"
        if local_goal is not None:
            goal_txt = f"goal(base)=({local_goal[0]:.2f},{local_goal[1]:.2f})"
        if plan is None:
            return f"WAITING {goal_txt} cmd=({self._cmd[0]:.2f},{self._cmd[1]:+.2f},{self._cmd[2]:+.2f})"
        vx, vy, wz = plan.command
        return (
            f"{plan.mode} {goal_txt} valid={plan.effective_samples}/{len(plan.candidates)} "
            f"cost={plan.best_cost:.2f} reject={plan.diagnostics} sel=({vx:.2f},{vy:+.2f},{wz:+.2f}) "
            f"cmd=({self._cmd[0]:.2f},{self._cmd[1]:+.2f},{self._cmd[2]:+.2f})"
        )

    def _draw_overlay(
        self,
        rgb: np.ndarray,
        traversable: np.ndarray,
        obstacle: np.ndarray,
        plan: Optional[MPPIResult],
        local_goal: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        out = rgb.copy()
        good = traversable > 0
        if good.any():
            out[good] = (
                0.50 * out[good] + 0.50 * np.array([55, 220, 70], np.float32)
            ).astype(np.uint8)
        if obstacle.any():
            out[obstacle] = (
                0.35 * out[obstacle] + 0.65 * np.array([255, 45, 45], np.float32)
            ).astype(np.uint8)

        return out

    # ----------------------------- cmd_vel safety timer
    def _goal_reached_now(self) -> bool:
        """Return the goal state safely while goal callbacks run in parallel."""
        with self._goal_lock:
            return self._goal_reached

    def _clear_active_control(self) -> None:
        """Stop and invalidate the currently scheduled MPPI sequence."""
        with self._command_lock:
            self._cmd = (0.0, 0.0, 0.0)
            self._active_controls = None
            self._active_control_started = None
            self._last_safe_bev_time = None

    def _activate_plan(self, plan: MPPIResult) -> None:
        """Schedule a collision-checked MPPI control sequence from now."""
        controls = None if plan.trajectory is None else plan.trajectory.controls
        if controls is None or len(controls) == 0:
            self._clear_active_control()
            return
        now = time.monotonic()
        with self._command_lock:
            self._cmd = tuple(map(float, controls[0]))
            self._active_controls = controls.astype(np.float32, copy=True)
            self._active_control_started = now
            self._last_safe_bev_time = now

    def _elapsed_control_steps(self, now: float) -> int:
        """How many entries of the active plan have physically elapsed."""
        with self._command_lock:
            if self._active_control_started is None:
                return 1
            return max(1, int((now - self._active_control_started) / self._mppi_dt))

    def _on_cmd_timer(self):
        if self._cmd_pub is None:
            return
        now = time.monotonic()
        with self._command_lock:
            stale = (
                self._last_safe_bev_time is None
                or (now - self._last_safe_bev_time) > self._max_bev_age
            )
            if stale or self._active_controls is None or self._active_control_started is None:
                vx, vy, wz = (0.0, 0.0, 0.0)
            else:
                index = int((now - self._active_control_started) / self._mppi_dt)
                # The terminal control is deliberately not repeated.  A
                # delayed DINO frame must cause a safe stop, never a
                # multi-second repeat of its last yaw command.
                if index >= len(self._active_controls):
                    vx, vy, wz = (0.0, 0.0, 0.0)
                else:
                    vx, vy, wz = map(float, self._active_controls[index])
        if self._goal_reached_now():
            vx, vy, wz = 0.0, 0.0, 0.0
        if self._sanity_ok is not True:
            vx, vy, wz = 0.0, 0.0, 0.0
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def stop(self):
        self._debug_window_running.clear()
        if self._debug_window_thread is not None and self._debug_window_thread.is_alive():
            self._debug_window_thread.join(timeout=1.0)
        self._clear_active_control()
        if self._cmd_pub is not None:
            msg = Twist()
            self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Dinov3NavNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

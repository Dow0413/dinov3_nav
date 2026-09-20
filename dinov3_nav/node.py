# -*- coding: utf-8 -*-
"""DINOv3 + RGB-D + footprint-aware local trajectory navigation.

Pipeline:
    RGB -> ground_seg/DINOv3 (+ optional SAM2/stairs) -> traversability image
    Depth + CameraInfo -> 3D points -> TF -> local BEV in planner frame
    BEV -> robot footprint inflation -> body-safe space
    Global goal -> TF -> planner frame
    Safe BEV + local goal -> trajectory rollout/scoring -> cmd_vel

The planner intentionally does not use the old image-space waypoint/gate controller.
"""

from __future__ import annotations

import time
from array import array
from math import hypot, radians
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener

from ground_seg import GroundSegmenter, SAM2MaskRefiner
from ground_seg.features import DEFAULT_CKPT
from ground_seg.sam2_refiner import DEFAULT_SAM2_CKPT

from .bev import BEVConfig, build_local_bev, transform_to_matrix
from .footprint import FootprintResult, build_body_safe_space
from .local_planner import LocalTrajectoryPlanner, PlannerConfig, Trajectory


_RGB_CHANNELS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
}


def image_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 uint8 RGB, without cv_bridge."""
    enc = msg.encoding.lower()
    channels = _RGB_CHANNELS.get(enc)
    if channels is None:
        raise ValueError(
            f"unsupported encoding {msg.encoding!r}; "
            "supported: rgb8/bgr8/rgba8/bgra8/mono8"
        )
    if msg.is_bigendian:
        raise ValueError("big-endian RGB images are not supported")

    h, w = int(msg.height), int(msg.width)
    row = np.frombuffer(msg.data, dtype=np.uint8, count=h * int(msg.step))
    arr = row.reshape(h, int(msg.step))[:, : w * channels].reshape(h, w, channels)

    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "rgba8":
        arr = arr[:, :, :3]
    elif enc == "bgra8":
        arr = arr[:, :, 2::-1]
    elif enc == "mono8":
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return np.ascontiguousarray(arr)


def depth_to_meters(msg: Image) -> np.ndarray:
    """sensor_msgs/Image depth -> HxW float32 meters."""
    enc = msg.encoding.lower()
    if enc == "32fc1":
        dtype = np.float32
        scale = 1.0
    elif enc == "16uc1":
        dtype = np.uint16
        scale = 0.001
    else:
        raise ValueError(
            f"unsupported depth encoding {msg.encoding!r}; supported: 32FC1/16UC1"
        )

    h, w = int(msg.height), int(msg.width)
    item = np.dtype(dtype).itemsize
    if int(msg.step) % item:
        raise ValueError(f"depth step={msg.step} is not divisible by dtype size={item}")

    row_width = int(msg.step) // item
    row = np.frombuffer(msg.data, dtype=dtype, count=h * row_width)
    arr = row.reshape(h, row_width)[:, :w].astype(np.float32)
    arr *= scale
    return np.ascontiguousarray(arr)


def gray_to_image_msg(mask: np.ndarray, header) -> Image:
    """bool/uint8 image -> sensor_msgs/Image mono8."""
    x = np.asarray(mask)
    if x.dtype == np.bool_:
        x = x.astype(np.uint8) * 255
    else:
        x = x.astype(np.uint8, copy=False)
        if x.size and x.max() <= 1:
            x = x * 255
    x = np.ascontiguousarray(x)

    out = Image()
    out.header = header
    out.height, out.width = x.shape[:2]
    out.encoding = "mono8"
    out.is_bigendian = 0
    out.step = int(out.width)
    out.data = array("B", x.tobytes())
    return out


def rgb_to_image_msg(rgb: np.ndarray, header) -> Image:
    x = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
    out = Image()
    out.header = header
    out.height, out.width = x.shape[:2]
    out.encoding = "rgb8"
    out.is_bigendian = 0
    out.step = int(out.width) * 3
    out.data = array("B", x.tobytes())
    return out


class Dinov3NavNode(Node):
    """DINOv3 traversability + local BEV + footprint-aware trajectory planner."""

    def __init__(self):
        super().__init__("dinov3_nav")
        p = self._declare_params()

        # ------------------------------------------------------------ perception
        self._gseg = GroundSegmenter(
            resolution=int(p["resolution"]),
            ckpt=str(p["ckpt"]),
            device=str(p["device"]),
            seed_rows_frac=float(p["seed_rows_frac"]),
            seed_cols_frac=float(p["seed_cols_frac"]),
            threshold=p["threshold"],
        )

        self._sam2: Optional[SAM2MaskRefiner] = None
        if bool(p["sam2.enable"]):
            try:
                self._sam2 = SAM2MaskRefiner(
                    checkpoint=str(p["sam2.ckpt"]),
                    device=str(p["device"]),
                )
            except Exception as e:
                self.get_logger().warning(f"SAM2 unavailable; keep DINO mask: {e}")

        self._stairs = None
        if bool(p["stairs.enable"]):
            try:
                from ground_seg import DINOStairsDetector

                self._stairs = DINOStairsDetector(
                    resolution=int(p["resolution"]),
                    threshold=float(p["stairs.threshold"]),
                    min_area_frac=float(p["stairs.min_area"]),
                    max_area_frac=float(p["stairs.max_area"]),
                    min_risers=int(p["stairs.min_risers"]),
                    device=str(p["device"]),
                    extractor=self._gseg._extractor,
                )
            except Exception as e:
                self.get_logger().warning(f"stairs detector unavailable: {e}")

        self._stairs_traversable = bool(p["stairs.treat_traversable"])

        # ------------------------------------------------------------ depth
        self._depth_enabled = bool(p["depth.enable"])
        self._safe_distance = float(p["depth.safe_distance"])
        self._bottom_exempt = float(p["depth.bottom_exempt_frac"])
        self._min_depth = float(p["depth.min_valid"])

        # ------------------------------------------------------------ frames / TF
        self._base_frame = str(p["base_frame"]).strip()
        self._camera_optical_frame = str(p["camera_optical_frame"]).strip()
        self._default_goal_frame = str(p["global.default_goal_frame"]).strip() or "odom"

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._camera_info: Optional[CameraInfo] = None
        self._camera_tf_cache: Optional[np.ndarray] = None
        self._camera_tf_frame: Optional[str] = None

        # ------------------------------------------------------------ BEV / footprint
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
        )
        self._bev_trav_threshold = float(p["bev.traversability_threshold"])

        self._robot_length = float(p["robot.length"])
        self._robot_width = float(p["robot.width"])
        self._robot_margin = float(p["robot.safety_margin"])

        # ------------------------------------------------------------ local planner
        self._planner = LocalTrajectoryPlanner(
            PlannerConfig(
                horizon=float(p["planner.horizon"]),
                dt=float(p["planner.dt"]),
                v_samples=tuple(float(x) for x in p["planner.v_samples"]),
                w_samples=tuple(float(x) for x in p["planner.w_samples"]),
                start_ignore_distance=float(p["planner.start_ignore_distance"]),
                clearance_cap=float(p["planner.clearance_cap"]),
                progress_weight=float(p["planner.progress_weight"]),
                traversability_weight=float(p["planner.traversability_weight"]),
                clearance_weight=float(p["planner.clearance_weight"]),
                heading_weight=float(p["planner.heading_weight"]),
                turn_weight=float(p["planner.turn_weight"]),
                smooth_weight=float(p["planner.smooth_weight"]),
                max_angular=float(p["control.max_angular"]),
                turn_in_place_angle=radians(float(p["planner.turn_in_place_angle_deg"])),
                turn_gain=float(p["planner.turn_gain"]),
            )
        )

        # ------------------------------------------------------------ control / global goal
        self._control_enabled = bool(p["control.enabled"])
        self._watchdog = float(p["control.watchdog"])
        self._global_enabled = bool(p["global.enable"])
        self._goal_tolerance = float(p["global.goal_tolerance"])

        self._goal: Optional[PoseStamped] = None
        self._goal_reached = False
        self._cmd: Tuple[float, float] = (0.0, 0.0)
        self._previous_w = 0.0
        self._last_trajectory: Optional[Trajectory] = None
        self._last_footprint: Optional[FootprintResult] = None

        # ------------------------------------------------------------ runtime state
        self._process_period = float(p["process_period"])
        self._last_process: Optional[float] = None
        self._last_result: Optional[float] = None
        self._frame_count = 0

        self._out_dir: Optional[Path] = None
        self._save_every = max(1, int(p["output.save_every"]))
        if bool(p["output.enable"]):
            base = (
                Path(str(p["output.dir"]))
                if str(p["output.dir"]).strip()
                else Path(__file__).resolve().parents[1] / "output"
            )
            self._out_dir = base / time.strftime("%Y%m%d_%H%M%S")
            self._out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------ publishers
        self._ground_pub = self.create_publisher(
            Image, str(p["ground_mask_topic"]), qos_profile_sensor_data
        )
        self._trav_pub = self.create_publisher(
            Image, str(p["traversable_mask_topic"]), qos_profile_sensor_data
        )
        self._safe_pub = self.create_publisher(
            Image, str(p["safe_mask_topic"]), qos_profile_sensor_data
        )
        self._obst_pub = self.create_publisher(
            Image, str(p["obstacle_mask_topic"]), qos_profile_sensor_data
        )
        self._cov_pub = self.create_publisher(Float32, str(p["coverage_topic"]), 10)

        self._overlay_pub = None
        if bool(p["publish_overlay"]):
            self._overlay_pub = self.create_publisher(
                Image, str(p["overlay_topic"]), qos_profile_sensor_data
            )

        self._cmd_pub = None
        if self._control_enabled:
            self._cmd_pub = self.create_publisher(Twist, str(p["cmd_vel_topic"]), 10)

        # ------------------------------------------------------------ subscriptions
        self.create_subscription(
            CameraInfo,
            str(p["camera_info_topic"]),
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        if self._depth_enabled:
            self._sub_rgb = Subscriber(
                self,
                Image,
                str(p["image_topic"]),
                qos_profile=qos_profile_sensor_data,
            )
            self._sub_depth = Subscriber(
                self,
                Image,
                str(p["depth_topic"]),
                qos_profile=qos_profile_sensor_data,
            )
            self._sync = ApproximateTimeSynchronizer(
                [self._sub_rgb, self._sub_depth],
                queue_size=int(p["depth.sync_queue"]),
                slop=float(p["depth.sync_slop"]),
            )
            self._sync.registerCallback(self._on_images)
        else:
            self.create_subscription(
                Image,
                str(p["image_topic"]),
                self._on_image,
                qos_profile_sensor_data,
            )

        if self._global_enabled:
            self.create_subscription(
                PoseStamped,
                str(p["goal_pose_topic"]),
                self._on_goal,
                10,
            )

        if self._cmd_pub is not None:
            self.create_timer(
                1.0 / float(p["control.cmd_rate"]),
                self._on_cmd_timer,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        self.get_logger().info(
            "dinov3_nav ready: "
            f"device={self._gseg.device} "
            f"sam2={self._sam2 is not None} "
            f"stairs={self._stairs is not None} "
            f"depth={self._depth_enabled} "
            f"base_frame='{self._base_frame}' "
            f"bev={self._bev_cfg.x_max - self._bev_cfg.x_min:.1f}x"
            f"{self._bev_cfg.y_max - self._bev_cfg.y_min:.1f}m "
            f"res={self._bev_cfg.resolution:.2f}m "
            f"robot={self._robot_length:.2f}x{self._robot_width:.2f}m "
            f"margin={self._robot_margin:.2f}m "
            f"control={self._control_enabled} global_goal={self._global_enabled}"
        )

    # ---------------------------------------------------------------- parameters
    def _declare_params(self) -> dict:
        defaults = {
            # topics
            "image_topic": "/zed/zed_node/rgb/image_rect_color",
            "depth_topic": "/zed/zed_node/depth/depth_registered",
            "camera_info_topic": "/zed/zed_node/rgb/camera_info",
            "cmd_vel_topic": "/cmd_vel",
            "goal_pose_topic": "/goal_pose",
            "ground_mask_topic": "/dinov3_nav/ground_mask",
            "traversable_mask_topic": "/dinov3_nav/traversable_mask",
            "safe_mask_topic": "/dinov3_nav/safe_mask",
            "obstacle_mask_topic": "/dinov3_nav/obstacle_mask",
            "overlay_topic": "/dinov3_nav/overlay",
            "coverage_topic": "/dinov3_nav/coverage",
            "publish_overlay": True,
            # frames
            "base_frame": "zed_camera_link",
            "camera_optical_frame": "",
            # robot
            "robot.width": 0.40,
            "robot.length": 0.65,
            "robot.safety_margin": 0.10,
            # depth
            "depth.enable": True,
            "depth.safe_distance": 1.0,
            "depth.bottom_exempt_frac": 0.35,
            "depth.min_valid": 0.20,
            "depth.sync_slop": 0.05,
            "depth.sync_queue": 10,
            # DINO / ground segmentation
            "ckpt": DEFAULT_CKPT,
            "resolution": 448,
            "device": "auto",
            "seed_rows_frac": 0.10,
            "seed_cols_frac": 0.60,
            "threshold": "otsu",
            # SAM2
            "sam2.enable": True,
            "sam2.ckpt": str(DEFAULT_SAM2_CKPT),
            # stairs
            "stairs.enable": False,
            "stairs.threshold": 0.45,
            "stairs.min_area": 0.008,
            "stairs.max_area": 0.35,
            "stairs.min_risers": 3,
            "stairs.treat_traversable": True,
            # BEV
            "bev.resolution": 0.05,
            "bev.x_min": 0.0,
            "bev.x_max": 4.0,
            "bev.y_min": -2.0,
            "bev.y_max": 2.0,
            "bev.max_depth": 6.0,
            "bev.pixel_stride": 2,
            "bev.min_observed_points": 1,
            "bev.min_obstacle_points": 2,
            "bev.traversability_threshold": 0.50,
            # planner
            "planner.horizon": 2.0,
            "planner.dt": 0.10,
            "planner.v_samples": [0.15, 0.30, 0.45],
            "planner.w_samples": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
            "planner.start_ignore_distance": 0.25,
            "planner.clearance_cap": 0.80,
            "planner.progress_weight": 2.0,
            "planner.traversability_weight": 1.0,
            "planner.clearance_weight": 1.2,
            "planner.heading_weight": 0.8,
            "planner.turn_weight": 0.25,
            "planner.smooth_weight": 0.20,
            "planner.turn_in_place_angle_deg": 35.0,
            "planner.turn_gain": 1.2,
            # control
            "control.enabled": True,
            "control.max_angular": 0.8,
            "control.cmd_rate": 10.0,
            "control.watchdog": 1.5,
            # global goal
            "global.enable": True,
            "global.default_goal_frame": "odom",
            "global.goal_tolerance": 0.35,
            # output/runtime
            "process_period": 0.10,
            "output.enable": True,
            "output.dir": "",
            "output.save_every": 1,
        }

        values = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            values[name] = self.get_parameter(name).value

        thr = str(values["threshold"]).strip()
        values["threshold"] = "otsu" if thr.lower() == "otsu" else float(thr)
        return values

    # ---------------------------------------------------------------- callbacks
    def _on_camera_info(self, msg: CameraInfo):
        self._camera_info = msg

    def _on_images(self, rgb_msg: Image, depth_msg: Image):
        self._process(rgb_msg, depth_msg)

    def _on_image(self, rgb_msg: Image):
        self._process(rgb_msg, None)

    def _on_goal(self, msg: PoseStamped):
        self._goal = msg
        self._goal_reached = False
        self.get_logger().info(
            f"new global goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
            f"frame='{msg.header.frame_id or self._default_goal_frame}'"
        )

    # ---------------------------------------------------------------- geometry helpers
    def _camera_matrix(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._camera_info is None:
            return None

        info = self._camera_info
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        h, w = shape
        if info.width > 0 and info.height > 0:
            sx = w / float(info.width)
            sy = h / float(info.height)
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy

        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _resolved_camera_frame(self) -> Optional[str]:
        if self._camera_optical_frame:
            return self._camera_optical_frame
        if self._camera_info is None:
            return None
        frame = str(self._camera_info.header.frame_id).strip()
        return frame or None

    def _camera_to_base_matrix(self) -> Optional[np.ndarray]:
        camera_frame = self._resolved_camera_frame()
        if camera_frame is None:
            return None

        if "optical" not in camera_frame.lower():
            self.get_logger().warning(
                f"camera frame '{camera_frame}' does not look like an optical frame. "
                "RGB-D back-projection uses optical convention (X right, Y down, Z forward). "
                "Set camera_optical_frame explicitly if needed.",
                throttle_duration_sec=10.0,
            )

        if self._camera_tf_cache is not None and self._camera_tf_frame == camera_frame:
            return self._camera_tf_cache

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.10),
            )
        except TransformException as e:
            self.get_logger().warning(
                f"camera TF unavailable: {self._base_frame} <- {camera_frame}: {e}",
                throttle_duration_sec=5.0,
            )
            return None

        T = transform_to_matrix(tf.transform)
        self._camera_tf_cache = T
        self._camera_tf_frame = camera_frame
        return T

    def _global_goal_xy_in_base(self) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """Transform the persistent global goal into the current planner/base frame."""
        if self._goal is None:
            return None, None

        goal_frame = str(self._goal.header.frame_id).strip() or self._default_goal_frame

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                goal_frame,
                Time(),
                timeout=Duration(seconds=0.10),
            )
        except TransformException as e:
            self.get_logger().warning(
                f"goal TF unavailable: {self._base_frame} <- {goal_frame}: {e}",
                throttle_duration_sec=5.0,
            )
            return None, None

        T = transform_to_matrix(tf.transform)
        p = self._goal.pose.position
        goal_world = np.array([float(p.x), float(p.y), float(p.z), 1.0], dtype=np.float32)
        goal_base = T @ goal_world

        gx = float(goal_base[0])
        gy = float(goal_base[1])
        return (gx, gy), hypot(gx, gy)

    # ---------------------------------------------------------------- perception / planning
    def _depth_obstacle(
        self,
        depth: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Simple near-obstacle evidence in image space."""
        h, w = shape
        obstacle = np.zeros((h, w), dtype=bool)
        if depth is None:
            return obstacle

        valid = np.isfinite(depth) & (depth > self._min_depth)
        obstacle = valid & (depth < self._safe_distance)

        # Avoid classifying the near floor at the image bottom as an obstacle.
        r0 = int((1.0 - self._bottom_exempt) * h)
        obstacle[r0:, :] = False
        return obstacle

    def _process(self, rgb_msg: Image, depth_msg: Optional[Image]):
        now = time.monotonic()
        if self._last_process is not None and now - self._last_process < self._process_period:
            return
        self._last_process = now

        try:
            rgb = image_to_rgb(rgb_msg)
        except ValueError as e:
            self.get_logger().warning(f"RGB conversion failed: {e}", throttle_duration_sec=10.0)
            return

        depth = None
        if depth_msg is not None:
            try:
                depth = depth_to_meters(depth_msg)
                if depth.shape[:2] != rgb.shape[:2]:
                    self.get_logger().warning(
                        f"depth shape {depth.shape[:2]} != RGB shape {rgb.shape[:2]}; ignore depth",
                        throttle_duration_sec=10.0,
                    )
                    depth = None
            except ValueError as e:
                self.get_logger().warning(
                    f"depth conversion failed: {e}", throttle_duration_sec=10.0
                )

        t0 = time.perf_counter()

        # Default to stop on every frame; only a valid planned trajectory can overwrite it.
        self._cmd = (0.0, 0.0)
        self._last_trajectory = None
        self._last_footprint = None

        # ---------------- perception
        try:
            prepared = self._gseg.prepare(rgb)
            ground = self._gseg.segment_prepared(prepared, depth=depth)

            if self._sam2 is not None:
                try:
                    refined = self._sam2.refine_prepared(prepared, ground.mask)
                    if refined.iou_with_coarse < 0.35:
                        self.get_logger().warning(
                            f"SAM2 low consistency IoU={refined.iou_with_coarse:.2f}; keep DINO mask",
                            throttle_duration_sec=10.0,
                        )
                    else:
                        ground.mask = refined.mask
                        ground.coverage = float((ground.mask > 0).mean())
                except Exception as e:
                    self.get_logger().warning(
                        f"SAM2 refinement failed; keep DINO mask: {e}",
                        throttle_duration_sec=10.0,
                    )

            sres = self._stairs.detect_prepared(prepared) if self._stairs is not None else None

            traversable = ground.mask.copy()
            if sres is not None and self._stairs_traversable and sres.mask.any():
                traversable = cv2.bitwise_or(traversable, sres.mask)

            obstacle = self._depth_obstacle(depth, rgb.shape[:2])
            image_safe = (traversable > 0) & ~obstacle

        except Exception as e:
            self.get_logger().error(
                f"perception failed: {e}", throttle_duration_sec=10.0
            )
            return

        # ---------------- local BEV + footprint + trajectory planner
        goal_distance: Optional[float] = None
        local_goal: Optional[Tuple[float, float]] = None
        bev_safe_cov = 0.0

        if depth is None:
            self.get_logger().warning(
                "local BEV planner requires registered depth; command is zero",
                throttle_duration_sec=5.0,
            )
        else:
            K = self._camera_matrix(depth.shape)
            T_base_camera = self._camera_to_base_matrix()

            if K is None:
                self.get_logger().warning(
                    "CameraInfo/intrinsics unavailable; command is zero",
                    throttle_duration_sec=5.0,
                )
            elif T_base_camera is None:
                # TF helper already logged the detailed reason.
                pass
            else:
                try:
                    valid_depth = (
                        np.isfinite(depth)
                        & (depth >= self._min_depth)
                        & (depth <= self._bev_cfg.max_depth)
                    )

                    # Anything with valid geometry but not traversable is obstacle evidence.
                    # Near-depth obstacle evidence is kept explicitly as a hard obstacle source.
                    obstacle_for_bev = obstacle | (valid_depth & ~(traversable > 0))

                    bev = build_local_bev(
                        traversable_mask=traversable,
                        obstacle_mask=obstacle_for_bev,
                        depth=depth,
                        K=K,
                        T_base_from_camera=T_base_camera,
                        cfg=self._bev_cfg,
                    )

                    footprint = build_body_safe_space(
                        bev=bev,
                        robot_length=self._robot_length,
                        robot_width=self._robot_width,
                        safety_margin=self._robot_margin,
                        traversability_threshold=self._bev_trav_threshold,
                    )
                    self._last_footprint = footprint
                    bev_safe_cov = float(footprint.safe.mean())

                    if self._global_enabled:
                        local_goal, goal_distance = self._global_goal_xy_in_base()
                    else:
                        local_goal = (self._bev_cfg.x_max - 0.5, 0.0)
                        goal_distance = float("inf")

                    self._goal_reached = (
                        goal_distance is not None
                        and goal_distance <= self._goal_tolerance
                    )

                    if not self._goal_reached and local_goal is not None:
                        trajectory = self._planner.plan(
                            bev=bev,
                            footprint=footprint,
                            goal_xy=local_goal,
                            previous_w=self._previous_w,
                        )

                        if trajectory is not None:
                            self._last_trajectory = trajectory
                            self._cmd = (float(trajectory.v), float(trajectory.w))
                            self._previous_w = float(trajectory.w)
                    else:
                        self._cmd = (0.0, 0.0)
                        if self._goal_reached:
                            self._previous_w = 0.0

                except Exception as e:
                    self.get_logger().error(
                        f"BEV/planner failed: {e}", throttle_duration_sec=10.0
                    )
                    self._cmd = (0.0, 0.0)

        # ---------------- publish debug outputs
        header = rgb_msg.header
        self._ground_pub.publish(gray_to_image_msg(ground.mask, header))
        self._trav_pub.publish(gray_to_image_msg(traversable, header))

        # This topic remains IMAGE-space debug only. Final body-safe planning happens in BEV.
        self._safe_pub.publish(gray_to_image_msg(image_safe, header))
        self._obst_pub.publish(gray_to_image_msg(obstacle, header))
        self._cov_pub.publish(Float32(data=float(ground.coverage)))

        overlay = None
        if self._overlay_pub is not None or self._out_dir is not None:
            overlay = self._draw_overlay(
                rgb=rgb,
                traversable=traversable,
                image_safe=image_safe,
                obstacle=obstacle,
                local_goal=local_goal,
                goal_distance=goal_distance,
                trajectory=self._last_trajectory,
                bev_safe_cov=bev_safe_cov,
            )
            if self._overlay_pub is not None:
                self._overlay_pub.publish(rgb_to_image_msg(overlay, header))

        self._last_result = time.monotonic()
        self._frame_count += 1

        if (
            self._out_dir is not None
            and overlay is not None
            and self._frame_count % self._save_every == 0
        ):
            try:
                cv2.imwrite(
                    str(self._out_dir / f"frame_{self._frame_count:06d}_nav.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
            except Exception as e:
                self.get_logger().warning(
                    f"failed to save overlay: {e}", throttle_duration_sec=10.0
                )

        if self._frame_count == 1 or self._frame_count % 20 == 0:
            elapsed = time.perf_counter() - t0
            thr = (
                f"{ground.threshold_used:.3f}"
                if ground.threshold_used is not None
                else "fallback"
            )

            if self._goal_reached and goal_distance is not None:
                state_txt = f"goal_reached d={goal_distance:.2f}m"
            elif self._last_trajectory is not None:
                state_txt = (
                    f"traj v={self._last_trajectory.v:.2f} "
                    f"w={self._last_trajectory.w:+.2f} "
                    f"score={self._last_trajectory.score:.2f}"
                )
            elif self._global_enabled and self._goal is None:
                state_txt = "waiting_goal"
            else:
                state_txt = "no_valid_trajectory"

            self.get_logger().info(
                f"frame {self._frame_count}: "
                f"ground_cov={ground.coverage:.2f} "
                f"bev_safe={bev_safe_cov:.2f} "
                f"thr={thr} {state_txt} "
                f"t={elapsed:.2f}s "
                f"cmd=({self._cmd[0]:.2f},{self._cmd[1]:+.2f})"
            )

    # ---------------------------------------------------------------- control
    def _on_cmd_timer(self):
        if self._cmd_pub is None:
            return

        now = time.monotonic()
        stale = self._last_result is None or (now - self._last_result > self._watchdog)
        lin, ang = (0.0, 0.0) if stale else self._cmd

        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self._cmd_pub.publish(msg)

    def publish_stop(self):
        if self._cmd_pub is not None:
            try:
                self._cmd_pub.publish(Twist())
            except Exception:
                pass

    # ---------------------------------------------------------------- visualization
    def _draw_overlay(
        self,
        rgb: np.ndarray,
        traversable: np.ndarray,
        image_safe: np.ndarray,
        obstacle: np.ndarray,
        local_goal: Optional[Tuple[float, float]],
        goal_distance: Optional[float],
        trajectory: Optional[Trajectory],
        bev_safe_cov: float,
    ) -> np.ndarray:
        out = rgb.copy()

        carved = (traversable > 0) & ~image_safe
        if carved.any():
            out[carved] = (
                0.45 * out[carved] + 0.55 * np.array([40, 90, 40])
            ).astype(np.uint8)

        if image_safe.any():
            out[image_safe] = (
                0.45 * out[image_safe] + 0.55 * np.array([60, 220, 60])
            ).astype(np.uint8)

        if obstacle.any():
            out[obstacle] = (
                0.45 * out[obstacle] + 0.55 * np.array([255, 60, 60])
            ).astype(np.uint8)

        lin, ang = self._cmd
        cv2.putText(
            out,
            f"cmd v={lin:.2f} w={ang:+.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 220, 0),
            2,
        )

        if self._goal_reached:
            state = "GOAL REACHED"
        elif trajectory is not None:
            state = f"traj score={trajectory.score:.2f} bev_safe={bev_safe_cov:.2f}"
        elif self._global_enabled and self._goal is None:
            state = "WAITING GLOBAL GOAL"
        else:
            state = "NO VALID TRAJECTORY"

        cv2.putText(
            out,
            state,
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 0) if trajectory is not None else (255, 80, 80),
            2,
        )

        if local_goal is not None:
            gx, gy = local_goal
            gtxt = f"goal(base) x={gx:.2f} y={gy:.2f}"
            if goal_distance is not None and np.isfinite(goal_distance):
                gtxt += f" d={goal_distance:.2f}"
            cv2.putText(
                out,
                gtxt,
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 220, 0),
                2,
            )

        return out


def main(args=None) -> int:
    rclpy.init(args=args)
    node = Dinov3NavNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0

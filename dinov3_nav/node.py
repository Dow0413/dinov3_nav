# -*- coding: utf-8 -*-
"""DINOv3 反应式视觉导航节点（CAT 风格 demo，无全局规划）。

数据流：

    /zed/zed_node/rgb/image_rect_color ─┐
                                        ├─ 时间同步 → 融合 → 局部 waypoint → /cmd_vel
    /zed/zed_node/depth/depth_registered┘

    RGB   → ground_seg（DINOv3 多原型相似度 + SAM2）→ 视觉地面证据
    Depth → 地面平面一致性 + 近距障碍
    融合  → 视觉地面与贴合地面平面的几何证据互补；再从可走区扣除障碍，
            只保留从脚下连续可达的区域（当前阶段不考虑机身宽度）
    目标  → 连通安全通路内"尽可能远、优先居中"的点 = 局部 waypoint

设计取舍（第一版优先简单、可解释、实时）：

- 深度与 DINO 地面掩码共同拟合免标定的地面平面。贴合该平面、但视觉
  相似度略低的像素可补回；高出平面的视觉误检会被抑制。近距障碍只在
  上部行判定，避免前视相机脚下地面深度过小而被误删。
- 当前阶段不对障碍做膨胀，也不检查机身走廊净空；只要求像素级地面从
  画面底部连续连到 waypoint。这样优先验证地面召回、局部目标选择和转向。
- 无 waypoint（没有足够宽的 gate）→ 零速；watchdog 超时 → 零速；退出
  前发零速。控制就是"转向对准 waypoint + 按其距离定速"，可解释。
- 每帧重规划（~6-10Hz），不记忆旧 waypoint；被挡时侧向 gate 自动胜出，
  即"向左/右搜索新的安全方向"。

话题与全部阈值见 config/params.yaml；核心分割算法在仓库包 ground_seg 中。
"""

from __future__ import annotations

import time
from array import array
from math import atan2, cos, hypot, pi, radians
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from ground_seg import GroundSegmenter, SAM2MaskRefiner
from ground_seg.features import DEFAULT_CKPT
from ground_seg.sam2_refiner import DEFAULT_SAM2_CKPT

_RGB_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def image_to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → (H, W, 3) uint8 RGB。手写转换，不依赖 cv_bridge
    （cv_bridge 在系统 ROS 侧，venv 里混装容易出版本坑）。"""
    enc = msg.encoding.lower()
    channels = _RGB_CHANNELS.get(enc)
    if channels is None:
        raise ValueError(
            f"unsupported encoding {msg.encoding!r} (支持 rgb8/bgr8/rgba8/bgra8/mono8)")
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
    return np.ascontiguousarray(arr)


def depth_to_meters(msg: Image) -> np.ndarray:
    """sensor_msgs/Image(深度) → (H, W) float32 米。支持 32FC1(米) 与
    16UC1(毫米，Gazebo depth 插件常用)。无效值保持 NaN/0，由调用方过滤。"""
    enc = msg.encoding.lower()
    if enc == "32fc1":
        dtype, scale = np.float32, 1.0
    elif enc == "16uc1":
        dtype, scale = np.uint16, 0.001
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r} (支持 32fc1/16uc1)")
    h, w = msg.height, msg.width
    item = np.dtype(dtype).itemsize
    if msg.step % item:
        raise ValueError(f"depth step {msg.step} 不是 {item} 的整数倍")
    row = np.frombuffer(msg.data, dtype=dtype, count=h * (msg.step // item))
    arr = row.reshape(h, msg.step // item)[:, :w].astype(np.float32) * scale
    return arr


def gray_to_image_msg(mask: np.ndarray, header) -> Image:
    out = Image()
    out.header = header
    out.height, out.width = mask.shape[:2]
    out.encoding = "mono8"
    out.is_bigendian = 0
    out.step = out.width
    out.data = array("B", mask.tobytes())
    return out


def rgb_to_image_msg(rgb: np.ndarray, header) -> Image:
    out = Image()
    out.header = header
    out.height, out.width = rgb.shape[:2]
    out.encoding = "rgb8"
    out.is_bigendian = 0
    out.step = out.width * 3
    out.data = array("B", rgb.tobytes())
    return out


class Dinov3NavNode(Node):
    """RGB+深度 → 安全可通行区域 → 最远安全 waypoint → /cmd_vel。"""

    def __init__(self):
        super().__init__("dinov3_nav")
        p = self._declare_params()

        # ------------------------------------------------------------ 模型
        self._gseg = GroundSegmenter(
            resolution=p["resolution"], ckpt=p["ckpt"], device=p["device"],
            seed_rows_frac=p["seed_rows_frac"], seed_cols_frac=p["seed_cols_frac"],
            threshold=p["threshold"],
        )
        self._sam2: Optional[SAM2MaskRefiner] = None
        if p["sam2.enable"]:
            try:
                self._sam2 = SAM2MaskRefiner(checkpoint=p["sam2.ckpt"], device=p["device"])
            except Exception as e:
                self.get_logger().warning(f"SAM2 不可用，保留 DINO 掩码：{e}")
        self._stairs = None
        if p["stairs.enable"]:
            try:
                from ground_seg import DINOStairsDetector

                self._stairs = DINOStairsDetector(
                    resolution=p["resolution"], threshold=p["stairs.threshold"],
                    min_area_frac=p["stairs.min_area"], max_area_frac=p["stairs.max_area"],
                    min_risers=p["stairs.min_risers"],
                    device=p["device"], extractor=self._gseg._extractor,
                )
            except Exception as e:
                self.get_logger().warning(f"楼梯头不可用，降级为纯地面：{e}")

        # -------------------------------------------------------- 几何常量
        self._hfov_rad = radians(float(p["robot.hfov"]))
        self._safe_distance = float(p["depth.safe_distance"])
        self._bottom_exempt = float(p["depth.bottom_exempt_frac"])
        self._min_depth = float(p["depth.min_valid"])

        # -------------------------------------------------------- 控制参数
        self._control_enabled = p["control.enabled"]
        self._cruise_linear = p["control.cruise_linear"]
        self._max_angular = p["control.max_angular"]
        self._k_angular = p["control.k_angular"]
        self._center_weight = p["control.center_weight"]
        self._stop_depth = p["control.stop_depth"]
        self._full_speed_depth = p["control.full_speed_depth"]
        self._min_coverage = p["control.min_coverage"]
        self._watchdog = p["control.watchdog"]
        self._turn_in_place_angle = radians(float(p["control.turn_in_place_angle_deg"]))

        # -------------------------------------------------------- 全局目标引导
        self._global_enabled = p["global.enable"]
        self._goal_tolerance = float(p["global.goal_tolerance"])
        self._goal_weight = float(p["global.goal_weight"])
        self._forward_weight = float(p["global.forward_weight"])
        self._camera_yaw = radians(float(p["camera.yaw_offset_deg"]))
        self._state_timeout = float(p["global.state_timeout"])

        # -------------------------------------------------------- 运行状态
        self._process_period = p["process_period"]
        self._stairs_traversable = p["stairs.treat_traversable"]
        self._cmd: Tuple[float, float] = (0.0, 0.0)
        self._last_process: Optional[float] = None   # time.monotonic()，节流用
        self._last_result: Optional[float] = None    # 最近一帧成功分割的时刻
        self._frame_count = 0
        self._last_wp: Optional[Tuple[float, float, float]] = None  # (row,col,depth) 供绘图
        self._last_width = 640                       # 最近一帧宽度，列→角度换算用
        self._odom: Optional[Odometry] = None
        self._odom_received: Optional[float] = None
        self._goal: Optional[PoseStamped] = None
        self._goal_received: Optional[float] = None
        self._goal_reached = False

        # 输出目录：每次启动按时间标签新建一个文件夹（如 output/20260920_153045），
        # 推理出的叠加图按帧号存成 frame_000001_ground.png。
        self._out_dir: Optional[Path] = None
        self._save_every = max(1, int(p["output.save_every"]))
        if p["output.enable"]:
            base = (Path(p["output.dir"]) if p["output.dir"]
                    else Path(__file__).resolve().parents[1] / "output")
            self._out_dir = base / time.strftime("%Y%m%d_%H%M%S")
            self._out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------ IO
        self._ground_pub = self.create_publisher(
            Image, p["ground_mask_topic"], qos_profile_sensor_data)
        self._trav_pub = self.create_publisher(
            Image, p["traversable_mask_topic"], qos_profile_sensor_data)
        self._safe_pub = self.create_publisher(
            Image, p["safe_mask_topic"], qos_profile_sensor_data)
        self._obst_pub = self.create_publisher(
            Image, p["obstacle_mask_topic"], qos_profile_sensor_data)
        self._cov_pub = self.create_publisher(Float32, p["coverage_topic"], 10)
        self._overlay_pub = (
            self.create_publisher(Image, p["overlay_topic"], qos_profile_sensor_data)
            if p["publish_overlay"] else None
        )
        self._cmd_pub = (
            self.create_publisher(Twist, p["cmd_vel_topic"], 10)
            if self._control_enabled else None
        )

        # 订阅：RGB 与深度做近似时间同步（registered 深度与 RGB 像素对齐）。
        # 两个 Subscriber 留在默认互斥组里（同步回调串行化），推理耗时不再
        # 饿死 cmd_vel 定时器——定时器单独成组跑在另一执行线程。
        if p["depth.enable"]:
            sub_rgb = Subscriber(self, Image, p["image_topic"],
                                 qos_profile=qos_profile_sensor_data)
            sub_depth = Subscriber(self, Image, p["depth_topic"],
                                   qos_profile=qos_profile_sensor_data)
            sync = ApproximateTimeSynchronizer(
                [sub_rgb, sub_depth], queue_size=10, slop=p["depth.sync_slop"])
            sync.registerCallback(self._on_images)
        else:
            self.create_subscription(
                Image, p["image_topic"], self._on_image,
                qos_profile_sensor_data)
        if self._global_enabled:
            self.create_subscription(Odometry, p["odom_topic"], self._on_odom, 10)
            self.create_subscription(PoseStamped, p["goal_pose_topic"], self._on_goal, 10)
        if self._cmd_pub is not None:
            self.create_timer(1.0 / p["control.cmd_rate"], self._on_cmd_timer,
                              callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info(
            f"dinov3_nav ready: ckpt={Path(p['ckpt']).name} resolution={p['resolution']} "
            f"device={self._gseg.device} sam2={self._sam2 is not None} "
            f"stairs={self._stairs is not None} depth={p['depth.enable']} "
            f"safe_dist={self._safe_distance:.2f}m "
            f"image='{p['image_topic']}' cmd_vel='{p['cmd_vel_topic']}' "
            f"control={self._control_enabled} global_goal={self._global_enabled}"
            + (f" output={self._out_dir}" if self._out_dir is not None else "")
        )

    # ----------------------------------------------------------- 参数

    def _declare_params(self) -> dict:
        defaults = {
            # 话题
            "image_topic": "/zed/zed_node/rgb/image_rect_color",
            "depth_topic": "/zed/zed_node/depth/depth_registered",
            "cmd_vel_topic": "/cmd_vel",
            "odom_topic": "/odometry",
            "goal_pose_topic": "/goal_pose",
            "ground_mask_topic": "/dinov3_nav/ground_mask",
            "traversable_mask_topic": "/dinov3_nav/traversable_mask",
            "safe_mask_topic": "/dinov3_nav/safe_mask",
            "obstacle_mask_topic": "/dinov3_nav/obstacle_mask",
            "overlay_topic": "/dinov3_nav/overlay",
            "coverage_topic": "/dinov3_nav/coverage",
            "publish_overlay": True,
            # 机器人 / 相机几何（列→角度、米→像素换算都用 hfov 近似内参）
            "robot.width": 0.40,
            "robot.hfov": 90.0,
            "camera.yaw_offset_deg": 0.0,
            # 深度安全
            "depth.enable": True,
            "depth.safe_distance": 1.0,
            "depth.inflation": 0.15,
            "depth.bottom_exempt_frac": 0.35,
            "depth.min_valid": 0.20,
            "depth.sync_slop": 0.05,
            # 模型（与 ground_seg/main.py 同一套）
            "ckpt": DEFAULT_CKPT,
            "resolution": 448,
            "device": "auto",
            "seed_rows_frac": 0.10,
            "seed_cols_frac": 0.60,
            "threshold": "otsu",
            # SAM2 边界精修
            "sam2.enable": True,
            "sam2.ckpt": str(DEFAULT_SAM2_CKPT),
            # 楼梯候选头（默认关，与 main.py 一致）
            "stairs.enable": False,
            "stairs.threshold": 0.45,
            "stairs.min_area": 0.008,
            "stairs.max_area": 0.35,
            "stairs.min_risers": 3,
            "stairs.treat_traversable": True,
            # 输出
            "output.enable": True,
            "output.dir": "",
            "output.save_every": 1,
            # 运行节奏
            "process_period": 0.1,
            # 控制律（waypoint 跟踪）
            "control.enabled": True,
            "control.cruise_linear": 0.6,
            "control.max_angular": 0.8,
            "control.k_angular": 1.2,
            "control.center_weight": 0.3,
            "control.stop_depth": 0.5,
            "control.full_speed_depth": 2.0,
            "control.min_coverage": 0.03,
            "control.cmd_rate": 10.0,
            "control.watchdog": 1.5,
            "control.turn_in_place_angle_deg": 35.0,
            # 全局目标（Odometry 与 PoseStamped 必须处于相同世界坐标系）
            "global.enable": True,
            "global.goal_tolerance": 0.35,
            "global.goal_weight": 1.2,
            "global.forward_weight": 1.0,
            "global.state_timeout": 1.0,
        }
        values = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            values[name] = self.get_parameter(name).value
        # threshold 参数统一为字符串：'otsu' 或固定余弦阈值（如 "0.55"）
        thr = str(values["threshold"]).strip()
        values["threshold"] = "otsu" if thr == "otsu" else float(thr)
        return values

    # ----------------------------------------------------------- 回调

    def _on_images(self, rgb_msg: Image, depth_msg: Image):
        self._process(rgb_msg, depth_msg)

    def _on_image(self, rgb_msg: Image):
        self._process(rgb_msg, None)

    def _on_odom(self, msg: Odometry):
        self._odom = msg
        self._odom_received = time.monotonic()

    def _on_goal(self, msg: PoseStamped):
        self._goal = msg
        self._goal_received = time.monotonic()
        self._goal_reached = False
        self.get_logger().info(
            f"new global goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
            f"frame='{msg.header.frame_id or '<unspecified>'}'")

    def _process(self, rgb_msg: Image, depth_msg: Optional[Image]):
        now = time.monotonic()
        if self._last_process is not None and now - self._last_process < self._process_period:
            return  # 上一帧还没处理完/间隔未到，丢弃（相机 30Hz，推理 ~6-10Hz）
        try:
            rgb = image_to_rgb(rgb_msg)
        except ValueError as e:
            self.get_logger().warning(f"图像转换失败：{e}", throttle_duration_sec=10.0)
            return
        depth = None
        if depth_msg is not None:
            try:
                depth = depth_to_meters(depth_msg)
                if depth.shape[:2] != rgb.shape[:2]:
                    self.get_logger().warning(
                        f"深度 {depth.shape[:2]} 与 RGB {rgb.shape[:2]} 尺寸不符，忽略深度",
                        throttle_duration_sec=10.0)
                    depth = None
            except ValueError as e:
                self.get_logger().warning(f"深度转换失败：{e}", throttle_duration_sec=10.0)
        self._last_process = now

        t0 = time.perf_counter()
        try:
            # ---- RGB：地面分割（与 ground_seg/main.py 同一流程）----
            prepared = self._gseg.prepare(rgb)          # 一次 DINOv3 前向，各头共用
            # registered depth 在 ground_seg 内部以 DINO 视觉掩码为可靠种子
            # 拟合地面平面：贴平面的弱视觉区域可被召回，明显高出平面的
            # 视觉误检会被抑制，不能只把 RGB 与深度生硬取交集。
            ground = self._gseg.segment_prepared(prepared, depth=depth)
            if self._sam2 is not None:
                try:
                    refined = self._sam2.refine_prepared(prepared, ground.mask)
                    if refined.iou_with_coarse < 0.35:
                        self.get_logger().warning(
                            f"SAM2 一致性低 IoU={refined.iou_with_coarse:.2f}，保留 DINO 掩码",
                            throttle_duration_sec=10.0)
                    else:
                        ground.mask = refined.mask
                        ground.coverage = float((ground.mask > 0).mean())
                except Exception as e:
                    self.get_logger().warning(f"SAM2 精修失败，回退 DINO 掩码：{e}",
                                              throttle_duration_sec=10.0)
            sres = self._stairs.detect_prepared(prepared) if self._stairs is not None else None
            traversable = ground.mask
            if sres is not None and self._stairs_traversable and sres.mask.any():
                traversable = cv2.bitwise_or(traversable, sres.mask)

            # ---- Depth：近距障碍（当前阶段不建模机器狗机身宽度/膨胀）----
            obstacle = self._depth_obstacle(depth, rgb.shape[:2])
            h, w = rgb.shape[:2]
            self._last_width = w
            blocked = obstacle

            # ---- 融合：视觉/平面互补地面 ∩ 无近距障碍 ∩ 从脚下连续可达 ----
            # 不做形态学膨胀或机身净空筛选，先把地面召回与转向行为调通。
            route = (traversable > 0) & ~blocked
            fused = self._bottom_connected(route)
        except Exception as e:
            self.get_logger().error(f"分割/融合失败：{e}", throttle_duration_sec=10.0)
            return
        elapsed = time.perf_counter() - t0

        # ---- waypoint 搜索与控制 ----
        safe_cov = float(fused.mean())
        goal_heading = None
        goal_distance = None
        if self._global_enabled:
            goal_heading, goal_distance = self._global_goal_in_base()
        wp = None
        self._goal_reached = (goal_distance is not None
                              and goal_distance <= self._goal_tolerance)
        if (not self._goal_reached and safe_cov >= self._min_coverage
                and (not self._global_enabled or goal_heading is not None)):
            wp = self._find_waypoint(fused, depth, goal_heading)
        self._last_wp = wp
        self._cmd = self._compute_command(wp)

        # ------------------------------------------------------------ 发布
        header = rgb_msg.header
        self._ground_pub.publish(gray_to_image_msg(ground.mask, header))
        self._trav_pub.publish(gray_to_image_msg(traversable, header))
        self._safe_pub.publish(gray_to_image_msg((fused * 255).astype(np.uint8), header))
        self._obst_pub.publish(gray_to_image_msg((blocked * 255).astype(np.uint8), header))
        self._cov_pub.publish(Float32(data=float(ground.coverage)))
        overlay = None
        if self._overlay_pub is not None or self._out_dir is not None:
            overlay = self._draw_overlay(rgb, ground.mask, fused, obstacle, wp)
            if self._overlay_pub is not None:
                self._overlay_pub.publish(rgb_to_image_msg(overlay, header))

        self._last_result = time.monotonic()
        self._frame_count += 1
        if (self._out_dir is not None and overlay is not None
                and self._frame_count % self._save_every == 0):
            try:
                cv2.imwrite(
                    str(self._out_dir / f"frame_{self._frame_count:06d}_ground.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as e:
                self.get_logger().warning(f"保存 ground.png 失败：{e}",
                                          throttle_duration_sec=10.0)
        if self._frame_count == 1 or self._frame_count % 20 == 0:
            thr = (f"{ground.threshold_used:.3f}"
                   if ground.threshold_used is not None else "fallback")
            n_warn = len(ground.warnings) + (len(sres.warnings) if sres is not None else 0)
            warn = f"  warnings={n_warn}" if n_warn else ""
            if self._goal_reached:
                wp_txt = f"goal_reached ({goal_distance:.2f}m)"
            elif wp is not None:
                _, col, gd = wp
                angle_deg = np.degrees(self._col_to_angle(col, w))
                goal_txt = (f" goal={np.degrees(goal_heading):+.0f}deg"
                            if goal_heading is not None else "")
                wp_txt = f"wp=({angle_deg:+.0f}deg,{gd:.1f}m){goal_txt}"
            else:
                wp_txt = "wp=none" if not self._global_enabled else "wp=none (waiting goal/odom)"
            lin, ang = self._cmd
            self.get_logger().info(
                f"frame {self._frame_count}: cov={ground.coverage:.2f} safe={safe_cov:.2f} "
                f"thr={thr} {wp_txt} t={elapsed:.2f}s cmd=({lin:.2f}, {ang:+.2f}){warn}")

    # ----------------------------------------------------------- 感知原语

    def _depth_obstacle(self, depth: Optional[np.ndarray],
                        shape: Tuple[int, int]) -> np.ndarray:
        """近距障碍掩码。只在上部 (1-bottom_exempt) 行判定：前视相机底部行
        的地面本身深度就小，全图判会把脚下删光；近距竖直物在上部行可见。"""
        h, w = shape
        obstacle = np.zeros((h, w), bool)
        if depth is None:
            return obstacle
        valid = np.isfinite(depth) & (depth > self._min_depth)
        obstacle = valid & (depth < self._safe_distance)
        obstacle[int((1.0 - self._bottom_exempt) * h):, :] = False
        return obstacle

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """归一化到 [-pi, pi]。"""
        return (angle + pi) % (2.0 * pi) - pi

    def _global_goal_in_base(self) -> Tuple[Optional[float], Optional[float]]:
        """将 world/odom 中的目标变为机器人 base 平面的方位角和距离。

        这里不悄悄猜测 map→odom TF：goal 与 odometry frame 不一致时直接停住，
        以免把 map 坐标误作 odom 坐标导致反向行驶。若两者 frame_id 都为空，
        则按调用方已保证坐标一致处理。
        """
        now = time.monotonic()
        if self._odom is None or self._goal is None:
            return None, None
        if (self._odom_received is None or self._goal_received is None
                or now - self._odom_received > self._state_timeout
                or now - self._goal_received > self._state_timeout):
            self.get_logger().warning("odometry/goal stale; stopping", throttle_duration_sec=5.0)
            return None, None
        odom_frame = self._odom.header.frame_id
        goal_frame = self._goal.header.frame_id
        if odom_frame and goal_frame and odom_frame != goal_frame:
            self.get_logger().error(
                f"goal frame '{goal_frame}' != odom frame '{odom_frame}'; "
                "publish goal in odom frame or add a TF transform",
                throttle_duration_sec=5.0)
            return None, None

        pose = self._odom.pose.pose
        dx = self._goal.pose.position.x - pose.position.x
        dy = self._goal.pose.position.y - pose.position.y
        distance = hypot(dx, dy)
        q = pose.orientation
        # 平面导航只取 quaternion 的 yaw；q 已由 Odometry 保证是单位四元数。
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        world_bearing = atan2(dy, dx)
        return self._wrap_angle(world_bearing - yaw), distance

    @staticmethod
    def _bottom_connected(mask: np.ndarray) -> np.ndarray:
        """只保留与画面底部种子带（底 10% 行 × 中间 60% 列）8 连通的区域：
        保证允许通过区域从"机器人脚下"连续延伸，排除远处孤立地面。"""
        h, w = mask.shape
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        r0 = int(0.90 * h)
        band = labels[r0:, int(0.2 * w):int(0.8 * w)]
        keep = {int(l) for l in np.unique(band) if l != 0}
        if not keep:
            return np.zeros_like(mask)
        return np.isin(labels, list(keep))

    def _find_waypoint(self, fused: np.ndarray, depth: Optional[np.ndarray],
                       goal_heading: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
        """在允许区域内找最远安全 waypoint。

        `fused` 已经过底部连通筛选，其中每个候选点都有从机器人脚下延伸
        过去的像素级连续地面。自底向上找连通段，得分 = 越远越高 + 居中
        偏好，取最优段中心。返回 (row, col, gate_depth)。
        """
        h, w = fused.shape
        half = (w - 1) / 2.0
        best = None
        for r in range(h - 2, int(0.25 * h), -8):       # 从底往上扫到 25% 画面高
            row = fused[r]
            if not row.any():
                continue
            edges = np.diff(np.pad(row.astype(np.int8), 1))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for c0, c1 in zip(starts, ends):
                if c1 - c0 < 3:                          # 单像素尖角不作为 waypoint
                    continue
                if depth is not None:
                    seg = depth[r, c0:c1]
                    ok = np.isfinite(seg) & (seg > 0.05)
                    gd = float(np.median(seg[ok])) if ok.any() else self._safe_distance
                else:
                    gd = self._safe_distance             # 无深度 → 按安全距离保守处理
                gd = max(gd, self._min_depth)
                center = (c0 + c1 - 1) / 2.0
                # 只在已连通的安全像素内选点：避障来自 fused 的约束；全局
                # 引导来自候选方向与全局目标方位的夹角。目标被障碍挡住时，
                # 不可行方向没有候选，得分最高的侧向可通行分支自然成为绕障点。
                forward = 1.0 - r / h
                image_angle = self._col_to_angle(center, w)
                base_heading = self._image_to_base_angle(image_angle)
                if goal_heading is None:
                    goal_alignment = 0.0
                else:
                    goal_alignment = cos(self._wrap_angle(base_heading - goal_heading))
                score = (self._forward_weight * forward
                         + self._goal_weight * goal_alignment
                         + self._center_weight * (1.0 - abs(center - half) / half))
                if best is None or score > best[0]:
                    best = (score, float(r), center, gd)
        if best is None:
            return None
        _, r, c, gd = best
        return (r, c, gd)

    # ----------------------------------------------------------- 控制律

    def _col_to_angle(self, col: float, w: int) -> float:
        """图像列 → 水平角度(rad)，右半画面为正。"""
        return (col - (w - 1) / 2.0) / ((w - 1) / 2.0) * self._hfov_rad / 2.0

    def _image_to_base_angle(self, image_angle: float) -> float:
        """光学图像方位 → base_link 方位。ROS 中左转为正，图像右侧为正。"""
        return self._wrap_angle(self._camera_yaw - image_angle)

    def _compute_command(self, wp: Optional[Tuple[float, float, float]]
                         ) -> Tuple[float, float]:
        """waypoint 跟踪：转向对准它，速度按它的距离在 stop/full 之间线性过渡。
        无 waypoint（没有足够宽的安全通道）→ 零速。"""
        if wp is None:
            return 0.0, 0.0
        _, col, gd = wp
        w = self._last_width
        target_angle = self._image_to_base_angle(self._col_to_angle(col, w))
        angular = float(np.clip(self._k_angular * target_angle,
                                -self._max_angular, self._max_angular))
        frac = np.clip((gd - self._stop_depth)
                       / max(self._full_speed_depth - self._stop_depth, 0.1), 0.0, 1.0)
        # waypoint 在大侧角时先原地对准，避免一边猛冲一边大转向。
        heading_speed = max(0.0, cos(target_angle))
        if abs(target_angle) >= self._turn_in_place_angle:
            heading_speed = 0.0
        return float(self._cruise_linear * frac * heading_speed), angular

    def _on_cmd_timer(self):
        now = time.monotonic()
        stale = (self._last_result is None
                 or now - self._last_result > self._watchdog)
        lin, ang = (0.0, 0.0) if stale else self._cmd
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self._cmd_pub.publish(msg)

    def publish_stop(self):
        """停机兜底：发布一次零速（节点退出前调用，避免底盘保持旧指令）。"""
        if self._cmd_pub is not None:
            try:
                self._cmd_pub.publish(Twist())
            except Exception:
                pass

    # ----------------------------------------------------------- 可视化

    def _draw_overlay(self, rgb: np.ndarray, ground_mask: np.ndarray,
                      fused: np.ndarray, obstacle: np.ndarray,
                      wp: Optional[Tuple[float, float, float]]) -> np.ndarray:
        """叠加图：绿=最终允许区域，深绿=被深度安全扣除的地面，红=近距障碍，
        黄圈+线=选中的 waypoint。ASCII 标注，避免热路径里的 PIL 中文渲染。"""
        out = rgb.copy()
        carved = (ground_mask > 0) & ~fused
        if carved.any():
            out[carved] = (0.45 * out[carved] + 0.55 * np.array([40, 90, 40])).astype(np.uint8)
        if fused.any():
            out[fused] = (0.45 * out[fused] + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
        if obstacle.any():
            out[obstacle] = (0.45 * out[obstacle] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        h, w = out.shape[:2]
        if wp is not None:
            r, c, gd = int(wp[0]), int(wp[1]), wp[2]
            cv2.circle(out, (c, r), 10, (255, 220, 0), 2)
            cv2.line(out, (w // 2, h - 1), (c, r), (255, 220, 0), 2)
            angle_deg = np.degrees(self._col_to_angle(c, w))
            cv2.putText(out, f"wp {angle_deg:+.0f}deg {gd:.1f}m", (c + 14, r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        else:
            cv2.putText(out, "NO SAFE WAYPOINT", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out


def main(args=None) -> int:
    rclpy.init(args=args)
    node = Dinov3NavNode()
    executor = MultiThreadedExecutor(num_threads=2)
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

# dinov3_nav — ground_seg 的 ROS 2（Jazzy）封装

订阅前视相机 RGB + 对齐深度，跑仓库包 `ground_seg` 的可通行地面分割
（DINOv3 种子相似度 + SAM2 边界精修，可选楼梯候选头），与深度近距障碍
融合出**安全可通行区域**，结合 `/odometry` 的当前位置和 `/goal_pose` 的
全局目标，在其上选"最远且朝向目标的局部 waypoint"，直接输出 `/cmd_vel`
（无全局地图的视觉局部避障，CAT 风格）。**核心算法仍在
`ground_seg/`，本包只做 ROS IO + 深度融合 + 控制律**，命令行批处理流程
（`python ground_seg/main.py`）不受影响。

## 数据流

```
/zed/zed_node/rgb/image_rect_color ─┐
                                    ├─ 时间同步 ─► 允许通过区域 ─► 最远安全 waypoint ─► /cmd_vel
/zed/zed_node/depth/depth_registered┘   （融合）      （gate 搜索）

/odometry + /goal_pose ────────────────► 目标在 base_link 中的方位 ─┘

RGB   → ground_seg（DINOv3 多原型 + SAM2）→ 视觉地面证据
Depth → 免标定地面平面一致性 + 近距障碍
融合  → 视觉/几何互补地面 ∩ 无近距障碍 ∩ 从脚下连续可达

──► /dinov3_nav/ground_mask        (mono8)
──► /dinov3_nav/traversable_mask   (mono8)
──► /dinov3_nav/safe_mask          (mono8，最终允许区域)
──► /dinov3_nav/obstacle_mask      (mono8，近距障碍观测)
──► /dinov3_nav/overlay            (rgb8，绿安全区/红障碍/黄圈 waypoint)
──► /dinov3_nav/coverage           (Float32)
──► output/<时间戳>/frame_*_ground.png
```

话题、模型、深度安全、控制增益全部在 [config/params.yaml](config/params.yaml) 里改。

## 构建与运行

```bash
cd /home/dow/DOW/dinov3/dinov3_nav_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select dinov3_nav
source install/setup.bash

ros2 launch dinov3_nav dinov3_nav.launch.py    # 或 ros2 run dinov3_nav dinov3_nav_node
```

**解释器说明**：节点可执行文件的 `#!` 指向仓库 `.venv` 的 python（torch
cu128 只在 venv 里，系统 python3 没有）；rclpy 由启动器按 `ROS_DISTRO`
从 `/opt/ros` 补进 `sys.path`，所以**必须先 source ROS** 再启动。不进
colcon 也可以直接 `./scripts/dinov3_nav_node` 跑（参数同 `ros2 run`）。

## 无 ZED 时联调

`image_replay` 会成对发布 RGB + 合成深度（地面深度斜坡，底 0.5m→顶 6m），
时间戳相同保证同步；`--obs x0,y0,x1,y1` 可在指定矩形注入 0.3m 近障碍测转向：

```bash
ros2 run dinov3_nav image_replay /home/dow/DOW/dinov3/test_images/galileo_test_2.jpg
# 注入画面中部近障碍（测避障转向）：
ros2 run dinov3_nav image_replay img.jpg --obs 500,80,800,350
ros2 topic echo /cmd_vel
ros2 run rqt_image_view rqt_image_view    # 看 /dinov3_nav/overlay（黄圈=waypoint）
```

## 输出留存

每次启动在包目录 `output/` 下按时间标签新建文件夹，把推理的绿色叠加图
逐帧存成 `frame_<帧号>_ground.png`：

```
output/20260920_153045/frame_000001_ground.png
```

`output.dir` 可改绝对路径，`output.save_every` 控制每几帧存一张（全存约
每分钟 100~200MB，长跑建议调大）。

## 安全可通行区域与 waypoint（`dinov3_nav/node.py`）

最终允许通过区域**不等于**"DINOv3 判为地面"，每帧按四条约束取交集：

```
允许区域 = DINOv3 地面 ∩ 与机器人底部连通 ∩ 深度上无近距障碍 ∩ 满足机器人安全宽度
```

1. **视觉—几何互补地面**：DINOv3 多原型 + SAM2 给出高召回视觉候选；以
   底部可靠候选拟合 `1/depth = a·row+b` 地面平面。贴合平面但 DINO 相似度
   略低的像素会被召回（阴影、远处、透视变化），明显比平面更近的视觉误检
   会被剔除（障碍物）。这不是 RGB/深度的简单交集。
2. **深度无近障碍**：`depth < safe_distance` 判障碍后从地面里扣除。**只在上部
   (1-bottom_exempt_frac) 行判定**——前视相机底部行的地面本身深度就小，
   全图判会把脚下删光；近距竖直物在上部行可见，后续的逐像素走廊净空
   同样保护底部走廊，也避免贴着桌腿/墙走
3. **路径连通**：只保留与画面底部种子带（底 10% 行 × 中间 60% 列）8
   连通的区域，排除孤立远处地面。当前调试阶段不做机器人宽度、走廊净空
   或障碍膨胀约束，先验证地面召回、局部目标和转向；这不是上机安全配置。

**waypoint 选择**：所有合格 gate 按「越远越高(权重 1.0) + 居中偏好
(center_weight)」打分，取最优段的中心——即"视野内最远的可通行目标点，
优先靠近行进方向"。前方被堵时侧向 gate 自动胜出（= 在可通行区域内向
左/右搜索新的安全方向）。每帧重规划（~6-10Hz）。

**控制律**：`angular.z = -k_angular · waypoint 角度`（列按 hfov 换算）；
`linear = cruise_linear · 按 waypoint 距离在 stop_depth/full_speed_depth
之间线性过渡`。无合格 waypoint、覆盖率过低或 watchdog 超时 → 零速；
节点退出前也发一次零速。

## 面向全局目标的局部避障

节点订阅 `nav_msgs/Odometry` 的 `/odometry` 与 `geometry_msgs/PoseStamped` 的
`/goal_pose`。二者必须使用同一个世界坐标系（通常为 `odom`）：节点用当前
位置/航向将全局目标转成 base_link 方位角，再在每帧已连通的可通行图像区域
中，按“更远 + 与目标方位更一致 + 不贴边”选择 waypoint。

障碍物会先从局部候选中移除；因此目标方向不通时，最优点会落在仍可通行的
侧向区域，实现边绕障边向目标推进。局部点偏角超过
`control.turn_in_place_angle_deg` 时机器人只转不前进。进入
`global.goal_tolerance` 范围、没有 goal/odom、状态超时，或两个话题 frame_id
不一致时，节点输出零速度。

上车前建议：先 `control.enabled: false` 用 overlay 核对安全区域质量
（绿=允许、深绿=被深度扣除的地面、红=近障碍、黄圈=waypoint），再把
`cruise_linear` 从小往上调。Gazebo 无深度时 `depth.enable: false`
退化成纯 RGB。机器人不能爬梯的话把 `stairs.treat_traversable`
改为 false（楼梯头开启时）。

## 包结构

```
dinov3_nav/                                 # 位于 dinov3_nav_ws/src/
├── package.xml / setup.py                  # ament_python 包定义（data_files 装 lib/ 保留 venv shebang）
├── config/params.yaml                      # 话题 + 模型 + 控制 + 输出参数
├── launch/dinov3_nav.launch.py
├── dinov3_nav/node.py                      # 节点本体（ROS IO + 控制律 + 输出留存）
├── output/<时间戳>/frame_*_ground.png      # 运行时生成
└── scripts/
    ├── dinov3_nav_node                     # 启动器：venv 解释器 + 补 rclpy 路径
    └── image_replay                        # 静态图发布成相机流（调试工具）
```

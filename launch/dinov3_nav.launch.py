# -*- coding: utf-8 -*-
"""启动 dinov3_nav 节点并载入 config/params.yaml。

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    ros2 launch dinov3_nav dinov3_nav.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("dinov3_nav"), "config", "params.yaml")
    return LaunchDescription([
        Node(
            package="dinov3_nav",
            executable="dinov3_nav_node",
            name="dinov3_nav",
            output="screen",
            parameters=[params_file],
        ),
    ])

# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("dinov3_nav"), "config", "params.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="dinov3_nav",
                executable="dinov3_nav_node",
                name="dinov3_nav",
                output="screen",
                parameters=[params_file],
            )
        ]
    )

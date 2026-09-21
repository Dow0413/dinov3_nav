# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("dinov3_nav"), "config", "params.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "show_mppi_debug",
                default_value="true",
                description=(
                    "Start the C++ image_view window for /dinov3_nav/bev_debug."
                ),
            ),
            Node(
                package="dinov3_nav",
                executable="dinov3_nav_node",
                name="dinov3_nav",
                output="screen",
                parameters=[params_file],
            ),
            # image_view is a ROS C++ executable linked against the system
            # OpenCV HighGUI/Qt build. It keeps visualization out of the
            # DINO Python venv, whose cv2 wheel is intentionally headless.
            Node(
                package="image_view",
                executable="image_view",
                name="dinov3_nav_mppi_debug",
                remappings=[("image", "/dinov3_nav/bev_debug")],
                condition=IfCondition(LaunchConfiguration("show_mppi_debug")),
                output="screen",
            ),
        ]
    )

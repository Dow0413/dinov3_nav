from setuptools import setup

package_name = "dinov3_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    # 可执行文件不能用 scripts= / entry_points 安装：setuptools 会把含
    # "python" 的 shebang 重写为构建时的解释器（系统 python3，没有 torch）。
    # 改为 data_files 直接落到 lib/<pkg>（ros2 run/ros2 launch 的查找路径），
    # 原样保留各脚本指向仓库 .venv 的 #!。
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
        ("share/" + package_name + "/launch", ["launch/dinov3_nav.launch.py"]),
        ("lib/" + package_name, ["scripts/dinov3_nav_node", "scripts/image_replay"]),
    ],
    maintainer="dow",
    maintainer_email="dow@todo.todo",
    description="DINOv3 traversable-ground segmentation as a ROS 2 node with cmd_vel control.",
    license="MIT",
)

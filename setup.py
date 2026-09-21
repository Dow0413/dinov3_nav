"""Setuptools metadata for the ROS 2 ament_python package."""

from glob import glob

from setuptools import find_packages, setup

package_name = "dinov3_nav"

setup(
    name=package_name,
    version="0.4.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # The wrapper retains the project virtualenv interpreter, which is
        # required by the DINOv3/PyTorch runtime.
        ("lib/" + package_name, ["scripts/dinov3_nav_node", "scripts/image_replay"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="dow",
    maintainer_email="dow@todo.todo",
    description="DINOv3 RGB-D traversability navigation with a footprint-safe MPPI local planner.",
    license="MIT",
    tests_require=["pytest"],
)

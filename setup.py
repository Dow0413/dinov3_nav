from setuptools import setup

package_name = "dinov3_nav"

setup(
    name=package_name,
    version="0.3.0",
    packages=[package_name],
    # Keep the project venv shebang of scripts/dinov3_nav_node unchanged.
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
        ("share/" + package_name + "/launch", ["launch/dinov3_nav.launch.py"]),
        ("lib/" + package_name, ["scripts/dinov3_nav_node", "scripts/image_replay"]),
    ],
    maintainer="dow",
    maintainer_email="dow@todo.todo",
    description="DINOv3 RGB-D capability-inspired traversability navigation with footprint-aware local planning.",
    license="MIT",
)

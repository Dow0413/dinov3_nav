# dinov3_nav/footprint.py

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

import cv2
import numpy as np

from .bev import BEVGrid


@dataclass
class FootprintResult:
    safe: np.ndarray
    blocked: np.ndarray
    inflated_blocked: np.ndarray
    radius_m: float


def robot_circumscribed_radius(
    length: float,
    width: float,
    safety_margin: float,
) -> float:
    """
    用矩形机器狗的外接圆作为 conservative footprint。
    """

    body_radius = hypot(length * 0.5, width * 0.5)

    return body_radius + safety_margin


def inflate_binary(
    mask: np.ndarray,
    radius_m: float,
    resolution: float,
) -> np.ndarray:

    radius_cells = max(
        1,
        int(ceil(radius_m / resolution)),
    )

    size = 2 * radius_cells + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    inflated = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return inflated > 0


def build_body_safe_space(
    bev: BEVGrid,
    robot_length: float,
    robot_width: float,
    safety_margin: float,
    traversability_threshold: float,
) -> FootprintResult:

    # 已观测但不可通行，也看成障碍。
    non_traversable = (
        bev.observed
        & (bev.traversability < traversability_threshold)
    )

    blocked = bev.obstacle | non_traversable

    radius = robot_circumscribed_radius(
        robot_length,
        robot_width,
        safety_margin,
    )

    inflated = inflate_binary(
        blocked,
        radius_m=radius,
        resolution=bev.cfg.resolution,
    )

    safe = (
        bev.observed
        & (bev.traversability >= traversability_threshold)
        & ~inflated
    )

    return FootprintResult(
        safe=safe,
        blocked=blocked,
        inflated_blocked=inflated,
        radius_m=radius,
    )
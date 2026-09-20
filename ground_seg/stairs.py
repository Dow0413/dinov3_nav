# -*- coding: utf-8 -*-
"""Zero-shot stairs detection with CLIP ViT-B-32 (OpenAI weights, offline).

Why a stairs head at all: the ground segmenter already keeps stairs OUT of
the traversable mask (riser shadows + perspective make them dissimilar to
flat floor), but that is an appearance mechanism, not recognition — the
robot cannot tell "stairs I must not climb" from "wall" in the non-ground
remainder, and nothing guarantees a shallow same-material landing stays
excluded. This head recognizes stairs explicitly and
:func:`subtract_from_ground` enforces the exclusion as a safety net.

Mechanism is DoorDetector's, wholesale: multi-scale sliding window over the
inference-resolution RGB, CLIP softmax mass on a stair-prompt ensemble,
geometry filtering — only the prompts and geometry defaults differ
(no aspect constraint: staircases come in all shapes; windows up to 224 px
so a nearby flight is seen as a whole; larger min area: stairs are big).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .doors import DEFAULT_CLIP_CKPT, DoorBox, DoorDetector, DoorResult
from .features import FeatureExtractor

# Tuned 2026-09-18 on 3 stair photos + 7 controls (see README stairs section):
# indoor-only negatives let outdoor grid textures (plaza/road) peak at 0.68;
# the outdoor + office-grid blocks below pulled every control under 0.42 while
# "a short flight of steps"/"interior stairs" lifted the mid-size indoor flight
# from 0.50 to 0.54 — hence threshold 0.48 splits both gaps by ~0.06.
DEFAULT_STAIRS_PROMPTS = [
    "a staircase", "stairs", "a flight of stairs",
    "a stairway", "steps", "a stone staircase",
    "a short flight of steps", "interior stairs",
]
DEFAULT_STAIRS_NEG_PROMPTS = [
    "a floor", "a flat floor", "a carpet", "a wall", "a doorway",
    "a ramp", "a ladder", "a railing", "a person", "a box",
    "a window", "a bookshelf", "a shelf", "a ceiling",
    # outdoor distractors: plaza/road grid textures look stair-like to CLIP
    "a road", "a sidewalk", "a paved plaza", "a building facade",
    "grass", "a tree", "a sculpture", "a fence",
    # office distractors: glass partition grids / empty shelving
    "a glass partition", "a bookcase", "a cabinet", "a window frame",
]

# DoorResult/DoorBox already serve as generic region-detection results
# (semseg.door_result_from_semseg does the same); aliases keep the API readable.
StairsResult = DoorResult
StairsBox = DoorBox


class StairsDetector(DoorDetector):
    """Sliding-window CLIP zero-shot stairs detection (see module docstring)."""

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        ckpt: str = DEFAULT_CLIP_CKPT,
        threshold: float = 0.48,
        stride: int = 48,
        scales: Sequence[int] = (96, 128, 160, 224),
        batch_size: int = 64,
        device: str = "auto",
        positive_prompts: Sequence[str] = DEFAULT_STAIRS_PROMPTS,
        negative_prompts: Sequence[str] = DEFAULT_STAIRS_NEG_PROMPTS,
        *,
        min_area_frac: float = 0.005,
        aspect_min: float = 0.0,          # no aspect prior: stairs vary
        min_bottom_frac: float = 0.35,    # stairs rise from the lower scene
        extractor: Optional[FeatureExtractor] = None,
    ):
        super().__init__(
            model_name=model_name, ckpt=ckpt, threshold=threshold,
            stride=stride, scales=scales, batch_size=batch_size, device=device,
            positive_prompts=positive_prompts, negative_prompts=negative_prompts,
            target_name="stair",
            min_area_frac=min_area_frac, aspect_min=aspect_min,
            min_bottom_frac=min_bottom_frac, extractor=extractor,
        )


def heat_stats(heat: np.ndarray) -> str:
    """One-line heatmap stats for threshold tuning (peak vs noise floor)."""
    return (f"max={heat.max():.3f} p99={np.percentile(heat, 99):.3f} "
            f"p95={np.percentile(heat, 95):.3f}")


def subtract_from_ground(
    ground_mask: np.ndarray, stairs_mask: np.ndarray, dilate_frac: float = 0.02
) -> Tuple[np.ndarray, float, int]:
    """Safety net: remove a dilated stairs region from the ground mask.

    The dilation band (~2% of image height) absorbs the independent
    upsampling boundary errors of the two heads. Returns
    (new_mask, new_coverage, removed_px); ground_mask itself is not mutated.
    """
    new = ground_mask.copy()
    if stairs_mask is None or not stairs_mask.any():
        return new, float((new > 0).mean()), 0
    k = max(3, int(round(dilate_frac * ground_mask.shape[0])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    danger = cv2.dilate((stairs_mask > 0).astype(np.uint8), kernel)
    removed = int(((new > 0) & (danger > 0)).sum())
    new[danger > 0] = 0
    return new, float((new > 0).mean()), removed

# -*- coding: utf-8 -*-
"""SAM2 boundary refinement for a coarse DINOv3 traversability mask.

DINOv3 decides *which* connected region is similar to the bottom ground
seeds; SAM2 decides *where its pixel boundary is*.  SAM2 is promptable rather
than semantic, so the generated positive/negative points remain anchored to
the DINO proposal and cannot silently turn into an unrelated object mask.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .features import PreparedImage

SAM2_REPO = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
DEFAULT_SAM2_CKPT = SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@dataclass
class SAM2Refinement:
    mask: np.ndarray
    score: float
    iou_with_coarse: float
    elapsed_s: float
    warnings: List[str] = field(default_factory=list)


class SAM2MaskRefiner:
    """Refine a binary DINO proposal with SAM2.1-Hiera-Tiny.

    The tiny checkpoint is sufficient for crisp object/plane boundaries and
    makes this optional stage practical on CPU.  The class loads lazily so a
    pure-DINO workflow remains dependency-free at import time.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_SAM2_CKPT,
        config: str = DEFAULT_SAM2_CONFIG,
        device: str = "auto",
        positive_points: int = 6,
        negative_points: int = 4,
    ):
        self.checkpoint = Path(checkpoint)
        self.config = config
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self.positive_points = positive_points
        self.negative_points = negative_points
        self._predictor = None

    def _ensure_model(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint}. "
                "Download sam2.1_hiera_tiny.pt or disable SAM2 refinement."
            )
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError("SAM2 is not installed; install third_party/sam2 first") from e
        self._predictor = SAM2ImagePredictor(
            build_sam2(self.config, str(self.checkpoint), device=self.device)
        )
        return self._predictor

    @staticmethod
    def _spread_points(mask: np.ndarray, count: int) -> np.ndarray:
        """Greedily choose interior pixels far apart from each other/boundary."""
        if count <= 0 or not mask.any():
            return np.empty((0, 2), np.float32)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        pts = []
        for _ in range(count):
            y, x = np.unravel_index(dist.argmax(), dist.shape)
            if dist[y, x] < 2:
                break
            pts.append((x, y))
            cv2.circle(dist, (int(x), int(y)), max(8, int(dist[y, x] * 1.4)), 0, -1)
        return np.asarray(pts, np.float32).reshape(-1, 2)

    def _prompts(self, coarse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fg = coarse > 0
        pos = self._spread_points(fg, self.positive_points)
        # Negative prompts only come from a narrow exterior ring.  Far-away
        # pixels might be another valid floor region and would over-constrain
        # SAM2; a ring teaches just the DINO proposal's local boundary.
        k = max(9, int(round(min(fg.shape) * 0.025))) | 1
        outer = cv2.dilate(fg.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ring = outer & ~fg
        neg = self._spread_points(ring, self.negative_points)
        points = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([
            np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)
        ])
        return points, labels

    def refine_prepared(self, prepared: PreparedImage, coarse_mask: np.ndarray) -> SAM2Refinement:
        t0 = time.perf_counter()
        coarse = (coarse_mask > 0).astype(np.uint8)
        if not coarse.any():
            return SAM2Refinement(coarse_mask.copy(), 0.0, 1.0, 0.0, ["empty DINO proposal"])
        points, labels = self._prompts(coarse)
        if not len(points):
            return SAM2Refinement(coarse_mask.copy(), 0.0, 1.0, 0.0, ["no valid SAM2 prompts"])

        predictor = self._ensure_model()
        rgb = np.asarray(prepared.pil)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
        ):
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=True
            )

        # SAM's confidence alone favours a visually salient sub-object.  Rank
        # candidates jointly by SAM confidence and agreement with DINO's
        # semantic proposal, rejecting implausibly tiny/huge masks.
        best, best_key = coarse, (-1.0, -1.0)
        for mask, score in zip(masks, scores):
            m = mask.astype(bool)
            area = m.mean()
            inter = np.logical_and(m, coarse).sum()
            union = np.logical_or(m, coarse).sum()
            iou = float(inter / max(union, 1))
            key = (0.65 * iou + 0.35 * float(score), float(score))
            if 0.02 <= area <= 0.92 and key > best_key:
                best, best_key = m.astype(np.uint8), key
        inter = np.logical_and(best > 0, coarse > 0).sum()
        union = np.logical_or(best > 0, coarse > 0).sum()
        return SAM2Refinement(
            (best.astype(np.uint8) * 255), float(best_key[1]),
            float(inter / max(union, 1)), time.perf_counter() - t0,
        )

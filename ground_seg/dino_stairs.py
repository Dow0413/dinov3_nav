# -*- coding: utf-8 -*-
"""A training-free stairs-candidate detector using only DINOv3 patch features.

This is deliberately a *structural* head, rather than pretending that an
ImageNet-pretrained backbone is a semantic stairs classifier.  A flight of
stairs produces several horizontally coherent changes between adjacent patch
rows (the risers).  We find those repeated changes in the final DINOv3 feature
map and turn their enclosing connected regions into a stairs-candidate mask.

It is useful as a first local/offline baseline and needs no model beyond the
ViT-S/B checkpoint already used by :class:`GroundSegmenter`.  For a production
semantic ``stairs`` label, use the ADE20K adapter once its matching ViT-7B
backbone checkpoint is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .features import FeatureExtractor, PreparedImage, get_extractor


@dataclass
class StairsCandidate:
    """Output of the DINO-only structural stairs detector.

    ``score_map`` is a normalized evidence map, not a calibrated probability.
    """

    mask: np.ndarray                    # original resolution uint8 {0, 255}
    score_map: np.ndarray               # original resolution float32 [0, 1]
    boxes: List[Tuple[int, int, int, int]]
    threshold: float
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class DINOStairsDetector:
    """Detect repeated horizontal riser structure in frozen DINOv3 features."""

    def __init__(
        self,
        resolution: int = 448,
        threshold: float = 0.45,
        min_area_frac: float = 0.008,
        max_area_frac: float = 0.35,
        min_risers: int = 3,
        device: str = "auto",
        extractor: Optional[FeatureExtractor] = None,
    ):
        self.resolution = resolution
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_risers = min_risers
        self._extractor = extractor or get_extractor(
            resolution=resolution, device=device
        )

    def detect(self, image) -> StairsCandidate:
        t0 = time.perf_counter()
        result = self.detect_prepared(self._extractor.prepare(image, self.resolution))
        result.elapsed_s = time.perf_counter() - t0
        return result

    def detect_prepared(self, prepared: PreparedImage) -> StairsCandidate:
        t0 = time.perf_counter()
        feats, h, w = self._extractor.features(prepared)
        grid = feats.reshape(h, w, -1)

        # Adjacent-row cosine distance: strong at a riser.  Normalising by a
        # robust percentile makes the threshold portable across DINO variants.
        vertical = 1.0 - (grid[:-1] * grid[1:]).sum(axis=-1)
        vertical = np.maximum(vertical, 0.0)
        p95 = float(np.percentile(vertical, 95))
        evidence = np.zeros((h, w), np.float32)
        if p95 > 1e-6:
            evidence[1:] = np.clip(vertical / p95, 0.0, 1.0)

        # A riser should extend sideways.  Horizontal opening rejects isolated
        # texture changes; vertical dilation joins multiple risers into a
        # single flight candidate.
        line = (evidence >= self.threshold).astype(np.uint8)
        line = cv2.morphologyEx(
            line, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 5), 1)),
        )
        support = cv2.dilate(
            line, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 6), 3))
        )
        support = cv2.morphologyEx(
            support, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
        )

        keep = np.zeros((h, w), np.uint8)
        boxes_patch: List[Tuple[int, int, int, int]] = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        min_area = max(4, int(round(self.min_area_frac * h * w)))
        max_area = int(round(self.max_area_frac * h * w))
        for label in range(1, n):
            x, y, bw, bh, area = stats[label]
            component_lines = line[labels == label]
            # The key distinction from a single shelf/curb: multiple separate
            # horizontal rows of evidence must occur in the same component.
            n_risers = int(np.count_nonzero(line[y:y + bh].any(axis=1)))
            if area < min_area or area > max_area or n_risers < self.min_risers:
                continue
            # Stairs seen by a forward camera belong below the upper third.
            if y + bh < h // 3:
                continue
            keep[labels == label] = 1
            boxes_patch.append((int(x), int(y), int(bw), int(bh)))

        # DINO's 16px patch grid is intentionally coarse.  Refine/add a
        # candidate only when the original image also contains a *stack* of
        # nearly horizontal edge segments.  This catches close stair flights
        # whose individual risers are clearer in pixels than in patch tokens.
        geometric, geometric_boxes = self._riser_geometry(prepared.rgb)
        if geometric.any():
            geometric_patch = cv2.resize(geometric, (w, h), interpolation=cv2.INTER_NEAREST)
            keep = cv2.bitwise_or(keep, geometric_patch)
            sx_patch, sy_patch = w / prepared.rgb.shape[1], h / prepared.rgb.shape[0]
            boxes_patch.extend([
                (int(round(x * sx_patch)), int(round(y * sy_patch)),
                 int(round(bw * sx_patch)), int(round(bh * sy_patch)))
                for x, y, bw, bh in geometric_boxes
            ])

        hr, wr = prepared.size
        w0, h0 = prepared.orig_size
        mask_r = cv2.resize(keep, (wr, hr), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask_r, (w0, h0), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        score_r = cv2.resize(evidence, (wr, hr), interpolation=cv2.INTER_LINEAR)
        score = cv2.resize(score_r, (w0, h0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        sx, sy = w0 / w, h0 / h
        boxes = [
            (int(round(x * sx)), int(round(y * sy)),
             int(round(bw * sx)), int(round(bh * sy)))
            for x, y, bw, bh in boxes_patch
        ]
        # Dedupe boxes created by the DINO and geometric paths.
        boxes = self._dedupe_boxes(boxes)
        warnings = []
        if not boxes:
            warnings.append("no repeated horizontal DINO feature boundaries found")
        return StairsCandidate(mask, score, boxes, self.threshold, warnings,
                               time.perf_counter() - t0)

    @staticmethod
    def _dedupe_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        result = []
        for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            x, y, w, h = box
            if any(
                (max(x, xx) < min(x + w, xx + ww) and max(y, yy) < min(y + h, yy + hh))
                for xx, yy, ww, hh in result
            ):
                continue
            result.append(box)
        return result

    @staticmethod
    def _riser_geometry(rgb: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Return regions supported by three or more horizontal image edges.

        Hough segments are never used individually: shelves, road markings and
        handrails often make one horizontal line, whereas a stair flight has a
        vertically stacked family of them in the same x-range.
        """
        H, W = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 130, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=max(20, W // 18),
            minLineLength=max(32, W // 8), maxLineGap=max(8, W // 35),
        )
        evidence = np.zeros((H, W), np.uint8)
        if lines is None:
            return evidence, []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = int(x2 - x1), int(y2 - y1)
            if abs(dx) < max(16, W // 12) or abs(dy) > 0.16 * abs(dx):
                continue
            cv2.line(evidence, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)

        # Join adjacent risers but do not bridge widely separated structures.
        support = cv2.dilate(
            evidence, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, W // 20), max(5, H // 28)))
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        mask = np.zeros_like(evidence)
        boxes = []
        for label in range(1, n):
            x, y, bw, bh, area = stats[label]
            # Count distinct original-edge rows, not dilated rows.
            rows = int(np.count_nonzero(evidence[y:y + bh, x:x + bw].any(axis=1)))
            if (rows < 3 or bw < 0.22 * W or bh < 0.08 * H or
                    area > 0.42 * H * W or y + bh < 0.30 * H):
                continue
            component = (labels == label).astype(np.uint8)
            # Filling only the component's envelope makes the output a usable
            # region mask rather than a set of thin edge strokes.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, W // 35), max(3, H // 40)))
            component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask[component > 0] = 1
            boxes.append((int(x), int(y), int(bw), int(bh)))
        return mask, boxes

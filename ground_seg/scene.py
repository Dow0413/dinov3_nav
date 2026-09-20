# -*- coding: utf-8 -*-
"""Full-scene unsupervised segmentation via K-means on DINOv3 patch features.

Produces a region map of the whole image (not only ground): every pixel gets a
region id.  Regions are *unnamed* — K-means clusters carry no semantics; the
semantic labels (ground / door) come from GroundSegmenter and DoorDetector.

Pipeline: patch features -> K-means (fixed seed) -> one-hot bilinear upsample
+ argmax (hole-free full-res cluster map) -> median smoothing -> per-cluster
connected components -> regions with bbox / area / palette color.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .features import DEFAULT_CKPT, DINOV3_REPO, FeatureExtractor, ImageLike, PreparedImage, get_extractor

# Tableau-ish palette, RGB.
PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (0, 128, 128), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (154, 99, 36), (255, 152, 175), (128, 128, 128),
]


@dataclass
class SceneRegion:
    id: int                        # dense region id (0..n-1)
    cluster: int                   # K-means cluster this region came from
    area_frac: float               # region area / image area, original resolution
    bbox: Tuple[int, int, int, int]  # x, y, w, h at original resolution
    color: Tuple[int, int, int]    # RGB palette color (indexed by cluster)


@dataclass
class SceneResult:
    labels: np.ndarray       # (H0, W0) int32 dense region ids, hole-free
    clusters: np.ndarray     # (H0, W0) int32 K-means cluster id per pixel
    regions: List[SceneRegion]
    k: int
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)


class SceneSegmenter:
    """Segment the whole image into feature-coherent regions (no semantics)."""

    MIN_AREA_FRAC = 0.001  # drop components smaller than 0.1% of the image

    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        k: int = 8,
        spatial_weight: float = 0.0,
        *,
        resolution: int = 448,
        device: str = "auto",
        repo_dir: str = DINOV3_REPO,
        ckpt: str = DEFAULT_CKPT,
    ):
        if extractor is None:
            extractor = get_extractor(
                resolution=resolution, device=device, repo_dir=repo_dir, ckpt=ckpt
            )
        self._extractor = extractor
        self.k = k
        self.spatial_weight = spatial_weight

    def prepare(self, image: ImageLike) -> PreparedImage:
        return self._extractor.prepare(image)

    def segment(self, image: ImageLike) -> SceneResult:
        t_start = time.perf_counter()
        result = self.segment_prepared(self.prepare(image))
        result.elapsed_s = time.perf_counter() - t_start
        return result

    def segment_prepared(self, prepared: PreparedImage) -> SceneResult:
        warnings: List[str] = []
        t0 = time.perf_counter()
        feats, h, w = self._extractor.features(prepared)
        k = min(self.k, h * w)
        if k < self.k:
            warnings.append(f"k clamped to {k} (only {h*w} patches)")

        data = feats
        if self.spatial_weight > 0:
            # Optional compactness prior: append scaled (row, col) coordinates.
            rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
            span = max(h, w)
            coords = np.stack(
                [rows.ravel() / span, cols.ravel() / span], axis=1
            ) * self.spatial_weight
            data = np.concatenate([feats, coords], axis=1)

        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        patch_labels = km.fit_predict(data).astype(np.int32).reshape(h, w)

        # One-hot bilinear upsample + argmax -> hole-free full-res cluster map.
        w0, h0 = prepared.orig_size
        onehot = np.zeros((h, w, k), np.float32)
        onehot[np.arange(h)[:, None], np.arange(w)[None, :], patch_labels] = 1.0
        onehot_t = torch.from_numpy(onehot).permute(2, 0, 1)[None]  # [1,k,h,w]
        up = F.interpolate(onehot_t, size=(h0, w0), mode="bilinear", align_corners=False)
        clusters = up[0].argmax(0).numpy().astype(np.uint8)  # k <= 255
        clusters = cv2.medianBlur(clusters, ksize=5).astype(np.int32)

        # Split each cluster into connected regions; drop specks; dense relabel.
        regions: List[SceneRegion] = []
        labels = np.zeros((h0, w0), np.int32)
        min_area = self.MIN_AREA_FRAC * h0 * w0
        next_id = 0
        for cluster in range(k):
            n_comp, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
                (clusters == cluster).astype(np.uint8), connectivity=8
            )
            for comp in range(1, n_comp):
                if stats[comp, cv2.CC_STAT_AREA] < min_area:
                    continue
                cx, cy, cw, ch = stats[comp, :4]
                labels[cc_labels == comp] = next_id
                regions.append(
                    SceneRegion(
                        id=next_id,
                        cluster=cluster,
                        area_frac=stats[comp, cv2.CC_STAT_AREA] / (h0 * w0),
                        bbox=(int(cx), int(cy), int(cw), int(ch)),
                        color=PALETTE[cluster % len(PALETTE)],
                    )
                )
                next_id += 1
        if next_id == 0:
            warnings.append("no regions survived min-area filter; labels all 0")

        return SceneResult(
            labels=labels,
            clusters=clusters,
            regions=regions,
            k=k,
            elapsed_s=time.perf_counter() - t0,
            warnings=warnings,
        )

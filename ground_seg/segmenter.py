# -*- coding: utf-8 -*-
"""Traversable-ground segmentation from a single forward-facing camera image.

Zero-training pipeline built on frozen DINOv3 ViT patch features.地面识别与
可通行性判断分离：本模块只负责**第一阶段 ground_mask**（图像中所有可能
属于地面的区域，高召回，不做 2D 连通删除——障碍物后方的地面仍然保留）；
"机器人当前真正能走"由下游（ROS 节点）结合深度障碍/底部连通/gate 宽度
再判定。

    bottom-of-image seeds → K 个 ground prototype（近/远/明/暗各成一簇）
    → max-over-prototypes 余弦相似度 → Otsu 阶梯 → 高置信区域迭代扩展
    → (可选) 深度几何融合：删"比该行地面平面近"的墙/家具，捞回
      "特征漂移但深度贴平面"的阴影/远处地面 → 小碎块降噪 → 上采样

设计要点：

- 单一 prototype 覆盖不了近处/远处/阴影地面的特征漂移，种子聚成多簇后
  取逐像素最大相似度；扩展靠迭代更新 prototype（上限 expand_iters，
  只吃 thr+margin 以上的高置信像素，种子永远锚定在池里，防错误扩散）。
- 深度只参与第一阶段的互补融合（见 depth_geometry.py）：语义删几何上
  不可能是地面的，几何救语义上漏掉的——不是简单取交集。
- `require_seed_connectivity=True` 可回退到旧行为（种子 8 连通域过滤），
  但那会把被柱子隔断的远处地面整块删掉，默认关闭。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .depth_geometry import (
    RowPlaneModel,
    classify,
    fit_row_plane,
    fuse,
    patch_depth_stats,
)
from .features import (  # noqa: F401  (re-exported for callers/tests)
    DEFAULT_CKPT,
    DINOV3_REPO,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
    FeatureExtractor,
    ImageLike,
    PreparedImage,
    get_extractor,
)

# 原型池上限（种子 + 高置信新 patch）：超出均匀下采样，控制 KMeans 规模
_MAX_PROTO_POOL = 384


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


@dataclass
class GroundMaskResult:
    mask: np.ndarray            # (H0, W0) uint8 {0, 255}, original resolution
    similarity: np.ndarray      # (H0, W0) float32, cosine similarity (upsampled)
    patch_mask: np.ndarray      # (h, w) bool, patch-grid mask
    threshold_used: Optional[float]
    coverage: float             # fraction of image pixels masked (0..1)
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    n_prototypes: int = 1       # 本帧实际使用的 ground prototype 数
    depth_used: bool = False    # 深度是否成功参与了第一阶段融合


class GroundSegmenter:
    """Segment the visible ground (stage-1, high-recall) in a forward-facing frame."""

    def __init__(
        self,
        resolution: int = 448,
        seed_rows_frac: float = 0.10,
        seed_cols_frac: float = 0.60,
        threshold: Union[str, float] = "otsu",
        device: str = "auto",
        repo_dir: str = DINOV3_REPO,
        ckpt: str = DEFAULT_CKPT,
        extractor: Optional[FeatureExtractor] = None,
        n_prototypes: int = 3,
        expand_iters: int = 3,
        expand_margin: float = 0.05,
        rescue_margin: float = 0.05,
        geo_tol_near: float = 0.15,
        geo_tol_on: float = 0.15,
        require_seed_connectivity: bool = False,
    ):
        self.resolution = resolution
        self.seed_rows_frac = seed_rows_frac
        self.seed_cols_frac = seed_cols_frac
        self.threshold = threshold
        self.n_prototypes = max(1, int(n_prototypes))
        self.expand_iters = max(1, int(expand_iters))
        self.expand_margin = float(expand_margin)
        self.rescue_margin = float(rescue_margin)
        self.geo_tol_near = float(geo_tol_near)
        self.geo_tol_on = float(geo_tol_on)
        self.require_seed_connectivity = bool(require_seed_connectivity)
        if extractor is None:
            extractor = get_extractor(
                resolution=resolution, device=device, repo_dir=repo_dir, ckpt=ckpt
            )
        self._extractor = extractor
        # Mirror extractor state for backward compatibility.
        self.repo_dir = extractor.repo_dir
        self.ckpt = extractor.ckpt
        self.device = extractor.device

    # --------------------------------------------------- delegated to extractor

    def _ensure_model(self) -> None:
        self._extractor._ensure_model()

    def _load_pil(self, image: ImageLike) -> Image.Image:
        return self._extractor._load_pil(image)

    def _preprocess(self, pil: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        return self._extractor._preprocess(pil, resolution=self.resolution)

    def _features(self, x: torch.Tensor) -> Tuple[np.ndarray, int, int]:
        return self._extractor._features(x)

    def prepare(self, image: ImageLike) -> PreparedImage:
        return self._extractor.prepare(image, resolution=self.resolution)

    # ------------------------------------------------------------------ seeds

    def _seed_window(self, h: int, w: int) -> Tuple[range, range]:
        rows = range(h - max(1, round(self.seed_rows_frac * h)), h)
        col_margin = round((1.0 - self.seed_cols_frac) / 2.0 * w)
        cols = range(max(0, col_margin), max(1, w - col_margin))
        return rows, cols

    def _seed_vector(self, feats: np.ndarray, h: int, w: int) -> np.ndarray:
        """旧的单 prototype 路径（trimmed mean），n_prototypes=1 时使用。"""
        rows, cols = self._seed_window(h, w)
        seeds = feats[[r * w + c for r in rows for c in cols]]  # [N, D]
        mean = seeds.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-8
        sims = seeds @ mean
        keep = sims >= np.quantile(sims, 0.20)  # drop most-dissimilar 20%
        seed_vec = seeds[keep].mean(axis=0)
        return seed_vec / (np.linalg.norm(seed_vec) + 1e-8)

    def _seed_prototypes(self, feats: np.ndarray, h: int, w: int) -> np.ndarray:
        """种子特征聚成 [K, D] 多 prototype：近/远/明/暗地面在 DINO 特征空间
        有漂移，单一平均向量覆盖不住。成员过少的小簇视为离群丢弃。"""
        rows, cols = self._seed_window(h, w)
        seeds = feats[[r * w + c for r in rows for c in cols]]  # [N, D]
        k = min(self.n_prototypes, len(seeds))
        if k <= 1 or len(seeds) < 3 * k:
            return self._seed_vector(feats, h, w)[None, :]
        from sklearn.cluster import KMeans  # 惰性导入：与 scene.py 同一用法

        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(seeds)
        protos = []
        for cid in range(k):
            members = seeds[labels == cid]
            if len(members) < max(2, round(0.10 * len(seeds))):
                continue  # 离群小簇：脚部 / 线缆 / 暗角
            v = members.mean(axis=0)
            protos.append(v / (np.linalg.norm(v) + 1e-8))
        if not protos:
            return self._seed_vector(feats, h, w)[None, :]
        return np.stack(protos)

    def _update_prototypes(
        self,
        feats: np.ndarray,
        fused: np.ndarray,
        sim: np.ndarray,
        thr_used: float,
        h: int,
        w: int,
        on_plane: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """迭代扩展：把 mask 内高置信 patch（sim ≥ thr+margin，或贴地面平面
        的救援 patch）并入原型池重新聚类。种子永远完整保留（锚定，防漂移）；
        池超上限均匀下采样（固定种子，结果可复现）。"""
        rows, cols = self._seed_window(h, w)
        seed_idx = np.asarray([r * w + c for r in rows for c in cols], dtype=np.int64)
        conf = fused & (sim >= thr_used + self.expand_margin)
        if on_plane is not None:
            conf = conf | (on_plane & (sim >= thr_used - self.rescue_margin))
        conf_idx = np.flatnonzero(conf.ravel())
        max_extra = _MAX_PROTO_POOL - len(seed_idx)
        if max_extra <= 0:
            pool = feats[seed_idx]
        else:
            if len(conf_idx) > max_extra:
                rng = np.random.default_rng(0)
                conf_idx = np.sort(rng.choice(conf_idx, size=max_extra, replace=False))
            pool = feats[np.concatenate([seed_idx, conf_idx])]

        k = min(self.n_prototypes, len(pool))
        if k <= 1:
            v = pool.mean(axis=0)
            return (v / (np.linalg.norm(v) + 1e-8))[None, :]
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(pool)
        protos = []
        for cid in range(k):
            members = pool[labels == cid]
            if len(members) < max(2, round(0.05 * len(pool))):
                continue
            v = members.mean(axis=0)
            protos.append(v / (np.linalg.norm(v) + 1e-8))
        if not protos:
            v = pool.mean(axis=0)
            protos = [v / (np.linalg.norm(v) + 1e-8)]
        return np.stack(protos)

    # ------------------------------------------------------------- similarity

    @staticmethod
    def _similarity_map(
        feats: np.ndarray, protos: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """逐 patch 对多 prototype 的最大余弦相似度（特征已 L2 归一化）。"""
        sims = feats @ protos.T               # [h*w, K]
        sim = sims.max(axis=1).reshape(h, w).astype(np.float32)
        return cv2.GaussianBlur(sim, (3, 3), 0)

    # -------------------------------------------------- binarize + connectivity

    def _threshold_ladder(self, sim: np.ndarray, warnings: List[str]) -> List[float]:
        if isinstance(self.threshold, str):
            if self.threshold != "otsu":
                raise ValueError(f"threshold must be 'otsu' or a float, got {self.threshold!r}")
            u8 = (np.clip(sim, 0.0, 1.0) * 255.0).astype(np.uint8)
            t_otsu = float(cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]) / 255.0
            return [t_otsu, t_otsu - 0.05, t_otsu - 0.10]
        return [float(self.threshold)]

    def _semantic_mask(self, sim: np.ndarray, thr: float, h: int, w: int
                       ) -> Tuple[np.ndarray, float]:
        """sim ≥ thr → patch mask（不做种子连通删除——阶段 1 允许含障碍物
        后方的地面），随后降噪清理。`require_seed_connectivity=True` 时回退
        旧的种子连通域过滤。返回 (mask, coverage)。"""
        fg = sim >= thr
        if self.require_seed_connectivity:
            fg = self._seed_components(fg, h, w)
        return self._cleanup(fg, h, w)

    def _seed_components(self, fg: np.ndarray, h: int, w: int) -> np.ndarray:
        """旧行为：只保留含种子 patch 的 8 连通域（会把断连的远处地面删掉）。"""
        n, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
        rows, cols = self._seed_window(h, w)
        seed_labels = {int(labels[r, c]) for r in rows for c in cols}
        seed_labels.discard(0)  # 0 = background
        if not seed_labels:
            return np.zeros((h, w), bool)
        return np.isin(labels, list(seed_labels))

    def _cleanup(self, mask: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, float]:
        """降噪（非断连删除）：3×3 开运算 + 去掉 <0.5% patch 面积的小碎块。"""
        m = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        ).astype(bool)
        min_area = max(4, round(0.005 * h * w))
        n2, labels2, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8
        )
        keep = {i for i in range(1, n2) if stats[i, cv2.CC_STAT_AREA] >= min_area}
        if keep:
            m = np.isin(labels2, list(keep))
        return m, float(m.mean())

    def _binarize(self, sim: np.ndarray, warnings: List[str], h: int, w: int
                  ) -> Tuple[np.ndarray, Optional[float]]:
        """Otsu 阶梯 + 覆盖率规则（与旧版一致）：覆盖率 <0.05 沿阶梯降档；
        >0.90 时收紧一档重试；全失败回底部种子条带。返回 (mask, thr_used)。"""
        accepted: Optional[Tuple[np.ndarray, float]] = None
        for thr in self._threshold_ladder(sim, warnings):
            patch_mask, coverage = self._semantic_mask(sim, thr, h, w)
            if coverage < 0.05:
                warnings.append(f"thr={thr:.3f}: coverage {coverage:.2f} < 0.05")
                continue
            if coverage > 0.90 and isinstance(self.threshold, str):
                # Otsu collapsed (dark / textureless floor) — retry tighter once.
                tighter = thr + 0.05
                pm2, cov2 = self._semantic_mask(sim, tighter, h, w)
                if 0.05 <= cov2 <= 0.90:
                    warnings.append(f"thr={thr:.3f}: coverage {coverage:.2f} > 0.90; "
                                    f"retry thr={tighter:.3f} accepted (coverage {cov2:.2f})")
                    patch_mask, thr = pm2, tighter
                else:
                    warnings.append(f"thr={thr:.3f}: similarity map near-degenerate "
                                    f"(coverage {coverage:.2f}); mask may over-extend")
            accepted = (patch_mask, thr)
            break

        if accepted is None:
            rows, cols = self._seed_window(h, w)
            patch_mask = np.zeros((h, w), bool)
            patch_mask[np.ix_(list(rows), list(cols))] = True
            warnings.append("no ground found at any threshold; "
                            "returning bottom-strip fallback")
            return patch_mask, None
        return accepted

    # ------------------------------------------------------------------ depth

    def _fit_plane(self, semantic: np.ndarray, z_patch: np.ndarray,
                   valid_patch: np.ndarray, h: int,
                   warnings: List[str]) -> Optional[RowPlaneModel]:
        """用第一轮语义 mask ∩ 底部 45% 行 ∩ 有效深度的 patch RANSAC 拟合
        地面距离-行号模型（只在迭代 0 拟合一次，不随后续漂移重拟合）。"""
        rows_grid = np.arange(h)[:, None].repeat(z_patch.shape[1], axis=1)
        cand = semantic & valid_patch & (rows_grid >= int(round(0.55 * h)))
        model = None
        if int(cand.sum()) >= 8:
            model = fit_row_plane(rows_grid[cand] + 0.5, z_patch[cand], h)
        if model is None:
            warnings.append("depth ground-plane fit failed; depth fusion skipped")
        return model

    # ---------------------------------------------------------------- pipeline

    def segment(self, image: ImageLike,
                depth: Optional[np.ndarray] = None) -> GroundMaskResult:
        t_start = time.perf_counter()
        prepared = self.prepare(image)
        result = self.segment_prepared(prepared, depth=depth)
        result.elapsed_s = time.perf_counter() - t_start  # include load+preprocess
        return result

    def segment_prepared(self, prepared: PreparedImage,
                         depth: Optional[np.ndarray] = None) -> GroundMaskResult:
        """第一阶段地面识别。depth: (H0, W0) float 米、与原图像素对齐
        （NaN/≤0 无效）；提供时参与互补融合（删几何误检 + 救特征漂移），
        缺失/形状不符时自动退回纯语义。DINOv3 前向只跑一次（feature_cache），
        迭代扩展只有 KMeans + 小矩阵乘的毫秒级开销。"""
        warnings: List[str] = []

        orig_size = prepared.orig_size  # (W0, H0)
        hr, wr = prepared.size
        feats, h, w = self._extractor.features(prepared)

        # ---- 深度 → patch 网格统计 ----
        z_patch = valid_patch = None
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            if d.ndim == 2 and d.shape == (orig_size[1], orig_size[0]):
                try:
                    z_patch, valid_patch = patch_depth_stats(d, hr, wr, h, w)
                except Exception as e:  # noqa: BLE001  单帧降级，不中断
                    warnings.append(f"depth stats failed: {e}")
            else:
                warnings.append(
                    f"depth shape {d.shape} != {(orig_size[1], orig_size[0])}, ignored")

        # ---- 迭代：多 prototype 语义 → 深度几何融合 ----
        protos = self._seed_prototypes(feats, h, w)
        sim = self._similarity_map(feats, protos, h, w)
        model: Optional[RowPlaneModel] = None
        above = on = None
        fused: Optional[np.ndarray] = None
        thr_used: Optional[float] = None
        prev: Optional[np.ndarray] = None
        for it in range(self.expand_iters):
            semantic, thr_used = self._binarize(sim, warnings, h, w)
            fused = semantic
            if z_patch is not None:
                if model is None:
                    model = self._fit_plane(semantic, z_patch, valid_patch, h, warnings)
                if model is not None:
                    above, on = classify(model, z_patch, valid_patch,
                                         self.geo_tol_near, self.geo_tol_on)
                    fused, _rescued = fuse(semantic, sim, thr_used,
                                           self.rescue_margin, above, on)
                    fused, _cov = self._cleanup(fused, h, w)  # 融合可能产生碎块
            if prev is not None and _iou(prev, fused) >= 0.98:
                break  # 收敛：再迭代 mask 基本不变
            prev = fused
            if it + 1 < self.expand_iters and thr_used is not None:
                protos = self._update_prototypes(
                    feats, fused, sim, thr_used, h, w,
                    on_plane=on if model is not None else None)
                sim = self._similarity_map(feats, protos, h, w)

        assert fused is not None  # 循环至少跑一次
        patch_mask = fused
        mask = self._upsample(patch_mask, hr, wr, orig_size)
        sim_full = cv2.resize(sim, orig_size, interpolation=cv2.INTER_LINEAR)
        final_coverage = float((mask > 0).mean())

        return GroundMaskResult(
            mask=mask,
            similarity=sim_full,
            patch_mask=patch_mask,
            threshold_used=thr_used,
            coverage=final_coverage,
            warnings=warnings,
            elapsed_s=0.0,  # caller (segment) fills in wall-clock time
            n_prototypes=int(len(protos)),
            depth_used=z_patch is not None,
        )

    def _upsample(
        self, patch_mask: np.ndarray, hr: int, wr: int, orig_size: Tuple[int, int]
    ) -> np.ndarray:
        m = torch.from_numpy(patch_mask.astype(np.float32))[None, None]
        m = F.interpolate(m, size=(hr, wr), mode="bilinear", align_corners=False)
        m = (m[0, 0].numpy() >= 0.5).astype(np.uint8)
        w0, h0 = orig_size
        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
        m = (m >= 0.5).astype(np.uint8)
        m = cv2.medianBlur(m, ksize=5)
        return (m * 255).astype(np.uint8)

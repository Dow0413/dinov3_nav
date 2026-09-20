# -*- coding: utf-8 -*-
"""深度几何一致性：免标定的"地面距离-行号"模型（纯 numpy/cv2，无 torch）。

前视相机看平地时，z-depth 只跟图像行号有关（小孔成像 + 地面平面）：

    z(r) = c / (r - r0)        等价  1/z = a·r + b   （a = 1/c > 0）

r0 是地平线行，c 吸收 fx·相机高度——**不需要相机内外参标定**。对平地上
采样出的 (行号, 深度) patch 做 RANSAC 线性拟合即可得到模型。

逐 patch 分类（z_pred 为该行的地面预测深度）：

    z < z_pred·(1-tol_near)   → 在地面之上（墙/家具/柱子——该行"太近"）
    |z - z_pred| ≤ tol_on     → 贴合地面平面（强地面几何证据，可用于召回）
    比 z_pred 远很多 / 无深度 → 中性（下坡、量测误差不惩罚）

与 DINOv3 语义的互补融合见 `fuse`：语义删"几何上不可能是地面"的，几何
捞回"特征漂移但深度在平面上"的——不是简单取交集。
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# 深度有效值范围（米）：ZED 32FC1 / Gazebo 16UC1 换算成米后统一在此过滤
_DEPTH_MIN_M = 0.05
_DEPTH_MAX_M = 60.0
# 一个 patch（16×16 深度采样）内有效像素占比低于此值 → 该 patch 深度无效
_PATCH_VALID_FRAC = 0.125


@dataclass
class RowPlaneModel:
    """1/z = a·r + b 的地面距离模型（r 为 patch 行号，从 0 起）。"""

    a: float
    b: float
    r0: float        # 地平线行号 = -b/a（r > r0 才可能是地面）
    n_inliers: int

    def predict(self, r):
        """行号（数组或标量）→ 预测地面 z-depth（米）。"""
        den = self.a * np.asarray(r, dtype=np.float64) + self.b
        return 1.0 / np.maximum(den, 1e-9)


# --------------------------------------------------------------- patch 统计

def patch_depth_stats(
    depth_full: np.ndarray, hr: int, wr: int, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """全分辨率深度 → patch 网格统计。

    depth_full: (H0, W0) float 米，NaN/越界视为无效。
    (hr, wr): 推理张量分辨率（= 16·patch 网格，精确整除）。
    返回 (z_patch[h, w] float32 米、NaN=无效, valid_patch[h, w] bool)。
    """
    z = np.asarray(depth_full, dtype=np.float32)
    finite = np.isfinite(z) & (z > _DEPTH_MIN_M) & (z < _DEPTH_MAX_M)
    z = np.where(finite, z, np.nan)
    # cv2.resize 不认 NaN：分别送数值图和有效标志（最近邻，保持像素对齐）
    zr = cv2.resize(np.nan_to_num(z, nan=0.0), (wr, hr), interpolation=cv2.INTER_NEAREST)
    fr = cv2.resize(finite.astype(np.float32), (wr, hr),
                    interpolation=cv2.INTER_NEAREST) > 0.5
    zr = np.where(fr, zr, np.nan)

    bh, bw = hr // h, wr // w          # 每 patch 的深度采样数（ViT/16 时 = 16）
    blocks = zr.reshape(h, bh, w, bw)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)   # 全 NaN patch 的中位数
        zmed = np.nanmedian(blocks, axis=(1, 3)).astype(np.float32)
        cnt = np.isfinite(blocks).sum(axis=(1, 3))
    valid = cnt >= max(1, round(_PATCH_VALID_FRAC * bh * bw))
    return np.where(valid, zmed, np.nan), valid


# ------------------------------------------------------------------- 拟合

def fit_row_plane(
    rows: np.ndarray,
    z: np.ndarray,
    grid_h: int,
    *,
    n_trials: int = 256,
    rel_tol: float = 0.15,
    min_points: int = 8,
    seed: int = 0,
) -> Optional[RowPlaneModel]:
    """对平地采样点 (patch 行号, 深度) RANSAC 拟合 1/z = a·r + b。

    约束：a > 0（越往下越近）；地平线 r0 = -b/a 必须在数据行之上
    （所有地面点都在地平线以下），且不在图像上方过远处。向量化批解
    两点采样，内点最小二乘重拟合。失败返回 None（调用方退回纯语义）。
    """
    rows = np.asarray(rows, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z) & (z > _DEPTH_MIN_M)
    rows, z = rows[ok], z[ok]
    n = len(rows)
    if n < min_points:
        return None
    iz = 1.0 / z

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_trials)
    j = rng.integers(0, n, n_trials)
    dr = rows[j] - rows[i]
    good = np.abs(dr) >= 1.0                       # 行号太近的两点定不出斜率
    i, j, dr = i[good], j[good], dr[good]
    if len(i) == 0:
        return None

    a = (iz[j] - iz[i]) / dr
    b = iz[i] - a * rows[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = -b / a
    valid = (a > 1e-4) & np.isfinite(a) & np.isfinite(b) & np.isfinite(r0) \
        & (r0 < rows.min()) & (r0 > -grid_h)       # 地平线在数据之上、图内附近
    if not valid.any():
        return None

    preds = a[:, None] * rows[None, :] + b[:, None]
    rel = np.abs(iz[None, :] - preds) / iz[None, :]
    inl = rel < rel_tol
    inl[~valid] = False
    counts = inl.sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] < max(4, min_points // 2):
        return None

    sel = inl[best]
    A = np.stack([rows[sel], np.ones(int(sel.sum()))], axis=1)
    coef, *_ = np.linalg.lstsq(A, iz[sel], rcond=None)
    a2, b2 = float(coef[0]), float(coef[1])
    r02 = -b2 / a2 if a2 > 1e-6 else np.inf
    if not (a2 > 1e-6 and r02 < rows.min() and r02 > -grid_h):
        a2, b2, r02 = float(a[best]), float(b[best]), float(r0[best])  # 离散解兜底

    recount = int((np.abs(iz - (a2 * rows + b2)) / iz < rel_tol).sum())
    return RowPlaneModel(a=a2, b=b2, r0=float(r02), n_inliers=recount)


# ----------------------------------------------------------------- 分类/融合

def classify(
    model: RowPlaneModel,
    z_patch: np.ndarray,
    valid_patch: np.ndarray,
    tol_near: float,
    tol_on: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐 patch 判 (above, on)：above=在地面之上（近于该行平面预测），
    on=贴合地面平面。地平线以上（模型无定义）与无效深度一律中性。"""
    h, w = z_patch.shape
    r = (np.arange(h, dtype=np.float64) + 0.5)[:, None]   # patch 中心行
    den = model.a * r + model.b
    below_horizon = den > 1e-6
    zp = 1.0 / np.where(below_horizon, den, 1.0)
    near = below_horizon & np.asarray(valid_patch, bool) & np.isfinite(z_patch)
    above = near & (z_patch < zp * (1.0 - tol_near))
    on = near & (np.abs(z_patch - zp) <= zp * tol_on)
    return above.astype(bool), on.astype(bool)


def fuse(
    semantic: np.ndarray,
    sim: np.ndarray,
    thr: Optional[float],
    rescue_margin: float,
    above: np.ndarray,
    on: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """语义 ∩ 几何 的互补融合（非纯交集）：

    ground = (语义 ∧ ¬above_plane) ∨ (sim ≥ thr - rescue_margin ∧ on_plane)

    第一项：深度压制"几何上不可能是地面"的误检（墙/家具/柱子）；
    第二项：深度捞回特征漂移到阈值之下的阴影/远处地面（召回救援）。
    返回 (fused, rescued)，rescued 仅供调试/可视化。
    """
    fused = np.asarray(semantic, bool) & ~above
    if thr is not None:
        rescued = (~fused) & on & (sim >= thr - rescue_margin)
    else:
        rescued = np.zeros_like(fused)
    return fused | rescued, rescued

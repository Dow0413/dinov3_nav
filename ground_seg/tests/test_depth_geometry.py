# -*- coding: utf-8 -*-
"""depth_geometry 离线单测（纯 numpy，不加载模型/GPU）。

运行：.venv/bin/python ground_seg/tests/test_depth_geometry.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ground_seg.depth_geometry import (  # noqa: E402
    RowPlaneModel, classify, fit_row_plane, fuse, patch_depth_stats,
)

H0, W0 = 720, 1280          # 原图（RGB/深度同尺寸）
HR, WR = 448, 768           # 推理张量分辨率（16 的倍数）
H, W = HR // 16, WR // 16   # patch 网格 28×48


def true_depth(row0=10.0, c=60.0):
    """张量尺度合成平地深度：tensor 行 t 处 z = c/(t − 16·row0)，再最近邻放大
    到原图尺寸——patch_depth_stats 先缩到 (HR,WR) 再分块，这样采样精确可期。"""
    import cv2
    t = np.arange(HR, dtype=np.float32)[:, None]
    den = t - row0 * 16.0          # den>1 保证 z < c 落进有效深度范围
    zt = np.where(den > 1.0, c / np.maximum(den, 1.0), np.nan).repeat(WR, axis=1)
    return cv2.resize(zt, (W0, H0), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS {name} {detail}")
        else:
            fail += 1
            print(f"  FAIL {name} {detail}")

    # ---- 1. patch_depth_stats：中位数下采样 + NaN 无效 ----
    # 1a 中位数精确性：直接喂张量尺寸（resize 恒等），期望 = 块中心行深度
    t = np.arange(HR, dtype=np.float32)[:, None]
    zt = np.where(t - 160.0 > 1.0, 60.0 / np.maximum(t - 160.0, 1.0),
                  np.nan).repeat(WR, axis=1)
    zp, valid = patch_depth_stats(zt, HR, WR, H, W)
    # 全有效块的中位数 = 第 8、9 个深度值的均值（z 随行单调）
    ts = np.arange(H, dtype=np.float64) * 16.0
    expect = 0.5 * (60.0 / (ts + 7.0 - 160.0) + 60.0 / (ts + 8.0 - 160.0))
    check("统计-中位数正确", np.allclose(zp[11:], expect[11:, None], rtol=1e-4),
          f"max_err={np.nanmax(np.abs(zp[11:] - expect[11:, None])):.5f}")

    # 1b 有效性 / NaN：走原图尺寸路径（含 nearest 缩放）
    z = true_depth()
    zp, valid = patch_depth_stats(z, HR, WR, H, W)
    check("统计-地平线下有效", valid[10:].all() and not valid[:10].any(),
          f"first_valid_row={int(np.argmax(valid)) // W}")
    z_nan = z.copy()
    z_nan[:200] = np.nan                      # 顶部大片无效
    zp2, valid2 = patch_depth_stats(z_nan, HR, WR, H, W)
    check("统计-顶部NaN无效", (~valid2[: H // 4]).all() and valid2[H // 2:].all())
    check("统计-无效为NaN", np.isnan(zp2[~valid2]).all())

    # ---- 2. RANSAC 平面拟合：噪声 + 墙离群点下恢复 r0/c ----
    rng = np.random.default_rng(7)
    rows = np.tile(np.arange(14, H, dtype=np.float64) + 0.5, 4)   # 地平线 r0=10 以下
    zc = 60.0 / (rows - 10.0) * (1.0 + rng.normal(0, 0.03, rows.shape))
    wall_rows = rows[: len(rows) // 5]
    wall_z = 0.5 * (60.0 / (wall_rows - 10.0))            # 墙：深度只有平面一半
    m = fit_row_plane(np.concatenate([rows, wall_rows]),
                      np.concatenate([zc, wall_z]), H)
    check("拟合-成功", m is not None)
    if m is not None:
        check("拟合-r0接近真值", abs(m.r0 - 10.0) < 0.5, f"r0={m.r0:.2f} (true 10)")
        c_fit = 1.0 / m.a
        check("拟合-c接近真值", abs(c_fit - 60.0) / 60.0 < 0.05,
              f"c={c_fit:.1f} (true 60)")
        check("拟合-墙被剔除", m.n_inliers <= len(zc) + 5,
              f"inliers={m.n_inliers} ground={len(zc)}")

    # ---- 3. 拟合失败路径 ----
    check("拟合-点太少None", fit_row_plane(np.array([5.0, 6.0]), np.array([2.0, 1.5]), H) is None)
    bad = fit_row_plane(np.arange(30, dtype=np.float64), np.full(30, 3.0), H)  # 常深度=无斜率
    check("拟合-常深度None", bad is None)

    # ---- 4. classify 真值表：above / on / 中性 ----
    model = RowPlaneModel(a=1.0 / 60.0, b=-10.0 / 60.0, r0=10.0, n_inliers=99)
    zg = np.full((H, W), np.nan, np.float32)
    r_c = (np.arange(H, dtype=np.float32) + 0.5)
    zrow = 60.0 / (r_c - 10.0)
    zg[:, :] = zrow[:, None]                                  # 全平面
    zg[20, 10] = zrow[20] * 0.5                               # 墙：太近
    zg[21, 10] = zrow[21] * 1.02                              # 平面附近
    zg[22, 10] = zrow[22] * 3.0                               # 太远（下坡/坏点）→ 中性
    zg[5, 5] = 2.0                                            # 地平线以上 → 中性
    valid = np.isfinite(zg)
    above, on = classify(model, zg, valid, tol_near=0.15, tol_on=0.15)
    check("分类-墙above", above[20, 10] and not on[20, 10])
    check("分类-平面on", on[21, 10] and not above[21, 10])
    check("分类-普通地面on", on[H // 2, W // 2] and not above[H // 2, W // 2])
    check("分类-太远中性", not above[22, 10] and not on[22, 10])
    check("分类-地平线上中性", not above[5, 5] and not on[5, 5])

    # ---- 5. fuse 互补融合真值表 ----
    sem = np.zeros((H, W), bool)
    sem[H // 2, W // 2] = True          # 普通语义地面 → 保留
    sem[20, 10] = True                  # 语义在墙上 → 被 above 删除
    sim = np.zeros((H, W), np.float32)
    sim[21, 10] = 0.47                  # 阴影地面：sim 略低于阈值但 on-plane
    thr, margin = 0.50, 0.05
    fused, rescued = fuse(sem, sim, thr, margin, above, on)
    check("融合-语义保留", fused[H // 2, W // 2])
    check("融合-墙被删", not fused[20, 10])
    check("融合-阴影被救援", fused[21, 10] and rescued[21, 10])
    sim[22, 10] = 0.47                  # sim 达 rescue 线但 on=False（太远中性）→ 不救援
    check("融合-非平面不救援", not fuse(sem, sim, thr, margin, above, on)[0][22, 10])
    fused_none, _ = fuse(sem, sim, None, margin, above, on)   # 无阈值（fallback）→ 只删不救
    check("融合-无thr不救援", not fused_none[21, 10] and fused_none[H // 2, W // 2])

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

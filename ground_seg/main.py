#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一入口：识别可通行地面与楼梯区域（仅用本地 DINOv3 权重）。

改下面的配置，然后（在仓库根目录）python ground_seg/main.py

每张图的结果存到 OUTPUT/<照片名>/ 文件夹（STAIRS=True 时 4 个文件）：
    <stem>_ground.png       地面（绿色）与楼梯候选（橙色）叠加图
    <stem>_ground_mask.png  DINO 地面二值掩码
    <stem>_stairs_mask.png  DINO 楼梯候选二值掩码
    <stem>_traversable_mask.png  地面 + 楼梯的可通行区域二值掩码
    <stem>_stairheat.jpg    楼梯结构证据热图（调试）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ============================ 在这里改配置 ============================
# 模型权重（变体从文件名自动推断，ViT-S/B/L 通用）：
CKPT = "dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#   快 3 倍的小模型： "dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# 输入图片：一个目录，或单张图，或多张的列表
IMAGES = "test_images"
# IMAGES = ["test_images/galileo_test_2.jpg"]

OUTPUT = "outputs"       # 输出根目录（每张图一个子文件夹）
RESOLUTION = 448         # 推理分辨率；地面检测 448 已足够（896 慢一倍，边缘略细）
DEVICE = "auto"          # auto / cuda / cpu

# 楼梯检测（只使用与地面头相同的 DINOv3 patch 特征；无额外模型/联网）：
# 楼梯语义不能从无标注 DINO 特征可靠地产生，先默认关闭候选头；开启时仅供调试。
STAIRS = False
STAIRS_THRESHOLD = 0.45  # 相邻 patch 行的 DINO 特征突变阈值（归一化证据）
STAIRS_MIN_AREA = 0.008
STAIRS_MAX_AREA = 0.35   # 过滤道路/整面栅格等覆盖半幅以上的结构误报
STAIRS_MIN_RISERS = 3

# SAM2 边界精修：DINOv3 产出语义候选，SAM2 只按候选生成的点提示贴合像素边界。
SAM2_REFINE = True
SAM2_CKPT = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"

# 地面检测调参（一般不用动）：
SEED_ROWS = 0.10         # 底部种子行占比（前视相机先验：地面在画面底部）
SEED_COLS = 0.60         # 种子列居中占比
THRESHOLD = "otsu"       # 'otsu' 或固定余弦阈值如 0.55

# 多 prototype / 迭代扩展 / 深度几何（第一阶段地面识别）：
N_PROTOTYPES = 3         # 种子聚成几簇 prototype（近/远/明/暗）；1 = 旧单向量
EXPAND_ITERS = 3         # 高置信区域迭代扩展轮数；1 = 不扩展
EXPAND_MARGIN = 0.05     # 并入原型池需 sim ≥ thr + 此余量（防错误扩散）
RESCUE_MARGIN = 0.05     # 深度贴平面的阴影地面救援带宽（thr − 此值）
GEO_TOL_NEAR = 0.15      # 比该行地面预测近 15% 以上 → 墙/家具，删除
GEO_TOL_ON = 0.15        # 与该行地面预测偏差 ≤15% → 贴平面，可救援
SEED_CONNECTIVITY = False  # True 回退旧行为（种子连通域过滤，会删断连远处地面）
USE_DEPTH = True         # 存在 <stem>_depth.npy（与原图同尺寸 float32 米）时
                         # 让深度参与地面识别；没有则自动纯 RGB
# =====================================================================

# 本文件在 <root>/ground_seg/ 内：把 <root> 放上 sys.path 才能 import ground_seg
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ground_seg import GroundSegmenter  # noqa: E402
from ground_seg.stairs import heat_stats  # noqa: E402
from ground_seg.visualize import ground_overlay, heatmap  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_inputs() -> list[Path]:
    if isinstance(IMAGES, (list, tuple)):
        return [Path(p) for p in IMAGES]
    p = Path(IMAGES)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
    return [p]


def build_stairs_detector(extractor):
    """None 表示配置关闭；该头不需要 CLIP 或网络。"""
    if not STAIRS:
        return None
    try:
        from ground_seg import DINOStairsDetector

        return DINOStairsDetector(
            resolution=RESOLUTION, threshold=STAIRS_THRESHOLD,
            min_area_frac=STAIRS_MIN_AREA, max_area_frac=STAIRS_MAX_AREA,
            min_risers=STAIRS_MIN_RISERS,
            device=DEVICE, extractor=extractor,
        )
    except Exception as e:
        print(f"warning: 楼梯检测不可用，已降级为纯地面流程（{e}）")
        return None


def build_sam2_refiner():
    if not SAM2_REFINE:
        return None
    try:
        from ground_seg import SAM2MaskRefiner

        return SAM2MaskRefiner(checkpoint=SAM2_CKPT, device=DEVICE)
    except Exception as e:
        print(f"warning: SAM2 不可用，保留原始 DINO 地面掩码（{e}）")
        return None


def load_depth(image_path: Path, prepared, warnings: list):
    """加载与原图同尺寸的 <stem>_depth.npy（float32 米，NaN/≤0 无效）。
    没有 sidecar 文件时返回 None（自动退回纯 RGB）。"""
    if not USE_DEPTH:
        return None
    dpath = image_path.with_name(image_path.stem + "_depth.npy")
    if not dpath.exists():
        warnings.append("no depth sidecar, RGB-only")
        return None
    depth = np.load(dpath).astype(np.float32)
    h0, w0 = prepared.pil.size[::-1]
    if depth.ndim != 2 or depth.shape != (h0, w0):
        warnings.append(f"depth {depth.shape} != image {(h0, w0)}, ignored")
        return None
    return depth


def main() -> int:
    files = collect_inputs()
    if not files:
        print(f"error: no input images found for {IMAGES!r}", file=sys.stderr)
        return 1

    gseg = GroundSegmenter(
        resolution=RESOLUTION, ckpt=CKPT, device=DEVICE,
        seed_rows_frac=SEED_ROWS, seed_cols_frac=SEED_COLS, threshold=THRESHOLD,
        n_prototypes=N_PROTOTYPES, expand_iters=EXPAND_ITERS,
        expand_margin=EXPAND_MARGIN, rescue_margin=RESCUE_MARGIN,
        geo_tol_near=GEO_TOL_NEAR, geo_tol_on=GEO_TOL_ON,
        require_seed_connectivity=SEED_CONNECTIVITY,
    )
    sdet = build_stairs_detector(gseg._extractor)
    sam2 = build_sam2_refiner()

    print(f"model={Path(CKPT).name}  resolution={RESOLUTION}  sam2={sam2 is not None}  stairs={sdet is not None}  "
          f"images={len(files)}  output={OUTPUT}/")
    for f in files:
        warnings: list[str] = []
        t0 = time.perf_counter()

        prepared = gseg.prepare(f)              # 两个头共用 DINOv3 特征/模型
        depth = load_depth(f, prepared, warnings)
        ground = gseg.segment_prepared(prepared, depth=depth)
        dino_ground_mask = ground.mask.copy()
        if sam2 is not None:
            try:
                refined = sam2.refine_prepared(prepared, ground.mask)
                ground.mask = refined.mask
                ground.coverage = float((ground.mask > 0).mean())
                if refined.iou_with_coarse < 0.35:
                    warnings.append(
                        f"SAM2 与 DINO 候选一致性低（IoU={refined.iou_with_coarse:.2f}），已保留 DINO 掩码"
                    )
                    ground.mask = dino_ground_mask
                    ground.coverage = float((ground.mask > 0).mean())
            except Exception as e:
                warnings.append(f"SAM2 精修失败，已回退 DINO 掩码：{e}")

        sres = None
        if sdet is not None:
            sres = sdet.detect_prepared(prepared)
        traversable = ground.mask.copy()
        if sres is not None:
            # 当前需求把楼梯也视为可通行。若机器人不能爬梯，只需下游使用
            # ground_mask，或将这里的 bitwise_or 改成保持 ground.mask。
            traversable = cv2.bitwise_or(traversable, sres.mask)

        out_dir = Path(OUTPUT) / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{f.stem}_dino_mask.png"), dino_ground_mask)
        cv2.imwrite(str(out_dir / f"{f.stem}_ground_mask.png"), ground.mask)
        cv2.imwrite(str(out_dir / f"{f.stem}_traversable_mask.png"), traversable)
        cv2.imwrite(str(out_dir / f"{f.stem}_ground.png"),
                    cv2.cvtColor(ground_overlay(np.asarray(prepared.pil), ground,
                                                stairs=sres.mask if sres else None),
                                 cv2.COLOR_RGB2BGR))
        if sres is not None:
            cv2.imwrite(str(out_dir / f"{f.stem}_stairs_mask.png"), sres.mask)
            cv2.imwrite(str(out_dir / f"{f.stem}_stairheat.jpg"),
                        cv2.cvtColor(heatmap(sres.score_map), cv2.COLOR_RGB2BGR))

        warnings += ground.warnings
        if sres is not None:
            warnings += sres.warnings
        thr = f"{ground.threshold_used:.3f}" if ground.threshold_used is not None else "fallback"
        stairs_part = (f"stairs={len(sres.boxes)} evidence[{heat_stats(sres.score_map)}]  "
                       if sres else "")
        warn = f"  warnings={warnings}" if warnings else ""
        print(f"{f.name}  coverage={ground.coverage:.2f}  thr={thr}  "
              f"protos={ground.n_prototypes}  depth={ground.depth_used}  {stairs_part}"
              f"t={time.perf_counter() - t0:.2f}s{warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

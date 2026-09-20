#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI: full-scene segmentation + traversable-ground + door detection.

Usage:
    python scripts/extract_scene.py INPUT [INPUT...] -o outputs \
        [--k 8] [--spatial-weight 0.0] [--doors auto|on|off] [--door-thr 0.75] \
        [--scales 96,128,160] [--stride 48] [--door-min-area 0.0005] \
        [--resolution 896] [--seed-rows 0.10] [--seed-cols 0.60] \
        [--threshold otsu|0.55] [--device auto|cuda|cpu]

INPUT: image path(s) or directory (jpg/jpeg/png, case-insensitive).
One DINOv3 forward is shared by all three heads. Per image prints:
    name coverage= thr= k= regions= doors= scores=[...] t=...s warnings=[...]
and writes a per-image folder <out>/<stem>/ containing <stem>_segments.png
(semantic labels) / _ground.png / _door.png / _nav.png / _mask.png
(+ _doorheat.jpg when door detection ran).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Defensive: huggingface.co is unreachable on this network; fail fast instead
# of hanging if the CLIP stack ever tries the hub (only local files are used).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

# Allow running from anywhere: put the ground_seg package's parent on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ground_seg import GroundSegmenter, SceneSegmenter  # noqa: E402
from ground_seg.features import DEFAULT_CKPT  # noqa: E402
from ground_seg.visualize import save_scene_all  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: skipping missing input {item}", file=sys.stderr)
    return files


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="image path(s) or directory")
    ap.add_argument("-o", "--output", default="outputs", help="output directory")
    ap.add_argument("--k", type=int, default=8, help="K-means region count")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="DINOv3 checkpoint path; variant (ViT-S/B/L/...) inferred from filename")
    ap.add_argument("--spatial-weight", type=float, default=0.0,
                    help="compactness prior weight (0 = appearance only)")
    ap.add_argument("--doors", default="auto", choices=["auto", "on", "off"],
                    help="door detection: auto = run, degrade if open_clip/weights missing")
    ap.add_argument("--door-thr", type=float, default=0.75,
                    help="door-ness threshold (uniform-prompt baseline ~0.375)")
    ap.add_argument("--scales", default="96,128,160",
                    help="comma-separated sliding-window sizes, small to large")
    ap.add_argument("--stride", type=int, default=48, help="sliding-window stride")
    ap.add_argument("--door-min-area", type=float, default=0.0005,
                    help="min door box area as a fraction of the image "
                         "(raise to ~0.02 to drop wall-panel false positives)")
    ap.add_argument("--resolution", type=int, default=896, help="shorter-side inference resolution")
    ap.add_argument("--seed-rows", type=float, default=0.10, help="bottom fraction of patch rows used as seeds")
    ap.add_argument("--seed-cols", type=float, default=0.60, help="central fraction of patch cols used as seeds")
    ap.add_argument("--threshold", default="otsu", help="'otsu' or a fixed cosine threshold like 0.55")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args()


def make_door_detector(args, warnings: list[str], extractor=None):
    """--doors on/off/auto handling. Returns DoorDetector or None."""
    if args.doors == "off":
        return None
    try:
        from ground_seg.doors import DoorDetector
        scales = tuple(int(s) for s in str(args.scales).split(",") if s.strip())
        det = DoorDetector(
            threshold=args.door_thr, stride=args.stride, scales=scales,
            min_area_frac=args.door_min_area,
            device=args.device, extractor=extractor,
        )
        det._ensure_model()  # fail fast: import + weights + text prototypes
        return det
    except Exception as e:  # ImportError or weight download failure
        if args.doors == "on":
            raise
        warnings.append(f"door detection disabled: {e}")
        return None


def main() -> int:
    args = parse_args()
    files = collect_inputs(args.inputs)
    if not files:
        print("error: no input images found", file=sys.stderr)
        return 1

    try:
        args.threshold = float(args.threshold)
    except ValueError:
        if args.threshold != "otsu":
            print(f"error: --threshold must be 'otsu' or a float, got {args.threshold!r}", file=sys.stderr)
            return 1

    setup_warnings: list[str] = []
    gseg = GroundSegmenter(
        resolution=args.resolution,
        seed_rows_frac=args.seed_rows,
        seed_cols_frac=args.seed_cols,
        threshold=args.threshold,
        ckpt=args.ckpt,
        device=args.device,
    )
    sseg = SceneSegmenter(
        extractor=gseg._extractor, k=args.k, spatial_weight=args.spatial_weight
    )
    ddet = make_door_detector(args, setup_warnings, extractor=gseg._extractor)
    for w in setup_warnings:
        print(f"warning: {w}", file=sys.stderr)

    for f in files:
        warnings: list[str] = []
        t0 = time.perf_counter()

        prepared = gseg.prepare(f)          # one resize + one DINOv3 forward,
        ground = gseg.segment_prepared(prepared)   # shared by all three heads
        scene = sseg.segment_prepared(prepared)
        doors = ddet.detect_prepared(prepared) if ddet is not None else None

        img_rgb = np.asarray(prepared.pil)
        # Per-image output folder keeps each photo's results self-contained.
        save_scene_all(ground, scene, doors, img_rgb, Path(args.output) / f.stem, f.stem)

        warnings += ground.warnings + scene.warnings + (doors.warnings if doors else [])
        thr = f"{ground.threshold_used:.3f}" if ground.threshold_used is not None else "fallback"
        scores = [round(b.score, 2) for b in doors.boxes] if doors else []
        n_doors = len(scores) if doors else "off"
        warn = f" warnings={warnings}" if warnings else ""
        print(
            f"{f.name}  coverage={ground.coverage:.2f}  thr={thr}  k={scene.k}  "
            f"regions={len(scene.regions)}  doors={n_doors}  scores={scores}  "
            f"t={time.perf_counter() - t0:.2f}s{warn}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

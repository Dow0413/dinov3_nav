#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI: extract traversable-ground masks from images.

Usage:
    python scripts/extract_ground.py INPUT [INPUT...] -o outputs \
        [--resolution 448] [--seed-rows 0.10] [--seed-cols 0.60] \
        [--threshold otsu|0.55] [--device auto|cuda|cpu]

INPUT: image path(s) or directory (jpg/jpeg/png, case-insensitive).
Per image prints: name coverage= thr= t= warnings=[...] and writes
<stem>_mask.png, <stem>_overlay.png, <stem>_simmap.jpg into -o.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running from anywhere: put the ground_seg package's parent on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ground_seg import GroundSegmenter  # noqa: E402
from ground_seg.visualize import save_all  # noqa: E402

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
    ap.add_argument("--resolution", type=int, default=448, help="shorter-side inference resolution")
    ap.add_argument("--seed-rows", type=float, default=0.10, help="bottom fraction of patch rows used as seeds")
    ap.add_argument("--seed-cols", type=float, default=0.60, help="central fraction of patch cols used as seeds")
    ap.add_argument("--threshold", default="otsu", help="'otsu' or a fixed cosine threshold like 0.55")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args()


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

    segmenter = GroundSegmenter(
        resolution=args.resolution,
        seed_rows_frac=args.seed_rows,
        seed_cols_frac=args.seed_cols,
        threshold=args.threshold,
        device=args.device,
    )

    for f in files:
        result = segmenter.segment(f)
        img_rgb = np.asarray(Image.open(f).convert("RGB"))
        save_all(result, img_rgb, args.output, f.stem)
        warn = f" warnings={result.warnings}" if result.warnings else ""
        thr = f"{result.threshold_used:.3f}" if result.threshold_used is not None else "fallback"
        print(
            f"{f.name}  coverage={result.coverage:.2f}  thr={thr}  "
            f"t={result.elapsed_s:.2f}s{warn}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

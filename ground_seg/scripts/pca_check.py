#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sanity check: PCA of DINOv3 patch features — ground cluster should be
visually distinct from walls/sky. Usage: scripts/pca_check.py IMAGE [OUT.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ground_seg import GroundSegmenter  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    image = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "pca_check.png"

    seg = GroundSegmenter()
    seg._ensure_model()
    pil = seg._load_pil(image)
    x, (hr, wr) = seg._preprocess(pil)
    feats, h, w = seg._features(x)

    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(feats)  # [h*w, 3]

    # Robust per-channel normalization (2-98 percentile).
    vis = np.zeros_like(proj)
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [2, 98])
        vis[:, c] = np.clip((proj[:, c] - lo) / (hi - lo + 1e-8), 0, 1)
    vis = (vis.reshape(h, w, 3) * 255).astype(np.uint8)

    vis_full = cv2.resize(vis, (wr, hr), interpolation=cv2.INTER_NEAREST)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(out, cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {out}")
    print(f"explained variance ratios: {pca.explained_variance_ratio_.round(3)}")
    print(f"saved {out}  ({h}x{w} patches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

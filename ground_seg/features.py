# -*- coding: utf-8 -*-
"""Shared DINOv3 feature extraction (single model per process).

`FeatureExtractor` owns everything common to the segmenters: model loading,
image loading/preprocessing and patch-feature extraction.  The model variant
(ViT-S/B/L/... 16) is inferred from the checkpoint filename — the constructor
must match the weights, a ViT-S skeleton cannot load ViT-B tensors.  `get_extractor()`
caches instances so GroundSegmenter / SceneSegmenter / DoorDetector share one
backbone on the GPU instead of one copy each.

`prepare()` returns a `PreparedImage` that keeps both the raw resized RGB
image (for CLIP crops — never feed CLIP the ImageNet-normalized tensor) and
the normalized tensor (for DINOv3), guaranteed pixel-aligned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# Local upstream checkout (read-only) and the only weights available offline.
DINOV3_REPO = "/home/dow/DOW/dinov3/dinov3"
DEFAULT_CKPT = (
    "/home/dow/DOW/dinov3/dinov3/checkpoints/"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def hub_entry_for(ckpt: str) -> str:
    """Map a DINOv3 checkpoint filename to its torch.hub entrypoint.

    e.g. .../dinov3_vitb16_pretrain_lvd1689m-*.pth -> "dinov3_vitb16".
    The hub entrypoint builds the architecture; it must match the weights
    (a dinov3_vits16 skeleton loading ViT-B tensors fails on every shape).
    """
    m = re.search(r"vit(?:7b|s|b|l|h)(?:16|8)(?:plus)?", Path(ckpt).name)
    if not m:
        raise ValueError(
            f"cannot infer model variant from checkpoint {ckpt!r}; expected a "
            "dinov3_vit{s|b|l|h|7b}{8|16}[plus]...pth filename"
        )
    return f"dinov3_{m.group(0)}"

PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class PreparedImage:
    pil: Image.Image            # EXIF-transposed RGB original (display / orig size)
    rgb: np.ndarray             # (Hr, Wr, 3) uint8 — resized, pre-normalization
    x: torch.Tensor             # (1, 3, Hr, Wr) ImageNet-normalized, on device
    size: Tuple[int, int]       # (Hr, Wr)
    orig_size: Tuple[int, int]  # (W0, H0)
    # Filled lazily: all heads processing this image share one DINO forward.
    feature_cache: Optional[Tuple[np.ndarray, int, int]] = None


class FeatureExtractor:
    """Load one DINOv3 backbone and turn images into L2-normalized patch features."""

    # One model per (repo, ckpt, device); `resolution` is per-call, not per-model.
    _instances: ClassVar[Dict[Tuple[str, str, str], "FeatureExtractor"]] = {}

    def __init__(
        self,
        resolution: int = 448,
        device: str = "auto",
        repo_dir: str = DINOV3_REPO,
        ckpt: str = DEFAULT_CKPT,
    ):
        self.resolution = resolution
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.repo_dir = repo_dir
        self.ckpt = ckpt
        self._model = None

    # ------------------------------------------------------------------ model

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            model = torch.hub.load(
                repo_or_dir=self.repo_dir,
                model=hub_entry_for(self.ckpt),
                source="local",
                weights=self.ckpt,
            )
            self._model = model.to(self.device).eval()
        return self._model

    # ------------------------------------------------------------- preprocess

    def _load_pil(self, image: ImageLike) -> Image.Image:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image.copy()
        elif isinstance(image, np.ndarray):
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"expected HxWx3 RGB array, got {arr.shape}")
            img = Image.fromarray(arr)
        else:
            raise TypeError(f"unsupported image type: {type(image)}")
        return ImageOps.exif_transpose(img).convert("RGB")

    def _resize(
        self, pil: Image.Image, resolution: int
    ) -> Tuple[Image.Image, int, int]:
        """Aspect-preserving resize to `resolution`, stretched to 16px multiples.

        Stretching (<= ~3%) rather than padding: bottom padding would fabricate
        seed rows of fake ground; cropping could clip the real bottom edge.
        """
        w0, h0 = pil.size
        scale = resolution / min(w0, h0)
        w1, h1 = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        wr = max(PATCH_SIZE, round(w1 / PATCH_SIZE) * PATCH_SIZE)
        hr = max(PATCH_SIZE, round(h1 / PATCH_SIZE) * PATCH_SIZE)
        return pil.resize((wr, hr), Image.BICUBIC), hr, wr

    def _preprocess(
        self, pil: Image.Image, resolution: int = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resized+normalized tensor [1,3,Hr,Wr] and (Hr, Wr)."""
        if resolution is None:
            resolution = self.resolution
        img, hr, wr = self._resize(pil, resolution)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
            IMAGENET_STD
        )[:, None, None]
        return x[None].to(self.device), (hr, wr)

    # ---------------------------------------------------------------- features

    def _features(self, x: torch.Tensor) -> Tuple[np.ndarray, int, int]:
        """L2-normalized last-layer patch features, [h*w, D] float32."""
        with torch.inference_mode():
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(x.device.type == "cuda")
            ):
                f = self._ensure_model().get_intermediate_layers(
                    x, n=1, reshape=True, norm=True
                )[0]  # [1, D, h, w]
            f = F.normalize(f.float(), dim=1)
        _, _, h, w = f.shape
        feats = f[0].reshape(f.shape[1], -1).T.cpu().numpy()  # [h*w, D]
        return feats, h, w

    # ------------------------------------------------------------- combined API

    def prepare(self, image: ImageLike, resolution: int = None) -> PreparedImage:
        """One image -> both CLIP-ready rgb and DINOv3-ready tensor, pixel-aligned."""
        if resolution is None:
            resolution = self.resolution
        pil = self._load_pil(image)
        img, hr, wr = self._resize(pil, resolution)
        rgb = np.asarray(img)  # (hr, wr, 3) uint8, same resize as x

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
            IMAGENET_STD
        )[:, None, None]
        x = x[None].to(self.device)

        return PreparedImage(
            pil=pil, rgb=rgb, x=x, size=(hr, wr), orig_size=pil.size
        )

    def features(self, prepared: PreparedImage) -> Tuple[np.ndarray, int, int]:
        if prepared.feature_cache is None:
            prepared.feature_cache = self._features(prepared.x)
        return prepared.feature_cache


def get_extractor(
    resolution: int = 448,
    device: str = "auto",
    repo_dir: str = DINOV3_REPO,
    ckpt: str = DEFAULT_CKPT,
) -> FeatureExtractor:
    """Process-wide singleton: one backbone load shared by all segmenters."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (repo_dir, ckpt, str(device))
    if key not in FeatureExtractor._instances:
        FeatureExtractor._instances[key] = FeatureExtractor(
            resolution=resolution, device=device, repo_dir=repo_dir, ckpt=ckpt
        )
    return FeatureExtractor._instances[key]

# -*- coding: utf-8 -*-
"""Semantic segmentation with the official DINOv3 ViT-L + Mask2Former ADE20K adapter.

Zero training on our side: Meta released frozen-backbone adapters trained on
ADE20K (150 classes) for the ViT-L backbone.  We map three of those classes to
navigation semantics — door(14), floor(3), stairs(46) — and convert the door
mask into the same DoorResult the CLIP head produces, so visualization and
downstream code are unchanged.

Both weight files live on the gated CDN (403 without an authorized link) and
must be fetched once into dinov3/checkpoints/:
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth    ViT-L backbone (~1.2GB)
    dinov3_vitl16_ade20k_m2f_head-bf307cb1.pth      M2F decoder head

Inference follows the official whole-image recipe (eval/segmentation/
inference.py): squash to 512x512, forward, merge per-query masks with softmaxed
class logits via einsum, upsample per-class probabilities to the original size.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# Defensive, consistent with the rest of the package.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .doors import DoorBox, DoorResult
from .features import DINOV3_REPO, IMAGENET_MEAN, IMAGENET_STD

ImageLike = Union[str, Path, Image.Image, np.ndarray]

DEFAULT_VITL_CKPT = (
    "/home/dow/DOW/dinov3/dinov3/checkpoints/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_M2F_CKPT = (
    "/home/dow/DOW/dinov3/dinov3/checkpoints/"
    "dinov3_vitl16_ade20k_m2f_head-bf307cb1.pth"
)

# ADE20K-150 class ids (0-indexed, standard ordering).
ADE_FLOOR, ADE_DOOR, ADE_STAIRS = 3, 14, 46
NAV_CLASSES = {"door": ADE_DOOR, "floor": ADE_FLOOR, "stairs": ADE_STAIRS}


@dataclass
class SemsegResult:
    """Per-image semantic segmentation at original resolution."""
    labels: np.ndarray                 # (H,W) int32 ADE20K argmax ids
    masks: dict[str, np.ndarray]      # name -> (H,W) bool, prob > threshold
    prob_maps: dict[str, np.ndarray]  # name -> (H,W) float32 in [0, 1]
    probs: dict[str, float]           # name -> mean prob inside its mask (0 if empty)
    threshold: float
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)


class SemanticSegmentor:
    """Official ADE20K adapter mapped onto door / floor / stairs."""

    def __init__(
        self,
        backbone_ckpt: str = DEFAULT_VITL_CKPT,
        head_ckpt: str = DEFAULT_M2F_CKPT,
        device: str = "auto",
        input_size: int = 512,          # official whole-image inference size
        mask_thr: float = 0.5,
        repo_dir: str = DINOV3_REPO,
        *,
        pretrained: bool = True,        # False = random init, smoke-test only
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backbone_ckpt = backbone_ckpt
        self.head_ckpt = head_ckpt
        self.input_size = input_size
        self.mask_thr = mask_thr
        self.repo_dir = repo_dir
        self.pretrained = pretrained
        self._model = None

    # ------------------------------------------------------------------ model

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            if self.pretrained:
                missing = [
                    p for p in (self.backbone_ckpt, self.head_ckpt)
                    if not Path(p).is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        "官方 ADE20K Adapter 权重缺失（CDN 有门禁，需授权链接手动下载）: "
                        + "; ".join(missing)
                    )
            # Same assembly as the official _make_dinov3_m2f_segmentor for
            # backbone_name="dinov3_vitl16" (that entry is not re-exported by
            # hubconf, so we build it from the public pieces directly).
            backbone = torch.hub.load(
                repo_or_dir=self.repo_dir,
                model="dinov3_vitl16",
                source="local",
                pretrained=self.pretrained,
                weights=self.backbone_ckpt,
            )
            if str(self.repo_dir) not in sys.path:
                sys.path.insert(0, str(self.repo_dir))
            from dinov3.eval.segmentation.models import build_segmentation_decoder

            model = build_segmentation_decoder(
                backbone_model=backbone,
                decoder_type="m2f",
                hidden_dim=2048,
                autocast_dtype=torch.bfloat16,
            )
            if self.pretrained:
                sd = torch.load(self.head_ckpt, map_location="cpu", weights_only=True)
                if "model" in sd:  # tolerate training-checkpoint wrapping
                    sd = sd["model"]
                missing, unexpected = model.load_state_dict(sd, strict=False)
                bad_missing = [k for k in missing if "backbone" not in k]
                if bad_missing or unexpected:
                    raise RuntimeError(
                        f"M2F head weights mismatch: missing={bad_missing[:5]} "
                        f"unexpected={list(unexpected)[:5]}"
                    )
            self._model = model.to(self.device).eval()
        return self._model

    # ------------------------------------------------------------- inference

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

    def segment(self, image: ImageLike) -> SemsegResult:
        """One image -> ADE20K argmax labels + door/floor/stairs masks."""
        t0 = time.perf_counter()
        pil = self._load_pil(image)
        w0, h0 = pil.size
        s = self.input_size

        img = pil.resize((s, s), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
            IMAGENET_STD
        )[:, None, None]
        x = x[None].to(self.device)

        with torch.inference_mode():
            pred = self._ensure_model().predict(x, rescale_to=(s, s))
            mask_cls = F.softmax(pred["pred_logits"], dim=-1)[..., :-1]  # [1,Q,150]
            mask_pred = pred["pred_masks"].sigmoid()                     # [1,Q,s,s]
            prob = torch.einsum(
                "bqc,bqhw->bchw", mask_cls.float(), mask_pred.float()
            )[0]                                                          # [150,s,s]
            labels_s = prob.argmax(0).cpu().numpy().astype(np.int32)

        # Upsample to the ORIGINAL resolution before thresholding, so masks
        # align pixel-perfectly with the other heads' outputs.
        masks: dict[str, np.ndarray] = {}
        prob_maps: dict[str, np.ndarray] = {}
        probs: dict[str, float] = {}
        for name, cid in NAV_CLASSES.items():
            p = cv2.resize(
                prob[cid].cpu().numpy(), (w0, h0), interpolation=cv2.INTER_LINEAR
            )
            m = p > self.mask_thr
            masks[name] = m
            prob_maps[name] = p.astype(np.float32)
            probs[name] = float(p[m].mean()) if m.any() else 0.0
        labels = cv2.resize(labels_s, (w0, h0), interpolation=cv2.INTER_NEAREST)

        warnings = []
        if not masks["door"].any():
            warnings.append(f"no door pixels above {self.mask_thr}")
        return SemsegResult(
            labels=labels, masks=masks, prob_maps=prob_maps, probs=probs,
            threshold=self.mask_thr,
            elapsed_s=time.perf_counter() - t0, warnings=warnings,
        )


def door_result_from_semseg(res: SemsegResult) -> DoorResult:
    """Adapt the semantic door mask to the CLIP head's result type.

    No geometric filters here — the class is trained semantically, so unlike
    CLIP there is no switch-panel noise to filter out.  Score = mean door
    probability inside each connected component.
    """
    t0 = time.perf_counter()
    door = res.masks["door"]
    door_p = res.prob_maps["door"]
    mask_u8 = door.astype(np.uint8) * 255

    boxes: list[DoorBox] = []
    n, comp = cv2.connectedComponents(door.astype(np.uint8), connectivity=8)
    for cid in range(1, n):
        m = comp == cid
        if m.sum() < 100:  # drop specks
            continue
        ys, xs = np.where(m)
        boxes.append(DoorBox(
            bbox=(int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
            score=float(door_p[m].mean()),
            area=int(m.sum()),
        ))
    boxes.sort(key=lambda b: b.area, reverse=True)
    return DoorResult(
        mask=mask_u8, heatmap=door_p, boxes=boxes, threshold=res.threshold,
        elapsed_s=time.perf_counter() - t0,
        warnings=list(res.warnings),
    )

# -*- coding: utf-8 -*-
"""Zero-shot door detection with CLIP ViT-B-32 (OpenAI weights, offline-capable).

No training and no labels: a multi-scale sliding window over the inference-
resolution RGB image is classified against a prompt ensemble ("a door" and
friends vs. walls/windows/furniture/...), producing a door-ness heatmap which
is thresholded, cleaned up and geometry-filtered into door instances.

Weights: OpenAI's original ViT-B-32 TorchScript checkpoint from
openaipublic.azureedge.net (huggingface.co is NOT reachable on this network),
converted once to a plain state_dict that open_clip loads via
``create_model_and_transforms("ViT-B-32-quickgelu", pretrained=<path>)``
(quickgelu matches the OpenAI activation).
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Defensive: huggingface.co is unreachable here; fail fast instead of hanging
# if anything in the stack ever tries the hub (we only use local files).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .features import FeatureExtractor, ImageLike, PreparedImage  # noqa: E402

CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)
CLIP_CACHE_DIR = os.path.expanduser("~/.cache/clip")
DEFAULT_CLIP_CKPT = os.path.join(CLIP_CACHE_DIR, "ViT-B-32-open_clip.pth")

DEFAULT_DOOR_PROMPTS = [
    "a door", "a doorway", "an open door", "a closed door",
    "double doors", "a glass door",
]
DEFAULT_NEG_PROMPTS = [
    "a wall", "a window", "a whiteboard", "a cabinet", "a bookshelf",
    "a monitor", "a poster", "a picture frame", "a person", "a box",
]
TEMPLATES = [
    "a photo of {}",
    "a photo of the {} in an office",
    "a photo of a {} in an indoor scene",
    "a close-up photo of a {}",
    "a blurred photo of a {}",
    "a low-resolution photo of a {}",
    "a bright photo of a {}",
]

# One shared CLIP model per (model_name, ckpt, device): door/stair detectors
# with the same weights reuse it instead of loading a second ~0.6GB copy.
_CLIP_CACHE: Dict[Tuple[str, str, str], Tuple] = {}


def ensure_clip_weights(path: str = DEFAULT_CLIP_CKPT) -> str:
    """Download + convert OpenAI ViT-B-32 weights once; returns state_dict path."""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    jit_path = os.path.join(os.path.dirname(path), "ViT-B-32.pt")
    if not os.path.isfile(jit_path):
        expected = CLIP_URL.split("/")[-2]  # sha256 prefix from the URL
        print(f"downloading CLIP ViT-B-32 (~354MB) from openaipublic.azureedge.net ...")
        urllib.request.urlretrieve(CLIP_URL, jit_path)
        digest = hashlib.sha256(open(jit_path, "rb").read()).hexdigest()
        if not digest.startswith(expected):
            os.remove(jit_path)
            raise RuntimeError("CLIP weight download failed sha256 check")
    sd = torch.jit.load(jit_path, map_location="cpu").state_dict()
    for k in ["input_resolution", "context_length", "vocab_size"]:
        sd.pop(k, None)
    torch.save(sd, path)
    print(f"converted CLIP weights -> {path}")
    return path


@dataclass
class DoorBox:
    bbox: Tuple[int, int, int, int]  # x, y, w, h at ORIGINAL resolution
    score: float                     # mean heatmap value inside the component
    area: int                        # pixel area at inference resolution


@dataclass
class DoorResult:
    mask: np.ndarray        # (H0, W0) uint8 {0, 255}, original resolution
    heatmap: np.ndarray     # (H0, W0) float32 in [0, 1], upsampled
    boxes: List[DoorBox]
    threshold: float
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)


class DoorDetector:
    """Sliding-window CLIP zero-shot door detection.

    The mechanism is target-agnostic: prompts and geometry filters are
    constructor parameters, so subclasses (e.g. StairsDetector) reuse it
    wholesale with different defaults.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        ckpt: str = DEFAULT_CLIP_CKPT,
        threshold: float = 0.75,
        stride: int = 48,
        scales: Sequence[int] = (96, 128, 160),
        batch_size: int = 64,
        device: str = "auto",
        positive_prompts: Sequence[str] = DEFAULT_DOOR_PROMPTS,
        negative_prompts: Sequence[str] = DEFAULT_NEG_PROMPTS,
        target_name: str = "door",
        *,
        min_area_frac: float = 0.0005,
        aspect_min: float = 1.1,
        min_bottom_frac: float = 0.35,
        extractor: Optional[FeatureExtractor] = None,
    ):
        self.model_name = model_name
        self.ckpt = ckpt
        self.threshold = threshold
        self.stride = stride
        self.scales = tuple(scales)
        self.batch_size = batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.positive_prompts = list(positive_prompts)
        self.negative_prompts = list(negative_prompts)
        self.target_name = target_name
        self.min_area_frac = min_area_frac
        self.aspect_min = aspect_min
        self.min_bottom_frac = min_bottom_frac
        self._extractor = extractor  # optional; only used by detect()
        self._model = None
        self._clip_preprocess = None
        self._protos = None  # [n_prompts, D], positives first
        self._n_pos = len(self.positive_prompts)

    # ------------------------------------------------------------------ model

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for CLIP-based detection "
                "(doors/stairs): pip install open_clip_torch"
            ) from e
        key = (self.model_name, self.ckpt, str(self.device))
        cached = _CLIP_CACHE.get(key)
        if cached is None:
            path = ensure_clip_weights(self.ckpt)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=path
            )
            cached = (model.to(self.device).eval(), preprocess)
            _CLIP_CACHE[key] = cached
        self._model, self._clip_preprocess = cached

        # Text prototypes: each prompt averaged over templates, L2-normalized.
        texts = [
            t.format(p)
            for p in (*self.positive_prompts, *self.negative_prompts)
            for t in TEMPLATES
        ]
        embs = []
        with torch.inference_mode():
            for i in range(0, len(texts), 64):
                tok = open_clip.tokenize(texts[i : i + 64]).to(self.device)
                e = self._model.encode_text(tok).float()
                embs.append(e)
        embs = torch.cat(embs)  # [P*T, D]
        embs = embs.reshape(len(self.positive_prompts) + len(self.negative_prompts),
                            len(TEMPLATES), -1).mean(1)
        self._protos = torch.nn.functional.normalize(embs, dim=-1)  # [P, D]

    # ------------------------------------------------------------- heat map

    def _windows(self, H: int, W: int, s: int) -> List[Tuple[int, int]]:
        if s > H or s > W:
            return []
        def axis(n: int) -> List[int]:
            pts = list(range(0, n - s + 1, self.stride))
            if pts[-1] != n - s:
                pts.append(n - s)
            return pts
        return [(y, x) for y in axis(H) for x in axis(W)]

    def heatmap_prepared(self, prepared: PreparedImage) -> np.ndarray:
        """Door-ness in [0,1] at inference resolution (Hr, Wr)."""
        self._ensure_model()
        rgb = prepared.rgb
        H, W = rgb.shape[:2]

        crops: List[Image.Image] = []
        coords: List[Tuple[int, int, int]] = []  # y, x, s
        for s in self.scales:
            for y, x in self._windows(H, W, s):
                crops.append(Image.fromarray(rgb[y : y + s, x : x + s]))
                coords.append((y, x, s))
        if not crops:
            return np.zeros((H, W), np.float32)

        scores = np.zeros(len(crops), np.float32)
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                batch = crops[i : i + self.batch_size]
                x = torch.stack([self._clip_preprocess(c) for c in batch]).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(self.device.type == "cuda")):
                    img_emb = self._model.encode_image(x).float()
                img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
                logits = self._model.logit_scale.exp().float() * (img_emb @ self._protos.T)
                probs = torch.softmax(logits, dim=1)
                scores[i : i + len(batch)] = probs[:, : self._n_pos].sum(1).cpu().numpy()

        accum = np.zeros((H, W), np.float32)
        count = np.zeros((H, W), np.float32)
        for (y, x, s), p in zip(coords, scores):
            accum[y : y + s, x : x + s] += p
            count[y : y + s, x : x + s] += 1
        heat = accum / np.maximum(count, 1e-6)
        return cv2.GaussianBlur(heat, (5, 5), 0)

    # -------------------------------------------------------------- pipeline

    def detect(self, image: ImageLike) -> DoorResult:
        if self._extractor is None:
            from .features import get_extractor
            self._extractor = get_extractor()
        t_start = time.perf_counter()
        result = self.detect_prepared(self._extractor.prepare(image))
        result.elapsed_s = time.perf_counter() - t_start
        return result

    def detect_prepared(self, prepared: PreparedImage) -> DoorResult:
        warnings: List[str] = []
        t0 = time.perf_counter()
        heat = self.heatmap_prepared(prepared)
        hr, wr = heat.shape
        w0, h0 = prepared.orig_size

        binary = (heat >= self.threshold).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2
        )
        n_comp, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        keep = np.zeros_like(binary, bool)
        boxes: List[DoorBox] = []
        min_area = self.min_area_frac * hr * wr
        sx, sy = w0 / wr, h0 / hr
        for comp in range(1, n_comp):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            x, y, cw, ch = stats[comp, :4]
            if area < min_area:
                continue
            if self.aspect_min > 0 and ch / max(cw, 1) < self.aspect_min:
                continue
            if (y + ch) < self.min_bottom_frac * hr:
                continue  # doors stand on/near the floor; reject sky/ceiling blobs
            keep[cc == comp] = True
            boxes.append(
                DoorBox(
                    bbox=(int(round(x * sx)), int(round(y * sy)),
                          int(round(cw * sx)), int(round(ch * sy))),
                    score=float(heat[cc == comp].mean()),
                    area=area,
                )
            )
        boxes.sort(key=lambda b: -b.area)

        if not boxes:
            warnings.append(f"no {self.target_name}s above threshold {self.threshold:.2f}")
            mask = np.zeros((h0, w0), np.uint8)
        else:
            mask_full = cv2.resize(keep.astype(np.float32), (w0, h0),
                                   interpolation=cv2.INTER_LINEAR)
            mask = ((mask_full >= 0.5).astype(np.uint8)) * 255

        heat_full = cv2.resize(heat, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return DoorResult(
            mask=mask,
            heatmap=heat_full,
            boxes=boxes,
            threshold=self.threshold,
            elapsed_s=time.perf_counter() - t0,
            warnings=warnings,
        )

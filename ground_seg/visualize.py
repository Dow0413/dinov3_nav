# -*- coding: utf-8 -*-
"""Visualization for ground/scene/door results (RGB arrays in/out).

Labels are Chinese ("门 0.85", "可通行地面 42%") via the system Noto CJK font,
falling back to ASCII cv2 text when no CJK font is installed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .doors import DoorBox, DoorResult
from .scene import PALETTE, SceneResult
from .segmenter import GroundMaskResult
from .semseg import SemsegResult

GREEN = (0, 220, 0)    # RGB — traversable ground / semantic floor
BLUE = (30, 90, 255)   # RGB — doors
ORANGE = (255, 150, 0) # RGB — stairs (robot must not climb)
WHITE = (255, 255, 255)

_CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


@lru_cache(maxsize=8)
def _cjk_font(size: int):
    for p in _CJK_FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return None


def put_text(img: np.ndarray, text: str, x: int, y: int,
             scale: int = 1, color=WHITE) -> np.ndarray:
    """Draw text with a black stroke (CJK-capable). Returns the new image."""
    font = _cjk_font(28 * scale)
    if font is None:  # no CJK font installed — ASCII fallback
        cv2.putText(img, text.encode("ascii", "replace").decode(), (x, y + 28 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color, 2, cv2.LINE_AA)
        return img
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.text((max(4, x), max(4, y)), text, font=font, fill=color,
           stroke_width=max(2, 2 * scale), stroke_fill=(0, 0, 0))
    return np.asarray(pil).copy()  # PIL hands back a read-only buffer


def overlay(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Green tint over masked pixels + white contour. Returns RGB uint8."""
    out = img_rgb.astype(np.float32)
    m = mask > 0
    out[m] = out[m] * (1.0 - alpha) + np.array(GREEN, np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, WHITE, thickness=max(1, out.shape[0] // 400))
    return out


def heatmap(sim: np.ndarray) -> np.ndarray:
    """Cosine-similarity map -> JET heatmap. Returns RGB uint8."""
    u8 = (np.clip(sim, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)[:, :, ::-1]


# ------------------------------------------------------------------- scene


def colorize_labels(labels: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Per-pixel region labels -> flat palette colors. Returns RGB uint8."""
    lut = np.array(colors, np.uint8)  # row i = color of label i
    return lut[np.clip(labels, 0, len(lut) - 1)]


def scene_overlay(
    img_rgb: np.ndarray, scene: SceneResult, alpha: float = 0.65
) -> np.ndarray:
    """Region-colored map blended over the photo. Returns RGB uint8."""
    colors = [r.color for r in scene.regions] + list(PALETTE)  # id -> color, spare tail
    flat = colorize_labels(scene.labels, colors)
    out = img_rgb.astype(np.float32) * (1.0 - alpha) + flat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------- doors


def _tint(img_rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = img_rgb.astype(np.float32)
    m = mask > 0
    out[m] = out[m] * (1.0 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def door_overlay(img_rgb: np.ndarray, result: DoorResult) -> np.ndarray:
    """Blue tint + contour + bbox/score per detected door. Returns RGB uint8."""
    out = _tint(img_rgb, result.mask, BLUE, 0.45)
    contours, _ = cv2.findContours(
        (result.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, WHITE, thickness=max(1, out.shape[0] // 400))
    scale = max(1, out.shape[0] // 400)
    for b in result.boxes:
        x, y, w, h = b.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), BLUE, thickness=max(2, out.shape[0] // 300))
        out = put_text(out, f"门 {b.score:.2f}", x + 6, y + 8, scale)
    return out


def door_heatmap(heat: np.ndarray) -> np.ndarray:
    """Door-ness map -> JET heatmap. Returns RGB uint8."""
    return heatmap(heat)


# ------------------------------------------------------------ single-target


def ground_overlay(
    img_rgb: np.ndarray,
    result: GroundMaskResult,
    stairs: np.ndarray | None = None,
) -> np.ndarray:
    """Ground location only: green tint + contour + coverage label.

    With `stairs` (binary mask), that region gets an orange tint + 台阶 label
    under the coverage text — the companion view to safety subtraction, where
    the caller has already removed the stair region from result.mask.
    """
    out = overlay(img_rgb, result.mask, alpha=0.35)
    if stairs is not None and stairs.any():
        out = _tint(out, stairs.astype(np.uint8) * 255, ORANGE, 0.45)
        out = _label_mask(out, stairs, "台阶")
    scale = max(1, out.shape[0] // 400)
    return put_text(out, f"可通行地面 {result.coverage:.0%}", 12, 12, scale)


# ----------------------------------------------------------------- semantic


def semantic_segments(
    img_rgb: np.ndarray,
    scene: SceneResult,
    ground: GroundMaskResult,
    doors: DoorResult | None = None,
    stairs: np.ndarray | None = None,
) -> np.ndarray:
    """Scene segmentation annotated with the semantics we DO have.

    K-means regions carry no object semantics and stay color-only; the heads
    add theirs: doors get blue boxes labeled 门 <score>, traversable ground
    gets a white contour labeled 可通行地面 <coverage>, stairs get an orange
    contour labeled 台阶.
    """
    out = scene_overlay(img_rgb, scene)
    scale = max(1, out.shape[0] // 400)

    if stairs is not None and stairs.any():
        out = _tint(out, stairs.astype(np.uint8) * 255, ORANGE, 0.45)
        out = _label_mask(out, stairs, "台阶")

    if doors is not None:
        for b in doors.boxes:
            x, y, w, h = b.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), BLUE,
                          thickness=max(2, out.shape[0] // 300))
            out = put_text(out, f"门 {b.score:.2f}", x + 6, y + 8, scale)

    m = (ground.mask > 0).astype(np.uint8)
    if m.any():
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [biggest], -1, WHITE,
                         thickness=max(2, out.shape[0] // 400))
        M = cv2.moments(biggest)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            out = put_text(out, f"可通行地面 {ground.coverage:.0%}",
                           cx - 90 * scale, cy - 16 * scale, scale)
    return out


def _label_mask(out: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    """Write `text` at the largest component's centroid with a white contour."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, [biggest], -1, WHITE,
                     thickness=max(2, out.shape[0] // 400))
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return out
    scale = max(1, out.shape[0] // 400)
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return put_text(out, text, cx - 60 * scale, cy - 16 * scale, scale)


def semantic_overlay(img_rgb: np.ndarray, sem: SemsegResult) -> np.ndarray:
    """Pure ADE20K semantic view: floor GREEN, door BLUE, stairs ORANGE,
    with a legend giving per-class coverage.  Replaces nothing — this is the
    "semantic segmentation" output the K-means head cannot provide."""
    out = img_rgb.copy()
    for name, color, alpha in [
        ("floor", GREEN, 0.30), ("door", BLUE, 0.45), ("stairs", ORANGE, 0.45),
    ]:
        m = sem.masks.get(name)
        if m is None or not m.any():
            continue
        out = _tint(out, m.astype(np.uint8) * 255, color, alpha)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, WHITE,
                         thickness=max(1, out.shape[0] // 400))
    n_door = len(_components(sem.masks["door"]))
    lines = [
        f"地板 {sem.masks['floor'].mean():.0%}",
        f"门 x{n_door}",
        f"台阶 {sem.masks['stairs'].mean():.0%}",
    ]
    scale = max(1, out.shape[0] // 400)
    for i, line in enumerate(lines):
        out = put_text(out, line, 12, 12 + i * 36 * scale, scale)
    return out


def _components(mask: np.ndarray) -> list:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return list(range(1, n))


# ---------------------------------------------------------------- navigation


def nav_overlay(
    img_rgb: np.ndarray,
    ground_mask: np.ndarray,
    door_result: DoorResult | None = None,
    stairs: np.ndarray | None = None,
) -> np.ndarray:
    """Navigation map: traversable ground GREEN, doors BLUE, stairs ORANGE.

    Purely visual: where masks overlap, later draws win on canvas; the
    underlying masks are returned unchanged by their segmenters.
    """
    out = overlay(img_rgb, ground_mask, alpha=0.35)
    if stairs is not None and stairs.any():
        out = _tint(out, stairs.astype(np.uint8) * 255, ORANGE, 0.45)
        out = _label_mask(out, stairs, "台阶")
    if door_result is not None and door_result.mask.any():
        out = _tint(out, door_result.mask, BLUE, 0.5)
        contours, _ = cv2.findContours(
            (door_result.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, WHITE, thickness=max(1, out.shape[0] // 400))
        for b in door_result.boxes:
            x, y, w, h = b.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), BLUE,
                          thickness=max(2, out.shape[0] // 300))
    return out


def save_all(
    result: GroundMaskResult,
    img_rgb: np.ndarray,
    out_dir: Path | str,
    stem: str,
) -> dict[str, Path]:
    """Write <stem>_mask.png / _overlay.png / _simmap.jpg. Returns saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mask": out_dir / f"{stem}_mask.png",
        "overlay": out_dir / f"{stem}_overlay.png",
        "simmap": out_dir / f"{stem}_simmap.jpg",
    }
    cv2.imwrite(str(paths["mask"]), result.mask)
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay(img_rgb, result.mask), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["simmap"]), cv2.cvtColor(heatmap(result.similarity), cv2.COLOR_RGB2BGR))
    return paths


def save_scene_all(
    ground: GroundMaskResult,
    scene: SceneResult,
    doors: DoorResult | None,
    img_rgb: np.ndarray,
    out_dir: Path | str,
    stem: str,
    stairs: np.ndarray | None = None,
    semantic: SemsegResult | None = None,
) -> dict[str, Path]:
    """Write the per-image result folder <out_dir>/:
        <stem>_segments.png  scene segmentation + semantic labels (门/可通行地面/台阶)
        <stem>_ground.png    ground location only (green + coverage)
        <stem>_door.png      door locations only (blue + boxes/scores)
        <stem>_nav.png       navigation overlay (ground green, doors blue, stairs orange)
        <stem>_mask.png      binary ground mask (for downstream navigation code)
        <stem>_doorheat.jpg  door-ness heatmap (debug)
        <stem>_semantic.png  ADE20K semantic view (地板/门/台阶), semseg only
    _door.png / _doorheat.jpg are omitted when door detection did not run;
    _semantic.png is written only when the ADE20K adapter ran.
    Returns saved paths keyed as segments/ground/door/nav/mask/doorheat/semantic.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "segments": out_dir / f"{stem}_segments.png",
        "ground": out_dir / f"{stem}_ground.png",
        "nav": out_dir / f"{stem}_nav.png",
        "mask": out_dir / f"{stem}_mask.png",
    }
    cv2.imwrite(str(paths["segments"]),
                cv2.cvtColor(semantic_segments(img_rgb, scene, ground, doors, stairs), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["ground"]),
                cv2.cvtColor(ground_overlay(img_rgb, ground), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["nav"]),
                cv2.cvtColor(nav_overlay(img_rgb, ground.mask, doors, stairs), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["mask"]), ground.mask)
    if doors is not None:
        paths["door"] = out_dir / f"{stem}_door.png"
        paths["doorheat"] = out_dir / f"{stem}_doorheat.jpg"
        cv2.imwrite(str(paths["door"]),
                    cv2.cvtColor(door_overlay(img_rgb, doors), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(paths["doorheat"]),
                    cv2.cvtColor(door_heatmap(doors.heatmap), cv2.COLOR_RGB2BGR))
    if semantic is not None:
        paths["semantic"] = out_dir / f"{stem}_semantic.png"
        cv2.imwrite(str(paths["semantic"]),
                    cv2.cvtColor(semantic_overlay(img_rgb, semantic), cv2.COLOR_RGB2BGR))
    return paths

"""
Detect a face with OpenCV Haar cascades and crop to FER-style 48×48 grayscale input.

Falls back to whole-image resize when no face is found (optional callers can treat
that as an error instead).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {path}")
    return _cascade


def _largest_face_box(
    gray: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (48, 48),
) -> Optional[Tuple[int, int, int, int]]:
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if faces is None or len(faces) == 0:
        return None
    # (x, y, w, h) with largest area
    best = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    return x, y, w, h


def _pad_box(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(img_w, x + w + px)
    y1 = min(img_h, y + h + py)
    return x0, y0, x1 - x0, y1 - y0


def pil_to_model_input_from_face(
    image: Image.Image,
    *,
    require_face: bool = True,
) -> Tuple[Optional[np.ndarray], Image.Image, Optional[Image.Image], Optional[str]]:
    """
    Build model input (1, 48, 48, 1) from the largest detected face.

    Returns:
        batch: float32 array or None if require_face and no face found
        display_rgb: RGB image for UI (original with optional rectangle)
        gradcam_base: RGB image aligned with model input (face zoom); use for Grad-CAM overlay
        error: short user message if require_face and no face
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    box = _largest_face_box(gray)
    display = rgb.copy()

    if box is None:
        if require_face:
            return None, Image.fromarray(display), None, "No face detected. Try better lighting, face the camera, or use a closer photo."
        # whole image (legacy behavior)
        small = np.array(image.convert("L").resize((48, 48)), dtype=np.float32) / 255.0
        batch = np.expand_dims(small, axis=(0, -1))
        gbase = image.convert("RGB").resize((192, 192), Image.BICUBIC)
        return batch, Image.fromarray(display), gbase, None

    x, y, fw, fh = box
    x0, y0, cw, ch = _pad_box(x, y, fw, fh, w, h)
    face_gray = gray[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=(0, -1))

    cv2.rectangle(display, (x0, y0), (x0 + cw, y0 + ch), (0, 255, 100), 2)
    # Grad-CAM base: same crop upscaled so heatmap aligns with facial structure
    gbase = Image.fromarray(face_gray).convert("RGB").resize((192, 192), Image.BICUBIC)
    return batch, Image.fromarray(display), gbase, None


def bgr_frame_to_model_input_from_face(
    img_bgr: np.ndarray,
    *,
    require_face: bool = False,
) -> np.ndarray:
    """
    Same as upload path but for BGR video frames. If no face and require_face,
    raises ValueError. If not require_face, falls back to full-frame resize.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    box = _largest_face_box(gray, min_size=(32, 32))

    if box is None:
        if require_face:
            raise ValueError("no_face")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("L").resize((48, 48))
        x = np.asarray(pil, dtype=np.float32) / 255.0
        return np.expand_dims(x, axis=(0, -1))

    x, y, fw, fh = box
    x0, y0, cw, ch = _pad_box(x, y, fw, fh, w, h)
    face_gray = gray[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=(0, -1))

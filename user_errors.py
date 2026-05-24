"""
Map exceptions to short user-facing messages (with optional technical detail).
"""

from __future__ import annotations

import logging
from typing import Tuple

from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)


def humanize_processing_error(exc: BaseException) -> Tuple[str, str]:
    """
    Return (short_message_for_user, technical_line_for_expander).
    """
    tech = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, FileNotFoundError):
        return (
            "The trained model file is missing. Add `Model/fer_model.h5` to the project or check the deployment checkout.",
            tech,
        )

    if isinstance(exc, UnidentifiedImageError):
        return (
            "This file is not a supported image. Try a JPG or PNG exported from your phone or camera.",
            tech,
        )

    if isinstance(exc, MemoryError):
        return (
            "Not enough memory to process this image. Try a smaller resolution (e.g. under 2000 px on the long side).",
            tech,
        )

    if isinstance(exc, OSError) and "image" in str(exc).lower():
        return (
            "The image could not be read. The file may be corrupted — try saving it again or use another photo.",
            tech,
        )

    # TensorFlow / Keras (import lazily to avoid extra cost on unrelated paths)
    try:
        import tensorflow as tf

        if isinstance(exc, (tf.errors.ResourceExhausted, tf.errors.OpError)):
            return (
                "The model hit a server or GPU memory limit. Wait a moment and try again, or use a smaller image.",
                tech,
            )
        if isinstance(exc, tf.errors.InvalidArgumentError):
            return (
                "The model received invalid input. Try a different photo (clear face, normal JPG/PNG).",
                tech,
            )
    except Exception:
        pass

    msg = str(exc).lower()
    if "model file not found" in msg or "fer_model" in msg:
        return (
            "The emotion model could not be loaded. Ensure `Model/fer_model.h5` exists next to the app.",
            tech,
        )
    if (
        "no conv2d" in msg
        or "gradcam" in msg
        or "never been called" in msg
        or "no defined output" in msg
        or "get_cmap" in msg
    ):
        return (
            "Explainability (Grad-CAM) could not run for this model build. Emotion prediction should still work.",
            tech,
        )
    if "cascade" in msg or "haar" in msg:
        return (
            "Face detection failed to initialize. Try turning off **Detect & crop face** in the sidebar.",
            tech,
        )

    logger.exception("Unhandled processing error")
    return (
        "Something unexpected went wrong while processing your image. See technical details below.",
        tech,
    )


def humanize_song_fetch_error(exc: BaseException) -> Tuple[str, str]:
    """Errors from iTunes / HTTP when loading recommendations."""
    tech = f"{type(exc).__name__}: {exc}"
    try:
        import requests

        if isinstance(exc, requests.Timeout):
            return ("The music service took too long to respond. Check your connection and tap **New songs** to retry.", tech)
        if isinstance(exc, requests.ConnectionError):
            return ("Could not reach the music service. Check your internet connection and try again.", tech)
        if isinstance(exc, requests.HTTPError):
            return ("The music service returned an error. Try again in a few minutes.", tech)
        if isinstance(exc, requests.RequestException):
            return ("Could not load song recommendations. Check your network and try **New songs**.", tech)
    except Exception:
        pass
    return ("Could not load song recommendations right now. Try **New songs** or check your connection.", tech)


def humanize_live_inference_error(exc: BaseException) -> str:
    """One-line message for live WebRTC overlay (no expander in thread context)."""
    summary, _ = humanize_processing_error(exc)
    return summary

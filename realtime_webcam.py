"""
Real-time webcam emotion overlay using streamlit-webrtc.

The video_frame_callback runs on a worker thread. TensorFlow inference is
serialized with a lock. Shared results are exposed via LiveEmotionState for
the main Streamlit thread (e.g. bar chart updates).
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import av
import cv2
import numpy as np
from PIL import Image

from emotion_detector import predict_emotion
from face_preprocess import bgr_frame_to_model_input_from_face
from user_errors import humanize_live_inference_error

# STUN helps WebRTC on remote hosts (e.g. Streamlit Community Cloud).
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}

# Run model every N frames to keep CPU reasonable (~7–15 FPS camera → ~2–4 inferences/s at N=4)
FRAME_SKIP = 4


@dataclass
class LiveEmotionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    emotion: str = ""
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    frame_n: int = 0


live_state = LiveEmotionState()
_predict_lock = threading.Lock()


def _bgr_to_model_input(img_bgr: np.ndarray) -> np.ndarray:
    """Prefer face crop; fall back to full-frame resize if no face (keeps stream smooth)."""
    return bgr_frame_to_model_input_from_face(img_bgr, require_face=False)


def make_video_frame_callback(use_ensemble: bool = False):
    """Return a callback for webrtc_streamer(video_frame_callback=...)."""

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with live_state.lock:
            live_state.frame_n += 1
            n = live_state.frame_n

        if n % FRAME_SKIP == 0:
            try:
                tensor = _bgr_to_model_input(img)
                with _predict_lock:
                    emotion, confidence, scores = predict_emotion(tensor, use_ensemble=use_ensemble)
                with live_state.lock:
                    live_state.emotion = emotion
                    live_state.confidence = float(confidence)
                    live_state.scores = dict(scores)
                    live_state.error = None
            except Exception as e:
                with live_state.lock:
                    live_state.error = humanize_live_inference_error(e)

        with live_state.lock:
            emotion = live_state.emotion
            confidence = live_state.confidence
            err = live_state.error

        if err:
            label = "Error — check logs"
            color = (0, 0, 255)
        elif emotion:
            label = f"{emotion.upper()}  {confidence:.1f}%"
            color = (80, 255, 80)
        else:
            label = "Starting camera..."
            color = (255, 255, 0)

        cv2.putText(
            img,
            label,
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback


def reset_live_state():
    """Clear shared state when switching modes (optional)."""
    with live_state.lock:
        live_state.emotion = ""
        live_state.confidence = 0.0
        live_state.scores = {}
        live_state.error = None
        live_state.frame_n = 0

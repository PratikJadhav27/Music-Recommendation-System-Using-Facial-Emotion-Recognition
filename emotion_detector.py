import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Emotion labels (FER-2013 order)
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Model")
BASELINE_PATH = os.path.join(_MODEL_DIR, "fer_model.h5")
BALANCED_PATH = os.path.join(_MODEL_DIR, "fer_balanced.h5")

# Cached Keras models keyed by "baseline" | "balanced"
_models: dict = {}


def ensemble_models_available() -> bool:
    """True when both checkpoint files exist on disk."""
    return os.path.isfile(BASELINE_PATH) and os.path.isfile(BALANCED_PATH)


def _load_model(path: str, cache_key: str):
    if cache_key not in _models:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        _models[cache_key] = load_model(path)
    return _models[cache_key]


def get_model():
    """Baseline CNN (used for Grad-CAM and single-model mode)."""
    return _load_model(BASELINE_PATH, "baseline")


def get_balanced_model():
    """Balanced CNN with class-weighted training."""
    return _load_model(BALANCED_PATH, "balanced")


def _softmax_vector(model, image_array: np.ndarray) -> np.ndarray:
    return model.predict(image_array, verbose=0)[0]


def predict_emotion(image_array, use_ensemble: bool = False):
    """
    Predict emotion and confidence scores.

    Args:
        image_array: (1, 48, 48, 1) float32 in [0, 1].
        use_ensemble: If True and both models exist, average their softmax outputs.
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid image array")

    try:
        baseline = get_model()
        predictions = _softmax_vector(baseline, image_array)

        if use_ensemble and ensemble_models_available():
            balanced = get_balanced_model()
            predictions = (predictions + _softmax_vector(balanced, image_array)) / 2.0

        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[emotion_index] * 100)
        confidence_scores = {
            emotion_labels[i]: round(float(predictions[i] * 100), 2)
            for i in range(len(emotion_labels))
        }
        return emotion_labels[emotion_index], confidence, confidence_scores
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"Error during emotion prediction: {str(e)}") from e

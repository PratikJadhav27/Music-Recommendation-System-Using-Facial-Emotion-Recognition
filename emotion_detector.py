import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Get absolute path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "fer_model.h5")

# Load trained model (cached globally)
_model = None

def get_model():
    """Load model once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model

def predict_emotion(image_array):
    """Predicts emotion and returns the top emotion with confidence scores."""
    try:
        if image_array is None or image_array.size == 0:
            raise ValueError("Invalid image array")
        
        model = get_model()
        predictions = model.predict(image_array, verbose=0)[0]  # Get the first (and only) sample's predictions
        emotion_index = np.argmax(predictions)  # Get the highest probability emotion
        confidence = predictions[emotion_index] * 100  # Convert to percentage

        # Create a dictionary of all emotions with their confidence scores
        confidence_scores = {emotion_labels[i]: round(predictions[i] * 100, 2) for i in range(len(emotion_labels))}

        return emotion_labels[emotion_index], confidence, confidence_scores
    except Exception as e:
        raise Exception(f"Error during emotion prediction: {str(e)}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "Model/fer_model.h5"
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_emotion(image_array):
    """Predicts emotion and returns the top emotion with confidence scores."""
    predictions = model.predict(image_array)[0]  # Get the first (and only) sample's predictions
    emotion_index = np.argmax(predictions)  # Get the highest probability emotion
    confidence = predictions[emotion_index] * 100  # Convert to percentage

    # Create a dictionary of all emotions with their confidence scores
    confidence_scores = {emotion_labels[i]: round(predictions[i] * 100, 2) for i in range(len(emotion_labels))}

    return emotion_labels[emotion_index], confidence, confidence_scores

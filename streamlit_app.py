import streamlit as st
import cv2
import numpy as np
from PIL import Image
from emotion_detector import predict_emotion
from spotify_recommendation import get_playlist_for_emotion
import tempfile
import os

# Streamlit UI
st.title("🎵 Music Recommendation System using Facial Emotion Recognition")

# Sidebar
st.sidebar.header("Upload or Capture Your Image")
option = st.sidebar.radio("Choose an option:", ("Upload an Image", "Capture via Webcam"))

# Function to capture image from webcam
def capture_webcam():
    """Captures image from webcam and saves it."""
    cap = cv2.VideoCapture(0)
    st.sidebar.text("Press 'Space' to capture, 'Q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Press 'Space' to Capture", frame)
        
        key = cv2.waitKey(1)
        if key == 32:  # Space key to capture
            img_path = "images/captured_image.jpg"
            cv2.imwrite(img_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return img_path
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None

# Handling Image Upload or Webcam Capture
image = None
if option == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "Capture via Webcam":
    if st.sidebar.button("Capture Image"):
        img_path = capture_webcam()
        if img_path:
            image = Image.open(img_path)

# Display Image and Predict Emotion
if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to OpenCV format
    img_array = np.array(image.convert("L").resize((48, 48))) / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Reshape for model
    
    # Predict emotion and confidence scores
    emotion, confidence, confidence_scores = predict_emotion(img_array)

    st.subheader(f"🎭 Detected Emotion: **{emotion.capitalize()}** ({confidence:.2f}% confidence)")

    # Display all confidence scores as a bar chart
    st.bar_chart(confidence_scores)

    # Fetch and Display Spotify Playlists
    st.subheader("🎵 Recommended Playlists for You:")
    playlists = get_playlist_for_emotion(emotion)
    
    for playlist in playlists:
        col1, col2 = st.columns([1, 4])
        with col1:
            if playlist["image"]:
                st.image(playlist["image"], width=100)
        with col2:
            st.markdown(f"[{playlist['name']}]({playlist['url']})")


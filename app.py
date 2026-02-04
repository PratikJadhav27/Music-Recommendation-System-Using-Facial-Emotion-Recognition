import streamlit as st

import numpy as np
from PIL import Image
from emotion_detector import predict_emotion
from spotify_recommendation import get_playlist_for_emotion
import os

# Streamlit UI
st.title("🎵 Music Recommendation System using Facial Emotion Recognition")

# Sidebar
st.sidebar.header("Upload or Capture Your Image")
option = st.sidebar.radio("Choose an option:", ("Upload an Image", "Capture via Webcam"))

# Note: Webcam capture in Streamlit requires a different approach
# Using st.camera_input instead of cv2 (which doesn't work in browser)
def capture_webcam():
    """Captures image from webcam using Streamlit's built-in camera input."""
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    picture = st.camera_input("Take a picture")
    if picture is not None:
        img = Image.open(picture)
        img_path = "images/captured_image.jpg"
        img.save(img_path)
        return img
    return None

# Handling Image Upload or Webcam Capture
image = None
if option == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "Capture via Webcam":
    image = capture_webcam()

# Display Image and Predict Emotion
if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    try:
        with st.spinner("🔄 Analyzing emotion..."):
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
            with st.spinner("🔄 Fetching playlists..."):
                playlists = get_playlist_for_emotion(emotion, confidence_scores)
            
            
            if playlists:
                for idx, playlist in enumerate(playlists):
                    col1, col2, col3 = st.columns([1, 5, 1])
                    
                    with col1:
                        if playlist["image"]:
                            st.image(playlist["image"], width=100)
                    
                    with col2:
                        st.markdown(f"**[{playlist['name']}]({playlist['url']})**")
                    
                    with col3:
                        # Feedback buttons
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            if st.button("👍", key=f"like_{idx}"):
                                from feedback_manager import log_feedback
                                success = log_feedback(
                                    emotion=emotion,
                                    confidence_scores=confidence_scores,
                                    playlist_name=playlist["name"],
                                    playlist_url=playlist["url"],
                                    rating=1
                                )
                                if success:
                                    st.success("Thanks for the feedback!", icon="✅")
                        
                        with feedback_col2:
                            if st.button("👎", key=f"dislike_{idx}"):
                                from feedback_manager import log_feedback
                                success = log_feedback(
                                    emotion=emotion,
                                    confidence_scores=confidence_scores,
                                    playlist_name=playlist["name"],
                                    playlist_url=playlist["url"],
                                    rating=-1
                                )
                                if success:
                                    st.info("Feedback noted!", icon="ℹ️")
            else:
                st.warning("⚠️ No playlists found. Please check your Spotify API credentials in the .env file.")
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model file not found. Please ensure fer_model.h5 is in the Model directory.")
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")


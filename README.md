# Music Recommendation System using Facial Emotion Recognition

This project is a web application that recommends music playlists based on the user's facial emotions. The system captures or accepts an uploaded image, detects the emotion displayed in the facial expression, and then recommends Spotify playlists that match the detected mood.

## Features

- Facial emotion recognition (7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Real-time webcam capture for emotion detection
- Image upload capability
- Spotify playlist recommendations based on detected emotions
- Visual confidence scores for emotion detection
- Streamlit-based user interface

## Technologies Used

- Python
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- Streamlit for web application
- Spotipy (Spotify API wrapper for Python)

## Installation

1. Clone this repository:
```
git clone https://github.com/your-username/music-recommendation-emotion-recognition.git
cd music-recommendation-emotion-recognition
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up Spotify API credentials:
   - Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com)
   - Create a new application to get your Client ID and Client Secret
   - Update the `CLIENT_ID` and `CLIENT_SECRET` variables in `spotify_auth.py`

4. Download the pre-trained model (if not included in the repository):
   - The trained model file `fer_model.h5` should be placed in the `Model` directory
   - If you want to retrain the model yourself, see the Dataset section below

## Dataset

This project uses the FER2013 dataset available on Kaggle:
[FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

To retrain the model:
1. Download the dataset from the Kaggle link above (requires a Kaggle account)
2. Extract and organize the dataset with the following structure:
```
dataset/
   train/
      Angry/
      Disgust/
      Fear/
      Happy/
      Neutral/
      Sad/
      Surprise/
   test/
      Angry/
      Disgust/
      Fear/
      Happy/
      Neutral/
      Sad/
      Surprise/
```
3. Run the training script: `python emotion_recognition.py`

## Usage

1. Ensure the pre-trained model `fer_model.h5` is in the correct location (`Model` directory)

2. Run the Streamlit application:
```
streamlit run streamlit_app.py
```

3. Access the web interface at `http://localhost:8501`

4. Choose to either upload an image or capture using your webcam

5. View your emotion analysis and recommended Spotify playlists

## Project Structure

- `emotion_recognition.py`: Script for training the CNN model on facial emotion dataset
- `emotion_detector.py`: Functions for emotion prediction using the trained model
- `spotify_auth.py`: Spotify API authentication
- `spotify_recommendation.py`: Playlist recommendation based on emotions
- `streamlit_app.py`: Main application with user interface
- `Model/fer_model.h5`: Pre-trained facial emotion recognition model

## Future Improvements

- Add real-time emotion tracking from video
- Improve model accuracy with more training data
- Implement user feedback system for playlist recommendations
- Add option to play music directly within the application
- Support for multiple music streaming platforms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Thanks to the Spotify API for playlist data
- Credits to the creators of the FER2013 dataset used for training

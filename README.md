# 🎵 Music Recommendation System using Facial Emotion Recognition

> An intelligent music recommendation system that analyzes facial expressions to suggest personalized song recommendations. Built with deep learning, probabilistic modeling, and a continuous feedback loop.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pratikjadhav27-music-recommendation-system-using-fac-app-bfderb.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Training Your Own Models](#training-your-own-models)
- [Project Structure](#project-structure)
- [Methodology & ML Approach](#methodology--ml-approach)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## 🎯 Overview

This project demonstrates **end-to-end machine learning** for emotion recognition and intelligent recommendation systems. Unlike typical "plug-and-play" ML projects, this system:

- **Trains custom CNN models** from scratch on the FER-2013 dataset
- **Handles real-world ML challenges** like class imbalance (Disgust: 1.5% of data)
- **Uses probabilistic modeling** instead of simple argmax classification
- **Implements a feedback loop** to collect user preferences for continuous improvement

---

## ✨ Key Features

### 🧠 **Advanced Emotion Recognition**
- **7 Emotions Detected**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Multiple Model Architectures**:
  - Baseline CNN (69.16% accuracy)
  - MobileNetV2 Transfer Learning (65% accuracy, 3x smaller)
  - Balanced CNN with Weighted Loss (66.52% accuracy, +20% Disgust recall)
- **Real-time Webcam & Image Upload Support**
- **Face Detection & Crop**: OpenCV Haar cascades find the largest face before 48×48 grayscale resize (toggle off for whole-image mode). Live webcam prefers a face crop and falls back to the full frame if none is found.
- **Low-Confidence Guard**: If the top emotion score is below a configurable threshold (default 40%), song recommendations stay hidden until you explicitly opt in — reducing misleading playlists on uncertain predictions.

### 🎶 **Intelligent Song Recommendations**
- **Probabilistic Mapping**: Blends songs based on emotion distribution (e.g., 70% Happy + 30% Sad)
- **iTunes API Integration**: Fetches real songs with album art — no account or API key required
- **30-second Audio Previews**: Listen to song snippets directly inside the app
- **Genre Diversity**: 5+ mood-based search terms per emotion for variety

### 🔍 **Explainability (Grad-CAM)**
- **Grad-CAM Heatmap Overlay**: Visualize which facial regions influenced the model’s prediction
- **On-demand Toggle**: Generate the heatmap only when needed to keep the app fast

### 📊 **User Feedback System**
- **Like/Dislike Buttons**: Collect user preferences per song
- **Data Logging**: Stores feedback in `data/feedback.csv` for future model improvements
- **Continuous Learning**: Foundation for a personalized recommendation engine

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Streamlit Web App)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌────────▼──────────┐
│  Image Input   │       │  Feedback Loop    │
│  (Upload/Cam)  │       │  (Like/Dislike)   │
└───────┬────────┘       └────────┬──────────┘
        │                         │
        │                         ▼
        │                ┌────────────────────┐
        │                │ feedback_manager.py│
        │                │  (CSV Logging)     │
        │                └────────────────────┘
        │
┌───────▼─────────────────────────────────────────┐
│         EMOTION DETECTION PIPELINE              │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. Preprocessing (48x48 grayscale)      │  │
│  │ 2. Model Inference (CNN/MobileNetV2)    │  │
│  │ 3. Softmax Probabilities (7 emotions)   │  │
│  └──────────────────────────────────────────┘  │
└───────┬─────────────────────────────────────────┘
        │
        │ {happy: 0.70, sad: 0.25, ...}
        │
┌───────▼─────────────────────────────────────────┐
│    PROBABILISTIC SONG RECOMMENDATION            │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. Identify Dominant Emotion (70%)      │  │
│  │ 2. Identify Secondary Emotion (25%)     │  │
│  │ 3. Fetch 3 songs  (Dominant  genre)    │  │
│  │ 4. Fetch 2 songs  (Secondary genre)    │  │
│  │ 5. Shuffle & Display with Preview      │  │
│  └──────────────────────────────────────────┘  │
└───────┬─────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│         iTunes Search API Integration          │
│  (Free · No key required · Album art + audio) │
└────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

### Baseline CNN (Custom Architecture)
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 69.16% |
| **Best Emotion** | Happy (F1: 0.89) |
| **Worst Emotion** | Fear (F1: 0.48) |
| **Model Size** | 86 MB |

### Balanced CNN (Class Weighted Loss)
| Metric | Value | Change |
|--------|-------|--------|
| **Overall Accuracy** | 66.52% | -2.6% |
| **Disgust Recall** | 72.07% | **+19.8%** ✅ |
| **Fear Recall** | 41.99% | +1.7% |
| **Model Size** | 86 MB | - |

**Key Insight**: Class weighting successfully improved minority class performance (Disgust) at an acceptable accuracy tradeoff.

### Per-Emotion Performance (Baseline)
| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Happy | 0.88 | 0.90 | 0.89 |
| Surprise | 0.78 | 0.81 | 0.80 |
| Disgust | 0.72 | 0.52 | 0.60 |
| Angry | 0.62 | 0.63 | 0.62 |
| Neutral | 0.59 | 0.72 | 0.65 |
| Sad | 0.59 | 0.58 | 0.58 |
| Fear | 0.61 | 0.40 | 0.48 |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

> **No API keys or external accounts required.** Song recommendations use the iTunes Search API which is completely free and open.

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/PratikJadhav27/Music-Recommendation-System-Using-Facial-Emotion-Recognition.git
cd Music-Recommendation-System-Using-Facial-Emotion-Recognition
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Ensure the pre-trained model is present**
   - The `Model/fer_model.h5` file should be present in the repo
   - If missing, see [Training Your Own Models](#training-your-own-models)

---

## 💻 Usage

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the App
1. **Choose Input Method**: Upload an image, take a single webcam snapshot, or use **Live Webcam (real-time)** for continuous video with an on-screen emotion overlay
2. **Capture/Upload**: Take a photo or select an image file (or click **START** in live mode and allow camera access)
3. **View Results**: See your detected emotion with confidence scores (updated live while streaming)
4. **Explore Songs**: Browse recommended songs with album art and 30-second audio previews (after upload/snapshot, or after you stop the live stream — songs use the last detected emotion)
5. **Provide Feedback**: Click 👍 or 👎 to help improve recommendations

**Live webcam note:** Real-time mode uses [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc). On some corporate or strict networks, WebRTC may need a TURN server (see the library docs). [Streamlit Community Cloud](https://streamlit.io/cloud) serves over HTTPS, which is required for camera access on the public internet.

---

## 🎓 Training Your Own Models

### 1. Download the FER-2013 Dataset
- Visit [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Download and extract to `dataset/` directory
- Expected structure:
```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```

### 2. Train Models
```bash
cd training

# Baseline CNN
python emotion_recognition.py

# Transfer Learning (MobileNetV2)
python train_transfer_learning.py

# Balanced Model (Class Weighting)
python train_balanced.py
```

### 3. Evaluate Models
```bash
# Evaluate baseline
python evaluate_model.py --model baseline

# Evaluate balanced
python evaluate_model.py --model balanced

# Evaluate MobileNetV2
python evaluate_model.py --model mobilenet
```

See [`training/README.md`](training/README.md) for detailed instructions.

---

## 📁 Project Structure

```
Music-Recommendation-System-Using-Facial-Emotion-Recognition/
├── training/                      # Model training scripts
│   ├── emotion_recognition.py     # Baseline CNN training
│   ├── train_transfer_learning.py # MobileNetV2 training
│   ├── train_balanced.py          # Balanced CNN training
│   ├── evaluate_model.py          # Model evaluation
│   └── README.md                  # Training documentation
├── Model/                         # Trained models & artifacts
│   ├── fer_model.h5               # Baseline model
│   ├── fer_balanced.h5            # Balanced model
│   ├── fer_mobilenet.h5           # MobileNetV2 model
│   ├── training_history.png       # Training curves
│   └── confusion_matrix.png       # Evaluation results
├── data/                          # User feedback logs (gitignored)
│   └── feedback.csv               # Like/Dislike data
├── emotion_detector.py            # Emotion prediction logic
├── face_preprocess.py             # Haar face detect → FER 48×48 input
├── realtime_webcam.py             # Live WebRTC video + on-frame emotion overlay
├── spotify_recommendation.py      # Probabilistic song recommendation (iTunes API)
├── feedback_manager.py            # Feedback logging system
├── app.py                         # Main Streamlit web application
├── requirements.txt               # Python dependencies
└── TRAINING_ANALYSIS.md           # Detailed model analysis
```

---

## 🧪 Methodology & ML Approach

### 1. **Model Training**
- **Architecture**: Custom CNN with BatchNormalization and Dropout
- **Data Augmentation**: Rotation, zoom, shift, flip (prevents overfitting)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Transfer Learning**: MobileNetV2 (ImageNet weights) for comparison

### 2. **Handling Class Imbalance**
- **Problem**: Disgust emotion only 1.5% of dataset
- **Solution**: Weighted categorical crossentropy (inverse frequency)
- **Result**: +20% recall improvement for Disgust

### 3. **Probabilistic Modeling**
- **Traditional Approach**: `argmax(predictions)` → single emotion
- **Our Approach**: Use full softmax distribution
- **Benefit**: Captures nuanced emotional states (e.g., "70% Happy, 30% Nostalgic")

### 4. **Feedback Loop**
- **Data Collection**: User ratings (Like/Dislike) logged to CSV per song
- **Future Use**: Train a personalization model (Phase 5)
- **ML Maturity**: Demonstrates understanding of data-centric AI

---

## ⚠️ Limitations & Future Work

### Known Limitations
1. **Dataset Bias**: FER-2013 is grayscale webcam images (limited diversity)
2. **Lighting Sensitivity**: Performance degrades in poor lighting
3. **Cultural Differences**: Emotion expression varies across cultures
4. **Fear/Sad Confusion**: These emotions share subtle facial features (40% recall for Fear)

### Future Improvements
- [ ] **Real-time Video Tracking**: Analyze emotion over time (temporal modeling)
- [ ] **Personalization Model**: Use feedback data to train a preference predictor
- [ ] **Multi-modal Input**: Combine facial expressions with voice tone
- [ ] **Explainability**: Add Grad-CAM visualizations to show which facial regions influenced the prediction
- [x] **Deployment**: Live app deployed on [Streamlit Community Cloud](https://pratikjadhav27-music-recommendation-system-using-fac-app-bfderb.streamlit.app)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **FER-2013 Dataset**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **iTunes Search API**: [Apple Developer](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/)
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Rapid web app development

---

## 📧 Contact

**Pratik Jadhav**  
GitHub: [@PratikJadhav27](https://github.com/PratikJadhav27)

---

**⭐ If you found this project helpful, please consider giving it a star!**

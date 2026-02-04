# 🎵 Music Recommendation System using Facial Emotion Recognition

> An intelligent music recommendation system that analyzes facial expressions to suggest personalized Spotify playlists. Built with deep learning, probabilistic modeling, and a continuous feedback loop.

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

**Perfect for ML/AI interviews** - demonstrates deep understanding of model training, evaluation, and deployment.

---

## ✨ Key Features

### 🧠 **Advanced Emotion Recognition**
- **7 Emotions Detected**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Multiple Model Architectures**:
  - Baseline CNN (69.16% accuracy)
  - MobileNetV2 Transfer Learning (65% accuracy, 3x smaller)
  - Balanced CNN with Weighted Loss (66.52% accuracy, +20% Disgust recall)
- **Real-time Webcam & Image Upload Support**

### 🎶 **Intelligent Playlist Recommendations**
- **Probabilistic Mapping**: Mixes playlists based on emotion distribution (e.g., 70% Happy + 30% Sad)
- **Spotify Integration**: Fetches real playlists via Spotify API
- **Genre Diversity**: 4+ genres per emotion for variety

### 📊 **User Feedback System**
- **Like/Dislike Buttons**: Collect user preferences
- **Data Logging**: Stores feedback in `data/feedback.csv` for future model improvements
- **Continuous Learning**: Foundation for personalized recommendation engine

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
│    PROBABILISTIC PLAYLIST RECOMMENDATION        │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. Identify Dominant Emotion (70%)      │  │
│  │ 2. Identify Secondary Emotion (25%)     │  │
│  │ 3. Fetch 3 playlists (Dominant genre)   │  │
│  │ 4. Fetch 2 playlists (Secondary genre)  │  │
│  │ 5. Shuffle & Display                    │  │
│  └──────────────────────────────────────────┘  │
└───────┬─────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│          SPOTIFY API INTEGRATION               │
│  (Search playlists by genre, return metadata) │
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
- Spotify Developer Account ([Get one here](https://developer.spotify.com))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/PratikJadhav27/Music-Recommendation-System-Using-Facial-Emotion-Recognition.git
cd Music-Recommendation-System-Using-Facial-Emotion-Recognition
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up Spotify API credentials**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy your **Client ID** and **Client Secret**
   - Create a `.env` file in the project root:
   ```env
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

5. **Download the pre-trained model** (if not included)
   - The `Model/fer_model.h5` file should be present
   - If missing, see [Training Your Own Models](#training-your-own-models)

---

## 💻 Usage

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the App
1. **Choose Input Method**: Upload an image or use your webcam
2. **Capture/Upload**: Take a photo or select an image file
3. **View Results**: See your detected emotion with confidence scores
4. **Explore Playlists**: Browse recommended Spotify playlists
5. **Provide Feedback**: Click 👍 or 👎 to help improve recommendations

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
├── data/                          # User feedback logs
│   └── feedback.csv               # Like/Dislike data
├── emotion_detector.py            # Emotion prediction logic
├── spotify_auth.py                # Spotify API authentication
├── spotify_recommendation.py      # Probabilistic playlist logic
├── feedback_manager.py            # Feedback logging system
├── app.py               # Main web application
├── requirements.txt               # Python dependencies
├── .env                           # API credentials (not in repo)
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
- **Data Collection**: User ratings (Like/Dislike) logged to CSV
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
- [ ] **Deployment**: Dockerize the application for easy deployment

---

## 🎤 Interview Talking Points

**"Tell me about a challenging ML project you've worked on."**

> "I built an emotion-based music recommendation system from scratch. The interesting challenge was handling class imbalance - the 'Disgust' emotion was only 1.5% of the dataset, causing the model to ignore it. I implemented weighted loss functions, which improved Disgust recall by 20% at a 2.6% accuracy tradeoff. This taught me the importance of precision-recall tradeoffs in real-world ML."

**"How do you evaluate ML models?"**

> "I use multiple metrics: accuracy, per-class precision/recall, F1-scores, and confusion matrices. For this project, I discovered that 'Fear' and 'Sad' were frequently confused due to similar facial features. This led me to explore transfer learning with MobileNetV2, which provided better feature extraction."

**"How do you think about production ML systems?"**

> "I moved beyond simple classification to probabilistic modeling - instead of just saying 'you're happy', the system understands 'you're 70% happy and 30% nostalgic', and mixes playlists accordingly. I also built a feedback loop to collect user preferences, creating a dataset for future personalization models. This demonstrates data-centric AI thinking."

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **FER-2013 Dataset**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Spotify API**: [Spotify for Developers](https://developer.spotify.com/)
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Rapid web app development

---

## 📧 Contact

**Pratik Jadhav**  
GitHub: [@PratikJadhav27](https://github.com/PratikJadhav27)

---

**⭐ If you found this project helpful, please consider giving it a star!**

# Training Scripts

This directory contains scripts for training and evaluating the emotion recognition model.

## Prerequisites

1. **Download FER-2013 Dataset**
   - Go to [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download and extract the dataset
   - Place the `train` and `test` folders in `../dataset/` directory

2. **Install Dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

## Scripts

### `emotion_recognition.py`
Trains the emotion recognition CNN model from scratch.

**Features:**
- Data augmentation (rotation, zoom, flip, shift)
- Improved CNN architecture with BatchNormalization
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Training visualization (accuracy/loss plots)

**Usage:**
```bash
cd training
python emotion_recognition.py
```

**Outputs:**
- `../Model/fer_model.h5` - Trained model
- `../Model/training_history.png` - Training curves

---

### `evaluate_model.py`
Evaluates the trained model on the test set.

**Features:**
- Classification report (Precision, Recall, F1 per emotion)
- Confusion matrix visualization
- Per-class accuracy breakdown
- Most confused emotion pairs

**Usage:**
```bash
cd training
python evaluate_model.py
```

**Outputs:**
- `../Model/classification_report.txt` - Detailed metrics
- `../Model/confusion_matrix.png` - Visual confusion matrix

---

### `train_transfer_learning.py`
Trains a MobileNetV2 model using transfer learning.

**Why Transfer Learning?**
- Uses weights pre-trained on ImageNet (millions of images)
- Better feature extraction than training from scratch
- Faster convergence and better generalizability

**Usage:**
```bash
cd training
python train_transfer_learning.py
```

**Outputs:**
- `../Model/fer_mobilenet.h5` - Trained MobileNetV2 model
- `../Model/training_history_mobilenet.png` - Training curves

---

## Expected Directory Structure

```
Music-Recommendation-System-Using-Facial-Emotion-Recognition/
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
├── training/
│   ├── emotion_recognition.py
│   ├── evaluate_model.py
│   └── README.md (this file)
└── Model/
    ├── fer_model.h5
    ├── training_history.png
    ├── classification_report.txt
    └── confusion_matrix.png
```

---

## Training Tips

1. **GPU Recommended**: Training on CPU will be slow. Use Google Colab if you don't have a GPU.
2. **Epochs**: Start with 25-30 epochs. EarlyStopping will prevent overfitting.
3. **Batch Size**: Adjust based on your GPU memory (32, 64, or 128).
4. **Monitoring**: Watch validation accuracy. If it plateaus early, the model might need tuning.

---

## Next Steps After Training

1. Run `evaluate_model.py` to get detailed metrics
2. Analyze the confusion matrix to understand which emotions are confused
3. Update the main app to use the new model
4. Consider Phase 2: Transfer Learning (MobileNet/ResNet) for better accuracy

"""
Model Evaluation Script
========================
This script evaluates the trained emotion recognition model on the test set.

Outputs:
- Classification Report (Precision, Recall, F1 per emotion)
- Confusion Matrix (visual and numerical)
- Overall accuracy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
MODEL_PATH = os.path.join("..", "Model", "fer_model.h5")
TEST_DIR = os.path.join("..", "dataset", "test")
OUTPUT_DIR = os.path.join("..", "Model")

# Parameters
IMG_SIZE = 48
BATCH_SIZE = 64

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Model path: {MODEL_PATH}")
print(f"Test directory: {TEST_DIR}")
print("=" * 60)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model file not found at {MODEL_PATH}")
    print("Please train the model first using emotion_recognition.py")
    exit(1)

# Check if test directory exists
if not os.path.exists(TEST_DIR):
    print(f"\nERROR: Test directory not found at {TEST_DIR}")
    print("Please ensure the FER-2013 dataset is properly set up")
    exit(1)

# Load the trained model
print("\nLoading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Prepare test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Important: don't shuffle for evaluation
)

print(f"\nTest samples: {test_generator.samples}")
print(f"Class indices: {test_generator.class_indices}")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Calculate overall accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"\n{'=' * 60}")
print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'=' * 60}")

# Generate classification report
print("\nCLASSIFICATION REPORT:")
print("-" * 60)
report = classification_report(
    true_classes, 
    predicted_classes, 
    target_names=EMOTIONS,
    digits=4
)
print(report)

# Save classification report to file
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, 'w') as f:
    f.write("EMOTION RECOGNITION MODEL - CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
print(f"\nClassification report saved to: {report_path}")

# Generate confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Emotion Recognition', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Save confusion matrix
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")

# Calculate and display per-class accuracy
print("\nPER-CLASS ACCURACY:")
print("-" * 60)
for i, emotion in enumerate(EMOTIONS):
    class_correct = cm[i, i]
    class_total = np.sum(cm[i, :])
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"{emotion.capitalize():12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_correct}/{class_total} samples")

# Identify most confused emotion pairs
print("\nMOST CONFUSED EMOTION PAIRS:")
print("-" * 60)
confusion_pairs = []
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        if i != j:
            confusion_pairs.append((cm[i, j], EMOTIONS[i], EMOTIONS[j]))

confusion_pairs.sort(reverse=True)
for count, true_emotion, pred_emotion in confusion_pairs[:5]:
    if count > 0:
        print(f"{true_emotion.capitalize()} → {pred_emotion.capitalize()}: {count} times")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"\nResults saved in: {OUTPUT_DIR}")
print("- classification_report.txt")
print("- confusion_matrix.png")

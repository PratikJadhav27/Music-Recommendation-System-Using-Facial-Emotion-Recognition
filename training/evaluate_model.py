"""
Model Evaluation Script
========================
This script evaluates the trained emotion recognition model on the test set.

Outputs:
- Classification Report (Precision, Recall, F1 per emotion)
- Confusion Matrix (visual and numerical)
- Overall accuracy

Usage:
    python evaluate_model.py                     # Evaluates default baseline model
    python evaluate_model.py --model mobilenet   # Evaluates MobileNetV2 model
    python evaluate_model.py --model balanced    # Evaluates Balanced model
    python evaluate_model.py --path ../Model/my_model.h5
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths (Relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Model")

# Default Models
MODELS = {
    "baseline": os.path.join(OUTPUT_DIR, "fer_model.h5"),
    "mobilenet": os.path.join(OUTPUT_DIR, "fer_mobilenet.h5"),
    "balanced": os.path.join(OUTPUT_DIR, "fer_balanced.h5")
}

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Models")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenet", "balanced"],
                        help="Select which standard model to evaluate")
    parser.add_argument("--path", type=str, help="Path to a custom model file (overrides --model)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine model path
    if args.path:
        MODEL_PATH = args.path
        model_name = "custom_model"
    else:
        MODEL_PATH = MODELS[args.model]
        model_name = args.model

    # Parameters
    IMG_SIZE = 48
    BATCH_SIZE = 64

    print("=" * 60)
    print(f"MODEL EVALUATION: {model_name.upper()}")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Test directory: {TEST_DIR}")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model file not found at {MODEL_PATH}")
        print("Please train the model first.")
        if args.model == "baseline":
            print("Run: python emotion_recognition.py")
        elif args.model == "mobilenet":
            print("Run: python train_transfer_learning.py")
        elif args.model == "balanced":
            print("Run: python train_balanced.py")
        return

    # Check if test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"\nERROR: Test directory not found at {TEST_DIR}")
        print("Please ensure the FER-2013 dataset is properly set up in ../dataset/test")
        return

    # Load the trained model
    print("\nLoading trained model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Prepare test data
    # Note: MobileNetV2 expects RGB (3 channels), Baseline expects Grayscale (1 channel)
    # We infer this from the model input shape
    input_shape = model.input_shape[1:] # (48, 48, 1) or (48, 48, 3)
    channels = input_shape[2]
    
    color_mode = 'grayscale' if channels == 1 else 'rgb'
    print(f"Model expects input shape: {input_shape} -> Using color_mode: {color_mode}")

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode=color_mode,
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
    report_filename = f"classification_report_{model_name}.txt"
    report_path = os.path.join(OUTPUT_DIR, report_filename)
    with open(report_path, 'w') as f:
        f.write(f"EMOTION RECOGNITION MODEL ({model_name.upper()}) - CLASSIFICATION REPORT\n")
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
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save confusion matrix
    cm_filename = f"confusion_matrix_{model_name}.png"
    cm_path = os.path.join(OUTPUT_DIR, cm_filename)
    # plt.savefig(cm_path, dpi=300, bbox_inches='tight') # Commented out to prevent path errors if backend missing
    try:
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")

    # Calculate and display per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("-" * 60)
    for i, emotion in enumerate(EMOTIONS):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"{emotion.capitalize():12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_correct}/{class_total} samples")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

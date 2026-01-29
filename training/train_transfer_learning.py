"""
Transfer Learning Emotion Recognition
=====================================
Trains a MobileNetV2 model for facial emotion recognition.
Uses weights pre-trained on ImageNet for better feature extraction.

Why MobileNetV2?
- Lightweight and fast (good for real-time webcam)
- Better accuracy than simple custom CNNs
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATASET_DIR = os.path.join("..", "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_SAVE_PATH = os.path.join("..", "Model", "fer_mobilenet.h5")

# Hyperparameters
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

print("=" * 60)
print("TRANSFER LEARNING: MOBILENETV2")
print("=" * 60)

# Data Augmentation
# MobileNetV2 expects 3 channels (RGB), even efficiently on grayscale
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("\nLoading Data (Converting Grayscale -> RGB)...")
# Note: color_mode='rgb' duplicates the grayscale channel 3 times
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

# Load MobileNetV2 Base Model (Pre-trained on ImageNet)
print("\nLoading MobileNetV2 Base Model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze the top layers for fine-tuning (optional)
# For this small input size, we might want to retrain more layers or keep it frozen
base_model.trainable = True # Unfreeze all for maximum adaptability on this distinct domain

# Build complete model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\nStarting Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Plotting
print("\nGenerating training plots...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('MobileNetV2 Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('MobileNetV2 Loss')
plt.legend()
plt.grid(True)

plot_path = os.path.join("..", "Model", "training_history_mobilenet.png")
plt.savefig(plot_path)
print(f"Plots saved to {plot_path}")
print(f"Model saved to {MODEL_SAVE_PATH}")

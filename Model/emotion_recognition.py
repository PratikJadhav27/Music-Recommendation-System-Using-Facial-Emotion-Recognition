import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path
data_dir = "dataset"  # Change this to the actual dataset path
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Define emotion categories
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
num_classes = len(emotions)

def load_data(data_dir):
    images = []
    labels = []
    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_dir, emotion)
        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (48, 48))  # Ensure correct size
            images.append(img)
            labels.append(label)
    images = np.array(images) / 255.0  # Normalize pixel values
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    labels = to_categorical(labels, num_classes=num_classes)
    return images, labels

# Load training and testing data
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# Print dataset shape
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Define CNN model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Save the model
model.save("fer_model.h5")


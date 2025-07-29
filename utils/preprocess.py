# utils/preprocess.py

import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Emotion names mapped to index
emotion_map = {
    0: "angry",
    1: "happy",
    2: "neutral",
    3: "sad",
    4: "surprise",
    5: "fear"  # 👈 Yeh add karo!
}


# Load images
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for emotion_name in os.listdir(folder_path):
        emotion_folder = os.path.join(folder_path, emotion_name)
        if not os.path.isdir(emotion_folder):
            continue
        label_index = emotion_map[emotion_name.lower()]
        for file in os.listdir(emotion_folder):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(emotion_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_index)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels, num_classes=7):
    images = images.astype('float32') / 255.0
    images = images.reshape(-1, 48, 48, 1)
    labels = to_categorical(labels, num_classes)
    return images, labels

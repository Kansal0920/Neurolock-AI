import sys
import os
import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# ✅ Path Fix: Add utils/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from preprocess import emotion_map

# ✅ Load Trained Model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model.h5'))
model = load_model(model_path)

# ✅ Initialize Pygame for voice alert
pygame.init()
pygame.mixer.init()

def speak_alert(text):
    print(f"🔊 {text}")  # For debug
    pygame.mixer.music.load("voice_alert.mp3")  # You can pre-generate TTS to MP3 or use vosk in real-time
    pygame.mixer.music.play()

# ✅ Start Webcam (built-in macOS iSight)
cap = cv2.VideoCapture(0)  # 0 works for most macs' built-in cams

if not cap.isOpened():
    print("🚫 Could not access the webcam.")
    sys.exit()

print("🎥 Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert to grayscale and resize to 48x48 (model input)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    predictions = model.predict(face)
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class} → Emotion: {emotion_map.get(int(predicted_class), 'Unknown')}")
    emotion_label = emotion_map[int(predicted_class)]

    # Display on screen
    cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('NEUROLOCK AI - Face Emotion Scanner 🔐🧠', frame)

    # Optional Alert
    if emotion_label == 'angry':  # 🔥 trigger security voice alert
        speak_alert("Suspicious Emotion Detected!")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()

import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_face
from app.voice_feedback import speak
import getpass

# Load the trained CNN model
model = load_model("models/cnn_model.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
last_emotion = None
last_spoken_time = 0

# 🎯 Step 1: Password authentication
password = getpass.getpass("🔐 Enter NEUROLOCK access password: ")
if password != "admin@123":  # You can change this
    print("❌ Incorrect password! Exiting.")
    exit()

print("✅ Password accepted.\n🔐 NEUROLOCK AI is now running... Press 'q' to quit.")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = preprocess_face(gray, x, y, w, h)
        if face_img is None:
            continue

        face_img = np.reshape(face_img, (1, 48, 48, 1))  # Only one batch, 48x48 image, 1 channel
        prediction = model.predict(face_img, verbose=0)

        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

        # 🎯 Speak only once for each new emotion, wait 2 seconds
        current_time = time.time()
        if emotion != last_emotion or (current_time - last_spoken_time) > 2:
            speak(f"Emotion detected: {emotion}")
            last_emotion = emotion
            last_spoken_time = current_time

    cv2.imshow("NEUROLOCK - Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

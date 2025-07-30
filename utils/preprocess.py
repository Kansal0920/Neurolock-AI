# utils/preprocess.py

import cv2
import numpy as np

def preprocess_face(gray_frame, x, y, w, h):
    # Crop face region with some padding to capture full face
    offset = 10
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(gray_frame.shape[1], x + w + offset)
    y2 = min(gray_frame.shape[0], y + h + offset)

    # Crop face
    face = gray_frame[y1:y2, x1:x2]

    # Resize to 48x48 as required by CNN
    face = cv2.resize(face, (48, 48))

    # Normalize to range [0, 1]
    face = face.astype("float32") / 255.0

    # Expand dimensions to match CNN input shape: (1, 48, 48, 1)
    face = np.expand_dims(face, axis=-1)  # shape (48, 48, 1)
    face = np.expand_dims(face, axis=0)   # shape (1, 48, 48, 1)

    return face

from keras.models import load_model

try:
    model = load_model("models/cnn_model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model failed to load:", e)

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("asl_custom_model.h5")

# Load label map (same order as in training)
with open("label_map.txt", "r") as f:
    label_list = [line.strip() for line in f]

def predict_letter(image):
    try:
        resized = cv2.resize(image, (28, 28))
        normalized = resized.astype("float32") / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)
        prediction = model.predict(reshaped, verbose=0)
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_index])
        return label_list[predicted_index], confidence
    except Exception as e:
        print(f"[predict_letter error] {e}")
        return "?", 0.0

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === CONFIG ===
DATASET_PATH = "dataset"
IMG_SIZE = 28
# Added HELLO gesture class
CLASSES = ['A', 'B', 'L', 'V', 'Y', 'HELLO']
label_map = {label: idx for idx, label in enumerate(CLASSES)}

# === Load Data ===
images, labels = [], []

for label in CLASSES:
    path = os.path.join(DATASET_PATH, label)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label_map[label])

images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
labels = to_categorical(labels, num_classes=len(CLASSES))

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

# === Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(CLASSES), activation='softmax')  # Now outputs 6 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train ===
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# === Save Model + Label Map ===
model.save("asl_custom_model.h5")
with open("label_map.txt", "w") as f:
    for label in CLASSES:
        f.write(f"{label}\n")

print("\n✅ Model saved as `asl_custom_model.h5` with updated HELLO class")

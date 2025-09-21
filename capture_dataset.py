import cv2
import os
import time

# Added HELLO gesture class
CLASSES = ['A', 'B', 'L', 'V', 'Y', 'HELLO']
DATASET_PATH = "dataset"

# Create folders if they don't exist
for label in CLASSES:
    os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)

cap = cv2.VideoCapture(0)
current_class = "A"
count = 0

print("📸 Press 's' to save current image")
print("🔤 Press A, B, L, V, Y, H to switch class (H = HELLO)")
print("❌ Press ESC to exit\n")

# ROI parameters
ROI_SIZE = 300
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1 = FRAME_WIDTH // 2 - ROI_SIZE // 2
y1 = FRAME_HEIGHT // 2 - ROI_SIZE // 2
x2 = x1 + ROI_SIZE
y2 = y1 + ROI_SIZE

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Draw centered ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Class: {current_class} | Count: {count}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Capture (Press s to save)", frame)
    cv2.imshow("ROI Preview", gray_roi)

    key = cv2.waitKey(1) & 0xFF

    # Save image
    if key == ord('s'):
        save_path = os.path.join(DATASET_PATH, current_class)
        filename = f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), gray_roi)
        count += 1
        print(f"✅ Saved image to {current_class}/{filename}")

    # Change current letter/gesture
    elif chr(key).upper() in [c[0] for c in CLASSES]:
        if chr(key).upper() == "H":
            current_class = "HELLO"
        else:
            current_class = chr(key).upper()
        count = len(os.listdir(os.path.join(DATASET_PATH, current_class)))
        print(f"🔄 Switched to: {current_class}")

    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

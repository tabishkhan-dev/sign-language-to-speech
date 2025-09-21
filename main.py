import cv2
import time
import mediapipe as mp
from asl_classifier import predict_letter
from tts import speak

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Camera init with fail-safe ===
cap = cv2.VideoCapture(0)  # Change index if it opens iPhone camera
if not cap.isOpened():
    print("❌ ERROR: Unable to access the camera. Exiting...")
    exit(1)

# === ROI setup ===
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ROI_SIZE = 300
x1 = FRAME_WIDTH // 2 - ROI_SIZE // 2
y1 = FRAME_HEIGHT // 2 - ROI_SIZE // 2
x2 = x1 + ROI_SIZE
y2 = y1 + ROI_SIZE

# === Load labels (letters only) ===
with open("label_map.txt", "r") as f:
    valid_labels = [line.strip() for line in f if line.strip() != "HELLO"]

sentence = ""
last_letter = None
last_time = 0

# === Stability tracking ===
stable_letter = None
stable_start = 0
stable_required_duration = 1.0
recently_accepted = False
accept_pause_duration = 0.5

# === HELLO motion tracking ===
wave_start_x = None
wave_end_x = None
wave_start_time = None

# --- Stricter palm checks ---
def is_palm_facing(hand_landmarks):
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]
    horiz_dist = abs(index_mcp.x - pinky_mcp.x)
    vert_diff = abs(index_mcp.y - pinky_mcp.y)
    return horiz_dist > 0.2 and vert_diff < 0.1  # Wide and flat

def is_hand_open(hand_landmarks):
    extended_fingers = 0
    finger_tips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            extended_fingers += 1
    thumb_tip = hand_landmarks.landmark[4]
    palm_center_x = (hand_landmarks.landmark[5].x + hand_landmarks.landmark[17].x) / 2
    thumb_extended = abs(thumb_tip.x - palm_center_x) > 0.1
    return extended_fingers == 4 and thumb_extended

def detect_full_wave(wrist_x, hand_landmarks):
    global wave_start_x, wave_end_x, wave_start_time
    if not is_palm_facing(hand_landmarks) or not is_hand_open(hand_landmarks):
        wave_start_x = None
        return False
    current_time = time.time()
    if wave_start_x is None and wrist_x <= 20:
        wave_start_x = wrist_x
        wave_end_x = wrist_x
        wave_start_time = current_time
        return False
    if wave_start_x is not None:
        wave_end_x = wrist_x
        if wave_end_x >= ROI_SIZE - 20 and (current_time - wave_start_time) <= 1.0:
            wave_start_x = None
            wave_end_x = None
            wave_start_time = None
            return True
        if wrist_x < wave_start_x or (current_time - wave_start_time) > 1.0:
            wave_start_x = None
            wave_end_x = None
            wave_start_time = None
    return False

# === Font ===
font = cv2.FONT_HERSHEY_SIMPLEX

print("🖐️ Show A, B, L, V, Y in the box, or open-palm wave left→right for HELLO")
print("⌨️ SPACE = space | S = speak | D = delete | ESC = exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read from camera. Exiting...")
        break
    frame = cv2.flip(frame, 1)
    overlay = frame.copy()
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_roi)

    now = time.time()
    letter, confidence = "?", 0.0
    if recently_accepted and now - last_time > accept_pause_duration:
        recently_accepted = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist_x_global = int(hand_landmarks.landmark[0].x * FRAME_WIDTH)
            wrist_x_local = wrist_x_global - x1
            if detect_full_wave(wrist_x_local, hand_landmarks) and not recently_accepted:
                sentence += "Hello "
                last_letter = "HELLO"
                last_time = now
                recently_accepted = True
                print("👋 HELLO detected")
                speak("Hello")
            else:
                letter, confidence = predict_letter(gray_roi)
                if not recently_accepted and letter in valid_labels:
                    if letter != stable_letter:
                        stable_letter = letter
                        stable_start = now
                    if now - stable_start >= stable_required_duration and confidence > 0.70:
                        sentence += letter
                        last_letter = letter
                        last_time = now
                        stable_letter = None
                        recently_accepted = True
            mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        stable_letter = None
        wave_start_x = None
        wave_end_x = None
        wave_start_time = None

    # --- ROI overlay ---
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)

    # --- Prediction info ---
    cv2.putText(frame, f"Prediction: {letter}", (10, 50), font, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), font, 1.2, (255, 255, 255), 3)

    # --- Confidence bar ---
    bar_width = int(confidence * 300)
    cv2.rectangle(frame, (10, 110), (10 + bar_width, 150), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, 110), (310, 150), (255, 255, 255), 3)

    # --- Stability indicator ---
    if stable_letter:
        held = now - stable_start
        cv2.putText(frame, f"Holding '{stable_letter}'... {held:.1f}s", (10, 190), font, 1.0, (200, 200, 0), 3)

    # --- Flash accepted gesture ---
    if now - last_time < 1.0:
        cv2.putText(frame, f"+{last_letter}", (FRAME_WIDTH // 2, 150), font, 2.5, (0, 255, 0), 5)

    # === Sentence display (top right) ===
    box_x1, box_y1 = FRAME_WIDTH - 480, 20
    box_x2, box_y2 = FRAME_WIDTH - 20, 90
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), -1)
    cv2.putText(frame, f"Sentence: {sentence}", (box_x1 + 10, 65), font, 1.2, (0, 0, 0), 5)  # Shadow
    cv2.putText(frame, f"Sentence: {sentence}", (box_x1 + 10, 65), font, 1.2, (0, 255, 255), 3)

    # === Bottom instruction bar ===
    bar_height = 60
    cv2.rectangle(frame, (0, FRAME_HEIGHT - bar_height), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    cv2.putText(frame, "SPACE = space  |  S = speak  |  D = delete  |  ESC = exit",
                (15, FRAME_HEIGHT - 18), font, 1.0, (0, 0, 0), 5)  # Shadow
    cv2.putText(frame, "SPACE = space  |  S = speak  |  D = delete  |  ESC = exit",
                (15, FRAME_HEIGHT - 18), font, 1.0, (0, 255, 255), 3)

    cv2.imshow("ASL Detection", frame)

    # --- Keyboard actions ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        speak(sentence)
        sentence = ""
    elif key == ord('d'):
        sentence = sentence[:-1]
    elif key == ord(' '):
        sentence += ' '

cap.release()
cv2.destroyAllWindows()

def count_fingers(landmarks):
    """
    Count fingers using landmark positions.
    Returns a list: [thumb, index, middle, ring, pinky]
    """
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    fingers = []

    # Thumb (horizontal logic)
    if landmarks.landmark[4].x < landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (vertical logic)
    for i in range(1, 5):
        if landmarks.landmark[tips_ids[i]].y < landmarks.landmark[pip_ids[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def recognize_gesture(landmarks):
    """
    Recognize basic static gestures using landmark logic.
    Returns gesture name or None.
    """
    fingers = count_fingers(landmarks)

    # Landmarks needed for directional logic
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]
    wrist = landmarks.landmark[0]

    # Directional check for "Thumbs Up"
    is_thumb_up = (
        thumb_tip.y < wrist.y and
        index_tip.y > thumb_tip.y and
        middle_tip.y > thumb_tip.y and
        ring_tip.y > thumb_tip.y and
        pinky_tip.y > thumb_tip.y
    )

    if all(f == 0 for f in fingers):
        return "Fist"
    elif is_thumb_up:
        return "Thumbs Up"
    elif all(f == 1 for f in fingers):
        return "Open Palm"
    else:
        return None

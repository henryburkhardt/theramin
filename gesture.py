"""Real-time hand gesture recognition for Jetson Orin Nano (MediaPipe Tasks API),
configured for a DJI Osmo Pocket 3 in webcam mode.

Needs the hand model next to this script:
  wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
PIPS = [6, 10, 14, 18]   # the joint below each tip


def fingers_up(lm, handedness):
    """Return [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down)."""
    # Thumb: compare x positions (direction depends on which hand it is)
    if handedness == "Right":
        thumb = 1 if lm[4].x < lm[3].x else 0
    else:
        thumb = 1 if lm[4].x > lm[3].x else 0
    others = [1 if lm[t].y < lm[p].y else 0 for t, p in zip(TIPS, PIPS)]
    return [thumb] + others


def classify(f):
    patterns = {
        (0, 0, 0, 0, 0): "Fist",
        (1, 1, 1, 1, 1): "Open palm",
        (0, 1, 0, 0, 0): "Point",
        (0, 1, 1, 0, 0): "Peace",
        (1, 0, 0, 0, 0): "Thumbs up",
        (1, 1, 0, 0, 1): "Rock on",
    }
    return patterns.get(tuple(f), f"{sum(f)} fingers")


def draw_hand(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return pts


def main():
    if not os.path.exists(MODEL):
        raise SystemExit(f"Model not found at {MODEL} - see the wget command at the top of this file.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # DJI Osmo Pocket 3 webcam mode: MJPG, smallest size is 1280x720
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start = time.monotonic()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed - check /dev/video0")
                break
            frame = cv2.flip(frame, 1)  # mirror view (delete if hands come out swapped)
            frame = cv2.resize(frame, (640, 360))  # smaller frames = faster tracking

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for lm, hand_info in zip(result.hand_landmarks, result.handedness):
                pts = draw_hand(frame, lm)
                side = hand_info[0].category_name
                gesture = classify(fingers_up(lm, side))
                x, y = pts[0]
                cv2.putText(frame, f"{side}: {gesture}", (x - 60, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

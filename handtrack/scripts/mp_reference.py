#!/usr/bin/env python3
"""MediaPipe Tasks HandLandmarker (CPU) reference output for a recorded clip.

Runs in ~/gesture-env (it needs the mediapipe package, which handtrack-env
doesn't have), on full-resolution frames, VIDEO mode, 2 hands.

  ~/gesture-env/bin/python scripts/mp_reference.py clips/bench results/mp_ref_bench.npz
"""
import argparse
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "hand_landmarker.task")


def load_clip(path):
    base = path[:-5] if path.endswith(".mjpg") else path
    data = open(base + ".mjpg", "rb").read()
    with open(base + ".csv") as f:
        next(f)
        rows = [list(map(int, line.split(","))) for line in f]
    return [data[off:off + size] for _seq, _tc, _td, off, size in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--min-detection", type=float, default=0.5)
    args = ap.parse_args()
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    frames = load_clip(args.clip)
    lms = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
    hand = np.full((len(frames), 2), -1, np.int8)     # 0 Left, 1 Right, -1 none
    times = []
    with vision.HandLandmarker.create_from_options(opts) as lmk:
        for i, j in enumerate(frames):
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            t = time.perf_counter()
            r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), i * 33)
            times.append((time.perf_counter() - t) * 1e3)
            for k, (lm, hd) in enumerate(zip(r.hand_landmarks[:2], r.handedness[:2])):
                lms[i, k] = [(p.x * w, p.y * h, p.z * w) for p in lm]
                hand[i, k] = 1 if hd[0].category_name == "Right" else 0
    np.savez(args.out, landmarks_px=lms, handedness=hand, ms=np.array(times))
    t = np.array(times[10:])
    print(f"{len(frames)} frames, MediaPipe CPU per frame p50 {np.percentile(t, 50):.1f} ms "
          f"p95 {np.percentile(t, 95):.1f} ms; frames with hands: {(hand[:, 0] >= 0).sum()}")


if __name__ == "__main__":
    main()

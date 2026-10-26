#!/usr/bin/env python3
"""Build (or reuse cached) TensorRT FP16 engines for every ONNX model we use.

  python scripts/build_engines.py            # build what's missing
  python scripts/build_engines.py --force    # rebuild everything
"""
import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from handtrack.trt_runner import build_engine  # noqa: E402

# (onnx path relative to models/, max batch)
MODELS = [
    ("mediapipe/hand_detector.onnx", 1),
    ("mediapipe/hand_landmarks_detector.onnx", 2),
    ("rtmpose/rtmdet_nano_hand.onnx", 1),
    ("rtmpose/rtmpose_m_hand_256.onnx", 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    for rel, max_batch in MODELS:
        onnx_path = os.path.join(ROOT, "models", rel)
        if not os.path.exists(onnx_path):
            print(f"skip {rel}: not found")
            continue
        t = time.monotonic()
        path = build_engine(onnx_path, fp16=not args.fp32, max_batch=max_batch, force=args.force)
        print(f"{rel} -> {os.path.relpath(path, ROOT)} ({time.monotonic() - t:.1f} s)")


if __name__ == "__main__":
    main()

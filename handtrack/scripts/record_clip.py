#!/usr/bin/env python3
"""Record raw MJPG frames from the camera for repeatable benchmarks.

Writes <name>.mjpg (concatenated JPEGs, byte-identical to what the camera sent)
and <name>.csv (one row per frame: sequence, t_capture_ns, t_dequeue_ns, offset, size).

Shows a live, mirrored preview window with the countdown and a REC timer when a
display is available (DISPLAY set); press q or Esc in it to stop early. The
camera is read and written in a separate thread, so the preview never costs
recorded frames.

  python scripts/record_clip.py clips/still_hand --seconds 20
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from handtrack.capture import LatestFrame  # noqa: E402
from handtrack.v4l2 import V4L2Capture  # noqa: E402
from handtrack.viewer import ensure_display  # noqa: E402


def load_clip(path):
    """Returns a list of (sequence, t_capture_ns, jpeg_bytes)."""
    base = path[:-5] if path.endswith(".mjpg") else path
    data = open(base + ".mjpg", "rb").read()
    frames = []
    with open(base + ".csv") as f:
        next(f)
        for line in f:
            seq, t_cap, _t_deq, off, size = map(int, line.split(","))
            frames.append((seq, t_cap, data[off:off + size]))
    return frames


class Recorder:
    def __init__(self, args):
        self.args = args
        self.preview = LatestFrame()         # (status_text, jpeg) for the preview
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.n = self.bytes = 0

    def run(self):
        a = self.args
        try:
            with V4L2Capture(a.device, a.width, a.height, a.fps, buffers=4) as cap, \
                    open(a.name + ".mjpg", "wb") as fm, open(a.name + ".csv", "w") as fc:
                fc.write("sequence,t_capture_ns,t_dequeue_ns,offset,size\n")
                start = time.monotonic() + a.countdown
                end = start + a.seconds
                last_k = None
                while not self.stop.is_set():
                    r = cap.read()
                    if r is None:
                        raise RuntimeError("camera timed out")
                    now = time.monotonic()
                    if now >= end:
                        break
                    seq, t_cap, t_deq, jpeg = r
                    if now < start:
                        k = int(start - now) + 1
                        if k != last_k:
                            print(f"recording in {k}...", flush=True)
                            last_k = k
                        self.preview.put((f"recording in {k}", jpeg))
                        continue
                    if self.n == 0:
                        print(f"RECORDING {a.seconds:g} s", flush=True)
                    fm.write(jpeg)
                    fc.write(f"{seq},{t_cap},{t_deq},{self.bytes},{len(jpeg)}\n")
                    self.bytes += len(jpeg)
                    self.n += 1
                    self.preview.put((f"REC {now - start:4.1f} / {a.seconds:g} s", jpeg))
        except Exception as e:
            self.error = e
        finally:
            self.done.set()


def show_preview(rec, title):
    import cv2
    import numpy as np
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    version = 0
    while not rec.done.is_set():
        r = rec.preview.wait_newer(version, timeout=0.1)
        if r is not None:
            version, (status, jpeg) = r
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            img = cv2.flip(img, 1)                      # mirrored, like a mirror
            rec_on = status.startswith("REC")
            if rec_on:
                cv2.circle(img, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(img, status, (44 if rec_on else 12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if rec_on else (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(title, img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            rec.stop.set()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="output path without extension")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--countdown", type=int, default=3, help="seconds to get into position")
    ap.add_argument("--no-preview", action="store_true", help="don't open a preview window")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.name)), exist_ok=True)
    rec = Recorder(args)
    worker = threading.Thread(target=rec.run, daemon=True)
    worker.start()
    if not args.no_preview and ensure_display():
        try:
            show_preview(rec, f"record_clip: {os.path.basename(args.name)} (q to stop)")
        except Exception as e:                      # no usable display: record anyway
            print(f"preview unavailable ({e}); recording without it", flush=True)
    worker.join()
    if rec.error:
        sys.exit(f"recording failed: {rec.error}")
    if rec.n == 0:
        sys.exit("no frames recorded")
    print(f"wrote {rec.n} frames ({rec.bytes / rec.n / 1024:.0f} KB avg) to {args.name}.mjpg/.csv")


if __name__ == "__main__":
    main()

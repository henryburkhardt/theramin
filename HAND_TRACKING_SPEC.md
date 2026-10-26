# Spec: Real-Time GPU Hand Tracking on Jetson Orin Nano

## 1. Goal

Build the lowest-latency, most stable hand and finger tracker possible on a Jetson Orin Nano, running entirely on-device and using the GPU (CUDA / TensorRT) rather than the CPU.

The tracker will later drive a musical instrument (a theremin/synth-style controller where finger and hand positions map to pitch, volume, timbre, etc.). **No audio or music code is in scope yet.** The deliverable is the tracking engine plus a clean data interface that a future instrument module can consume.

Priorities, in order:

1. **Latency** – time from a frame arriving on the Jetson to landmarks being available to a consumer.
2. **Stability** – low jitter on fingertip positions when the hand is still; no flicker between hands, no dropped tracks during fast motion.
3. **Accuracy** – 21 keypoints per hand, comparable to or better than MediaPipe Hands.
4. **Throughput** – keep up with every camera frame; never build a queue.

## 2. Current state (already working)

- `~/theramin/gesture.py`: MediaPipe Tasks `HandLandmarker` (v0.10.33), CPU only (TFLite XNNPACK), threaded capture keeping only the newest frame, on-screen FPS / inference / frame-age readout. This is the **baseline** to beat. Do not delete or break it.
- Python venv at `~/gesture-env` (Python 3.12) with mediapipe, opencv (pip build, **no GStreamer support**), numpy<2, sounddevice.
- OS reported as Ubuntu 24.04, kernel `6.8.12-tegra` (aarch64). The exact JetPack / L4T version is not confirmed. See §4.
- Model file `~/theramin/hand_landmarker.task` is present.

## 3. Hardware

- **Compute:** Jetson Orin Nano developer kit, 8 GB. Run in MAXN SUPER power mode with `sudo jetson_clocks` for all benchmarks (clocks reset on reboot).
- **Camera:** DJI Osmo Pocket 3 in UVC webcam mode, on a tripod, at `/dev/video0` (`/dev/video1` is metadata only). Connected through a USB 2.0 hub on bus 001. Formats it offers:
  - MJPG: 1280x720 @ 30/25, 1920x1080 @ 30/25/24, portrait 720x1280 and 1080x1920, 3840x2160 @ 30/25/24
  - H264: same sizes; 3840x2160 also at 60/50/48
  - There is **no uncompressed (YUYV) mode**, so every frame must be decoded.
  - `v4l2-ctl` streaming at MJPG 1280x720 measured about 38 fps (nominal 30).
- **Display:** a monitor on the Jetson. The developer usually works over SSH / VS Code Remote. GUI windows from an SSH shell need `DISPLAY=:0` (or `:1`) and `XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority`, with a desktop session logged in on the Jetson.

## 4. First task: verify the environment, then report

Before writing tracking code, inspect the system and write the findings to `ENVIRONMENT.md`. Don't assume versions; check them:

- JetPack / L4T version (`cat /etc/nv_tegra_release`, `dpkg -l | grep -E "nvidia-jetpack|nvidia-l4t-core"`)
- CUDA, cuDNN, TensorRT versions; whether TensorRT Python bindings are importable from system Python 3.12
- `trtexec` location
- Availability of: VPI, nvJPEG / the NVJPG hardware JPEG decoder, GStreamer with `nvv4l2decoder` / `nvjpegdec`, `jetson-utils`
- Whether `onnxruntime-gpu` (with the TensorRT execution provider) is installable for this JetPack and Python version, for example from NVIDIA's / Jetson AI Lab's Jetson-specific pip indexes
- Current power mode (`sudo nvpmodel -q`), free disk space, RAM

Then propose the concrete toolchain (which inference runtime, which decode path) in a short plan **before** large installs. Use a new venv such as `~/handtrack-env`; create it with `--system-site-packages` if that's needed to reach the system TensorRT bindings. Do not modify system Python packages, reflash the device, or change JetPack.

## 5. Architecture

A pipelined, multi-threaded design where no stage blocks another and stale frames are always dropped, never queued.

```
[capture] -> [decode] -> [GPU preprocess] -> [detect/track + landmarks (TensorRT)] -> [postprocess + filter] -> [publish]
                                                                                                         \-> [optional viewer]
```

### 5.1 Capture and decode

- Read MJPG at 1280x720 @ 30 from `/dev/video0` via V4L2 with the smallest possible driver buffer count.
- Prefer **hardware JPEG decode** (NVJPG via GStreamer `nvv4l2decoder mjpeg=1` / `nvjpegdec`, nvJPEG, or jetson-utils) so decoded frames land in GPU memory without a CPU round trip. Fall back to CPU decode (OpenCV / libjpeg-turbo) only if hardware decode isn't available, and record the cost of that.
- Benchmark the alternative of H264 decode via NVDEC, and report whether it's lower latency than MJPG for this camera.
- Timestamp every frame at arrival with `time.monotonic_ns()`. Carry that timestamp through every stage.
- A capture thread holds only the latest frame; consumers always take the newest one.

### 5.2 Models: run a short bake-off

Evaluate at least these two candidates, both exported to ONNX and built as **TensorRT FP16 engines**:

**A. MediaPipe hand models on TensorRT** (preferred starting point). Use the palm detection and hand landmark (full) models from MediaPipe, converted TFLite -> ONNX (existing community conversions, e.g. PINTO model zoo, are fine if the license allows) -> TensorRT. Reimplement the pipeline around them: anchor decoding, NMS, rotated ROI from palm keypoints, affine crop, landmark projection back to image space, handedness output.

**B. RTMPose hand** (MMPose / rtmlib, 21-keypoint hand model) with a lightweight hand detector (e.g. RTMDet-nano hand) on TensorRT.

Optional C: a YOLO pose model trained on a 21-keypoint hand dataset (e.g. Ultralytics hand-keypoints). Only include it if pretrained hand weights are available; don't train on the Jetson.

Pick the winner on measured latency, jitter, and accuracy (§7), and document the decision in `DECISIONS.md`. Check and record each model's license.

### 5.3 Tracking strategy (critical for latency)

- Use **detect once, then track**: run the hand detector only when fewer than the expected number of hands are tracked or confidence drops. Otherwise derive the next frame's ROI from the previous frame's landmarks and run only the landmark model. This is how MediaPipe stays fast and should be preserved.
- Support up to **2 hands** (an instrument will commonly use both). When 2 hands are present, batch both ROIs into a single landmark inference call if the engine supports batch size 2.
- Keep stable **track IDs and handedness** across frames; don't let left/right labels swap when hands cross or briefly occlude.
- Handle the camera image being mirrored or not via a config flag.

### 5.4 GPU efficiency

- Do resize, normalization, and the affine ROI crop on the GPU (CUDA kernels via CuPy, VPI, or TensorRT plugins). Avoid host <-> device copies of full frames.
- Use pinned memory, CUDA streams, and preallocated buffers. No per-frame allocations in the hot path.
- FP16 engines by default. INT8 is a stretch goal only if accuracy holds up (measure it).
- Cache built TensorRT engines on disk; rebuild automatically if the TensorRT version changes.
- If Python overhead turns out to be a measurable share of latency, report it and propose a C++ or hybrid path rather than switching silently.

### 5.5 Filtering for instrument control

- Apply a **One Euro filter** per landmark coordinate (configurable `min_cutoff`, `beta`, `d_cutoff`). This trades jitter against lag, which is exactly the trade-off an instrument needs.
- Expose **both raw and filtered** landmarks to consumers.
- Optional, behind a flag: short-horizon **prediction** (constant velocity, or the filter's derivative) to compensate a configurable number of milliseconds of pipeline latency. Report how much it helps vs. how much overshoot it causes.

### 5.6 Output interface

Define a typed data structure published once per processed frame:

```python
@dataclass
class HandFrame:
    frame_id: int
    t_capture_ns: int        # when the frame arrived from the camera
    t_publish_ns: int        # when landmarks were ready
    hands: list[Hand]

@dataclass
class Hand:
    track_id: int
    handedness: str          # "Left" / "Right"
    score: float
    landmarks_px: np.ndarray        # (21, 3) raw, pixel x,y + relative z
    landmarks_norm: np.ndarray      # (21, 3) raw, normalized 0..1
    landmarks_filtered: np.ndarray  # (21, 3) One Euro filtered, normalized
    velocity: np.ndarray            # (21, 2) normalized units per second
```

- An in-process API: `tracker.subscribe(callback)` and/or `tracker.latest()`.
- An optional **UDP sender** (OSC or a compact binary/JSON format, chosen and documented) so a separate synth process can consume the stream later. Off by default. No audio code.

### 5.7 Viewer

- A separate, optional viewer thread showing the skeleton, track IDs, handedness, and live stats: camera fps, processing fps, per-stage ms, and end-to-end latency.
- The viewer must **never slow the tracker**. It renders the latest result at whatever rate it can, and the tracker runs at full speed with the viewer on or off.
- A `--headless` mode with no window, stats printed to the terminal once per second.

## 6. Repository layout (suggested)

```
~/theramin/handtrack/
  README.md            # setup, running, troubleshooting (incl. DISPLAY/XAUTHORITY)
  ENVIRONMENT.md       # findings from §4
  DECISIONS.md         # model choice, decode path, trade-offs, with numbers
  BENCHMARKS.md        # results tables from §7
  config.yaml          # camera, model, filter, output settings
  handtrack/           # package: capture, decode, preprocess, models, tracker, filter, output, viewer
  scripts/
    build_engines.py   # ONNX -> TensorRT, with caching
    benchmark.py       # repeatable benchmarks on recorded clips and live camera
    record_clip.py     # record raw MJPG test clips for repeatable testing
  run.py               # main entry point: python run.py [--headless] [--udp HOST:PORT]
  tests/
```

## 7. Measurement and acceptance criteria

Build measurement first, so every optimization is judged by numbers.

**Per-stage timing:** decode, preprocess, detector, landmark model, postprocess/filter, publish. Report mean, p50, p95, and p99 over at least 1,000 frames.

**On-device latency:** `t_publish - t_capture` per frame.

**Repeatable test clips:** record a few raw MJPG clips (slow movement, fast movement, two hands crossing, hand entering and leaving the frame, still hand for jitter) and benchmark against them as well as the live camera.

**Jitter:** standard deviation of filtered fingertip positions (landmarks 4, 8, 12, 16, 20) on the still-hand clip, in pixels.

**Baseline comparison:** run the existing MediaPipe CPU pipeline on the same clips and report its numbers side by side.

**Targets** (MAXN SUPER, `jetson_clocks`, 1280x720 input, 2 hands, viewer off):

| Metric | Target |
|---|---|
| Processing rate | Keeps up with every camera frame (≥ 30 fps), with headroom to reach ≥ 60 fps on recorded clips |
| On-device latency, p95 (capture -> landmarks ready) | ≤ 20 ms; stretch ≤ 12 ms |
| Landmark model inference, per hand | ≤ 5 ms |
| CPU usage | Clearly lower than the MediaPipe baseline |
| Fingertip jitter (still hand, filtered) | Lower than baseline, with filter settings documented |
| Track stability | No handedness swaps or ID switches on the two-hands-crossing clip |

If a target can't be met, document why with measurements and the best achieved result.

**Glass-to-glass estimate:** document a simple method to estimate total latency including the camera itself (e.g. film a millisecond timer or a blinking LED alongside the viewer), and report the Pocket 3's own contribution. The camera's internal processing (stabilization, MJPG encoding) is likely a significant part of the total that no on-device software can remove.

## 8. Hardware recommendations (report only, don't buy)

After benchmarking, include a short section in `DECISIONS.md` on whether a different camera would reduce latency meaningfully. For example: a USB 3 or CSI global-shutter camera at 60–120 fps with uncompressed output, or a camera on a USB 3 port rather than the USB 2 hub. Base this on the measured split between camera latency and on-device latency.

## 9. Working agreements

- Work in milestones and stop for review at the end of each one:
  1. Environment report and toolchain plan (§4)
  2. Capture + decode path with timing, compared against the current OpenCV path
  3. Model bake-off with numbers; chosen model running end to end on TensorRT
  4. Tracking, filtering, output interface, viewer
  5. Benchmarks, docs, cleanup
- Ask before running anything that needs `sudo`, installs system packages, or downloads more than about 1 GB.
- Keep `~/theramin/gesture.py` and `~/gesture-env` working.
- Prefer small, verifiable steps; run things on the device and report real output rather than assuming success.
- Everything runs locally; no cloud inference.

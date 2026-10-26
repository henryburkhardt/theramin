# Benchmarks

Conditions for all runs: Jetson Orin Nano 8 GB, MAXN_SUPER, `jetson_clocks` active (GPU 1020 MHz, CPU 1.728 GHz),
DJI Osmo Pocket 3 in UVC webcam mode on USB 2.0 (480M), 1280x720 MJPG @ 30 fps, ~200 KB/frame.
Raw numbers are in `results/*.json`. Reproduce with `scripts/benchmark_decode.py`.

## Milestone 2: capture + decode (2026-10-26)

### Timestamps

All are CLOCK_MONOTONIC, comparable with `time.monotonic_ns()`.

| Name | Meaning |
|---|---|
| `t_capture` | Kernel V4L2 buffer timestamp. uvcvideo flags it `MONOTONIC \| SRC_SOE` with `hwtimestamps=0`: it's taken when the **first USB packet** of the frame arrives |
| `t_dequeue` | Userspace receives the complete compressed frame |
| `t_decoded` | Decoded RGB(A) frame is in GPU memory (CUDA stream synchronized) |
| `t_consumer` | The next pipeline stage receives the frame from the newest-frame slot |

`transfer = t_dequeue - t_capture` is mostly the USB transfer of the ~200 KB JPEG. `total = t_consumer - t_capture`.
None of this includes the camera's own exposure/processing/encoding time before the first packet leaves it; that needs the glass-to-glass test (Milestone 5).

### Live camera, 1000 frames per path, consumer keeps up

`python scripts/benchmark_decode.py --frames 1000` → `results/decode_live.json`. Latencies in ms.

| Path | fps | CPU cores | transfer p50 / p95 | decode→GPU p50 / p95 / p99 | **total p50 / p95 / p99** |
|---|---|---|---|---|---|
| `opencv`: `cv2.VideoCapture.read()` (gesture.py path) + RGB upload | 30.0 | 0.28 | — | — | 12.55 / 12.95 / 13.39 |
| `v4l2-cpu`: own V4L2 capture, `cv2.imdecode` + upload | 30.0 | 0.28 | 3.99 / 4.07 | 8.35 / 8.54 / 8.70 | 12.44 / 12.70 / 12.86 |
| **`v4l2-nvjpeg`: own V4L2 capture, nvJPEG (CUDA) into GPU memory** | 30.0 | **0.14** | 3.99 / 4.04 | **5.25 / 5.35 / 5.97** | **9.29 / 9.44 / 10.05** |
| `gst-nvv4l2-mjpeg`: v4l2src → nvv4l2decoder mjpeg=1 (NVJPG) → nvvidconv → appsink → upload | 30.0 | **0.10** | 4.08 / 4.25 | 6.72 / 7.04 / 7.57 | 10.90 / 11.33 / 11.93 |
| `gst-nvjpegdec`: v4l2src → nvjpegdec → nvvidconv → appsink → upload | 30.0 | 0.73 | 4.08 / 5.44 | 28.02 / 31.16 / 39.34 | 32.25 / 39.23 / 111.96 |

No frames were dropped on any path. The handoff from the decode thread to the consumer adds < 0.1 ms.

### Live camera with a slow consumer (simulated 40 ms model stage)

`--work-ms 40 --frames 300` → `results/decode_live_slowconsumer.json`. This is roughly what happens when the stage after decode is slower than the camera, as with MediaPipe on the CPU.

| Path | Frames consumed/s | **total p50 / p95 (ms)** |
|---|---|---|
| `opencv` (4 driver buffers, reads in order, as gesture.py does) | 20.4 | **179.8 / 195.1** |
| `v4l2-nvjpeg` (2 buffers, newest-frame slot) | 24.9 | 26.5 / 40.8 |
| `gst-nvv4l2-mjpeg` (appsink max-buffers=1 drop=true) | 24.9 | 27.9 / 42.4 |

With in-order reads, a slow consumer always gets the oldest of OpenCV's 4 queued frames, which adds about 170 ms. With newest-frame capture, the added latency is just the age of the newest frame (0–33 ms).

### Max decode throughput, recorded clip, 1000 decodes back to back

`scripts/record_clip.py clips/bench --seconds 10` (309 frames, 205 KB avg), then `--clip clips/bench --frames 1000` → `results/decode_clip.json`.

| Decoder | fps | CPU cores | per-frame p50 / p95 (ms) |
|---|---|---|---|
| `cv2.imdecode` + RGB + upload | 96 | 1.06 | 10.36 / 10.64 |
| nvJPEG (CUDA, hybrid backend) | 203 | 0.93 | 4.86 / 5.30 |
| GStreamer `nvv4l2decoder mjpeg=1` (NVJPG engine) → RGBA in system memory | **447** | 0.85 | — (pipelined) |
| GStreamer `nvjpegdec` → RGBA in system memory | 132 | 0.54 | — (pipelined) |

All four decoders clear the ≥ 60 fps headroom target for decode alone. The GStreamer numbers include Python's per-buffer map/copy, which accounts for most of their CPU use.

### H264 (not measurable)

The Pocket 3 advertises H264 at every size, but its UVC descriptor reports a max frame size of 0. As a result:
- `VIDIOC_REQBUFS` fails with EINVAL, and the kernel logs a vb2 `WARN_ON` (harmless).
- GStreamer `v4l2src` fails with "Video device did not suggest any buffer size".
- With buffers sized manually via `VIDIOC_CREATE_BUFS`, streaming starts but no frames arrive.

So NVDEC H264 vs MJPEG can't be compared on this camera on Linux. The decoder side works (synthetic H264 decodes on NVDEC; see ENVIRONMENT.md). The `gst-h264` paths stay in the benchmark as opt-in for a future camera.

## Milestone 3: model bake-off (2026-10-26)

Same conditions as above. Clips (15 s each, 1280x720 MJPG, recorded with `scripts/record_clip.py`): `still_hand`, `slow`, `fast`, `two_hands_crossing`, `enter_leave`. They're in `clips/`, not in git.

- Trackers: max 2 hands, TensorRT FP16, frames decoded with nvJPEG and processed back to back.
- Reference: MediaPipe Tasks HandLandmarker on the CPU (`scripts/mp_reference.py`, `~/gesture-env`), full-resolution frames, 2 hands.
- Reproduce with `scripts/benchmark_models.py`. Raw numbers: `results/models_clips.json`, `results/models_live_*.json`.

### Engines (FP16, 1000 runs each, inference only)

| Model | Input | p50 / p95 / p99 ms |
|---|---|---|
| MediaPipe palm detector | 192x192 | 1.73 / 1.75 / 2.84 |
| MediaPipe hand landmark (full), 1 hand | 224x224 | **1.13** / 1.17 / 2.21 |
| MediaPipe hand landmark, 2 hands batched | 2x224x224 | 1.65 / 1.67 / 2.77 |

TensorRT vs ONNX on real frames: landmark max |diff| 0.43 px in 224-crop space (~0.5 px in the frame); presence and handedness scores ≤ 0.001.

### Tracker results per clip

`trk` = tracker time per frame (GPU warp + inference + host post-processing), not including decode. `det%` = share of frames where the palm detector ran. It runs whenever fewer than 2 hands are tracked, which is every frame in one-hand clips. `err` = mean landmark distance to the MediaPipe CPU reference (px). `ids`/`sw`/`flip` = track IDs used, ID switches, handedness flips within a track. `jitter` = high-frequency fingertip motion (residual after a 9-frame moving average, px); it's only meaningful on `still_hand`.

**A. MediaPipe models on TensorRT** (`handtrack/mediapipe_hands.py`)

| Clip | decode+track fps | trk p50 / p95 / p99 ms | det% | hands / ref | err px (tips) | ids / sw / flip | jitter px |
|---|---|---|---|---|---|---|---|
| still_hand | 94.8 | 4.81 / 5.66 / 7.50 | 100 | 449 / 449 | 0.59 (0.60) | 1 / 0 / 0 | **0.44** |
| slow | 95.9 | 4.81 / 5.95 / 7.22 | 76 | 557 / 449 * | 0.59 (0.60) | 2 / 0 / 0 | 2.78 |
| fast | 91.5 | 5.55 / 6.54 / 7.48 | 92 | 487 / 475 | 0.85 (0.92) | 5 / 2 / 1 | 21.17 |
| two_hands_crossing | 104.9 | 3.01 / 5.29 / 6.27 | 19 | 817 / 856 | 0.87 (0.90) | 9 / 2 / 1 | 7.51 |
| enter_leave | 96.2 | 4.85 / 6.34 / 7.84 | 98 | 345 / 338 | 0.82 (0.91) | 6 / 0 / 2 | 3.56 |

\* The 108 extra hand-frames on `slow` are a real hand: the person typing in the background (checked visually). The reference missed it. An instrument needs a minimum hand size / play-area filter (Milestone 4).

**B. RTMPose-m hand 256 + RTMDet-nano hand on TensorRT** (`handtrack/rtmpose_hands.py`, detector every frame)

| Clip | decode+track fps | trk p50 / p95 / p99 ms | hands / ref | err px (tips) | ids / sw | jitter px |
|---|---|---|---|---|---|---|
| still_hand | 76.4 | 7.08 / 9.83 / 10.66 | 449 / 449 | 6.37 (7.00) | 1 / 0 | 0.80 |
| slow | 76.7 | 7.09 / 10.15 / 10.81 | 449 / 449 | 7.54 (7.76) | 1 / 0 | 5.13 |
| fast | 76.6 | 7.08 / 10.09 / 10.68 | 449 / 475 | 11.09 (12.46) | 3 / 0 | 22.53 |
| two_hands_crossing | 67.5 | 8.34 / 10.88 / 11.64 | 827 / 856 | 14.44 (17.36) | 12 / 1 | 22.92 |
| enter_leave | 85.4 | 7.03 / 9.05 / 10.52 | 310 / 338 | 9.78 (12.44) | 8 / 2 | 4.59 |

RTMPose has no handedness output. With detect-once-then-track (next ROI from the keypoint box), its still-hand jitter rose to 22 px, because the crop oscillates. So it runs the detector every frame.

**MediaPipe CPU reference** (per-frame time; some clips ran while the GPU benchmarks competed for the CPU, so ~100 ms values are inflated, and the clean baseline will be redone in Milestone 5)

| Clip | per-frame p50 / p95 ms | hand-frames | still-hand jitter px |
|---|---|---|---|
| still_hand | 71.2 / 72.8 | 449 | 0.44 |
| two_hands_crossing | 65.3 / 71.4 | 856 | — |

### Live, end to end (camera → landmarks ready), 40 s, MediaPipe-TRT

`t_publish - t_capture` (kernel V4L2 timestamp to landmarks on the host), ~2 hands in view:

| Decode source | fps | p50 / p95 / p99 ms |
|---|---|---|
| **V4L2 + nvJPEG** | 29.9 | **11.93 / 12.93 / 14.37** |
| GStreamer NVJPG (`gst-nvv4l2-mjpeg`) | 29.3 | 13.88 / 15.44 / 16.62 |

Against the spec: p95 ≤ 20 ms is **met**. The stretch goal (≤ 12 ms) is met at p50 but not at p95. About 4 ms of the total is USB transfer of the frame.

## Milestone 4: tracking, filtering, output (2026-10-26)

Full tracker (`handtrack/tracker.py`, config defaults) run offline on the clips with `scripts/eval_tracking.py` → `results/tracking_eval.json`. Live numbers come from `run.py`.

### Track stability (before → after Milestone 4)

`ids` = distinct track IDs; `switches` = frames where the set of IDs changed while the hand count stayed the same; `flips` = handedness changes within a track; `jumps` = a track's wrist moving > 150 px in one frame (an identity passed to another hand).

| Clip | ids | switches | flips | jumps | hand-frames |
|---|---|---|---|---|---|
| still_hand | 1 → 1 | 0 → 0 | 0 → 0 | 0 | 449 |
| slow | 2 → **1** | 0 → 0 | 0 → 0 | 0 | 557 → **449** (background hand filtered) |
| fast | 5 → **1** | 2 → **0** | 1 → **0** | 0 | 449 |
| two_hands_crossing | 9 → **2** | 2 → 1 * | 1 → **0** | **0** | 816 |
| enter_leave | 6 → **2** | 0 → 2 * | 2 → **0** | 0 | 319 |

\* The remaining switches are frames where one hand disappears (occluded or out of frame) and the other one is the visible hand: a correct change, not an identity error (0 jumps).

What changed:
- **Identity follows handedness** (`one_person: true`). The per-frame handedness output is decisive: on the crossing clip, 815 of 817 hand-frames were below 0.1 or above 0.9.
- **Duplicates are dropped.** Two ROIs confidently on the same hand count as one.
- **Hand-size filter:** `min_palm_size` 0.09 × frame height. Your hands measured ≥ 75 px; background hands ≤ 52 px.

### One Euro filter

The parameter sweep is in `results/filter_sweep.json`. Jitter = high-frequency fingertip motion on `still_hand`; lag = time shift that best aligns filtered with raw on `fast`.

| min_cutoff | beta | still jitter px | fast lag ms | fast filtered-raw px |
|---|---|---|---|---|
| raw (no filter) | | 0.44 | 0 | 0 |
| 0.3 | 3 | 0.08 | 51 | 28.0 |
| 0.3 | 10 | 0.12 | 30 | 15.8 |
| **0.3** | **30** | **0.17** | **< 1 frame** | **8.5** |
| 1.0 | 30 | 0.20 | < 1 frame | 7.8 |
| 0.3 | 100 | 0.23 | < 1 frame | 3.9 |

The default is `min_cutoff 0.3, beta 30`: 2.6× less jitter at rest, with no measurable lag on fast movement. On slow movement the filter lags about 25 ms (`slow` clip); that's the One Euro trade-off at low speed.

### Prediction (`filter.predict_ms`)

Fingertip error vs. the raw position one frame (33 ms) later:

| Clip | prediction 33 ms | filtered, no prediction | raw (current frame) | prediction p95 |
|---|---|---|---|---|
| slow | **4.7 px** | 10.2 px | 5.9 px | 11.1 px |
| fast | 22.3 px | 24.9 px | **17.2 px** | 55.0 px (overshoot) |

At 50 ms (2 frames): slow 7.7 px (vs 11.8 raw), fast 38.8 px (vs 33.9 raw). Prediction compensates the filter's slow-movement lag but overshoots fast flicks, so it's **off by default**; 33 ms is a reasonable setting for slow, expressive playing.

### Deferred detection

With one hand tracked and `max_hands: 2`, the palm detector looks for the second hand. Running it **after** publishing the landmarks (`tracking.defer_detection: true`) instead of before:

| Clip | time to landmarks p50 / p95 / p99 ms, detector before | detector deferred | hand-frames lost |
|---|---|---|---|
| still_hand | 4.66 / 6.10 / 7.22 | **2.17 / 2.47 / 4.29** | 0 |
| fast | 5.42 / 6.38 / 8.35 | **2.42 / 3.61 / 4.40** | 0 |
| two_hands_crossing | 2.98 / 5.00 / 6.25 | 2.97 / 3.99 / 4.80 | 2 |
| enter_leave | 4.73 / 6.09 / 7.16 | **2.30** / 5.07 / 5.21 | 1 |

A newly appearing hand is picked up one frame later; tracked hands never wait for the detector.

### Live, end to end (`run.py` with viewer, two hands)

camera 30.0 fps, processing 30.0 fps, viewer 29.9 fps; **capture → landmarks p50 12.45 ms, p95 13.35 ms**. Stages (p50):
- USB transfer 3.99 ms
- decode 5.12 ms
- landmarks 2.57 ms (both hands batched)
- post + filter 0.21 ms
- the detector didn't run, since both hands were tracked

With no hands in view (the detector runs every frame), latency is p50 13.7 / p95 14.7 ms.

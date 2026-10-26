# Decisions

## D1. Capture + decode path (Milestone 2, 2026-10-26)

**Decision: direct V4L2 capture (2 driver buffers, own thread) → nvJPEG (CUDA) decoding straight into a CuPy buffer → newest-frame slot.** Implemented in `handtrack/v4l2.py`, `handtrack/nvjpeg.py`, and `MjpegSource` in `handtrack/capture.py`.

**Fallback: GStreamer `nvv4l2decoder mjpeg=1` (NVJPG hardware engine)**, as `GstSource("gst-nvv4l2-mjpeg")`.

Numbers are in BENCHMARKS.md, Milestone 2.

### Why

- **Lowest latency.** Kernel timestamp → RGB frame in GPU memory: p50 9.3 ms, p95 9.4 ms, p99 10.1 ms. That's 3.3 ms better than the current OpenCV path (12.6 ms) and 1.6 ms better than the NVJPG GStreamer path (10.9 ms). Of the 9.3 ms, about 4.0 ms is USB transfer of the frame, which no decoder can remove, so decode→GPU is 5.3 ms.
- **The frame lands in GPU memory directly.** The GStreamer path has to go NVMM → system memory (a copy into Python) → pinned buffer → GPU, which costs about 1.5 ms. That copy could be removed with NvBufSurface/EGL-CUDA interop, but that needs a C extension or jetson-utils. Revisit this only if decode becomes the bottleneck.
- **Full control and observability.** We set the driver buffer count ourselves, get the kernel timestamp and the sequence number (so drops are counted), and get the raw JPEG bytes (which the clip recorder uses).
- **Low CPU use.** 0.14 cores at 30 fps, half the OpenCV path's 0.28. NVJPG is lower still (0.10).
- **Plenty of headroom.** nvJPEG decodes at 203 fps back to back, against the 60 fps target.

### Trade-offs and when to switch

- nvJPEG uses the GPU's CUDA cores (IDCT and colour conversion) and about one CPU core per 200 fps for Huffman decoding. NVJPG uses a dedicated engine and leaves the GPU free for TensorRT. Once the landmark models run (Milestone 3), re-measure end-to-end with both sources. If nvJPEG measurably slows inference, switch to `gst-nvv4l2-mjpeg`; it's a one-line change, since both sources produce the same `Frame`.
- nvJPEG's hardware backend (the NVJPG engine) isn't exposed on Orin: `nvjpegCreateEx(HARDWARE)` returns `ARCH_MISMATCH`. The only way to reach the hardware engine is the Jetson Multimedia API or GStreamer.
- `nvjpegdec` (GStreamer) is rejected: 28 ms decode, 0.73 cores, and spikes up to 112 ms.
- H264/NVDEC can't be used with this camera (see BENCHMARKS.md), so MJPG is the only input format.

### Rule adopted: newest frame only, everywhere

The slow-consumer test shows why the spec's "drop stale frames, never queue" rule matters. `gesture.py` reads with OpenCV's default 4 driver buffers. When its processing is slower than the camera, every frame it gets is about 4 frames old: 180 ms of latency, compared with 26 ms for newest-frame capture under the same load.

Note: `gesture.py` is a plain `cap.read()` loop, not the threaded newest-frame capture that spec §2 describes. It's left unchanged, as agreed.

### D1 follow-up (Milestone 3)

Re-measured live with the TensorRT models loaded: nvJPEG gives p95 12.9 ms capture→landmarks, against 15.4 ms for the NVJPG GStreamer path. nvJPEG's use of the GPU doesn't hurt inference measurably, so D1 stands.

## D2. Hand model (Milestone 3, 2026-10-26)

**Decision: MediaPipe palm detector + hand landmark "full" models, converted to ONNX and run as TensorRT FP16 engines, with the MediaPipe Tasks graph reimplemented around them** (`handtrack/mediapipe_hands.py`). Numbers are in BENCHMARKS.md, Milestone 3.

### Why

- **Faster.** Tracker p50 4.8 ms vs 7.1 ms for RTMPose; landmark model 1.1 ms per hand (1.65 ms for two batched) vs 3.3 ms. Live capture→landmarks p95 is 12.9 ms.
- **Steadier.** Still-hand fingertip jitter is 0.44 px vs 0.80 px for RTMPose, before any filtering.
- **Has handedness.** It's a model output, as the spec requires. RTMPose has none.
- **Detect once, then track works.** The landmark output gives a stable next ROI. RTMPose oscillated when tracked that way, so it has to run its detector every frame.
- **Accuracy.** There's no ground truth, so agreement with MediaPipe isn't proof of accuracy. But it matches MediaPipe CPU within 0.6–0.9 px mean. Visual checks of the frames where the two candidates disagree most showed no case where RTMPose was right and MediaPipe wrong; on motion blur both fail.

### Implementation notes that matter

- **Border handling decides accuracy.** Pad the palm-detector letterbox with zeros, but replicate edge pixels for landmark crops. With zero-padded crops, a hand at the frame edge was off by 19 px mean vs MediaPipe; with replicate it's 2.4 px.
- **Normalization.** Inputs are [0, 1] floats; the models embed no normalization metadata, and [0, 1] reproduces MediaPipe.
- **Conversion.** TFLite → ONNX with tf2onnx (opset 17) in a separate `~/convert-env` (TensorFlow 2.21), so `handtrack-env`'s numpy 1.x pins are untouched. The landmark ONNX gets a dynamic batch dimension for 2-hand batching; the detector stays at batch 1.
- **Engine caching.** `scripts/build_engines.py`, cached in `engines/`, keyed by ONNX hash + TensorRT version + GPU.

### Rejected

- **RTMPose-m hand + RTMDet-nano** (Apache-2.0): slower, no handedness, more jitter, and it can't use detect-once-then-track. Kept in `handtrack/rtmpose_hands.py` for reference.
- **trt_pose_hand** (NVIDIA-AI-IOT, MIT; also mirrored without a license in make2explore/Real-Time-Hand-Pose-Estimation-on-Jetson-Nano): not evaluated. It's a bottom-up model on the whole frame squashed to 224x224, so hands are a few dozen pixels across. It has no handedness, and it needs PyTorch + trt_pose's plugin on JetPack 7.
- **YOLO hand-keypoints** (optional candidate C in the spec): skipped. Ultralytics weights are AGPL-3.0 and would need PyTorch for export.

### Licenses

- MediaPipe hand models: Apache-2.0 (google-ai-edge/mediapipe).
- RTMPose / RTMDet: Apache-2.0 (open-mmlab).

### Open issues for Milestone 4

- **Track stability isn't there yet.** On `two_hands_crossing`: 9 track IDs, 2 ID switches, 1 handedness flip; on `enter_leave`, 2 flips. The spec target is none on the crossing clip. Fix with handedness smoothing per track, plus re-association by position/velocity when hands overlap.
- **Background hands are picked up.** A typing person in the background was tracked. Add a minimum hand size and/or play-area config.
- **The detector runs every frame when only one hand is visible** (because max_hands = 2). That's cheap (≈ 2.9 ms), but a detector interval while ≥ 1 hand is tracked would cut the one-hand tracker time from ~4.8 to ~2 ms.

## D3. Track identity and handedness (Milestone 4, 2026-10-26)

**Decision: with `one_person: true` (the default), a track's identity *is* its handedness.** There's one stable ID for the player's left hand and one for the right. Each frame's label comes from the landmark model's handedness output, with a short per-ROI average that resets when an ROI clearly lands on the other hand.

- **Why.** Plain ROI tracking let an ROI slide onto the other hand during a crossing. The track then carried the wrong hand's history, which caused ID switches and a handedness flip. The model's handedness, though, flips immediately when that happens, and it's decisive (>99% of frames below 0.1 or above 0.9). Result on the crossing clip: 2 IDs, 0 flips, 0 jumps (was 9 IDs, 2 switches, 1 flip).
- **Duplicates.** Two ROIs confidently on the same hand are merged, keeping the higher presence, so the detector can find the hand that was lost.
- **General case** (`one_person: false`, or more than 2 hands): tracks follow their ROI, and new ROIs take over recently lost tracks by wrist proximity (`reuse_seconds`).
- **`camera_mirrored`** maps MediaPipe's selfie-convention labels to the player's actual hands. Default `false`; **verify once with the viewer** (README).
- **The filter resets** when a track reappears after > 0.2 s or jumps > 10% of the frame, so it never smears from an old position.

## D4. Filtering (Milestone 4)

One Euro filter per coordinate, on normalized coordinates, timed with `t_capture_ns`. Defaults are `min_cutoff 0.3`, `beta 30`, `d_cutoff 1.0`, picked from a 25-point sweep: 2.6× less jitter at rest, with no measurable lag on fast movement (BENCHMARKS.md). Raw, filtered and velocity (the filter's derivative estimate) are all published. Prediction (`predict_ms`) is off by default: it helps slow movement but overshoots fast flicks.

## D5. Deferred palm detection (Milestone 4)

When hands are already tracked, the palm detector (looking for an additional hand) runs after the frame's landmarks are published, on the same frame, during the idle time before the next frame arrives. This saves ~2.5 ms p50 on the latency path with one hand tracked (4.7 → 2.2 ms). The cost: a new hand appears one frame later.

## D6. Output format (Milestone 4)

**OSC over UDP** by default: the native protocol of SuperCollider, Max/MSP, Pure Data and most synth environments. It's one bundle per frame (frame info + one message per hand), with filtered landmarks and velocities as floats, ~1.2 KB, so it fits one datagram. **Compact JSON** is the alternative for other consumers. Both send from a dedicated thread with newest-only semantics. The format is documented in README.md.

## D7. Viewer and frame buffers (Milestone 4)

- **The viewer never touches the GPU frame.** It decodes the frame's original JPEG on the CPU at half resolution, on the GUI thread, at its own rate. That keeps the tracker at full speed with the viewer on (measured: 30.0 fps processing, viewer 29.9 fps, p95 latency 13.35 ms).
- **Decoded frames rotate through 3 GPU buffers.** Before this, the decode thread wrote each frame into the single buffer the tracker might still be reading: a latent race, harmless only because tracking (~8 ms) was shorter than a frame period (33 ms).

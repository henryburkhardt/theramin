# handtrack

Low-latency hand tracking on a Jetson Orin Nano, for driving a theremin-style instrument.

The pipeline: DJI Osmo Pocket 3 (UVC, MJPG 1280x720 @ 30) → direct V4L2 capture → nvJPEG decode on the GPU → MediaPipe palm detector + hand-landmark models as TensorRT FP16 engines → stable tracks with handedness → One Euro filter → in-process API and optional OSC/JSON over UDP.

Live latency from the frame arriving on the Jetson to landmarks being ready is about **12–13 ms** (p50 12.5, p95 13.4 with two hands). That doesn't include the camera's own internal delay. Details are in [BENCHMARKS.md](BENCHMARKS.md); the reasons behind each choice are in [DECISIONS.md](DECISIONS.md); the system is described in [ENVIRONMENT.md](ENVIRONMENT.md).

## Setup

You need JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16); see ENVIRONMENT.md.

```bash
python3 -m venv --system-site-packages ~/handtrack-env    # system TensorRT, VPI, cv2 with GStreamer
~/handtrack-env/bin/pip install -r requirements.txt
```

The models (`models/`, not in git) are recreated with `scripts/fetch_models.sh`. That needs a separate `~/convert-env` with TensorFlow + tf2onnx, kept apart so it doesn't disturb `handtrack-env`'s numpy pins. TensorRT engines are built and cached in `engines/` automatically on the first run (about 5 minutes), or ahead of time:

```bash
~/handtrack-env/bin/python scripts/build_engines.py
```

For benchmarks, pin the clocks after each boot with `sudo jetson_clocks`.

## Running

```bash
cd ~/theramin/handtrack
~/handtrack-env/bin/python run.py                         # viewer on the Jetson's monitor
~/handtrack-env/bin/python run.py --headless              # stats printed once per second
~/handtrack-env/bin/python run.py --udp 127.0.0.1:9000    # also send OSC over UDP
~/handtrack-env/bin/python run.py --udp 192.168.1.20:9000 --udp-format json
```

Viewer keys:
- **q** / Esc: quit
- **m**: mirror the view
- **r**: also show the raw (unfiltered) skeleton
- **s**: save a snapshot PNG

Settings are in `config.yaml`; the defaults and their meanings are in `handtrack/config.py`.

### Check handedness once

Whether "Left"/"Right" mean your actual hands depends on whether the camera image is mirrored (`tracking.camera_mirrored`). Run the viewer and raise only your **right** hand. If it's labelled `Left`, flip `camera_mirrored` in `config.yaml`.

## Using the results

In-process:

```python
from handtrack import config
from handtrack.tracker import TrackingService

svc = TrackingService(config.load()).start()
svc.subscribe(lambda hf: ...)    # called on the tracking thread for every frame: keep it short
hf = svc.latest()                # or poll the newest HandFrame
svc.stop()
```

```python
@dataclass
class HandFrame:
    frame_id: int; t_capture_ns: int; t_publish_ns: int; hands: list[Hand]; stage_ms: dict

@dataclass
class Hand:
    track_id: int                   # stable; with one_person, one ID per hand (left/right)
    handedness: str                 # "Left" / "Right"
    score: float                    # hand presence
    landmarks_px: np.ndarray        # (21, 3) raw, pixels (+ relative z)
    landmarks_norm: np.ndarray      # (21, 3) raw, normalized 0..1
    landmarks_filtered: np.ndarray  # (21, 3) One Euro filtered, normalized
    velocity: np.ndarray            # (21, 2) normalized units per second
    handedness_score: float
    landmarks_predicted: np.ndarray | None   # when filter.predict_ms > 0
```

Timestamps are `CLOCK_MONOTONIC` ns (`time.monotonic_ns()`). `t_capture_ns` is the kernel's V4L2 timestamp for the frame's arrival. Landmark order is MediaPipe's (0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip).

### UDP (OSC, default)

One bundle per processed frame:

| Address | Arguments |
|---|---|
| `/handtrack/frame` | `i` frame_id, `h` t_capture_ns, `h` t_publish_ns, `i` n_hands |
| `/handtrack/hand` | `i` track_id, `s` handedness, `f` score, 63 × `f` filtered landmarks (x0 y0 z0 … x20 y20 z20, normalized), 42 × `f` velocity (vx0 vy0 …) |

There's one `/handtrack/hand` message per hand; about 1.2 KB per frame with two hands. The JSON format (`--udp-format json`) sends one object per datagram: `{"f", "tc", "tp", "hands": [{"id", "h", "s", "lm": [63], "v": [42]}]}`. The sender has its own thread and always sends the newest frame, so a slow network never delays tracking.

## Tools

| Script | Purpose |
|---|---|
| `scripts/record_clip.py NAME --seconds 15` | Record raw MJPG test clips (live preview, countdown) |
| `scripts/benchmark_decode.py` | Capture/decode paths (Milestone 2) |
| `scripts/benchmark_models.py` | Model bake-off on clips, or `--live` latency (Milestone 3) |
| `scripts/eval_tracking.py` | Track stability, jitter, lag, prediction; `--sweep` for filter tuning (Milestone 4) |
| `scripts/mp_reference.py` | MediaPipe CPU reference on a clip (run with `~/gesture-env`) |
| `scripts/build_engines.py`, `scripts/fetch_models.sh` | Engines and models |

Tests: `~/handtrack-env/bin/python -m pytest -q tests`

## Troubleshooting

- **No window over SSH.** The viewer and the recorder look for the desktop session themselves: they use `DISPLAY=:1` (or `:0`) and `XAUTHORITY=/run/user/$UID/gdm/Xauthority`. A desktop session must be logged in on the Jetson. Otherwise run with `--headless`. The window appears on the Jetson's monitor, not your laptop. `Gtk-Message: Failed to load module "canberra-gtk-module"` is harmless.
- **`/dev/video0` missing.** Select **Webcam** on the Pocket 3's screen after plugging it in. USB ID `2ca3:0023` is webcam mode; `2ca3:0020` is its storage/network mode, which has no video.
- **Camera drops off with `-71` errors in `journalctl -k`.** Plug it directly into the Jetson, not into the keyboard/mouse hub.
- **H264 doesn't work.** That's expected: the Pocket 3's UVC H264 mode is unusable on Linux (see BENCHMARKS.md). Use MJPG.
- **Latency higher than expected.** Run `sudo jetson_clocks` and check `cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/*/cur_freq` (1020000000).
- **Background people's hands get tracked.** Raise `tracking.min_palm_size` or set `tracking.play_area`.
- **numpy errors importing cv2.** `handtrack-env` must keep numpy < 2 (NVIDIA's cv2 is built against 1.x); `requirements.txt` pins it.

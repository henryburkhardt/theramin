# Environment report

First checked on 2026-10-15 (read-only). Updated on 2026-10-26 after the JetPack install and the creation of `~/handtrack-env`.

## Platform

| Item | Finding | How checked |
|---|---|---|
| L4T | **R39.2.1** (GCID 46758480, built 2026-08-07), `nvidia-l4t-core 39.2.1` | `/etc/nv_tegra_release`, `dpkg -l` |
| JetPack | **7.2.1**, `nvidia-jetpack 7.2.1-b49` **installed** (the full meta-package) | `dpkg -l` |
| OS / kernel | Ubuntu 24.04.4 LTS, `6.8.12-1021-tegra` aarch64 | `lsb_release`, `uname` |
| GPU | Orin (sm_87), nvgpu driver 595.78, CUDA 13.2 capability, FP16 supported by TensorRT | `nvidia-smi`, CuPy, TRT builder |
| Power mode | **MAXN_SUPER** (mode 2) | `nvpmodel -q` |
| Clocks | GPU idles at 306 MHz of 1020 MHz max, so `jetson_clocks` is **not** active. Run `sudo jetson_clocks` before benchmarks | devfreq sysfs |
| CPU | 6 cores | `nproc` |
| RAM | 7.3 GiB total, ~3.8 GiB available, **no swap** | `free -h` |
| Disk | 57 GB root, **26 GB free** (was 44 GB before the JetPack install) | `df -h` |
| apt repos | `repo.download.nvidia.com/jetson/{common,som,ffmpeg} r39.2` | `/etc/apt/sources.list.d/` |

## Compute stack (system, from apt)

| Component | Version |
|---|---|
| CUDA toolkit | 13.2 (`/usr/local/cuda`, `nvcc` 13.2) |
| cuDNN | 9.20.0 (`libcudnn9-cuda-13`) |
| TensorRT | 10.16.2.10 (`libnvinfer10`, `libnvonnxparsers10`). Python bindings importable from system Python 3.12 |
| `trtexec` | `/usr/bin/trtexec` |
| nvJPEG | `libnvjpeg-13-2` 13.1.0 (+ dev headers) |
| NVRTC / cuBLAS | `cuda-nvrtc-13-2` 13.2.86, `libcublas-13-2` 13.4.1 |
| VPI | 4.1.4 (`python3.12-vpi4`). It prints "Failed to open PVA device node" on import. That's harmless: the Orin Nano has no PVA, and CUDA/CPU/VIC backends still work |
| OpenCV (system) | NVIDIA `nvidia-opencv` **4.8.0**, built **with GStreamer**, needs numpy 1.x |
| jetson-utils | Not installed (source build only; not needed by the plan) |
| onnxruntime-gpu | Not installed (no JetPack 7 wheel; not needed, since we use TensorRT directly) |

**JetPack meta-package side effect:** as predicted, it replaced Ubuntu's `libopencv-*-dev 4.6.0` with NVIDIA's `libopencv-dev 4.8.0`. Ubuntu's 4.6.0 runtime libraries (`libopencv-*406t64`) are still installed.

## Video / decode

| Item | Finding |
|---|---|
| NVJPG hardware engines | Two present (`15380000.nvjpg`, `15540000.nvjpg`), plus NVDEC (`15480000.nvdec`) and VIC |
| GStreamer | 1.24.2 with `nvidia-l4t-gstreamer` 39.2.1. Elements: `nvv4l2decoder` (`mjpeg`, `disable-dpb`, `enable-max-performance`), `nvjpegdec`, `nvvidconv`, `nvv4l2camerasrc`, `v4l2src`, `jpegdec` |
| HW MJPEG decode works | **Yes.** `videotestsrc ! jpegenc ! nvv4l2decoder mjpeg=1` and `... ! nvjpegdec` both decoded 300 frames at 1280x720 |
| HW H264 decode works | **Yes.** `x264enc ! h264parse ! nvv4l2decoder` decoded 100 frames on NVDEC |
| Camera H264 over UVC | **Not usable.** The Pocket 3 reports a max frame size of 0 for H264, so `REQBUFS` fails (EINVAL plus a harmless vb2 kernel `WARN_ON`), and `v4l2src` errors out. With buffers sized via `CREATE_BUFS`, no frames arrive. MJPG is the only usable format |
| nvJPEG hardware backend | **Not available on Orin** (`nvjpegCreateEx(HARDWARE)` → `ARCH_MISMATCH`). CUDA backends work. The NVJPG engine is only reachable via GStreamer / the Multimedia API |
| GStreamer from Python | Yes, in system Python and in `~/handtrack-env` (`gi` + Gst 1.24.2). Also through system cv2 4.8 (`CAP_GSTREAMER`) |
| Camera | Pocket 3 in UVC webcam mode (`2ca3:0023`; `2ca3:0020` is its default network/storage mode, which has no video). `/dev/video0` = video, `/dev/video1` = metadata. Formats: MJPG 1280x720/1920x1080/portrait @ 30/25(/24), 3840x2160 @ 30/25/24; H264 the same plus 3840x2160 @ 60/50/48. No YUYV |
| Camera streaming (2026-10-26) | `v4l2-ctl` MJPG 1280x720, 2 buffers, 300 frames: **29.97 fps**, no errors. Live MJPG through `nvv4l2decoder mjpeg=1` and through `nvjpegdec`: 150 frames each, clean |
| USB | Camera at `1-2.4`, high speed (480M). The dev kit's USB-A ports sit behind an onboard hub on bus 001, so this path is expected even when the camera is plugged into the board. Behind the external keyboard/mouse hub, the camera hit repeated `-71` (EPROTO) errors and dropped off the bus; plugged into the board directly, it's stable |

## Python environments

### `~/handtrack-env` (new, for this project)

Created with `python3 -m venv --system-site-packages ~/handtrack-env`. The pinned pip packages are in `handtrack/requirements.txt`.

| Package | Version | Source |
|---|---|---|
| tensorrt | 10.16.2.10 | system |
| vpi | 4.1.4 | system |
| cv2 | 4.8.0 (GStreamer: yes) | system (NVIDIA) |
| PyGObject / Gst | 3.48.2 / 1.24.2 | system |
| scipy | 1.11.4 | system |
| numpy | **1.26.4** (pinned `<2`) | pip |
| cupy-cuda13x | **13.6.0** | pip |
| onnx | 1.23.2 | pip |
| ml-dtypes | 0.5.4 | pip |
| PyYAML, python-osc | 6.0.1, 1.10.2 | pip |

**Why numpy 1.x:** NVIDIA's system cv2 4.8 and Ubuntu's scipy 1.11 are compiled against numpy 1.x and fail to import with numpy 2 (`numpy.core.multiarray failed to import`). Because of that, CuPy is pinned to 13.6.0 (14.x requires numpy ≥ 2) and ml-dtypes to 0.5.x (0.6 requires numpy ≥ 2; onnx 1.23 needs ≥ 0.5.4). `opencv-python-headless` was deliberately **not** installed: it would shadow the system cv2 and lose GStreamer support.

Smoke test, all passing: every import above works. A CuPy `RawKernel` compiles via NVRTC and runs on the Orin (sm 87). `tensorrt.Builder` reports FP16 support. `Gst.init` succeeds.

### `~/convert-env` (model conversion only)

Created in Milestone 3 for TFLite → ONNX conversion: tensorflow 2.21, tf2onnx 1.17, onnxruntime 1.31, ai-edge-litert 2.3 (numpy 2.x), 1.7 GB. It's only needed by `scripts/fetch_models.sh`, and can be deleted once the models are final.

### `~/gesture-env` (baseline, untouched)

Python 3.12.3, mediapipe 1.1.1 (the spec says 0.10.33; the installed version is newer), opencv-contrib-python 4.11.0 (no GStreamer), numpy 1.26.4.

## Toolchain plan

The 10-15 plan stands, and its dependencies are now all installed:

- **Inference:** TensorRT 10.16 through its Python API. Engines are built from ONNX with `trtexec` or the Python builder, cached on disk and keyed by TRT version + GPU.
- **GPU buffers / preprocessing:** CuPy 13.6 `RawKernel`s (affine ROI crop, resize, normalize). VPI is available too if needed.
- **Decode, to be benchmarked in Milestone 2:**
  1. Primary: GStreamer `v4l2src` → `nvv4l2decoder mjpeg=1` (NVJPG) → `nvvidconv` → appsink, through PyGObject.
  2. Alternative: V4L2 capture → nvJPEG on the GPU into a CuPy buffer.
  3. Baseline: OpenCV CPU decode (the `gesture.py` path).
  4. H264 via `nvv4l2decoder` (NVDEC), timed against MJPEG, with `disable-dpb=true`.

### Outstanding before Milestone 2

- Camera: done (see Video / decode).
- `sudo jetson_clocks`: verified active on 2026-10-26 (GPU at 1020 MHz, CPU at 1.728 GHz). Rerun after every reboot.

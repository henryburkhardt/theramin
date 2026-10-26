#!/usr/bin/env python3
"""Milestone 2 benchmark: camera capture + decode paths.

Live mode (default) measures, per frame, from the kernel's V4L2 timestamp:
  transfer   t_dequeue - t_capture   (USB transfer + wakeup; camera-side, mostly fixed)
  decode     t_decoded - t_dequeue   (decode + colour conversion + into GPU memory)
  total      t_consumer - t_capture  (what the next pipeline stage sees)
plus delivered fps, frames the driver/pipeline dropped, and process CPU usage.

Clip mode (--clip) decodes recorded frames back to back to get each decoder's
maximum throughput (the >=60 fps headroom question).

  python scripts/benchmark_decode.py --frames 1000
  python scripts/benchmark_decode.py --clip clips/bench --frames 1000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

# gst-h264 / gst-h264-dpb exist but the Pocket 3's UVC H264 mode is unusable on Linux
# (reports sizeimage 0, REQBUFS fails, no frames even with CREATE_BUFS), so they're opt-in.
LIVE_PATHS = ["opencv", "v4l2-cpu", "v4l2-nvjpeg", "gst-nvv4l2-mjpeg", "gst-nvjpegdec"]
CLIP_DECODERS = ["cpu", "nvjpeg", "gst-nvv4l2-mjpeg", "gst-nvjpegdec"]


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return {"mean": x.mean(), "p50": np.percentile(x, 50), "p95": np.percentile(x, 95),
            "p99": np.percentile(x, 99), "max": x.max(), "n": len(x)}


def fmt(s):
    return "      n/a" if s is None else f"{s['mean']:6.2f} {s['p50']:6.2f} {s['p95']:6.2f} {s['p99']:6.2f}"


def save_preview(img, path):
    import cupy as cp
    import cv2
    a = cp.asnumpy(img)
    cv2.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGBA2BGR if a.shape[2] == 4 else cv2.COLOR_RGB2BGR))


def run_opencv(args):
    """The gesture.py path: cv2.VideoCapture(V4L2, MJPG) .read() -> BGR on CPU,
    with OpenCV's default buffer count, then (for parity) RGB upload to the GPU."""
    import cupy as cp
    import cv2
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.opencv_buffers:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.opencv_buffers)
    stream = cp.cuda.Stream(non_blocking=True)
    gpu = cp.empty((args.height, args.width, 3), cp.uint8)
    recs, last = [], None
    for i in range(args.warmup + args.frames):
        ok, bgr = cap.read()
        t_read = time.monotonic_ns()
        if not ok:
            raise RuntimeError("cv2 read failed")
        # V4L2 backend: POS_MSEC is the kernel buffer timestamp (CLOCK_MONOTONIC) in ms
        t_cap = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)
        gpu.set(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), stream=stream)
        stream.synchronize()
        t_gpu = time.monotonic_ns()
        if i == args.warmup:
            t0, c0 = time.monotonic(), time.process_time()
        if i >= args.warmup:
            recs.append({"total_cpu": (t_read - t_cap) / 1e6, "total": (t_gpu - t_cap) / 1e6})
        last = gpu
        if args.work_ms:
            time.sleep(args.work_ms / 1e3)
    wall, cpu = time.monotonic() - t0, time.process_time() - c0
    if args.preview:
        save_preview(last, os.path.join(args.preview, "opencv.png"))
    cap.release()
    return {"records": recs, "fps": len(recs) / wall, "cpu_cores": cpu / wall,
            "buffers": int(args.opencv_buffers or 4)}


def run_source(name, args):
    from handtrack.capture import GstSource, MjpegSource
    if name == "v4l2-cpu":
        src = MjpegSource(args.device, args.width, args.height, args.fps, args.buffers, decoder="cpu")
    elif name == "v4l2-nvjpeg":
        src = MjpegSource(args.device, args.width, args.height, args.fps, args.buffers, decoder="nvjpeg")
    else:
        src = GstSource(name, args.device, args.width, args.height, args.fps)
    try:
        version, recs, seqs, last = 0, [], [], None
        for i in range(args.warmup + args.frames):
            r = src.wait_newer(version, timeout=3.0)
            if r is None:
                raise RuntimeError("no frame within 3 s")
            version, f = r
            t_now = time.monotonic_ns()
            if i == args.warmup:
                t0, c0 = time.monotonic(), time.process_time()
            if i >= args.warmup:
                rec = {"decode": (f.t_decoded_ns - f.t_dequeue_ns) / 1e6 if f.t_dequeue_ns else None,
                       "transfer": (f.t_dequeue_ns - f.t_capture_ns) / 1e6 if f.t_dequeue_ns else None,
                       "handoff": (t_now - f.t_decoded_ns) / 1e6,
                       "total": (t_now - f.t_capture_ns) / 1e6}
                recs.append(rec)
                seqs.append(f.frame_id)
            last = f.image
            if args.work_ms:
                time.sleep(args.work_ms / 1e3)
        wall, cpu = time.monotonic() - t0, time.process_time() - c0
        if args.preview:
            save_preview(last, os.path.join(args.preview, f"{name}.png"))
        gaps = int(np.sum(np.maximum(np.diff(seqs) - 1, 0))) if len(seqs) > 1 else 0
        out = {"records": recs, "fps": len(recs) / wall, "cpu_cores": cpu / wall, "dropped": gaps}
        if hasattr(src, "cap"):
            out["buffers"] = src.cap.num_buffers
            out["timestamp_flags"] = hex(src.cap.timestamp_flags or 0)
        return out
    finally:
        src.stop()


def run_clip(args):
    """Back-to-back decode of recorded frames: max throughput per decoder."""
    import cupy as cp
    import cv2
    from record_clip import load_clip
    frames = [j for _, _, j in load_clip(args.clip)]
    n = min(args.frames, len(frames) * 50)
    results = {}
    for name in args.paths or CLIP_DECODERS:
        stream = cp.cuda.Stream(non_blocking=True)
        if name == "cpu":
            host = np.empty((args.height, args.width, 3), np.uint8)
            gpu = cp.empty_like(cp.asarray(host))

            def dec(j):
                cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB, dst=host)
                gpu.set(host, stream=stream)
                stream.synchronize()
        elif name == "nvjpeg":
            from handtrack.nvjpeg import NvJpegDecoder
            nd = NvJpegDecoder(args.width, args.height, "hybrid", stream)

            def dec(j):
                nd.decode(j)
                stream.synchronize()
        else:
            results[name] = clip_gst(name, frames, n, args)
            continue
        for j in frames[:20]:
            dec(j)
        ts = []
        t0, c0 = time.monotonic(), time.process_time()
        for i in range(n):
            t = time.perf_counter_ns()
            dec(frames[i % len(frames)])
            ts.append((time.perf_counter_ns() - t) / 1e6)
        wall = time.monotonic() - t0
        results[name] = {"per_frame": stats(ts), "fps": n / wall,
                         "cpu_cores": (time.process_time() - c0) / wall}
    return results


def clip_gst(name, frames, n, args):
    """appsrc -> hardware JPEG decoder -> nvvidconv -> RGBA -> appsink, pushed as
    fast as the pipeline accepts (max throughput, not per-frame latency)."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    dec = {"gst-nvv4l2-mjpeg": "nvv4l2decoder mjpeg=1 enable-max-performance=true",
           "gst-nvjpegdec": "nvjpegdec"}[name]
    p = Gst.parse_launch(
        f"appsrc name=src is-live=false format=time block=true max-bytes=2000000 "
        f"caps=image/jpeg,width={args.width},height={args.height},framerate=30/1 ! {dec} ! "
        f"nvvidconv ! video/x-raw,format=RGBA ! appsink name=sink sync=false max-buffers=4")
    src, sink = p.get_by_name("src"), p.get_by_name("sink")
    count = {"n": 0}

    def on_sample(s):
        smp = s.emit("pull-sample")
        b = smp.get_buffer()
        ok, mi = b.map(Gst.MapFlags.READ)
        b.unmap(mi)
        count["n"] += 1
        return Gst.FlowReturn.OK
    sink.set_property("emit-signals", True)
    sink.connect("new-sample", on_sample)
    p.set_state(Gst.State.PLAYING)
    t0, c0 = time.monotonic(), time.process_time()
    for i in range(n):
        buf = Gst.Buffer.new_wrapped(frames[i % len(frames)])
        buf.pts = i * Gst.SECOND // 30
        src.emit("push-buffer", buf)
    src.emit("end-of-stream")
    p.get_bus().timed_pop_filtered(60 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    wall = time.monotonic() - t0
    p.set_state(Gst.State.NULL)
    return {"per_frame": None, "fps": count["n"] / wall, "cpu_cores": (time.process_time() - c0) / wall,
            "decoded": count["n"], "pushed": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", help=f"live: {LIVE_PATHS}; clip: {CLIP_DECODERS}")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--buffers", type=int, default=2, help="V4L2 driver buffers for v4l2-* paths")
    ap.add_argument("--opencv-buffers", type=int, default=0, help="0 = OpenCV default (4)")
    ap.add_argument("--work-ms", type=float, default=0,
                    help="sleep this long per frame to simulate a slow model stage")
    ap.add_argument("--clip", help="recorded clip (from record_clip.py) for throughput mode")
    ap.add_argument("--preview", help="directory to save one decoded frame per path")
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args()
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.clip:
        res = run_clip(args)
        print(f"\nThroughput on {args.clip}, {args.frames} decodes, back to back")
        print(f"{'decoder':18s} {'fps':>7s} {'cpu cores':>9s}   per-frame ms mean/p50/p95/p99")
        for k, v in res.items():
            print(f"{k:18s} {v['fps']:7.1f} {v['cpu_cores']:9.2f}   {fmt(v['per_frame'])}")
    else:
        res = {}
        for name in args.paths or LIVE_PATHS:
            print(f"running {name} ...", flush=True)
            try:
                res[name] = run_opencv(args) if name == "opencv" else run_source(name, args)
            except Exception as e:
                print(f"  {name} failed: {e}")
                res[name] = {"error": str(e)}
        print(f"\nLive camera, {args.width}x{args.height} MJPG @ {args.fps}, {args.frames} frames per path"
              + (f", simulated model stage {args.work_ms:g} ms/frame" if args.work_ms else ""))
        print("latencies in ms: mean p50 p95 p99, from the kernel V4L2 timestamp")
        print(f"{'path':18s} {'fps':>5s} {'drop':>4s} {'cpu':>5s} | {'transfer':^27s} | {'decode->GPU':^27s} | {'total (consumer)':^27s}")
        for k, v in res.items():
            if "error" in v:
                print(f"{k:18s} ERROR {v['error']}")
                continue
            r = v["records"]
            col = lambda key: stats([x[key] for x in r if x.get(key) is not None])  # noqa: E731
            v["summary"] = {key: col(key) for key in ("transfer", "decode", "handoff", "total", "total_cpu")}
            print(f"{k:18s} {v['fps']:5.1f} {v.get('dropped', 0):4d} {v['cpu_cores']:5.2f} | "
                  f"{fmt(v['summary']['transfer'])} | {fmt(v['summary']['decode'])} | {fmt(v['summary']['total'])}")
    if args.out:
        for v in res.values():
            v.pop("records", None)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res},
                      f, indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

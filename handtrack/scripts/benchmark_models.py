#!/usr/bin/env python3
"""Milestone 3 bake-off: hand trackers on recorded clips (and optionally live).

For each tracker x clip (frames decoded with nvJPEG, processed back to back):
  timing     per-stage ms (detector when it ran, landmark/pose, tracker total) and
             decode+track throughput
  agreement  landmark px error vs the MediaPipe CPU reference (scripts/mp_reference.py),
             hand-frames found vs reference
  jitter     still_hand clip: per-fingertip std over the clip, and high-frequency jitter
             (residual after a centred 9-frame moving average), px, raw (unfiltered)
  stability  track IDs used, ID switches, handedness flips within a track

  python scripts/benchmark_models.py --out results/models_clips.json
  python scripts/benchmark_models.py --live --seconds 30 --trackers mediapipe
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLIPS = ["still_hand", "slow", "fast", "two_hands_crossing", "enter_leave"]
TIPS = [4, 8, 12, 16, 20]


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "n": int(len(x))}


def engine(prefix):
    found = sorted(glob.glob(os.path.join(ROOT, "engines", f"{prefix}-*.engine")), key=os.path.getmtime)
    if not found:
        sys.exit(f"no engine for {prefix}; run scripts/build_engines.py")
    return found[-1]


def make_tracker(name, max_hands):
    if name == "mediapipe":
        from handtrack.mediapipe_hands import MediaPipeHands
        return MediaPipeHands(engine("hand_detector"), engine("hand_landmarks_detector"), max_hands=max_hands)
    if name == "rtmpose":
        from handtrack.rtmpose_hands import RtmPoseHands
        return RtmPoseHands(engine("rtmdet_nano_hand"), engine("rtmpose_m_hand_256"), max_hands=max_hands)
    raise ValueError(name)


def jitter(track):
    """track: (T, 5, 2) fingertip px for one hand over consecutive frames."""
    std = float(np.mean(np.linalg.norm(track.std(0), axis=1)))
    k = 9
    if len(track) <= k:
        return std, None
    kernel = np.ones(k) / k
    smooth = np.array([[np.convolve(track[:, t, d], kernel, mode="valid") for d in range(2)]
                       for t in range(track.shape[1])]).transpose(2, 0, 1)   # (T-k+1, 5, 2)
    resid = track[k // 2: k // 2 + len(smooth)] - smooth
    hf = float(np.mean(np.sqrt((resid ** 2).sum(-1).mean(0))))
    return std, hf


def run_clip(tracker, clip, ref, decoder):
    from record_clip import load_clip
    frames = load_clip(os.path.join(ROOT, "clips", clip))
    tracker.reset()
    for _, _, j in frames[:5]:                                  # warm-up (kernel compile etc.)
        tracker.process(decoder.decode(j))
    tracker.reset()
    det_ms, lm_ms, total_ms, e2e_ms = [], [], [], []
    errs, n_ours, ids, switches, flips = [], 0, set(), 0, 0
    last_label, prev_ids = {}, None
    tips = {}                                                  # track_id -> list of (frame, (5,2))
    t0 = time.monotonic()
    for i, (_, _, j) in enumerate(frames):
        ts = time.perf_counter()
        img = decoder.decode(j)
        hands = tracker.process(img)
        e2e_ms.append((time.perf_counter() - ts) * 1e3)
        t = tracker.last_times
        total_ms.append(t.total_ms)
        if t.det_ran:
            det_ms.append(t.detect_ms)
        if t.landmark_ms:
            lm_ms.append(t.landmark_ms)
        n_ours += len(hands)
        cur_ids = {h.track_id for h in hands}
        if prev_ids is not None and len(cur_ids) == len(prev_ids) and cur_ids != prev_ids:
            switches += 1
        prev_ids = cur_ids
        for h in hands:
            ids.add(h.track_id)
            if h.handedness != "Unknown":
                if h.track_id in last_label and last_label[h.track_id] != h.handedness:
                    flips += 1
                last_label[h.track_id] = h.handedness
            tips.setdefault(h.track_id, []).append((i, h.landmarks_px[TIPS, :2].copy()))
            if ref is not None:
                rl, rh = ref["landmarks_px"][i], ref["handedness"][i]
                cand = [k for k in range(2) if rh[k] >= 0]
                if cand:
                    k = min(cand, key=lambda k: np.linalg.norm(rl[k, 0, :2] - h.landmarks_px[0, :2]))
                    d = np.linalg.norm(rl[k, :, :2] - h.landmarks_px[:, :2], axis=1)
                    if d[0] < 100:                             # same hand, not a mismatch
                        errs.append(d)
    wall = time.monotonic() - t0
    out = {"frames": len(frames), "fps_decode_track": len(frames) / wall,
           "detect_ms": stats(det_ms), "detector_ran_frac": len(det_ms) / len(frames),
           "landmark_ms": stats(lm_ms), "track_total_ms": stats(total_ms), "decode_track_ms": stats(e2e_ms),
           "hand_frames": n_ours, "track_ids": len(ids), "id_switches": switches, "handedness_flips": flips}
    if ref is not None:
        out["ref_hand_frames"] = int((ref["handedness"] >= 0).sum())
        if errs:
            e = np.array(errs)
            out["err_px"] = {"mean": float(e.mean()), "p95": float(np.percentile(e, 95)),
                             "wrist": float(e[:, 0].mean()), "tips": float(e[:, TIPS].mean())}
    # jitter on the longest continuous track
    best = max(tips.values(), key=len) if tips else []
    if len(best) > 30:
        idx = np.array([f for f, _ in best])
        runs = np.split(np.arange(len(idx)), np.where(np.diff(idx) != 1)[0] + 1)
        run = max(runs, key=len)
        std, hf = jitter(np.stack([best[r][1] for r in run]))
        out["tip_std_px"], out["tip_jitter_px"], out["jitter_frames"] = std, hf, len(run)
    return out


def ref_jitter(ref):
    """Same jitter metrics for the MediaPipe CPU reference (hand slot 0)."""
    lm = ref["landmarks_px"][:, 0][:, TIPS, :2]
    ok = ~np.isnan(lm).any(axis=(1, 2))
    idx = np.where(ok)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    run = max(runs, key=len)
    return jitter(lm[run])


def run_live(tracker, args):
    from handtrack.capture import GstSource, MjpegSource
    src = MjpegSource(decoder="nvjpeg") if args.source == "nvjpeg" else GstSource("gst-nvv4l2-mjpeg")
    lat, trk, nh = [], [], []
    try:
        version = 0
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            r = src.wait_newer(version, timeout=3.0)
            if r is None:
                raise RuntimeError("no frame")
            version, f = r
            hands = tracker.process(f.image)
            t_pub = time.monotonic_ns()
            lat.append((t_pub - f.t_capture_ns) / 1e6)
            trk.append(tracker.last_times.total_ms)
            nh.append(len(hands))
    finally:
        src.stop()
    return {"source": args.source, "frames": len(lat), "fps": len(lat) / args.seconds,
            "capture_to_landmarks_ms": stats(lat[30:]), "track_total_ms": stats(trk[30:]),
            "hands_per_frame": float(np.mean(nh))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="*", default=["mediapipe", "rtmpose"])
    ap.add_argument("--clips", nargs="*", default=CLIPS)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--source", choices=["nvjpeg", "gst"], default="nvjpeg")
    ap.add_argument("--out")
    args = ap.parse_args()
    import cupy as cp
    from handtrack.nvjpeg import NvJpegDecoder
    res = {}
    for name in args.trackers:
        tracker = make_tracker(name, args.max_hands)
        if args.live:
            res[name] = run_live(tracker, args)
            r = res[name]
            print(f"{name:10s} live/{r['source']}: {r['fps']:.1f} fps, hands/frame {r['hands_per_frame']:.2f}, "
                  f"capture->landmarks p50 {r['capture_to_landmarks_ms']['p50']:.2f} "
                  f"p95 {r['capture_to_landmarks_ms']['p95']:.2f} p99 {r['capture_to_landmarks_ms']['p99']:.2f} ms")
            continue
        decoder = NvJpegDecoder(1280, 720, stream=tracker.stream)
        res[name] = {}
        for clip in args.clips:
            rp = os.path.join(ROOT, "results", f"mp_ref_{clip}.npz")
            ref = np.load(rp) if os.path.exists(rp) else None
            res[name][clip] = run_clip(tracker, clip, ref, decoder)
            if ref is not None and "ref" not in res:
                res.setdefault("ref", {})
            if ref is not None:
                std, hf = ref_jitter(ref)
                res.setdefault("ref", {})[clip] = {"ms": stats(ref["ms"][10:]), "tip_std_px": std,
                                                   "tip_jitter_px": hf,
                                                   "hand_frames": int((ref["handedness"] >= 0).sum())}
        decoder.close()
        del tracker
        cp.get_default_memory_pool().free_all_blocks()

    if not args.live:
        f2 = lambda v: "   -  " if v is None else f"{v:6.2f}"  # noqa: E731
        for name in args.trackers:
            print(f"\n== {name} ==")
            print(f"{'clip':20s} {'fps':>6s} {'trk p50':>7s} {'p95':>6s} {'p99':>6s} {'det p50':>7s} {'det%':>5s} "
                  f"{'lm p50':>6s} {'hands':>5s}/{'ref':<5s} {'err':>6s} {'tips':>6s} {'ids':>3s} {'sw':>3s} {'flip':>4s} "
                  f"{'tipStd':>6s} {'jitter':>6s}")
            for clip, r in res[name].items():
                e = r.get("err_px", {})
                print(f"{clip:20s} {r['fps_decode_track']:6.1f} {r['track_total_ms']['p50']:7.2f} "
                      f"{r['track_total_ms']['p95']:6.2f} {r['track_total_ms']['p99']:6.2f} "
                      f"{f2(r['detect_ms']['p50'] if r['detect_ms'] else None):>7s} {r['detector_ran_frac']*100:5.0f} "
                      f"{f2(r['landmark_ms']['p50'] if r['landmark_ms'] else None)} "
                      f"{r['hand_frames']:5d}/{r.get('ref_hand_frames', 0):<5d} {f2(e.get('mean'))} {f2(e.get('tips'))} "
                      f"{r['track_ids']:3d} {r['id_switches']:3d} {r['handedness_flips']:4d} "
                      f"{f2(r.get('tip_std_px'))} {f2(r.get('tip_jitter_px'))}")
        if "ref" in res:
            print("\n== MediaPipe CPU reference (full-res frames, 2 hands) ==")
            for clip, r in res["ref"].items():
                print(f"{clip:20s} per-frame p50 {r['ms']['p50']:6.1f} ms p95 {r['ms']['p95']:6.1f}  "
                      f"hands {r['hand_frames']:4d}  tipStd {r['tip_std_px']:6.2f}  jitter {r['tip_jitter_px']:6.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

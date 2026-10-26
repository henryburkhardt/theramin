#!/usr/bin/env python3
"""Milestone 4: track stability and filter behaviour on recorded clips.

Runs the full HandTracker (stable IDs, handedness smoothing, One Euro filter)
offline on clips and reports:
  ids / switches / flips   distinct track IDs, frames where the set of IDs changed
                           while the hand count stayed the same, handedness changes
                           within a track, frames with two hands sharing a label
  jitter                   still_hand: high-frequency fingertip jitter (px), raw vs
                           filtered (residual after a centred 9-frame moving average)
  lag                      fast: mean distance (px) between filtered and raw fingertips,
                           and the lag (ms) that best aligns filtered with raw
  prediction               with --predict-ms: error of predicted fingertips vs the raw
                           position predict_ms later (px), vs no prediction

  python scripts/eval_tracking.py
  python scripts/eval_tracking.py --sweep            # filter parameter sweep
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_models import CLIPS, TIPS, jitter  # noqa: E402
from record_clip import load_clip  # noqa: E402

from handtrack import config as cfgmod  # noqa: E402
from handtrack.nvjpeg import NvJpegDecoder  # noqa: E402
from handtrack.tracker import HandTracker  # noqa: E402

_cache = {}


def decoded(clip, dec):
    """Raw frames for a clip (decoded per run; nvJPEG is fast)."""
    if clip not in _cache:
        _cache[clip] = load_clip(os.path.join(ROOT, "clips", clip))
    return _cache[clip]


def run(tracker, clip, dec):
    tracker.reset()
    frames = decoded(clip, dec)
    out = []
    for seq, t_cap, j in frames:
        out.append(tracker.process(dec.decode(j), seq, t_cap))
        tracker.after_publish()
    return out


def stability(hfs):
    """jumps: a track's wrist moving > 150 px between consecutive frames, i.e. the
    ID passed to another hand (an identity swap the ID-set metrics can't see)."""
    ids, switches, flips, same_label, jumps = set(), 0, 0, 0, 0
    last_label, prev, last_pos = {}, None, {}
    for i, hf in enumerate(hfs):
        cur = {h.track_id for h in hf.hands}
        ids |= cur
        if prev is not None and len(cur) == len(prev) and cur != prev:
            switches += 1
        prev = cur
        labels = [h.handedness for h in hf.hands]
        if len(labels) == 2 and labels[0] == labels[1]:
            same_label += 1
        for h in hf.hands:
            if h.track_id in last_label and last_label[h.track_id] != h.handedness:
                flips += 1
            last_label[h.track_id] = h.handedness
            lp = last_pos.get(h.track_id)
            if lp is not None and lp[0] == i - 1 and np.linalg.norm(h.landmarks_px[0, :2] - lp[1]) > 150:
                jumps += 1
            last_pos[h.track_id] = (i, h.landmarks_px[0, :2])
    return {"ids": len(ids), "switches": switches, "flips": flips, "jumps": jumps, "same_label_frames": same_label,
            "hand_frames": sum(len(hf.hands) for hf in hfs)}


def longest_track(hfs, key):
    """(T, 5, 2) fingertips in px for the longest continuous run of one track id."""
    W, H = 1280, 720
    best = {}
    for i, hf in enumerate(hfs):
        for h in hf.hands:
            best.setdefault(h.track_id, []).append((i, getattr(h, key)[TIPS, :2] * [W, H]))
    if not best:
        return None, None
    seq = max(best.values(), key=len)
    idx = np.array([i for i, _ in seq])
    runs = np.split(np.arange(len(idx)), np.where(np.diff(idx) != 1)[0] + 1)
    run_ = max(runs, key=len)
    return idx[run_], np.stack([seq[r][1] for r in run_])


def lag_metrics(hfs):
    idx, raw = longest_track(hfs, "landmarks_norm")
    _, filt = longest_track(hfs, "landmarks_filtered")
    if raw is None or len(raw) < 20:
        return {}
    dist = float(np.linalg.norm(filt - raw, axis=2).mean())
    # lag: shift raw forward by k frames, find k minimising error vs filtered (sub-frame via parabola)
    errs = [np.linalg.norm(filt[k:] - raw[:len(raw) - k], axis=2).mean() for k in range(0, 6)]
    k = int(np.argmin(errs))
    if 0 < k < 5:
        a, b, c = errs[k - 1], errs[k], errs[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return {"filt_raw_dist_px": dist, "lag_ms": float(k) * 1000 / 30}


def predict_metrics(hfs, predict_ms):
    """Predicted (filtered + v*dt) vs the raw position predict_ms later."""
    W, H = 1280, 720
    k = max(1, round(predict_ms / (1000 / 30)))
    tracks = {}
    for i, hf in enumerate(hfs):
        for h in hf.hands:
            tracks.setdefault(h.track_id, {})[i] = h
    e_pred, e_filt, e_raw = [], [], []
    for frames in tracks.values():
        for i, h in frames.items():
            later = frames.get(i + k)
            if later is None or h.landmarks_predicted is None:
                continue
            target = later.landmarks_norm[TIPS, :2] * [W, H]
            e_pred.append(np.linalg.norm(h.landmarks_predicted[TIPS, :2] * [W, H] - target, axis=1).mean())
            e_filt.append(np.linalg.norm(h.landmarks_filtered[TIPS, :2] * [W, H] - target, axis=1).mean())
            e_raw.append(np.linalg.norm(h.landmarks_norm[TIPS, :2] * [W, H] - target, axis=1).mean())
    if not e_pred:
        return {}
    return {"frames_ahead": k, "err_pred_px": float(np.mean(e_pred)), "err_filtered_px": float(np.mean(e_filt)),
            "err_raw_now_px": float(np.mean(e_raw)), "err_pred_p95": float(np.percentile(e_pred, 95))}


def evaluate(cfg, dec_holder, clips, verbose=True):
    tracker = HandTracker(cfg)
    if "dec" not in dec_holder:
        dec_holder["dec"] = NvJpegDecoder(cfg.camera.width, cfg.camera.height, stream=tracker.stream)
    dec = dec_holder["dec"]
    dec.stream = tracker.stream
    res = {}
    for clip in clips:
        hfs = run(tracker, clip, dec)
        r = stability(hfs)
        if clip == "still_hand":
            for key in ("landmarks_norm", "landmarks_filtered"):
                _, tr = longest_track(hfs, key)
                r["jitter_raw_px" if key == "landmarks_norm" else "jitter_filtered_px"] = jitter(tr)[1]
        if clip in ("fast", "slow"):
            r.update(lag_metrics(hfs))
        if cfg.filter.predict_ms and clip in ("fast", "slow"):
            r["predict"] = predict_metrics(hfs, cfg.filter.predict_ms)
        res[clip] = r
        if verbose:
            print(f"  {clip:20s} " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in r.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=CLIPS)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--predict-ms", type=float, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like filter.beta=10 tracking.detect_interval=3")
    ap.add_argument("--out")
    args = ap.parse_args()
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=")
        sec, key = k.split(".")
        overrides.setdefault(sec, {})[key] = json.loads(v) if v[0] in "[{0123456789-tfn" else v
    if args.predict_ms:
        overrides.setdefault("filter", {})["predict_ms"] = args.predict_ms
    holder = {}
    if args.sweep:
        rows = []
        for mc in (0.3, 0.6, 1.0, 1.5, 2.5):
            for beta in (1.0, 3.0, 10.0, 30.0, 100.0):
                o = {**overrides, "filter": {**overrides.get("filter", {}), "min_cutoff": mc, "beta": beta}}
                r = evaluate(cfgmod.load(overrides=o), holder, ["still_hand", "fast"], verbose=False)
                rows.append((mc, beta, r["still_hand"]["jitter_filtered_px"], r["fast"].get("lag_ms"),
                             r["fast"].get("filt_raw_dist_px")))
                print(f"min_cutoff {mc:4.1f} beta {beta:6.1f}: still jitter {rows[-1][2]:.3f} px | fast lag "
                      f"{rows[-1][3]:5.1f} ms, filtered-raw {rows[-1][4]:5.2f} px", flush=True)
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=1)
        return
    res = evaluate(cfgmod.load(overrides=overrides), holder, args.clips)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

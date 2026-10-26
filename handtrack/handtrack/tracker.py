"""Hand tracking engine: stable tracks, handedness, filtering, and the public API.

  HandTracker   per-frame logic on a decoded GPU image (usable offline on clips)
  TrackingService
                camera source + HandTracker in a thread; subscribe()/latest()

Coordinates: landmarks_px are camera-image pixels; *_norm/*_filtered are
normalized 0..1 by image width/height (z scaled like x); velocity is normalized
units per second. Coordinates are always in the camera image as captured;
only the handedness labels depend on tracking.camera_mirrored.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import config as cfgmod
from .filters import OneEuroFilter
from .mediapipe_hands import MediaPipeHands
from .trt_runner import build_engine


@dataclass
class Hand:
    track_id: int
    handedness: str                 # "Left" / "Right": the person's actual hand
    score: float                    # landmark-model hand presence
    landmarks_px: np.ndarray        # (21, 3) raw, pixel x, y + relative z (pixels)
    landmarks_norm: np.ndarray      # (21, 3) raw, normalized 0..1
    landmarks_filtered: np.ndarray  # (21, 3) One Euro filtered, normalized
    velocity: np.ndarray            # (21, 2) normalized units per second
    handedness_score: float = 0.5   # smoothed confidence of `handedness`
    landmarks_predicted: Optional[np.ndarray] = None  # (21, 3) if filter.predict_ms > 0


@dataclass
class HandFrame:
    frame_id: int
    t_capture_ns: int               # kernel timestamp of the frame's arrival from the camera
    t_publish_ns: int               # when landmarks were ready
    hands: list = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)


class _Track:
    def __init__(self, tid, p_right, label, cfg):
        self.id = tid
        self.p = p_right            # EMA of the model's P("Right"), MediaPipe convention
        self.label = label          # MediaPipe-convention label
        self.filter = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self.last_seen_ns = 0
        self.wrist = None


class HandTracker:
    def __init__(self, cfg=None, stream=None):
        self.cfg = cfg or cfgmod.load()
        c, t = self.cfg, self.cfg.tracking
        det = build_engine(cfgmod.resolve(c.models.detector), max_batch=1)
        lm = build_engine(cfgmod.resolve(c.models.landmarks), max_batch=max(2, t.max_hands))
        h = c.camera.height
        area = None
        if t.play_area:
            x0, y0, x1, y1 = t.play_area
            area = (x0 * c.camera.width, y0 * h, x1 * c.camera.width, y1 * h)
        self.model = MediaPipeHands(det, lm, max_hands=t.max_hands, min_detection=t.min_detection,
                                    min_presence=t.min_presence, detect_interval=t.detect_interval,
                                    min_palm_px=t.min_palm_size * h, play_area=area, stream=stream)
        self.stream = self.model.stream
        self.reset()

    def reset(self):
        self.model.reset()
        self.tracks = {}            # stable id -> _Track
        self.by_roi = {}            # MediaPipeHands ROI id -> stable id
        self.label_tracks = {}      # one_person: MediaPipe-convention label -> _Track
        self.roi_p = {}             # one_person: ROI id -> smoothed P("Right")
        self._next_id = 0

    def _track_for_label(self, label):
        tr = self.label_tracks.get(label)
        if tr is None:
            tr = _Track(self._next_id, 0.9 if label == "Right" else 0.1, label, self.cfg.filter)
            self._next_id += 1
            self.label_tracks[label] = self.tracks[tr.id] = tr
        return tr

    def _assign_one_person(self, raw_hands):
        """One person, up to two hands: identity follows handedness. The landmark
        model's per-frame handedness is decisive (>99% of frames below 0.1 or above
        0.9 on our clips), and when an ROI slides onto the other hand during a crossing
        it flips at once, so a short per-ROI average that resets on a clear flip
        tracks the right hand. Two ROIs confidently on the same hand are a duplicate:
        keep the more confident one and free the other ROI."""
        a = self.cfg.tracking.handedness_alpha
        self.roi_p = {r: p for r, p in self.roi_p.items() if r in {h.track_id for h in raw_hands}}
        for h in raw_hands:
            e = self.roi_p.get(h.track_id)
            e = h.p_right if e is None or abs(h.p_right - e) > 0.5 else (1 - a) * e + a * h.p_right
            self.roi_p[h.track_id] = e
        labels = ["Right" if self.roi_p[h.track_id] > 0.5 else "Left" for h in raw_hands]
        if len(raw_hands) == 2 and labels[0] == labels[1]:
            e0, e1 = (self.roi_p[h.track_id] for h in raw_hands)
            if min(abs(e0 - 0.5), abs(e1 - 0.5)) > 0.3:          # both sure: same hand twice
                drop = min(range(2), key=lambda i: raw_hands[i].presence)
                gone = raw_hands[drop].track_id
                self.model.tracked = [(r, rect) for r, rect in self.model.tracked if r != gone]
                raw_hands = [raw_hands[1 - drop]]
                labels = [labels[1 - drop]]
            else:                                                # unsure: likelier Right wins
                labels = ["Right", "Left"] if e0 >= e1 else ["Left", "Right"]
        out = []
        for h, label in zip(raw_hands, labels):
            tr = self._track_for_label(label)
            tr.p = self.roi_p[h.track_id]
            out.append((h, tr))
        return out

    # ---- identity ---------------------------------------------------------------
    def _new_or_reused(self, raw, active_labels, t_ns):
        t = self.cfg.tracking
        label = "Right" if raw.p_right > 0.5 else "Left"
        if t.one_person and t.max_hands == 2 and active_labels:
            other = next(iter(active_labels))
            label = "Left" if other == "Right" else "Right"
        lost = [tr for tr in self.tracks.values()
                if tr.id not in self.by_roi.values() and (t_ns - tr.last_seen_ns) < t.reuse_seconds * 1e9]
        if t.one_person:
            lost = [tr for tr in lost if tr.label == label]
        else:
            w = raw.landmarks_px[0, :2] / [self.cfg.camera.width, self.cfg.camera.height]
            lost = [tr for tr in lost if tr.wrist is not None and np.linalg.norm(tr.wrist - w) < 0.15]
        if lost:
            tr = max(lost, key=lambda tr: tr.last_seen_ns)
            tr.filter.reset()       # it reappeared somewhere else; don't smear from the old spot
            return tr
        tr = _Track(self._next_id, 0.9 if label == "Right" else 0.1, label, self.cfg.filter)
        self._next_id += 1
        self.tracks[tr.id] = tr
        return tr

    def _update_handedness(self, tr, p_right):
        t = self.cfg.tracking
        tr.p = (1 - t.handedness_alpha) * tr.p + t.handedness_alpha * p_right
        if tr.label == "Left" and tr.p > 0.5 + t.handedness_hysteresis:
            tr.label = "Right"
        elif tr.label == "Right" and tr.p < 0.5 - t.handedness_hysteresis:
            tr.label = "Left"

    def _assign_by_roi(self, raw_hands, t_ns):
        """General case: a track follows its ROI; new ROIs may take over a recently
        lost track (by handedness when one_person, else by wrist position)."""
        seen_rois = {h.track_id for h in raw_hands}
        self.by_roi = {r: s for r, s in self.by_roi.items() if r in seen_rois}
        active = {}
        for raw in raw_hands:                       # known ROIs first, then new ones
            if raw.track_id in self.by_roi:
                active[raw.track_id] = self.tracks[self.by_roi[raw.track_id]]
        for raw in raw_hands:
            if raw.track_id not in self.by_roi:
                tr = self._new_or_reused(raw, {a.label for a in active.values()}, t_ns)
                self.by_roi[raw.track_id] = tr.id
                active[raw.track_id] = tr
        for raw in raw_hands:
            self._update_handedness(active[raw.track_id], raw.p_right)
        horizon = self.cfg.tracking.reuse_seconds * 1e9
        self.tracks = {k: v for k, v in self.tracks.items()
                       if v.id in self.by_roi.values() or t_ns - v.last_seen_ns < horizon}
        return [(r, active[r.track_id]) for r in raw_hands]

    # ---- per frame ----------------------------------------------------------------
    def process(self, image, frame_id, t_capture_ns, defer_detect=None):
        """defer_detect (default: tracking.defer_detection): see MediaPipeHands.process;
        call after_publish() once the HandFrame has been handed out."""
        c = self.cfg
        W, H = c.camera.width, c.camera.height
        if defer_detect is None:
            defer_detect = c.tracking.defer_detection
        raw_hands = self.model.process(image, defer_detect=defer_detect)
        t0 = time.perf_counter()
        if c.tracking.one_person and c.tracking.max_hands <= 2:
            pairs = self._assign_one_person(raw_hands)
        else:
            pairs = self._assign_by_roi(raw_hands, t_capture_ns)

        hands = []
        t_s = t_capture_ns / 1e9
        scale = np.array([W, H, W], np.float64)
        for raw, tr in pairs:
            norm = raw.landmarks_px / scale
            # Reset the filter if the track reappears after a gap or jumps (a different
            # physical hand took over the ID); otherwise it would smear across the frame.
            if tr.wrist is not None and (t_capture_ns - tr.last_seen_ns > 0.2e9
                                         or np.linalg.norm(norm[0, :2] - tr.wrist) > 0.1):
                tr.filter.reset()
            filt, vel = tr.filter(norm, t_s)
            pred = None
            if c.filter.predict_ms:
                pred = filt.copy()
                pred[:, :2] += vel[:, :2] * c.filter.predict_ms / 1e3
            label = tr.label if c.tracking.camera_mirrored else ("Left" if tr.label == "Right" else "Right")
            hands.append(Hand(tr.id, label, raw.presence, raw.landmarks_px, norm.astype(np.float32),
                              filt.astype(np.float32), vel[:, :2].astype(np.float32),
                              handedness_score=float(tr.p if tr.label == "Right" else 1 - tr.p),
                              landmarks_predicted=None if pred is None else pred.astype(np.float32)))
            tr.last_seen_ns = t_capture_ns
            tr.wrist = norm[0, :2]
        hands.sort(key=lambda h: h.track_id)
        mt = self.model.last_times
        stage = {"detect": mt.detect_ms if mt.det_ran else 0.0, "landmarks": mt.landmark_ms,
                 "post_filter": (time.perf_counter() - t0) * 1e3, "detector_ran": mt.det_ran}
        return HandFrame(frame_id, t_capture_ns, time.monotonic_ns(), hands, stage)

    def after_publish(self):
        """Runs deferred palm detection (if any) for the frame just processed."""
        return self.model.detect_deferred()


class _Rolling:
    def __init__(self, n=300):
        self.v = deque(maxlen=n)

    def add(self, x):
        self.v.append(x)

    def pct(self, q):
        return float(np.percentile(self.v, q)) if self.v else float("nan")

    def mean(self):
        return float(np.mean(self.v)) if self.v else float("nan")


class TrackingService:
    """Camera -> decode -> HandTracker in a background thread. Publishes one
    HandFrame per processed frame. Callbacks run on the tracking thread, so keep
    them short (hand work to another thread, as the UDP sender does)."""

    def __init__(self, cfg=None):
        from .capture import GstSource, LatestFrame, MjpegSource
        self.cfg = cfg or cfgmod.load()
        c = self.cfg.camera
        self.tracker = HandTracker(self.cfg)
        if self.cfg.decode == "gst":
            self.source = GstSource("gst-nvv4l2-mjpeg", c.device, c.width, c.height, c.fps)
        else:
            self.source = MjpegSource(c.device, c.width, c.height, c.fps, c.buffers, decoder="nvjpeg")
        self._latest = LatestFrame()            # (HandFrame, jpeg or None) for viewers
        self._callbacks = []
        self._stop = threading.Event()
        self.error = None
        self._stats = {k: _Rolling() for k in ("transfer", "decode", "detect", "detect_deferred", "landmarks",
                                                "post_filter", "publish", "track_total", "latency")}
        self._cam_times = deque(maxlen=120)
        self._proc_times = deque(maxlen=120)
        self._det_ran = deque(maxlen=300)
        self._warmup = 30           # first frames include CUDA/NVRTC start-up; keep them out of stats
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.source.stop()

    def subscribe(self, callback: Callable[[HandFrame], None]):
        self._callbacks.append(callback)

    def latest(self) -> Optional[HandFrame]:
        r = self._latest.wait_newer(0, timeout=0)
        return r[1][0] if r else None

    def wait_newer(self, version, timeout=1.0):
        """For viewers: (version, (HandFrame, jpeg)) newer than version, or None."""
        if self.error:
            raise self.error
        return self._latest.wait_newer(version, timeout)

    def _run(self):
        try:
            version = 0
            while not self._stop.is_set():
                r = self.source.wait_newer(version, timeout=0.5)
                if r is None:
                    continue
                version, f = r
                t0 = time.perf_counter()
                hf = self.tracker.process(f.image, f.frame_id, f.t_capture_ns)
                t1 = time.perf_counter()
                # stage_ms is complete before subscribers see the frame
                hf.stage_ms["track_total"] = (t1 - t0) * 1e3
                if f.t_dequeue_ns:
                    hf.stage_ms["transfer"] = (f.t_dequeue_ns - f.t_capture_ns) / 1e6
                    hf.stage_ms["decode"] = (f.t_decoded_ns - f.t_dequeue_ns) / 1e6
                hf.stage_ms["latency"] = (hf.t_publish_ns - hf.t_capture_ns) / 1e6
                for cb in self._callbacks:
                    cb(hf)
                t2 = time.perf_counter()
                hf.stage_ms["publish"] = (t2 - t1) * 1e3
                # Off the latency path: palm detection deferred from this frame.
                deferred = self.tracker.after_publish()
                if deferred:
                    hf.stage_ms["detect_deferred"] = deferred
                if self._warmup:
                    self._warmup -= 1
                else:
                    for k, s in self._stats.items():
                        if k in hf.stage_ms and not (k == "detect" and not hf.stage_ms.get("detector_ran")):
                            s.add(hf.stage_ms[k])
                    self._cam_times.append(f.t_capture_ns)
                    self._proc_times.append(time.monotonic_ns())
                    self._det_ran.append(bool(hf.stage_ms.get("detector_ran")) or bool(deferred))
                self._latest.put((hf, getattr(f, "jpeg", None)))
        except Exception as e:
            self.error = e

    @staticmethod
    def _rate(ts):
        return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def stats(self):
        s = {k: {"mean": v.mean(), "p50": v.pct(50), "p95": v.pct(95)} for k, v in self._stats.items()}
        s["camera_fps"] = self._rate(self._cam_times)
        s["processing_fps"] = self._rate(self._proc_times)
        s["detector_ran_frac"] = float(np.mean(self._det_ran)) if self._det_ran else 0.0
        return s

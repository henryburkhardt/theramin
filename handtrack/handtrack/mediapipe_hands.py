"""MediaPipe Hands (palm detector + hand landmark "full") on TensorRT.

Re-implements the MediaPipe Tasks HandLandmarker graph around TensorRT engines.
The constants follow MediaPipe's calculators: SsdAnchors, TensorsToDetections,
weighted NMS, DetectionsToRects, RectTransformation, HandLandmarksToRect,
LandmarkProjection. All geometry is in source-image pixels; a rect is
(cx, cy, w, h, rotation_rad).

Per frame (detect once, then track):
  1. hands tracked from the previous frame give their next ROI from landmarks;
  2. only if fewer than max_hands are tracked, run the palm detector and add
     ROIs that don't overlap tracked ones;
  3. one batched landmark call for all ROIs; drop hands below min_presence.
"""
import math
import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from .gpu_ops import AffineWarper
from .trt_runner import TrtModel

DET_SIZE = 192
LM_SIZE = 224
# Landmarks HandLandmarksToRect uses: wrist, thumb CMC/MCP/IP, and MCP+PIP of each finger.
PARTIAL = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
# Palm landmarks (wrist, thumb CMC, finger MCPs) for the hand-size filter.
PALM = [0, 1, 5, 9, 13, 17]


def palm_size(lm):
    """Longest side of the palm landmarks' bounding box, in pixels."""
    p = lm[PALM, :2]
    return float((p.max(0) - p.min(0)).max())


def ssd_anchors():
    """SsdAnchorsCalculator for palm_detection_full: 192x192, strides 8,16,16,16,
    fixed anchor size, 2 anchors per cell -> 24*24*2 + 12*12*6 = 2016."""
    strides = [8, 16, 16, 16]
    anchors = []
    i = 0
    while i < len(strides):
        n = 0
        j = i
        while j < len(strides) and strides[j] == strides[i]:
            n += 2  # aspect ratio 1.0 + interpolated scale, per layer
            j += 1
        grid = math.ceil(DET_SIZE / strides[i])
        for y in range(grid):
            for x in range(grid):
                for _ in range(n):
                    anchors.append(((x + 0.5) / grid, (y + 0.5) / grid))
        i = j
    return np.array(anchors, np.float32)


def _norm_angle(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def _transform_rect(cx, cy, w, h, rot, scale, shift_y):
    """RectTransformation (shift in the rect's rotated frame, square_long, scale)."""
    cx += -h * shift_y * math.sin(rot)
    cy += h * shift_y * math.cos(rot)
    s = max(w, h) * scale
    return cx, cy, s, s, rot


def rect_to_matrix(rect, size):
    """Affine (2x3) mapping output pixel (u, v) in a size x size crop -> source pixel."""
    cx, cy, w, h, rot = rect
    c, s = math.cos(rot), math.sin(rot)
    a, b = w / size, h / size
    return np.array([[a * c, -b * s, cx - 0.5 * w * c + 0.5 * h * s],
                     [a * s, b * c, cy - 0.5 * w * s - 0.5 * h * c]], np.float32)


def weighted_nms(boxes, kps, scores, iou_thresh, max_out):
    """MediaPipe WEIGHTED non-max suppression. boxes (N, 4) as x0, y0, x1, y1."""
    order = np.argsort(-scores)
    boxes, kps, scores = boxes[order], kps[order], scores[order]
    out = []
    remaining = np.arange(len(scores))
    while len(remaining) and len(out) < max_out:
        top = boxes[remaining[0]]
        b = boxes[remaining]
        ix0, iy0 = np.maximum(top[0], b[:, 0]), np.maximum(top[1], b[:, 1])
        ix1, iy1 = np.minimum(top[2], b[:, 2]), np.minimum(top[3], b[:, 3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / ((top[2] - top[0]) * (top[3] - top[1]) + area - inter + 1e-9)
        group = remaining[iou > iou_thresh]
        w = scores[group][:, None]
        out.append((((boxes[group] * w).sum(0) / w.sum()),
                    ((kps[group].reshape(len(group), -1) * w).sum(0) / w.sum()).reshape(-1, 2),
                    scores[remaining[0]]))
        remaining = remaining[iou <= iou_thresh]
    return out


def _rect_iou(r1, r2):
    """IoU of the axis-aligned bounds of two rotated rects (cheap overlap test)."""
    def bounds(r):
        cx, cy, w, h, rot = r
        c, s = abs(math.cos(rot)), abs(math.sin(rot))
        hw, hh = (w * c + h * s) / 2, (w * s + h * c) / 2
        return cx - hw, cy - hh, cx + hw, cy + hh
    a, b = bounds(r1), bounds(r2)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class HandResult:
    track_id: int                # internal ROI id; stable IDs come from tracker.HandTracker
    handedness: str              # MediaPipe convention (assumes a mirrored/selfie image)
    handedness_score: float      # confidence of `handedness`; p_right = model's P("Right")
    presence: float
    landmarks_px: np.ndarray     # (21, 3): x, y in source pixels, z relative (pixels)
    world: np.ndarray            # (21, 3) metric world landmarks from the model
    rect: tuple                  # ROI the landmarks came from
    p_right: float = 0.5


@dataclass
class StageTimes:
    det_ran: bool = False
    detect_ms: float = 0.0       # warp + inference + decode/NMS (0 if not run)
    landmark_ms: float = 0.0     # warp + batched inference + projection
    total_ms: float = 0.0
    n_rois: int = 0
    extra: dict = field(default_factory=dict)


class MediaPipeHands:
    """detect_interval: while at least one hand is tracked but fewer than max_hands,
    run the palm detector only every N frames (1 = every frame, like MediaPipe).
    min_palm_px / play_area (x0, y0, x1, y1 pixels) reject small (background) hands
    and hands outside the playing area, both for new detections and tracked hands."""

    def __init__(self, det_engine, lm_engine, max_hands=2, min_detection=0.5, min_presence=0.5,
                 nms_iou=0.3, track_iou=0.5, detect_interval=1, min_palm_px=0.0, play_area=None,
                 stream=None):
        self.stream = stream or cp.cuda.Stream(non_blocking=True)
        self.det = TrtModel(det_engine, self.stream)
        self.lm = TrtModel(lm_engine, self.stream)
        assert self.lm.max_batch >= max_hands, "landmark engine batch < max_hands"
        self.det_in = self.det.inputs["input_1"]
        self.lm_in = self.lm.inputs["input_1"]
        # MediaPipe pads the detector letterbox with zeros but replicates edge
        # pixels for landmark crops; zero-padded crops cost 19 px mean error
        # vs MediaPipe on a hand at the frame edge (2.4 px with replicate).
        self.det_warp = AffineWarper(1, self.stream)
        self.lm_warp = AffineWarper(max_hands * 2, self.stream, replicate=True)
        self.anchors = ssd_anchors()
        self.max_hands, self.min_detection, self.min_presence = max_hands, min_detection, min_presence
        self.nms_iou, self.track_iou = nms_iou, track_iou
        self.detect_interval, self.min_palm_px, self.play_area = detect_interval, min_palm_px, play_area
        self._since_detect = 0
        self._deferred_image = None
        self.deferred_ms = 0.0
        self.tracked = []     # [(track_id, rect)] ROIs for the next frame
        self._next_id = 0
        self.last_times = StageTimes()

    def reset(self):
        self.tracked = []
        self._since_detect = 0
        self._deferred_image = None

    def _in_play_area(self, x, y):
        a = self.play_area
        return a is None or (a[0] <= x <= a[2] and a[1] <= y <= a[3])

    # ---- palm detection ----------------------------------------------------
    def _detect(self, image):
        H, W = image.shape[:2]
        s = DET_SIZE / max(W, H)                      # letterbox, keep aspect ratio
        pad_x, pad_y = (DET_SIZE - W * s) / 2, (DET_SIZE - H * s) / 2
        m = np.array([[1 / s, 0, -pad_x / s], [0, 1 / s, -pad_y / s]], np.float32)
        self.det_warp(image, self.det_in, [m])
        self.det.run(1)
        raw = cp.asnumpy(self.det.outputs["Identity"][0], stream=self.stream)
        logits = cp.asnumpy(self.det.outputs["Identity_1"][0, :, 0], stream=self.stream)
        self.stream.synchronize()
        scores = 1 / (1 + np.exp(-np.clip(logits, -100, 100)))
        keep = scores >= self.min_detection
        if not keep.any():
            return []
        raw, scores, anc = raw[keep], scores[keep], self.anchors[keep]
        cxy = raw[:, 0:2] / DET_SIZE + anc
        wh = raw[:, 2:4] / DET_SIZE
        boxes = np.concatenate([cxy - wh / 2, cxy + wh / 2], 1)
        kps = raw[:, 4:18].reshape(-1, 7, 2) / DET_SIZE + anc[:, None, :]
        rects = []
        for box, kp, score in weighted_nms(boxes, kps, scores, self.nms_iou, self.max_hands):
            # undo letterbox: normalized 192-square coords -> source pixels
            box = (box * DET_SIZE - [pad_x, pad_y, pad_x, pad_y]) / s
            kp = (kp * DET_SIZE - [pad_x, pad_y]) / s
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            w, h = box[2] - box[0], box[3] - box[1]
            # The palm box is a bit larger than the palm landmarks' bbox; 0.8 keeps
            # borderline hands for the landmark-based size check after inference.
            if max(w, h) < 0.8 * self.min_palm_px or not self._in_play_area(cx, cy):
                continue
            rot = _norm_angle(math.pi / 2 - math.atan2(-(kp[2, 1] - kp[0, 1]), kp[2, 0] - kp[0, 0]))
            rects.append((_transform_rect(cx, cy, w, h, rot, 2.6, -0.5), float(score)))
        return rects

    # ---- landmarks ----------------------------------------------------------
    @staticmethod
    def rect_from_landmarks(lm):
        """HandLandmarksToRect + RectTransformation(scale 2.0, shift_y -0.1)."""
        p = lm[PARTIAL, :2]
        x0, y0 = lm[0, 0], lm[0, 1]
        x1 = ((lm[5, 0] + lm[13, 0]) / 2 + lm[9, 0]) / 2
        y1 = ((lm[5, 1] + lm[13, 1]) / 2 + lm[9, 1]) / 2
        rot = _norm_angle(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
        ac = (p.max(0) + p.min(0)) / 2
        c, s = math.cos(-rot), math.sin(-rot)
        d = p - ac
        proj = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
        pmin, pmax = proj.min(0), proj.max(0)
        pc = (pmin + pmax) / 2
        cr, sr = math.cos(rot), math.sin(rot)
        cx = pc[0] * cr - pc[1] * sr + ac[0]
        cy = pc[0] * sr + pc[1] * cr + ac[1]
        w, h = pmax - pmin
        return _transform_rect(cx, cy, w, h, rot, 2.0, -0.1)

    def _landmarks(self, image, rects):
        n = len(rects)
        self.lm_warp(image, self.lm_in, [rect_to_matrix(r, LM_SIZE) for r in rects])
        self.lm.run(n)
        lms = cp.asnumpy(self.lm.outputs["Identity"][:n], stream=self.stream).reshape(n, 21, 3)
        pres = cp.asnumpy(self.lm.outputs["Identity_1"][:n, 0], stream=self.stream)
        hand = cp.asnumpy(self.lm.outputs["Identity_2"][:n, 0], stream=self.stream)
        world = cp.asnumpy(self.lm.outputs["Identity_3"][:n], stream=self.stream).reshape(n, 21, 3)
        self.stream.synchronize()
        out = []
        for i, (cx, cy, w, h, rot) in enumerate(rects):
            a = lms[i, :, 0] / LM_SIZE - 0.5
            b = lms[i, :, 1] / LM_SIZE - 0.5
            c, s = math.cos(rot), math.sin(rot)
            px = np.stack([cx + a * w * c - b * h * s,
                           cy + a * w * s + b * h * c,
                           lms[i, :, 2] / LM_SIZE * w], 1)
            out.append((px, float(pres[i]), float(hand[i]), world[i]))
        return out

    def detect_deferred(self):
        """Run the detection process() deferred, on the same frame; returns ms spent."""
        image, self._deferred_image = self._deferred_image, None
        if image is None:
            self.deferred_ms = 0.0
            return 0.0
        td = time.perf_counter()
        self._since_detect = 0
        for rect, _score in self._detect(image):
            if len(self.tracked) >= self.max_hands:
                break
            if all(_rect_iou(rect, r) < self.track_iou for _, r in self.tracked):
                self.tracked.append((self._next_id, rect))
                self._next_id += 1
        self.deferred_ms = (time.perf_counter() - td) * 1e3
        return self.deferred_ms

    # ---- per frame ----------------------------------------------------------
    def process(self, image, defer_detect=False):
        """image: CuPy (H, W, 3|4) uint8 RGB(A). Returns a list of HandResult.

        defer_detect: if hands are already tracked, don't run the palm detector
        before the landmarks; detect_deferred() runs it afterwards (once the
        results are published) and adds new hands for the next frame. Tracked
        hands then never wait for the detector; a newly appearing second hand
        is picked up one frame later."""
        t0 = time.perf_counter()
        times = StageTimes()
        rois = list(self.tracked)                          # [(track_id, rect)]
        self._since_detect += 1
        due = not rois or (len(rois) < self.max_hands and self._since_detect >= self.detect_interval)
        self._deferred_image = image if (due and rois and defer_detect) else None
        if due and self._deferred_image is None:
            self._since_detect = 0
            td = time.perf_counter()
            dets = self._detect(image)
            times.det_ran = True
            times.detect_ms = (time.perf_counter() - td) * 1e3
            for rect, _score in dets:
                if len(rois) >= self.max_hands:
                    break
                if all(_rect_iou(rect, r) < self.track_iou for _, r in rois):
                    rois.append((self._next_id, rect))
                    self._next_id += 1
        hands = []
        if rois:
            tl = time.perf_counter()
            results = self._landmarks(image, [r for _, r in rois])
            times.landmark_ms = (time.perf_counter() - tl) * 1e3
            for (tid, rect), (px, pres, hand, world) in zip(rois, results):
                if pres < self.min_presence or palm_size(px) < self.min_palm_px \
                        or not self._in_play_area(px[0, 0], px[0, 1]):
                    continue
                label, score = ("Right", hand) if hand > 0.5 else ("Left", 1 - hand)
                hands.append(HandResult(tid, label, score, pres, px, world, rect, p_right=hand))
        # Next-frame ROIs from landmarks; drop duplicates (two ROIs locked on one hand).
        hands.sort(key=lambda h: -h.presence)
        kept, nxt = [], []
        for h in hands:
            r = self.rect_from_landmarks(h.landmarks_px)
            if all(_rect_iou(r, r2) < self.track_iou for r2 in nxt):
                kept.append(h)
                nxt.append(r)
        self.tracked = [(h.track_id, r) for h, r in zip(kept, nxt)]
        if self._deferred_image is not None and len(self.tracked) >= self.max_hands:
            self._deferred_image = None                    # filled up meanwhile; nothing to find
        times.n_rois = len(rois)
        times.total_ms = (time.perf_counter() - t0) * 1e3
        self.last_times = times
        return kept

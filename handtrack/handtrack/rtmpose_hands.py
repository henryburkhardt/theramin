"""RTMPose-m hand (21 keypoints, SimCC, 256x256) + RTMDet-nano hand detector on TensorRT.

Pre/post-processing follows the models' mmdeploy pipeline.json files and rtmlib:
  detector: letterbox to 320 (top-left, pad 114), BGR, mean/std, NCHW; NMS is in the engine.
  pose:     bbox -> centre/scale with padding 1.25 and square aspect, affine crop to 256,
            RGB mean/std, NCHW; SimCC argmax / 2, score = min(max_x, max_y).
By default the detector runs every frame (as rtmlib does) and detections are matched
to the previous frame's hands by IoU to keep track IDs. track_from_keypoints=True
instead feeds each hand's keypoint box back as the next ROI (detect once, then
track), but RTMPose is sensitive to the crop and that loop oscillates: on the
still-hand clip it raised fingertip jitter from 0.8 px to 22 px.
The models have no handedness output; hands are reported as "Unknown".
"""
import time

import cupy as cp
import numpy as np

from .gpu_ops import AffineWarper
from .mediapipe_hands import HandResult, StageTimes
from .trt_runner import TrtModel

DET_SIZE = 320
POSE_SIZE = 256
PADDING = 1.25
MEAN_RGB, STD_RGB = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)


def _box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_to_center_scale(box):
    x0, y0, x1, y1 = box
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    s = max(x1 - x0, y1 - y0) * PADDING          # square model input
    return c, s


class RtmPoseHands:
    def __init__(self, det_engine, pose_engine, max_hands=2, min_detection=0.3, min_kp_score=0.3,
                 track_iou=0.5, track_from_keypoints=False, stream=None):
        self.stream = stream or cp.cuda.Stream(non_blocking=True)
        self.det = TrtModel(det_engine, self.stream)
        self.pose = TrtModel(pose_engine, self.stream)
        assert self.pose.max_batch >= max_hands
        self.det_in = self.det.inputs["input"]
        self.pose_in = self.pose.inputs["input"]
        self.det_warp = AffineWarper(1, self.stream, mean=MEAN_RGB[::-1], std=STD_RGB[::-1],
                                     channels=(2, 1, 0), nchw=True, border=114.0)
        self.pose_warp = AffineWarper(max_hands * 2, self.stream, mean=MEAN_RGB, std=STD_RGB, nchw=True)
        self.max_hands, self.min_detection, self.min_kp_score = max_hands, min_detection, min_kp_score
        self.track_iou = track_iou
        self.track_from_keypoints = track_from_keypoints
        self.tracked = []       # [(track_id, box)]
        self._next_id = 0
        self.last_times = StageTimes()

    def reset(self):
        self.tracked = []

    def _detect(self, image):
        H, W = image.shape[:2]
        r = min(DET_SIZE / W, DET_SIZE / H)
        self.det_warp(image, self.det_in, [[[1 / r, 0, 0], [0, 1 / r, 0]]])
        self.det.run(1)
        self.stream.synchronize()
        dets = cp.asnumpy(self.det.dynamic_output("dets")[0])
        dets = dets[dets[:, 4] >= self.min_detection]
        dets = dets[np.argsort(-dets[:, 4])]
        return [(d[:4] / r, float(d[4])) for d in dets]

    def _pose(self, image, boxes):
        n = len(boxes)
        mats, cs = [], []
        for box in boxes:
            c, s = box_to_center_scale(box)
            k = s / POSE_SIZE
            mats.append([[k, 0, c[0] - s / 2], [0, k, c[1] - s / 2]])
            cs.append((c, s))
        self.pose_warp(image, self.pose_in, mats)
        self.pose.run(n)
        sx = self.pose.outputs["simcc_x"][:n]
        sy = self.pose.outputs["simcc_y"][:n]
        ix, iy = cp.argmax(sx, axis=2), cp.argmax(sy, axis=2)
        vx, vy = cp.max(sx, axis=2), cp.max(sy, axis=2)
        ix, iy, vx, vy = (cp.asnumpy(a, stream=self.stream) for a in (ix, iy, vx, vy))
        self.stream.synchronize()
        out = []
        for i, (c, s) in enumerate(cs):
            kp = np.stack([ix[i], iy[i]], 1).astype(np.float32) / 2.0     # simcc_split_ratio 2
            kp = kp / POSE_SIZE * s + (c - s / 2)
            score = np.minimum(vx[i], vy[i])
            out.append((kp, score))
        return out

    @staticmethod
    def box_from_keypoints(kp):
        x0, y0 = kp.min(0)
        x1, y1 = kp.max(0)
        return np.array([x0, y0, x1, y1], np.float32)

    def process(self, image):
        t0 = time.perf_counter()
        times = StageTimes()
        if self.track_from_keypoints:
            rois = list(self.tracked)
        else:
            rois, prev = [], list(self.tracked)
        if len(rois) < self.max_hands:
            td = time.perf_counter()
            dets = self._detect(image)
            times.det_ran = True
            times.detect_ms = (time.perf_counter() - td) * 1e3
            for box, _score in dets:
                if len(rois) >= self.max_hands:
                    break
                if all(_box_iou(box, b) < self.track_iou for _, b in rois):
                    if not self.track_from_keypoints and prev:
                        # keep the ID of the best-overlapping hand from the last frame
                        j = max(range(len(prev)), key=lambda k: _box_iou(box, prev[k][1]))
                        if _box_iou(box, prev[j][1]) > 0.3:
                            rois.append((prev.pop(j)[0], box))
                            continue
                    rois.append((self._next_id, box))
                    self._next_id += 1
        hands = []
        if rois:
            tl = time.perf_counter()
            results = self._pose(image, [b for _, b in rois])
            times.landmark_ms = (time.perf_counter() - tl) * 1e3
            for (tid, box), (kp, score) in zip(rois, results):
                presence = float(score.mean())
                if presence < self.min_kp_score:
                    continue
                px = np.concatenate([kp, np.zeros((21, 1), np.float32)], 1)
                hands.append(HandResult(tid, "Unknown", 0.0, presence, px, np.zeros((21, 3), np.float32),
                                        tuple(box)))
        hands.sort(key=lambda h: -h.presence)
        kept, nxt = [], []
        for h in hands:
            b = self.box_from_keypoints(h.landmarks_px[:, :2])
            if all(_box_iou(b, b2) < self.track_iou for b2 in nxt):
                kept.append(h)
                nxt.append(b)
        self.tracked = [(h.track_id, b) for h, b in zip(kept, nxt)]
        times.n_rois = len(rois)
        times.total_ms = (time.perf_counter() - t0) * 1e3
        self.last_times = times
        return kept

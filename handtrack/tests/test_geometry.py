import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
cv2 = pytest.importorskip("cv2")

from handtrack.gpu_ops import AffineWarper  # noqa: E402
from handtrack.mediapipe_hands import (MediaPipeHands, rect_to_matrix, ssd_anchors,  # noqa: E402
                                       weighted_nms)


def _cv_matrix(m):
    """Our matrices use pixel centres at +0.5; OpenCV's at integers."""
    m = np.asarray(m, np.float64).copy()
    m[:, 2] += 0.5 * m[:, 0] + 0.5 * m[:, 1] - 0.5
    return m


@pytest.mark.parametrize("replicate", [False, True])
def test_warp_matches_opencv(replicate):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 256, (120, 160, 3), np.uint8), (7, 7), 0)
    rect = (30.0, 40.0, 90.0, 90.0, 0.6)          # partly outside the image
    m = rect_to_matrix(rect, 64)
    stream = cp.cuda.Stream(non_blocking=True)
    out = cp.zeros((1, 64, 64, 3), cp.float32)
    AffineWarper(1, stream, replicate=replicate)(cp.asarray(img), out, [m])
    stream.synchronize()
    got = cp.asnumpy(out[0]) * 255
    ref = cv2.warpAffine(img, _cv_matrix(m), (64, 64), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT).astype(np.float32)
    diff = np.abs(got - ref)
    assert diff.mean() < 0.5 and diff.max() < 4


def test_warp_mean_std_nchw_bgr():
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30            # RGB
    stream = cp.cuda.Stream(non_blocking=True)
    out = cp.zeros((1, 3, 4, 4), cp.float32)
    w = AffineWarper(1, stream, mean=(1, 2, 3), std=(2, 4, 8), channels=(2, 1, 0), nchw=True)
    w(cp.asarray(img), out, [[[1, 0, 0], [0, 1, 0]]])
    stream.synchronize()
    o = cp.asnumpy(out[0])
    assert np.allclose(o[0], (30 - 1) / 2) and np.allclose(o[1], (20 - 2) / 4) and np.allclose(o[2], (10 - 3) / 8)


def test_ssd_anchor_count_and_range():
    a = ssd_anchors()
    assert a.shape == (2016, 2)
    assert a.min() > 0 and a.max() < 1
    assert np.allclose(a[0], [0.5 / 24, 0.5 / 24]) and np.allclose(a[1], a[0])


def test_weighted_nms_merges_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    kps = np.zeros((3, 7, 2), np.float32)
    kps[1] += 1
    scores = np.array([0.9, 0.6, 0.8], np.float32)
    out = weighted_nms(boxes, kps, scores, 0.3, 5)
    assert len(out) == 2
    box, kp, score = out[0]
    assert score == pytest.approx(0.9)
    assert np.allclose(box, (boxes[0] * 0.9 + boxes[1] * 0.6) / 1.5)
    assert np.allclose(kp, 0.6 / 1.5)


def test_rect_from_landmarks_upright_hand():
    # Synthetic upright hand: wrist at bottom, fingers pointing up (-y).
    lm = np.zeros((21, 3), np.float32)
    lm[:, 0] = 100 + np.tile([-20, -10, 0, 10, 20], 5)[:21]
    lm[:, 1] = 200 - np.repeat(np.arange(5), 5)[:21] * 20
    lm[0] = [100, 220, 0]
    lm[5], lm[9], lm[13] = [90, 160, 0], [100, 160, 0], [110, 160, 0]
    cx, cy, w, h, rot = MediaPipeHands.rect_from_landmarks(lm)
    assert abs(rot) < 1e-6                 # wrist->MCPs points straight up = no rotation
    assert w == pytest.approx(h)           # square
    # shift_y = -0.1 moves the centre towards the fingers (up)
    assert cy < 200
    # a 90-degree-rotated hand gives a 90-degree rect
    c, s = 0.0, 1.0
    rot_lm = lm.copy()
    rot_lm[:, 0] = 100 + (lm[:, 0] - 100) * c - (lm[:, 1] - 200) * s
    rot_lm[:, 1] = 200 + (lm[:, 0] - 100) * s + (lm[:, 1] - 200) * c
    assert abs(abs(MediaPipeHands.rect_from_landmarks(rot_lm)[4]) - math.pi / 2) < 1e-5

import math
import types

import numpy as np
import pytest

from handtrack import config as cfgmod
from handtrack.filters import OneEuroFilter
from handtrack.mediapipe_hands import HandResult


def test_one_euro_passes_constant_and_smooths_noise():
    f = OneEuroFilter(min_cutoff=0.3, beta=0.0)
    rng = np.random.default_rng(0)
    xs = 0.5 + rng.normal(0, 0.01, 300)
    out = [f(np.array([x]), i / 30)[0][0] for i, x in enumerate(xs)]
    assert np.std(out[30:]) < 0.3 * np.std(xs)
    f = OneEuroFilter(1.0, 0.0)
    for i in range(10):
        y, dy = f(np.array([2.0]), i / 30)
    assert y[0] == pytest.approx(2.0) and dy[0] == pytest.approx(0.0)


def test_one_euro_beta_reduces_lag():
    t = np.arange(60) / 30
    x = np.where(t > 1.0, 1.0, 0.0)                    # step at t = 1 s
    lag = {}
    for beta in (0.0, 30.0):
        f = OneEuroFilter(0.3, beta)
        y = np.array([f(np.array([v]), ti)[0][0] for v, ti in zip(x, t)])
        lag[beta] = int(np.argmax(y > 0.5)) - int(np.argmax(x > 0.5))
    assert lag[30.0] < lag[0.0]


def test_one_euro_ignores_non_increasing_time():
    f = OneEuroFilter()
    f(np.array([1.0]), 1.0)
    y, _ = f(np.array([5.0]), 1.0)
    assert y[0] == 1.0


class _FakeModel:
    """Stands in for MediaPipeHands: returns scripted HandResults."""
    def __init__(self):
        self.script, self.tracked, self.last_times = [], [], types.SimpleNamespace(
            det_ran=False, detect_ms=0.0, landmark_ms=0.0)

    def reset(self):
        self.tracked = []

    def process(self, image, defer_detect=False):
        return self.script.pop(0)

    def detect_deferred(self):
        return 0.0


def _hand(roi, p_right, x, presence=0.99):
    lm = np.zeros((21, 3), np.float32)
    lm[:, 0], lm[:, 1] = x + np.arange(21), 300 + np.arange(21)
    return HandResult(roi, "Right" if p_right > 0.5 else "Left", 0.9, presence, lm, lm, (x, 300, 200, 200, 0),
                      p_right=p_right)


def _tracker(**tracking):
    from handtrack.tracker import HandTracker
    t = HandTracker.__new__(HandTracker)
    t.cfg = cfgmod.load(overrides={"tracking": tracking})
    t.model = _FakeModel()
    t.reset()
    return t


def test_one_person_ids_follow_handedness_through_a_crossing():
    t = _tracker()
    frames = [
        [_hand(0, 0.98, 300), _hand(1, 0.02, 900)],    # two hands
        [_hand(0, 0.98, 500)],                         # occlusion: one ROI left
        [_hand(0, 0.03, 600)],                         # ROI 0 slid onto the other hand
        [_hand(0, 0.03, 650), _hand(2, 0.97, 450)],    # separated again, new ROI
    ]
    t.model.script = list(frames)
    out = [t.process(None, i, int(i * 33e6), defer_detect=False) for i in range(len(frames))]
    ids = [{h.track_id: h.handedness for h in hf.hands} for hf in out]
    right_id = next(k for k, v in ids[0].items() if v == "Left")   # MediaPipe "Right" -> player's left
    assert ids[1] == {right_id: "Left"}
    assert len(ids[2]) == 1 and right_id not in ids[2]               # ID moved with the hand
    assert ids[3] == ids[0]                                          # same two IDs, same labels


def test_one_person_drops_duplicate_roi_on_same_hand():
    t = _tracker()
    t.model.tracked = [(0, None), (1, None)]
    t.model.script = [[_hand(0, 0.03, 600, presence=0.9), _hand(1, 0.05, 610, presence=0.99)]]
    hf = t.process(None, 0, 0, defer_detect=False)
    assert len(hf.hands) == 1
    assert [r for r, _ in t.model.tracked] == [1]


def test_camera_mirrored_flag_controls_labels():
    for mirrored, expect in ((False, "Left"), (True, "Right")):
        t = _tracker(camera_mirrored=mirrored)
        t.model.script = [[_hand(0, 0.98, 300)]]
        assert t.process(None, 0, 0, defer_detect=False).hands[0].handedness == expect


def test_filter_resets_when_track_jumps():
    t = _tracker()
    t.model.script = [[_hand(0, 0.98, 100)], [_hand(0, 0.98, 1000)]]
    t.process(None, 0, 0, defer_detect=False)
    hf = t.process(None, 1, int(33e6), defer_detect=False)
    h = hf.hands[0]
    assert np.allclose(h.landmarks_filtered[:, :2], h.landmarks_norm[:, :2])   # no smear from x=100


def test_osc_and_json_encoding_roundtrip():
    from pythonosc.osc_bundle import OscBundle
    from handtrack.output_udp import encode_json, encode_osc
    from handtrack.tracker import Hand, HandFrame
    lm = np.random.default_rng(0).random((21, 3)).astype(np.float32)
    h = Hand(7, "Left", 0.95, lm * 100, lm, lm, lm[:, :2])
    hf = HandFrame(42, 123456789012345, 123456789912345, [h])
    msgs = list(OscBundle(encode_osc(hf)))
    assert msgs[0].address == "/handtrack/frame" and msgs[0].params == [42, 123456789012345, 123456789912345, 1]
    p = msgs[1].params
    assert msgs[1].address == "/handtrack/hand" and p[:2] == [7, "Left"] and len(p) == 3 + 63 + 42
    assert np.allclose(p[3:66], lm.reshape(-1), atol=1e-6)
    import json
    d = json.loads(encode_json(hf))
    assert d["f"] == 42 and d["hands"][0]["id"] == 7 and len(d["hands"][0]["lm"]) == 63

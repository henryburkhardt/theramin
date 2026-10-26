"""Configuration: defaults merged with config.yaml, exposed with attribute access."""
import copy
import os
from types import SimpleNamespace

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "camera": {"device": "/dev/video0", "width": 1280, "height": 720, "fps": 30, "buffers": 2},
    "decode": "nvjpeg",                       # nvjpeg | gst
    "models": {
        "detector": "models/mediapipe/hand_detector.onnx",
        "landmarks": "models/mediapipe/hand_landmarks_detector.onnx",
    },
    "tracking": {
        "max_hands": 2,
        "min_detection": 0.5,
        "min_presence": 0.5,
        "detect_interval": 1,
        "defer_detection": True,              # with hands tracked, detect after publishing
        "min_palm_size": 0.09,                # fraction of frame height (0.09 = 65 px at 720p)
        "play_area": None,                    # [x0, y0, x1, y1] normalized, or null = whole frame
        "one_person": True,                   # 2 hands = one left + one right
        "camera_mirrored": False,             # is the camera image already mirrored (selfie view)?
        "handedness_alpha": 0.15,             # EMA weight of each frame's handedness
        "handedness_hysteresis": 0.15,        # label changes only past 0.5 +- this
        "reuse_seconds": 1.0,                 # a lost track's ID can come back within this
    },
    "filter": {"min_cutoff": 0.3, "beta": 30.0, "d_cutoff": 1.0, "predict_ms": 0.0},
    "output": {"udp": None, "udp_format": "osc"},
    "viewer": {"mirror": True, "show_raw": False},
}


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.path.join(ROOT, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    _merge(cfg, overrides or {})
    return _ns(cfg)


def resolve(path):
    """Paths in the config are relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

#!/usr/bin/env bash
# Recreate models/ (not in git): MediaPipe TFLite -> ONNX, plus RTMPose/RTMDet ONNX.
# Needs ~/convert-env (python3 -m venv ~/convert-env && ~/convert-env/bin/pip install tensorflow tf2onnx onnx)
# Then build engines: ~/handtrack-env/bin/python scripts/build_engines.py
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=../hand_landmarker.task
mkdir -p models/mediapipe models/rtmpose

python3 - "$TASK" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for name in ("hand_detector.tflite", "hand_landmarks_detector.tflite"):
    open(f"models/mediapipe/{name}", "wb").write(z.read(name))
PY
for m in hand_detector hand_landmarks_detector; do
  ~/convert-env/bin/python -m tf2onnx.convert --tflite models/mediapipe/$m.tflite \
      --output models/mediapipe/$m.onnx --opset 17
done

# RTMPose (bake-off candidate B; not used by the chosen pipeline)
BASE=https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk
TMP=$(mktemp -d)
curl -fsSL -o "$TMP/det.zip" $BASE/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip
curl -fsSL -o "$TMP/pose.zip" $BASE/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip
python3 - "$TMP" <<'PY'
import sys, zipfile
t = sys.argv[1]
for zipname, out in (("det.zip", "rtmdet_nano_hand"), ("pose.zip", "rtmpose_m_hand_256")):
    z = zipfile.ZipFile(f"{t}/{zipname}")
    for i in z.namelist():
        if i.endswith("end2end.onnx"):
            open(f"models/rtmpose/{out}.onnx", "wb").write(z.read(i))
        elif i.endswith("pipeline.json"):
            open(f"models/rtmpose/{out}.pipeline.json", "wb").write(z.read(i))
PY
rm -rf "$TMP"
echo "models ready"

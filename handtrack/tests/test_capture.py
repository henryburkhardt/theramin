import ctypes as C
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from handtrack import v4l2
from handtrack.capture import LatestFrame

HERE = os.path.dirname(os.path.abspath(__file__))


def test_v4l2_struct_sizes_match_kernel_abi():
    assert C.sizeof(v4l2.Buffer) == 88
    assert v4l2.Buffer.timestamp.offset == 24
    assert v4l2.Buffer.sequence.offset == 56
    assert v4l2.Buffer.m.offset == 64
    assert C.sizeof(v4l2.Format) == 208
    assert v4l2.Format.fmt.offset == 8


@pytest.mark.skipif(not shutil.which("gcc") or not os.path.exists("/usr/include/linux/videodev2.h"),
                    reason="needs gcc and kernel headers")
def test_ioctl_numbers_match_header(tmp_path):
    src = tmp_path / "n.c"
    src.write_text('#include <stdio.h>\n#include <linux/videodev2.h>\nint main(){printf("%lu %lu %lu %lu %lu %lu %lu %lu",'
                   "(unsigned long)VIDIOC_S_FMT,(unsigned long)VIDIOC_REQBUFS,(unsigned long)VIDIOC_QUERYBUF,"
                   "(unsigned long)VIDIOC_QBUF,(unsigned long)VIDIOC_DQBUF,(unsigned long)VIDIOC_STREAMON,"
                   "(unsigned long)VIDIOC_STREAMOFF,(unsigned long)VIDIOC_S_PARM);}")
    subprocess.run(["gcc", "-o", tmp_path / "n", src], check=True)
    got = list(map(int, subprocess.run([tmp_path / "n"], capture_output=True, text=True, check=True).stdout.split()))
    assert got == [v4l2.VIDIOC_S_FMT, v4l2.VIDIOC_REQBUFS, v4l2.VIDIOC_QUERYBUF, v4l2.VIDIOC_QBUF,
                   v4l2.VIDIOC_DQBUF, v4l2.VIDIOC_STREAMON, v4l2.VIDIOC_STREAMOFF, v4l2.VIDIOC_S_PARM]


def test_latest_frame_returns_only_newest():
    slot = LatestFrame()
    for i in range(5):
        slot.put(i)
    version, item = slot.wait_newer(0)
    assert (version, item) == (5, 4)
    assert slot.wait_newer(version, timeout=0.01) is None


def test_latest_frame_wakes_waiter():
    slot = LatestFrame()
    got = []
    t = threading.Thread(target=lambda: got.append(slot.wait_newer(0, timeout=2)))
    t.start()
    time.sleep(0.05)
    slot.put("x")
    t.join()
    assert got == [(1, "x")]


def test_nvjpeg_matches_opencv():
    cp = pytest.importorskip("cupy")
    cv2 = pytest.importorskip("cv2")
    from handtrack.nvjpeg import NvJpegDecoder
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), np.uint8), (15, 15), 0)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    dec = NvJpegDecoder(1280, 720)
    gpu = dec.decode(jpg.tobytes())
    dec.stream.synchronize()
    out = cp.asnumpy(gpu)
    ref = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    dec.close()
    assert np.abs(out.astype(int) - ref).mean() < 1.5

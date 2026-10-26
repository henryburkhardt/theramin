"""Frame sources: camera -> decoded RGB(A) frame in GPU memory, newest-frame-only.

Every source runs its own thread(s) and publishes into a LatestFrame slot; a
consumer calls source.wait_newer(version) and always gets the newest frame.
Nothing queues: if the consumer is slow, older frames are overwritten.

Timestamps (all CLOCK_MONOTONIC ns, comparable with time.monotonic_ns()):
  t_capture_ns  kernel V4L2 buffer timestamp (uvcvideo: start of the frame's USB transfer)
  t_dequeue_ns  when userspace received the compressed frame
  t_decoded_ns  when the decoded frame was ready in GPU memory (stream synchronized)
"""
import threading
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

from .v4l2 import V4L2Capture

# Decoded frames rotate through this many GPU buffers. A consumer working on
# frame N must not see it overwritten by frame N+1's decode; with newest-only
# delivery, 3 buffers give it two full frame periods (66 ms at 30 fps).
RING = 3


@dataclass
class Frame:
    frame_id: int          # driver sequence number (gaps = frames dropped before us)
    t_capture_ns: int
    t_dequeue_ns: int
    t_decoded_ns: int
    image: object          # cupy (H, W, 3) RGB or (H, W, 4) RGBA uint8, one of RING rotating
                           # buffers: valid for ~2 frame periods, copy it to keep longer
    jpeg: bytes = None     # the compressed frame (MjpegSource), e.g. for a viewer


class LatestFrame:
    """Single-slot mailbox. put() overwrites; wait_newer(version) returns
    (version, item) for anything newer than `version`, or None on timeout."""

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._version = 0

    def put(self, item):
        with self._cond:
            self._item = item
            self._version += 1
            self._cond.notify_all()

    def wait_newer(self, version, timeout=1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._version > version, timeout):
                return None
            return self._version, self._item


class _Source:
    def __init__(self):
        self.slot = LatestFrame()
        self._stop = threading.Event()
        self.error = None
        self._threads = []

    def _spawn(self, fn, name):
        def run():
            try:
                fn()
            except Exception as e:  # surfaced to the consumer via .error
                self.error = e
                self._stop.set()
        t = threading.Thread(target=run, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def wait_newer(self, version, timeout=1.0):
        """Returns (version, Frame) newer than `version` (start with 0), or None."""
        if self.error:
            raise self.error
        return self.slot.wait_newer(version, timeout)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self._close()

    def _close(self):
        pass


class MjpegSource(_Source):
    """Direct V4L2 capture thread + decode thread.
    decoder: "nvjpeg" (CUDA, into GPU memory) or "cpu" (cv2.imdecode + upload)."""

    def __init__(self, device="/dev/video0", width=1280, height=720, fps=30, buffers=2,
                 decoder="nvjpeg", nvjpeg_backend="hybrid"):
        super().__init__()
        self.cap = V4L2Capture(device, width, height, fps, buffers)
        self.raw = LatestFrame()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._ring = [cp.empty((height, width, 3), cp.uint8) for _ in range(RING)]
        self._ri = 0
        if decoder == "nvjpeg":
            from .nvjpeg import NvJpegDecoder
            self._dec = NvJpegDecoder(width, height, nvjpeg_backend, self.stream)
            self._decode = self._dec.decode
        elif decoder == "cpu":
            import cv2
            self._cv2 = cv2
            self._pinned = cp.cuda.alloc_pinned_memory(width * height * 3)
            self._host = np.frombuffer(self._pinned, np.uint8, width * height * 3).reshape(height, width, 3)
            self._decode = self._decode_cpu
        else:
            raise ValueError(decoder)
        self._spawn(self._capture_loop, "v4l2-capture")
        self._spawn(self._decode_loop, "decode")

    def _decode_cpu(self, jpeg, out):
        cv2 = self._cv2
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=self._host)
        out.set(self._host, stream=self.stream)
        return out

    def _capture_loop(self):
        while not self._stop.is_set():
            r = self.cap.read(timeout_ms=200)
            if r is not None:
                self.raw.put(r)

    def _decode_loop(self):
        version = 0
        while not self._stop.is_set():
            r = self.raw.wait_newer(version, timeout=0.2)
            if r is None:
                continue
            version, (seq, t_cap, t_deq, jpeg) = r
            img = self._decode(jpeg, self._ring[self._ri])
            self._ri = (self._ri + 1) % RING
            self.stream.synchronize()
            self.slot.put(Frame(seq, t_cap, t_deq, time.monotonic_ns(), img, jpeg))

    def _close(self):
        self.cap.close()
        if hasattr(self, "_dec"):
            self._dec.close()


GST_PIPELINES = {
    # NVJPG hardware engine through the V4L2 decoder API
    "gst-nvv4l2-mjpeg": ("image/jpeg,width={w},height={h},framerate={fps}/1 ! "
                         "nvv4l2decoder mjpeg=1 enable-max-performance=true"),
    # NVJPG hardware engine through libnvjpeg (Jetson multimedia, not CUDA nvJPEG)
    "gst-nvjpegdec": "image/jpeg,width={w},height={h},framerate={fps}/1 ! nvjpegdec",
    # H264 on NVDEC; disable-dpb skips the decoded picture buffer (needs a stream without B-frames)
    "gst-h264": ("video/x-h264,width={w},height={h},framerate={fps}/1 ! h264parse ! "
                 "nvv4l2decoder disable-dpb=true enable-max-performance=true"),
    "gst-h264-dpb": ("video/x-h264,width={w},height={h},framerate={fps}/1 ! h264parse ! "
                     "nvv4l2decoder enable-max-performance=true"),
}


class GstSource(_Source):
    """v4l2src -> hardware decoder -> nvvidconv (VIC) -> RGBA in system memory ->
    appsink (newest only) -> pinned host buffer -> GPU. The NVMM -> CPU -> GPU hop
    is the known cost of this path from Python (no NvBufSurface/EGL interop here)."""

    def __init__(self, kind, device="/dev/video0", width=1280, height=720, fps=30):
        super().__init__()
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self.Gst = Gst
        desc = (f"v4l2src name=src device={device} ! "
                + GST_PIPELINES[kind].format(w=width, h=height, fps=fps)
                + " ! nvvidconv ! video/x-raw,format=RGBA ! "
                "appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false")
        self.pipeline = Gst.parse_launch(desc)
        self.sink = self.pipeline.get_by_name("sink")
        self.stream = cp.cuda.Stream(non_blocking=True)
        n = width * height * 4
        self._pinned = cp.cuda.alloc_pinned_memory(n)
        self._host = np.frombuffer(self._pinned, np.uint8, n)
        self._ring = [cp.empty((height, width, 4), cp.uint8) for _ in range(RING)]
        self._ri = 0
        # Userspace arrival time per PTS, recorded on v4l2src's output pad.
        self._dequeue_ns = {}
        src_pad = self.pipeline.get_by_name("src").get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_src_buffer)
        self.pipeline.set_state(Gst.State.PLAYING)
        self._spawn(self._pull_loop, "gst-pull")

    def _on_src_buffer(self, pad, info):
        buf = info.get_buffer()
        self._dequeue_ns[buf.pts] = (time.monotonic_ns(), buf.offset)
        if len(self._dequeue_ns) > 64:
            for k in sorted(self._dequeue_ns)[:32]:
                del self._dequeue_ns[k]
        return self.Gst.PadProbeReturn.OK

    def _pull_loop(self):
        Gst = self.Gst
        bus = self.pipeline.get_bus()
        n = 0
        while not self._stop.is_set():
            msg = bus.pop_filtered(Gst.MessageType.ERROR)
            if msg:
                err, dbg = msg.parse_error()
                raise RuntimeError(f"GStreamer: {err.message} ({dbg})")
            sample = self.sink.emit("try-pull-sample", 200 * Gst.MSECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            t_capture = self.pipeline.get_base_time() + buf.pts
            t_deq, offset = self._dequeue_ns.get(buf.pts, (0, None))
            ok, mi = buf.map(Gst.MapFlags.READ)
            try:
                self._host[:] = np.frombuffer(mi.data, np.uint8)
            finally:
                buf.unmap(mi)
            gpu = self._ring[self._ri]
            self._ri = (self._ri + 1) % RING
            gpu.set(self._host.reshape(gpu.shape), stream=self.stream)
            self.stream.synchronize()
            # v4l2src's buffer offset is the driver sequence number; fall back to a counter.
            fid = offset if offset not in (None, Gst.BUFFER_OFFSET_NONE) else n
            n += 1
            self.slot.put(Frame(fid, t_capture, t_deq, time.monotonic_ns(), gpu))

    def _close(self):
        self.pipeline.set_state(self.Gst.State.NULL)

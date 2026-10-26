"""Minimal V4L2 MJPG capture via ioctl + mmap, no OpenCV.

Gives us what cv2.VideoCapture hides: the driver buffer count, the kernel's
per-frame timestamp (CLOCK_MONOTONIC, same clock as time.monotonic_ns()),
the driver sequence number (to count dropped frames), and the raw JPEG bytes.
Struct layouts and ioctl numbers were checked against <linux/videodev2.h>
on aarch64 (sizeof v4l2_buffer = 88, v4l2_format = 208).
"""
import ctypes as C
import fcntl
import mmap
import os
import select
import time

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0
V4L2_BUF_FLAG_TIMESTAMP_MASK = 0xE000
V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC = 0x2000
V4L2_BUF_FLAG_TSTAMP_SRC_MASK = 0x70000
V4L2_BUF_FLAG_TSTAMP_SRC_SOE = 0x10000
V4L2_BUF_FLAG_ERROR = 0x40


def fourcc(s):
    return ord(s[0]) | ord(s[1]) << 8 | ord(s[2]) << 16 | ord(s[3]) << 24


class Timeval(C.Structure):
    _fields_ = [("tv_sec", C.c_long), ("tv_usec", C.c_long)]


class Timecode(C.Structure):
    _fields_ = [("type", C.c_uint32), ("flags", C.c_uint32), ("frames", C.c_uint8),
                ("seconds", C.c_uint8), ("minutes", C.c_uint8), ("hours", C.c_uint8),
                ("userbits", C.c_uint8 * 4)]


class BufferM(C.Union):
    _fields_ = [("offset", C.c_uint32), ("userptr", C.c_ulong), ("planes", C.c_void_p), ("fd", C.c_int32)]


class Buffer(C.Structure):
    _fields_ = [("index", C.c_uint32), ("type", C.c_uint32), ("bytesused", C.c_uint32),
                ("flags", C.c_uint32), ("field", C.c_uint32), ("timestamp", Timeval),
                ("timecode", Timecode), ("sequence", C.c_uint32), ("memory", C.c_uint32),
                ("m", BufferM), ("length", C.c_uint32), ("reserved2", C.c_uint32),
                ("request_fd", C.c_int32)]


class PixFormat(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("width", "height", "pixelformat", "field", "bytesperline",
                                          "sizeimage", "colorspace", "priv", "flags", "ycbcr_enc",
                                          "quantization", "xfer_func")]


class FormatU(C.Union):
    _fields_ = [("pix", PixFormat), ("raw_data", C.c_uint8 * 200), ("_align", C.c_void_p)]


class Format(C.Structure):
    _fields_ = [("type", C.c_uint32), ("fmt", FormatU)]


class RequestBuffers(C.Structure):
    _fields_ = [("count", C.c_uint32), ("type", C.c_uint32), ("memory", C.c_uint32),
                ("capabilities", C.c_uint32), ("flags", C.c_uint8), ("reserved", C.c_uint8 * 3)]


class Fract(C.Structure):
    _fields_ = [("numerator", C.c_uint32), ("denominator", C.c_uint32)]


class CaptureParm(C.Structure):
    _fields_ = [("capability", C.c_uint32), ("capturemode", C.c_uint32), ("timeperframe", Fract),
                ("extendedmode", C.c_uint32), ("readbuffers", C.c_uint32), ("reserved", C.c_uint32 * 4)]


class StreamParmU(C.Union):
    _fields_ = [("capture", CaptureParm), ("raw_data", C.c_uint8 * 200)]


class StreamParm(C.Structure):
    _fields_ = [("type", C.c_uint32), ("parm", StreamParmU)]


assert C.sizeof(Buffer) == 88 and C.sizeof(Format) == 208
assert C.sizeof(RequestBuffers) == 20 and C.sizeof(StreamParm) == 204

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_S_PARM = 0xC0CC5616


class V4L2Capture:
    """Streams MJPG from a UVC device. read() returns one frame per call:
    (sequence, t_kernel_ns, t_dequeue_ns, jpeg_bytes)."""

    def __init__(self, device="/dev/video0", width=1280, height=720, fps=30, buffers=2,
                 pixfmt="MJPG"):
        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        try:
            self._setup(width, height, fps, buffers, pixfmt)
        except Exception:
            os.close(self.fd)
            raise

    def _setup(self, width, height, fps, buffers, pixfmt):
        f = Format(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
        f.fmt.pix.width, f.fmt.pix.height = width, height
        f.fmt.pix.pixelformat, f.fmt.pix.field = fourcc(pixfmt), V4L2_FIELD_ANY
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, f)
        if (f.fmt.pix.width, f.fmt.pix.height, f.fmt.pix.pixelformat) != (width, height, fourcc(pixfmt)):
            raise RuntimeError(f"camera refused {width}x{height} {pixfmt}: got "
                               f"{f.fmt.pix.width}x{f.fmt.pix.height}")
        self.width, self.height = width, height

        p = StreamParm(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
        p.parm.capture.timeperframe.numerator, p.parm.capture.timeperframe.denominator = 1, fps
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, p)
        tpf = p.parm.capture.timeperframe
        self.fps = tpf.denominator / tpf.numerator if tpf.numerator else float(fps)

        rb = RequestBuffers(count=buffers, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, rb)
        self.num_buffers = rb.count
        self.maps = []
        for i in range(rb.count):
            b = Buffer(index=i, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
            self.maps.append(mmap.mmap(self.fd, b.length, mmap.MAP_SHARED, mmap.PROT_READ,
                                       offset=b.m.offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        self._buf = Buffer(type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
        self._poll = select.poll()
        self._poll.register(self.fd, select.POLLIN)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, C.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
        self.streaming = True
        self.timestamp_flags = None

    def read(self, timeout_ms=1000):
        """Blocks until the next frame. Returns None on timeout. Frames the
        driver flagged as corrupt are requeued and skipped."""
        while True:
            if not self._poll.poll(timeout_ms):
                return None
            b = self._buf
            try:
                fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
            except BlockingIOError:
                continue
            t_dequeue = time.monotonic_ns()
            if self.timestamp_flags is None:
                self.timestamp_flags = b.flags & (V4L2_BUF_FLAG_TIMESTAMP_MASK | V4L2_BUF_FLAG_TSTAMP_SRC_MASK)
            bad = b.flags & V4L2_BUF_FLAG_ERROR or b.bytesused == 0
            data = None if bad else self.maps[b.index][:b.bytesused]
            seq = b.sequence
            t_kernel = b.timestamp.tv_sec * 1_000_000_000 + b.timestamp.tv_usec * 1000
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
            if not bad:
                return seq, t_kernel, t_dequeue, data

    def close(self):
        if getattr(self, "streaming", False):
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, C.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            self.streaming = False
        for m in getattr(self, "maps", []):
            m.close()
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

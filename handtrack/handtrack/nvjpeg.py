"""nvJPEG (CUDA) decoder via ctypes, decoding straight into a CuPy device buffer.

On Orin the hardware backend (NVJPG engine) is not available through nvJPEG
(nvjpegCreateEx(HARDWARE) returns NVJPEG_STATUS_ARCH_MISMATCH); the default
backend does Huffman decoding on the CPU and IDCT/colour conversion on the GPU.
"""
import ctypes as C

import cupy as cp

NVJPEG_OUTPUT_RGBI = 5
BACKENDS = {"default": 0, "hybrid": 1, "gpu_hybrid": 2, "hardware": 3}


class _Image(C.Structure):
    _fields_ = [("channel", C.c_void_p * 4), ("pitch", C.c_size_t * 4)]


_lib = None


def _load():
    global _lib
    if _lib is None:
        _lib = C.CDLL("libnvjpeg.so.13")
    return _lib


class NvJpegDecoder:
    """decode(jpeg_bytes) -> (H, W, 3) uint8 RGB CuPy array, enqueued on `stream`.
    The output buffer is preallocated and reused, so copy it if you keep it
    past the next decode."""

    def __init__(self, width, height, backend="hybrid", stream=None):
        lib = _load()
        self.lib = lib
        self.handle, self.state = C.c_void_p(), C.c_void_p()
        self._check(lib.nvjpegCreateEx(BACKENDS[backend], None, None, 0, C.byref(self.handle)), "create")
        self._check(lib.nvjpegJpegStateCreate(self.handle, C.byref(self.state)), "state")
        self.stream = stream or cp.cuda.Stream(non_blocking=True)
        self.out = cp.empty((height, width, 3), cp.uint8)
        self.img = _Image()
        self.img.channel[0] = self.out.data.ptr
        self.img.pitch[0] = width * 3

    @staticmethod
    def _check(status, what):
        if status:
            raise RuntimeError(f"nvJPEG {what} failed with status {status}")

    def decode(self, jpeg, out=None):
        """Decode into `out` (a (H, W, 3) uint8 CuPy array) or the internal buffer."""
        out = self.out if out is None else out
        self.img.channel[0] = out.data.ptr
        self._check(self.lib.nvjpegDecode(self.handle, self.state, jpeg, C.c_size_t(len(jpeg)),
                                          NVJPEG_OUTPUT_RGBI, C.byref(self.img),
                                          C.c_void_p(self.stream.ptr)), "decode")
        return out

    def close(self):
        if self.state:
            self.lib.nvjpegJpegStateDestroy(self.state)
            self.state = C.c_void_p()
        if self.handle:
            self.lib.nvjpegDestroy(self.handle)
            self.handle = C.c_void_p()

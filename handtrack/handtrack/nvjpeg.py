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


# Standard Huffman tables (JPEG spec Annex K.3). Many UVC webcams (e.g. Logitech) send MJPEG frames
# without a DHT segment and rely on the decoder to assume these; nvJPEG doesn't, so they're added back.
def _dht(cls_id, bits, vals):
    assert sum(bits) == len(vals)
    return bytes([cls_id] + bits + vals)


_DHT_BODY = b"".join([
    _dht(0x00, [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], list(range(12))),
    _dht(0x01, [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], list(range(12))),
    _dht(0x10, [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], list(bytes.fromhex(
        "01020300041105122131410613516107227114328191a1082342b1c11552d1f0"
        "2433627282090a161718191a25262728292a3435363738393a434445464748494a"
        "535455565758595a636465666768696a737475767778797a838485868788898a"
        "92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7"
        "c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"))),
    _dht(0x11, [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77], list(bytes.fromhex(
        "000102031104052131061241510761711322328108144291a1b1c109233352f0"
        "156272d10a162434e125f11718191a262728292a35363738393a434445464748494a"
        "535455565758595a636465666768696a737475767778797a82838485868788898a"
        "92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7"
        "c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa"))),
])
_DHT = b"\xff\xc4" + (len(_DHT_BODY) + 2).to_bytes(2, "big") + _DHT_BODY


def add_huffman_tables(jpeg):
    """Return `jpeg` with the standard Huffman tables inserted if it has none before its scan."""
    sos = jpeg.find(b"\xff\xda")
    if sos < 0 or jpeg.find(b"\xff\xc4", 0, sos) >= 0:
        return jpeg
    return jpeg[:sos] + _DHT + jpeg[sos:]


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
        jpeg = add_huffman_tables(jpeg)
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

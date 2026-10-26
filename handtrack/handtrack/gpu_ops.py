"""GPU preprocessing kernels (CuPy RawKernel)."""
import cupy as cp
import numpy as np

_WARP_SRC = r"""
extern "C" __global__
void warp_affine_bilinear(const unsigned char* __restrict__ src, int sw, int sh, int sc,
                          float* __restrict__ dst, int dw, int dh,
                          const float* __restrict__ M, int replicate, float border,
                          float mul0, float mul1, float mul2, float add0, float add1, float add2,
                          int p0, int p1, int p2, int nchw)
{
    // One thread per output pixel; blockIdx.z = batch item.
    // M (6 floats per item) maps output pixel coords to source pixel coords,
    // both measured with pixel centres at +0.5.
    // Output channel k = source channel p_k, scaled: value * mul_k + add_k.
    // Out-of-image samples read `border` (raw 0..255 value) unless replicate.
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (u >= dw || v >= dh) return;
    const float* m = M + b * 6;
    float uc = u + 0.5f, vc = v + 0.5f;
    float x = m[0] * uc + m[1] * vc + m[2] - 0.5f;
    float y = m[3] * uc + m[4] * vc + m[5] - 0.5f;
    if (replicate) {  // clamp to the image: edge pixels extend outward
        x = fminf(fmaxf(x, 0.f), (float)(sw - 1));
        y = fminf(fmaxf(y, 0.f), (float)(sh - 1));
    }
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float w00 = (1.f - fx) * (1.f - fy), w01 = fx * (1.f - fy), w10 = (1.f - fx) * fy, w11 = fx * fy;
    bool in00 = x0 >= 0 && x0 < sw && y0 >= 0 && y0 < sh;
    bool in01 = x0 + 1 >= 0 && x0 + 1 < sw && y0 >= 0 && y0 < sh;
    bool in10 = x0 >= 0 && x0 < sw && y0 + 1 >= 0 && y0 + 1 < sh;
    bool in11 = x0 + 1 >= 0 && x0 + 1 < sw && y0 + 1 >= 0 && y0 + 1 < sh;
    const int perm[3] = {p0, p1, p2};
    const float mul[3] = {mul0, mul1, mul2}, add[3] = {add0, add1, add2};
    for (int k = 0; k < 3; ++k) {
        int c = perm[k];
        float acc = 0.f;
        acc += w00 * (in00 ? src[((size_t)y0 * sw + x0) * sc + c] : border);
        acc += w01 * (in01 ? src[((size_t)y0 * sw + x0 + 1) * sc + c] : border);
        acc += w10 * (in10 ? src[((size_t)(y0 + 1) * sw + x0) * sc + c] : border);
        acc += w11 * (in11 ? src[((size_t)(y0 + 1) * sw + x0 + 1) * sc + c] : border);
        size_t o = nchw ? (((size_t)b * 3 + k) * dh + v) * dw + u
                        : (((size_t)b * dh + v) * dw + u) * 3 + k;
        dst[o] = acc * mul[k] + add[k];
    }
}
"""
_warp = cp.RawKernel(_WARP_SRC, "warp_affine_bilinear")


class AffineWarper:
    """Crops/resizes/rotates a uint8 (H, W, C>=3) RGB(A) image into a float32
    tensor, (N, h, w, 3) or with nchw=True (N, 3, h, w). Matrices are (N, 2, 3)
    mapping output pixel -> source pixel.

    Default normalization is x / 255 (MediaPipe). For mean/std models pass
    mean/std in output channel order; channels=(2, 1, 0) makes BGR from RGB.
    Outside the image: `border` (0..255), or the nearest edge pixel with replicate=True."""

    def __init__(self, max_batch, stream, replicate=False, mean=(0, 0, 0), std=(255, 255, 255),
                 channels=(0, 1, 2), nchw=False, border=0.0):
        self.stream = stream
        self.replicate = replicate
        self.mul = [np.float32(1.0 / s) for s in std]
        self.add = [np.float32(-m / s) for m, s in zip(mean, std)]
        self.perm = [np.int32(c) for c in channels]
        self.nchw = np.int32(nchw)
        self.border = np.float32(border)
        self._pinned = cp.cuda.alloc_pinned_memory(max_batch * 6 * 4)
        self._host_m = np.frombuffer(self._pinned, np.float32, max_batch * 6).reshape(max_batch, 6)
        self._dev_m = cp.empty((max_batch, 6), cp.float32)

    def __call__(self, image, out, matrices):
        n = len(matrices)
        if n == 0:
            return
        self._host_m[:n] = np.asarray(matrices, np.float32).reshape(n, 6)
        self._dev_m[:n].set(self._host_m[:n], stream=self.stream)
        h, w = out.shape[2:4] if self.nchw else out.shape[1:3]
        sh, sw, sc = image.shape
        with self.stream:
            _warp(((w + 15) // 16, (h + 15) // 16, n), (16, 16, 1),
                  (image, np.int32(sw), np.int32(sh), np.int32(sc), out, np.int32(w), np.int32(h),
                   self._dev_m, np.int32(self.replicate), self.border,
                   *self.mul, *self.add, *self.perm, self.nchw))

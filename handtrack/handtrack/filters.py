"""One Euro filter (Casiez et al., CHI 2012), vectorized over any array shape.

Low speed -> low cutoff (min_cutoff, Hz): strong smoothing, little jitter.
High speed -> cutoff rises by beta * |speed|: little lag. d_cutoff smooths the
speed estimate itself. Units: x in whatever the caller uses (we use normalized
image coordinates), speed in those units per second.
"""
import math

import numpy as np


def _alpha(dt, cutoff):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        """x: array, t: seconds. Returns (filtered x, filtered derivative)."""
        x = np.asarray(x, np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy(), self.dx.copy()
        dt = t - self.t
        if dt <= 0:
            return self.x.copy(), self.dx.copy()
        dx = (x - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        tau = 1.0 / (2 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy(), self.dx.copy()

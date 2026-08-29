"""Backend-agnostic analytic integrands with closed-form integrals.

Each integrand takes a points tensor of shape ``(N, dim)`` and returns ``(N,)``
using ``autoray.numpy`` so the identical code runs on every backend. Expected
values are exact (or exact-in-the-limit with negligible truncation on the stated
domain), so tests assert against ground truth rather than a reference run.

The functions and their integrals mirror how torchquad is actually used in the
wild: smooth 1D densities (Bayesian stats), high-dimensional sums (physics /
cosmology), peaked/tailed kernels, and complex-valued Fourier transforms
(characteristic functions).
"""

import math

from autoray import numpy as anp

# --- 1D smooth, deterministic-rule anchors -------------------------------------


def sin_1d(x):
    """sin(x)."""
    return anp.sin(x[:, 0])


def lorentzian_1d(x):
    """Lorentzian 1/(1+x^2) — peaked with slow tails."""
    return 1.0 / (1.0 + x[:, 0] ** 2)


def normal_pdf_1d(x):
    """Standard-normal density exp(-x^2/2)/sqrt(2*pi)."""
    return anp.exp(-0.5 * x[:, 0] ** 2) / math.sqrt(2.0 * math.pi)


# (name, fn, domain, exact integral)
SIN_ON_0_HALFPI = ("sin on [0,pi/2]", sin_1d, [[0.0, math.pi / 2.0]], 1.0)
SIN_ON_0_5 = ("sin on [0,5]", sin_1d, [[0.0, 5.0]], 1.0 - math.cos(5.0))
LORENTZIAN = ("lorentzian on [-1,1]", lorentzian_1d, [[-1.0, 1.0]], math.pi / 2.0)
NORMAL_PDF = (
    "normal pdf on [-8,8]",
    normal_pdf_1d,
    [[-8.0, 8.0]],
    math.erf(8.0 / math.sqrt(2.0)),
)

SMOOTH_1D = [SIN_ON_0_HALFPI, SIN_ON_0_5, LORENTZIAN, NORMAL_PDF]


# --- high-dimensional anchor ---------------------------------------------------


def sum_sin(x):
    """sum_i sin(x_i) — separable, smooth; the tutorial's high-dim example."""
    return anp.sum(anp.sin(x), axis=1)


def sum_sin_expected(dim):
    """Exact integral of sum_i sin(x_i) over [0,1]^dim."""
    return dim * (1.0 - math.cos(1.0))


# --- canonical 2D Monte Carlo case (README / tutorial) -------------------------


def sin_plus_exp_2d(x):
    """sin(x0) + exp(x1) — the most-copied torchquad snippet."""
    return anp.sin(x[:, 0]) + anp.exp(x[:, 1])


SIN_PLUS_EXP_DOMAIN = [[0.0, 1.0], [-1.0, 1.0]]
# integral = (1-cos 1)*2 + (e - e^-1)*1
SIN_PLUS_EXP_EXPECTED = (1.0 - math.cos(1.0)) * 2.0 + (math.e - 1.0 / math.e)


# --- complex-valued Fourier / characteristic-function anchors ------------------


def cauchy_char_integrand(f):
    """exp(-|t|) * exp(-i f t); integral over R is 2/(1+f^2) (real)."""

    def integrand(x):
        t = x[:, 0]
        return anp.exp(-anp.abs(t)) * anp.exp(-1j * f * t)

    return integrand


def gaussian_char_integrand(f):
    """exp(-t^2) * exp(-i f t); integral over R is sqrt(pi)*exp(-f^2/4) (real)."""

    def integrand(x):
        t = x[:, 0]
        return anp.exp(-(t**2)) * anp.exp(-1j * f * t)

    return integrand

"""Monte Carlo and VEGAS: accuracy, high-dimensional scaling, and determinism.

Stochastic integrators are what torchquad users reach for above ~4D (physics,
cosmology, finance). These assert statistical accuracy at fixed seeds and, per
the project's reproducibility guarantee, bit-for-bit determinism.
"""

import numpy as np
import pytest

from torchquad import MonteCarlo, VEGAS, set_up_backend

import _integrands as ig
from conftest import rel_error, to_numpy


def test_monte_carlo_2d_canonical_float32():
    """The README's sin(x)+exp(y) 2D case in float32 — the most-copied snippet."""
    set_up_backend("torch", data_type="float32")
    mc = MonteCarlo()
    result = mc.integrate(
        ig.sin_plus_exp_2d,
        dim=2,
        N=1_000_000,
        integration_domain=ig.SIN_PLUS_EXP_DOMAIN,
        seed=0,
        backend="torch",
    )
    error = rel_error(result, ig.SIN_PLUS_EXP_EXPECTED)
    assert error < 1e-2, f"MC 2D float32: rel err {error:.2e}"


def test_monte_carlo_10d_accuracy(backend_f64):
    """Plain MC on a 10D separable integrand stays within MC error, all backends."""
    mc = MonteCarlo()
    domain = [[0.0, 1.0]] * 10
    result = mc.integrate(
        ig.sum_sin, dim=10, N=1_000_000, integration_domain=domain, seed=0, backend=backend_f64
    )
    error = rel_error(result, ig.sum_sin_expected(10))
    assert error < 2e-2, f"MC 10D ({backend_f64}): rel err {error:.2e}"


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_vegas_10d_accuracy(backend):
    """VEGAS on a 10D integrand (torch/numpy only) — do not assert tighter than ~1e-3."""
    pytest.importorskip(backend)
    set_up_backend(backend, data_type="float64")
    vegas = VEGAS()
    domain = [[0.0, 1.0]] * 10
    result = vegas.integrate(
        ig.sum_sin, dim=10, N=200_000, integration_domain=domain, seed=0, backend=backend
    )
    error = rel_error(result, ig.sum_sin_expected(10))
    assert error < 5e-3, f"VEGAS 10D ({backend}): rel err {error:.2e}"


def test_monte_carlo_determinism(backend_f64):
    """Same seed reproduces the integral bit-for-bit (reproducibility guarantee)."""
    mc = MonteCarlo()
    domain = [[0.0, 1.0]] * 3
    kwargs = dict(dim=3, N=50_000, integration_domain=domain, seed=1234, backend=backend_f64)
    first = to_numpy(mc.integrate(ig.sum_sin, **kwargs))
    second = to_numpy(mc.integrate(ig.sum_sin, **kwargs))
    assert np.array_equal(first, second), (
        f"MC not deterministic on {backend_f64}: {first} != {second}"
    )

"""Contract shared by every sampler that can be passed to ``MonteCarlo`` as ``rng``.

Each sampler promises points in the half-open ``[0, 1)`` so the integrand is
never evaluated on the upper domain boundary, and promises to leave the
caller's global random state alone. Both are checked here in float32, where
rounding makes them easy to break, and on every backend.
"""

import autoray as ar
import numpy as np
import pytest
import torch

from torchquad import Halton, Lattice, RandomizedLatinHypercube, Sobol

BACKENDS = ["numpy", "torch", "jax", "tensorflow"]
SAMPLERS = [Sobol, Halton, Lattice, RandomizedLatinHypercube]
# The largest count Lattice accepts. In float32 the spacing next to 2**20 is
# 1/16, coarse enough for a naive construction to round onto exactly 1.
NUMBER_OF_POINTS = 2**20


# JAX only has float64 once x64 is switched on, which depends on which tests ran
# before this one, so it is checked in float32 alone.
RANGE_CASES = [
    (backend, dtype_name)
    for backend in BACKENDS
    for dtype_name in ["float32", "float64"]
    if (backend, dtype_name) != ("jax", "float64")
]


@pytest.mark.parametrize("backend,dtype_name", RANGE_CASES)
@pytest.mark.parametrize("sampler_class", SAMPLERS)
def test_points_stay_below_one(backend, dtype_name, sampler_class):
    """No sampler may return exactly 1.0, in particular not in float32."""
    dtype = ar.to_backend_dtype(dtype_name, like=backend)
    largest = 0.0
    # Several seeds: a rounding failure hits a few percent of the draws.
    for seed in range(8):
        points = ar.to_numpy(
            sampler_class(backend=backend, seed=seed).uniform([NUMBER_OF_POINTS, 3], dtype)
        )
        assert points.dtype == np.dtype(dtype_name)
        assert points.min() >= 0.0
        largest = max(largest, float(points.max()))
    assert largest < 1.0


# An unseeded Sobol sampler is left out: torch's SobolEngine takes its scrambling
# from the global generator when no seed is given, which is how Sobol has
# behaved since 0.6.0 and what makes it reproducible under torch.manual_seed.
ISOLATION_CASES = [
    (sampler_class, seed)
    for sampler_class in SAMPLERS
    for seed in [None, 0]
    if (sampler_class, seed) != (Sobol, None)
]


@pytest.mark.parametrize("sampler_class,seed", ISOLATION_CASES)
def test_sampling_leaves_global_torch_rng_untouched(sampler_class, seed):
    """Drawing points must neither seed nor advance PyTorch's global generator."""
    with torch.random.fork_rng(), torch.device("cpu"):
        torch.manual_seed(987)
        state_before = torch.get_rng_state().clone()
        sampler_class(backend="torch", seed=seed).uniform([1024, 2], torch.float64)
        assert torch.equal(state_before, torch.get_rng_state())


def test_lattice_points_exact_in_jax_32_bit_mode():
    """The lattice integers overflow int32 in JAX's default mode yet stay exact.

    ``index * z`` reaches 2**40. Without x64 JAX computes it modulo 2**32, which
    is harmless only because the number of points is a power of two; the
    unshifted points must therefore equal the NumPy (int64) ones bit for bit.
    """
    from jax.experimental import disable_x64

    size = [NUMBER_OF_POINTS, 6]
    reference = Lattice(backend="numpy", shift=False).uniform(size, np.float32)
    # Earlier tests may have enabled x64 globally; force the 32-bit mode here.
    with disable_x64():
        points = Lattice(backend="jax", shift=False).uniform(
            size, ar.to_backend_dtype("float32", like="jax")
        )
    np.testing.assert_array_equal(ar.to_numpy(points), reference)

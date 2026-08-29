"""Shared fixtures and helpers for the release-testing suite.

Unlike ``tests/``, this suite runs against an *installed* torchquad
(``pip install -e .``), so imports are plain ``import torchquad`` with no
``sys.path`` manipulation. See ``release_testing/README.md`` for the intent.
"""

import numpy as np
import pytest

from torchquad import set_up_backend

# Deterministic rules run on every backend. VEGAS is torch/numpy only and the
# JIT path excludes numpy; those tests narrow the set themselves.
ALL_BACKENDS = ["numpy", "torch", "jax", "tensorflow"]

# Below this magnitude an "expected" value is treated as zero and rel_error
# falls back to absolute error (e.g. odd-integrand cancellation checks).
_ZERO_DENOM_ATOL = 1e-12


@pytest.fixture(params=ALL_BACKENDS)
def backend_f64(request):
    """Yield each installed backend configured for float64, skipping the rest."""
    backend = request.param
    pytest.importorskip(backend)
    set_up_backend(backend, data_type="float64")
    return backend


def to_numpy(value):
    """Convert any backend tensor (possibly on GPU or carrying grad) to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def rel_error(result, expected):
    """Relative error of ``result`` against a scalar ground truth ``expected``.

    Falls back to absolute error when ``expected`` is ~0 (e.g. odd-integrand
    cancellation checks). Complex-safe.

    Args:
        result: Backend scalar/array returned by an integrator.
        expected (complex or float): Closed-form ground truth.

    Returns:
        float: Relative (or absolute, near zero) error magnitude.
    """
    is_complex = np.iscomplexobj(expected) or np.iscomplexobj(to_numpy(result))
    dtype = np.complex128 if is_complex else np.float64
    result = np.asarray(to_numpy(result), dtype=dtype).reshape(())
    denom = abs(expected)
    if denom < _ZERO_DENOM_ATOL:
        return float(abs(result - expected))
    return float(abs(result - expected) / denom)

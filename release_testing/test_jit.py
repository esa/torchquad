"""JIT-compiled repeated quadrature matches the eager path.

Users who evaluate the same integrator over many domains use
``get_jit_compiled_integrate`` (JAX/TF ``jit``, torch). A deterministic rule has
no RNG, so the compiled path must reproduce the eager path essentially
bit-for-bit — real parity, not just "both are roughly right".
"""

import math

import autoray as ar
import pytest
from autoray import numpy as anp

from torchquad import Simpson, set_up_backend

import _integrands as ig
from conftest import rel_error


@pytest.mark.parametrize("backend", ["torch", "jax", "tensorflow"])
def test_jit_matches_eager_simpson(backend):
    """JIT-compiled Simpson reproduces the eager result and the analytic value."""
    pytest.importorskip(backend)
    set_up_backend(backend, data_type="float64")
    simpson = Simpson()
    # Build the domain via autoray's public API (the compiled path takes a
    # backend tensor, not a Python list).
    dtype = ar.to_backend_dtype("float64", like=backend)
    domain = anp.array([[0.0, 2.0]], like=backend, dtype=dtype)
    n_points = 10001

    eager = simpson.integrate(
        ig.sin_1d, dim=1, N=n_points, integration_domain=domain, backend=backend
    )
    jit_integrate = simpson.get_jit_compiled_integrate(
        dim=1, N=n_points, integration_domain=domain, backend=backend
    )
    compiled = jit_integrate(ig.sin_1d, domain)

    expected = 1.0 - math.cos(2.0)  # int_0^2 sin(x) dx
    # Deterministic rule -> compiled and eager must agree to ~float64 rounding.
    assert rel_error(compiled, float(eager)) < 1e-10, f"JIT and eager Simpson disagree ({backend})"
    assert rel_error(compiled, expected) < 1e-8, f"JIT Simpson wrong ({backend})"

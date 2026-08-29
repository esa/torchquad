"""Cross-backend agreement and integer-domain regression tests.

These guard two correctness properties that are easy to regress and that hide
between backends (see REVIEW.md section 1):

* The same (integrator, integrand, domain) must agree across every installed
  backend in fp64 -- a property, not a hope. Dtype-promotion and
  ``meshgrid``-shape bugs surface here.
* An integer-dtype ``integration_domain`` must not silently integrate to zero
  (issue #180); it has to be promoted to float and produce the correct result.
"""

from importlib.util import find_spec

import numpy as np
import pytest
import autoray as ar
from autoray import numpy as anp
from autoray import to_backend_dtype

from torchquad.integration.trapezoid import Trapezoid
from torchquad.integration.simpson import Simpson
from torchquad.integration.boole import Boole
from torchquad.integration.gaussian import GaussLegendre
from torchquad.utils.set_up_backend import set_up_backend
from helper_functions import setup_test_for_backend

# Deterministic integrators only: stochastic methods (MonteCarlo, VEGAS) draw
# from per-backend RNGs, so their results cannot agree bit-for-bit across
# backends and are covered by their own seed-reproducibility tests instead.
_DETERMINISTIC_INTEGRATORS = [
    (Trapezoid, 101),
    (Simpson, 101),
    (Boole, 101),
    (GaussLegendre, 101),
]

# Iterate JAX last: JAX and TensorFlow both preallocate all GPU memory, and
# initializing JAX before TensorFlow causes OOM on GPU CI (see the ordering in
# utils_integration_test.py).
_ALL_BACKENDS = ["numpy", "torch", "tensorflow", "jax"]


def _installed_backends():
    """Return the installed backends, in the safe _ALL_BACKENDS iteration order."""
    return [backend for backend in _ALL_BACKENDS if find_spec(backend) is not None]


def _to_float(result):
    """Convert a scalar backend tensor to a Python float via NumPy."""
    return float(np.asarray(ar.to_numpy(result)))


def test_cross_backend_agreement():
    """Deterministic integrators must agree across all installed backends in fp64.

    Both integrands are linear per dimension, so every rule here integrates them
    exactly; that lets the test anchor to the analytic value as well as to
    cross-backend agreement (so a shared, identically-wrong result cannot pass).
    """
    backends = _installed_backends()
    if len(backends) < 2:
        pytest.skip("Need at least two installed backends to compare")

    # (integrand, dim, domain, analytic result). Linear per dimension -> exact.
    cases = [
        (lambda x: x[:, 0], 1, [[0.0, 2.0]], 2.0),  # integral of x over [0, 2]
        (
            lambda x: x[:, 0] + 2.0 * x[:, 1],  # exercises meshgrid (issue #214)
            2,
            [[0.0, 1.0], [0.0, 2.0]],
            5.0,
        ),
    ]

    # This test mutates global backend/precision state; restore the library
    # default (numpy/float32) on exit so a later test relying on it is unaffected.
    try:
        for integrand, dim, domain, expected in cases:
            for integrator_cls, N in _DETERMINISTIC_INTEGRATORS:
                results = {}
                for backend in backends:
                    set_up_backend(backend, "float64")
                    integration_domain = anp.array(
                        domain, like=backend, dtype=to_backend_dtype("float64", like=backend)
                    )
                    result = integrator_cls().integrate(
                        integrand, dim=dim, N=N, integration_domain=integration_domain
                    )
                    results[backend] = _to_float(result)

                reference = results[backends[0]]
                for backend, value in results.items():
                    assert abs(value - reference) < 1e-11, (
                        f"{integrator_cls.__name__} (dim={dim}): {backend} disagrees with "
                        f"{backends[0]} ({value} vs {reference})"
                    )
                    assert abs(value - expected) < 1e-9, (
                        f"{integrator_cls.__name__} (dim={dim}): {backend} result "
                        f"{value} != {expected}"
                    )
    finally:
        set_up_backend("numpy", "float32")


def _integer_domain_test(backend):
    """An integer-dtype domain must be promoted to float, not yield 0 (#180).

    Args:
        backend (str): Numerical backend under test.
    """
    dtype_int = to_backend_dtype("int64", like=backend)
    # Integer bounds [0, 3]; f(x) = x integrates exactly to 4.5 for every rule.
    integration_domain = anp.array([[0, 3]], like=backend, dtype=dtype_int)

    for integrator_cls, N in _DETERMINISTIC_INTEGRATORS:
        result = integrator_cls().integrate(
            lambda x: x[:, 0], dim=1, N=N, integration_domain=integration_domain
        )
        value = _to_float(result)
        assert value != 0.0, (
            f"{integrator_cls.__name__} integrated an integer domain to 0 (issue #180)"
        )
        assert abs(value - 4.5) < 1e-4, (
            f"{integrator_cls.__name__}: integer-domain result {value} != 4.5"
        )


# Defined in the safe _ALL_BACKENDS order (JAX last) to avoid GPU OOM.
test_integer_domain_numpy = setup_test_for_backend(_integer_domain_test, "numpy", None)
test_integer_domain_torch = setup_test_for_backend(_integer_domain_test, "torch", None)
test_integer_domain_tensorflow = setup_test_for_backend(_integer_domain_test, "tensorflow", None)
test_integer_domain_jax = setup_test_for_backend(_integer_domain_test, "jax", None)


if __name__ == "__main__":
    test_cross_backend_agreement()
    test_integer_domain_numpy()
    test_integer_domain_torch()
    test_integer_domain_tensorflow()
    test_integer_domain_jax()
    print("All cross-backend tests passed!")

"""Tests for passing extra integrand arguments via ``args`` (issues #187, #188).

Every integrator forwards ``args`` to the integrand as ``fn(points, *args)``, so a
parametric integrand can be integrated without wrapping it in a lambda.
"""

import numpy as np

from torchquad.integration.trapezoid import Trapezoid
from torchquad.integration.simpson import Simpson
from torchquad.integration.boole import Boole
from torchquad.integration.gaussian import GaussLegendre
from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.vegas import VEGAS
from helper_functions import setup_test_for_backend

_ALPHA = 3.0
_BETA = 2.0
_DOMAIN = [[0.0, 1.0]]
# integral over [0, 1] of (alpha * x + beta) is alpha/2 + beta.
_EXPECTED = _ALPHA * 0.5 + _BETA


def _to_float(result):
    """Convert a scalar backend tensor (possibly on GPU) to a Python float."""
    if hasattr(result, "cpu"):
        result = result.cpu()
    return float(np.asarray(result))


def _parametric(x, alpha, beta):
    """A parametric integrand exercising multi-argument ``*args`` unpacking."""
    return alpha * x[:, 0] + beta


def _args_deterministic_test(backend, dtype_name=None):
    """Grid integrators must forward args; the integrand here is exact for them."""
    for integrator_cls, N in [(Trapezoid, 101), (Simpson, 101), (Boole, 101), (GaussLegendre, 32)]:
        result = _to_float(
            integrator_cls().integrate(
                _parametric, dim=1, N=N, integration_domain=_DOMAIN, args=(_ALPHA, _BETA)
            )
        )
        assert abs(result - _EXPECTED) < 1e-9, (
            f"{integrator_cls.__name__} with args gave {result}, expected {_EXPECTED}"
        )


def _args_default_none_test(backend, dtype_name=None):
    """The default args=None must behave exactly like binding the parameters up front."""
    with_args = _to_float(
        Simpson().integrate(
            _parametric, dim=1, N=101, integration_domain=_DOMAIN, args=(_ALPHA, _BETA)
        )
    )
    bound = _to_float(
        Simpson().integrate(
            lambda x: _parametric(x, _ALPHA, _BETA), dim=1, N=101, integration_domain=_DOMAIN
        )
    )
    assert with_args == bound


def _args_monte_carlo_test(backend, dtype_name=None):
    """MonteCarlo must forward args."""
    result = _to_float(
        MonteCarlo().integrate(
            _parametric, dim=1, N=10000, integration_domain=_DOMAIN, seed=0, args=(_ALPHA, _BETA)
        )
    )
    assert abs(result - _EXPECTED) < 0.05, f"MonteCarlo with args gave {result}"


def _args_vegas_test(backend, dtype_name=None):
    """VEGAS must forward args (numpy and torch only)."""
    result = _to_float(
        VEGAS().integrate(
            _parametric, dim=1, N=20000, integration_domain=_DOMAIN, seed=0, args=(_ALPHA, _BETA)
        )
    )
    assert abs(result - _EXPECTED) < 0.05, f"VEGAS with args gave {result}"


test_args_deterministic_numpy = setup_test_for_backend(_args_deterministic_test, "numpy", "float64")
test_args_deterministic_torch = setup_test_for_backend(_args_deterministic_test, "torch", "float64")
test_args_deterministic_tensorflow = setup_test_for_backend(
    _args_deterministic_test, "tensorflow", "float64"
)
test_args_deterministic_jax = setup_test_for_backend(_args_deterministic_test, "jax", "float64")

test_args_default_none_numpy = setup_test_for_backend(_args_default_none_test, "numpy", "float64")
test_args_default_none_torch = setup_test_for_backend(_args_default_none_test, "torch", "float64")

test_args_monte_carlo_numpy = setup_test_for_backend(_args_monte_carlo_test, "numpy", "float64")
test_args_monte_carlo_torch = setup_test_for_backend(_args_monte_carlo_test, "torch", "float64")
test_args_monte_carlo_tensorflow = setup_test_for_backend(
    _args_monte_carlo_test, "tensorflow", "float64"
)
test_args_monte_carlo_jax = setup_test_for_backend(_args_monte_carlo_test, "jax", "float64")

# VEGAS supports numpy and torch only.
test_args_vegas_numpy = setup_test_for_backend(_args_vegas_test, "numpy", "float64")
test_args_vegas_torch = setup_test_for_backend(_args_vegas_test, "torch", "float64")


if __name__ == "__main__":
    for _backend in ["numpy", "torch", "tensorflow", "jax"]:
        _args_deterministic_test(_backend)
        _args_monte_carlo_test(_backend)
    _args_default_none_test("numpy")
    for _backend in ["numpy", "torch"]:
        _args_vegas_test(_backend)
    print("All args tests passed!")

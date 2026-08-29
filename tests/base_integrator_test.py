import warnings

import numpy as np
import pytest
from autoray import numpy as anp
from autoray import to_backend_dtype, to_numpy

from torchquad.integration.base_integrator import BaseIntegrator
from helper_functions import setup_test_for_backend

NUM_POINTS = 16


def _make_points_and_weights(backend, dtype_name, num_points=NUM_POINTS):
    """Build sample points and quadrature weights for the given backend.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        dtype_name ("float32" or "float64"): Floating point precision
        num_points (int, optional): Number of sample points. Defaults to 16.

    Returns:
        backend tensor: Sample points of shape (num_points, 1)
        backend tensor: Weights of shape (num_points,)
    """
    dtype = to_backend_dtype(dtype_name, like=backend)
    points = anp.array(
        np.linspace(0.0, 1.0, num_points).reshape(num_points, 1), like=backend, dtype=dtype
    )
    weights = anp.array(np.linspace(0.5, 2.0, num_points), like=backend, dtype=dtype)
    return points, weights


def _run_no_mutation_tests(backend, dtype_name):
    """evaluate_integrand must not modify the tensor the integrand returned.

    The integrand's return value belongs to the caller, not to torchquad. Applying
    the quadrature weights in place mutated it, which broke PyTorch autograd for
    any integrand whose backward pass reads its own output and silently corrupted
    tensors that an integrand caches and reuses across calls.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        dtype_name ("float32" or "float64"): Floating point precision
    """
    points, weights = _make_points_and_weights(backend, dtype_name)
    returned = {}

    def integrand(x):
        values = anp.exp(x[:, 0])
        returned["tensor"] = values
        return values

    result, num_points = BaseIntegrator.evaluate_integrand(integrand, points, weights=weights)

    unweighted = np.exp(to_numpy(points)[:, 0])
    assert num_points == points.shape[0]
    # The tensor the integrand handed over is still what it returned.
    assert np.allclose(to_numpy(returned["tensor"]), unweighted)
    # ...and the weights were nonetheless applied to the value torchquad returns.
    assert np.allclose(to_numpy(result), unweighted * to_numpy(weights))


def _run_no_mutation_multidim_tests(backend, dtype_name):
    """The same guarantee for an integrand returning more than one value per point.

    This exercises the branch of evaluate_integrand that reshapes and repeats the
    weights before applying them.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        dtype_name ("float32" or "float64"): Floating point precision
    """
    points, weights = _make_points_and_weights(backend, dtype_name)
    returned = {}

    def integrand(x):
        values = anp.stack([anp.exp(x[:, 0]), anp.exp(2.0 * x[:, 0])], axis=1)
        returned["tensor"] = values
        return values

    result, _ = BaseIntegrator.evaluate_integrand(integrand, points, weights=weights)

    x = to_numpy(points)[:, 0]
    unweighted = np.stack([np.exp(x), np.exp(2.0 * x)], axis=1)
    assert to_numpy(result).shape == (NUM_POINTS, 2)
    assert np.allclose(to_numpy(returned["tensor"]), unweighted)
    assert np.allclose(to_numpy(result), unweighted * to_numpy(weights)[:, None])


def _run_weight_dtype_tests(backend, dtype_name):
    """Applying the weights must not downcast them to the integrand's dtype.

    An in-place multiply takes the dtype of its left operand, so an integrand
    returning float32 under float64 precision silently discarded the weights'
    precision. The result now keeps the wider dtype, and the mismatch is warned
    about rather than passed over in silence.

    Restricted to numpy and torch: TensorFlow refuses to multiply mixed dtypes
    outright, so the mismatch cannot be constructed there at all.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        dtype_name ("float32" or "float64"): Floating point precision
    """
    points, weights = _make_points_and_weights(backend, dtype_name)
    narrower = to_backend_dtype("float32", like=backend)

    def integrand(x):
        return anp.astype(anp.exp(x[:, 0]), narrower)

    with pytest.warns(UserWarning, match="quadrature weights"):
        result, _ = BaseIntegrator.evaluate_integrand(integrand, points, weights=weights)
    # The weights' precision survives rather than being truncated to the integrand's.
    assert to_numpy(result).dtype == to_numpy(weights).dtype


def _run_matching_dtype_is_quiet_tests(backend, dtype_name):
    """An integrand that matches the configured precision must not warn.

    The mismatch warning has to stay off the ordinary path, including for complex
    integrands: complex128 carries the same precision as the float64 weights it is
    multiplied by, so pairing them loses nothing and must not be flagged.

    Args:
        backend (string): Numerical backend, e.g. "torch"
        dtype_name ("float32" or "float64"): Floating point precision
    """
    points, weights = _make_points_and_weights(backend, dtype_name)

    def real_integrand(x):
        return anp.exp(x[:, 0])

    def complex_integrand(x):
        return anp.astype(anp.exp(x[:, 0]), to_backend_dtype("complex128", like=backend))

    for integrand in (real_integrand, complex_integrand):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BaseIntegrator.evaluate_integrand(integrand, points, weights=weights)


test_no_mutation_numpy = setup_test_for_backend(_run_no_mutation_tests, "numpy", "float64")
test_no_mutation_torch = setup_test_for_backend(_run_no_mutation_tests, "torch", "float64")
test_no_mutation_jax = setup_test_for_backend(_run_no_mutation_tests, "jax", "float64")
test_no_mutation_tensorflow = setup_test_for_backend(
    _run_no_mutation_tests, "tensorflow", "float64"
)

test_no_mutation_multidim_numpy = setup_test_for_backend(
    _run_no_mutation_multidim_tests, "numpy", "float64"
)
test_no_mutation_multidim_torch = setup_test_for_backend(
    _run_no_mutation_multidim_tests, "torch", "float64"
)
test_no_mutation_multidim_jax = setup_test_for_backend(
    _run_no_mutation_multidim_tests, "jax", "float64"
)
test_no_mutation_multidim_tensorflow = setup_test_for_backend(
    _run_no_mutation_multidim_tests, "tensorflow", "float64"
)

test_weight_dtype_numpy = setup_test_for_backend(_run_weight_dtype_tests, "numpy", "float64")
test_weight_dtype_torch = setup_test_for_backend(_run_weight_dtype_tests, "torch", "float64")

test_matching_dtype_is_quiet_numpy = setup_test_for_backend(
    _run_matching_dtype_is_quiet_tests, "numpy", "float64"
)
test_matching_dtype_is_quiet_torch = setup_test_for_backend(
    _run_matching_dtype_is_quiet_tests, "torch", "float64"
)

if __name__ == "__main__":
    # used to run this test individually
    test_no_mutation_numpy()
    test_no_mutation_torch()
    test_no_mutation_jax()
    test_no_mutation_tensorflow()
    test_no_mutation_multidim_numpy()
    test_no_mutation_multidim_torch()
    test_no_mutation_multidim_jax()
    test_no_mutation_multidim_tensorflow()
    test_weight_dtype_numpy()
    test_weight_dtype_torch()
    test_matching_dtype_is_quiet_numpy()
    test_matching_dtype_is_quiet_torch()

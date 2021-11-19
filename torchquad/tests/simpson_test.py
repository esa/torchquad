import sys

sys.path.append("../")

import warnings

from integration.simpson import Simpson
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level
from integration_test_utils import compute_integration_test_errors


def _run_simpson_tests(backend):
    """Test the integrate function in integration.Simpson for the given backend."""

    simp = Simpson()

    # 1D Tests
    N = 100001

    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # Polynomials up to degree 3 can be integrated almost exactly with Simpson.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 3e-11
    for error in errors:
        assert error < 1e-7

    N = 3  # integration points, here 3 for order check (3 points should lead to almost 0 err for low order polynomials)
    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # All polynomials up to degree = 3 should be 0
    # If this breaks, check if test functions in integration_test_utils changed.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 1e-15

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors, funcs = compute_integration_test_errors(
            simp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True, backend=backend
        )
    print(f"3D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 3 or err < 1e-12
    for error in errors:
        assert error < 5e-6

    # 10D Tests
    N = 3 ** 10
    errors, funcs = compute_integration_test_errors(
        simp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True, backend=backend
    )
    print(f"10D Simpson Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for error in errors:
        assert error < 5e-9


def test_integrate_numpy():
    """Test the integrate function in integration.Simpson with Numpy"""
    set_log_level("INFO")
    _run_simpson_tests("numpy")


def test_integrate_torch():
    """Test the integrate function in integration.Simpson with Torch"""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double", backend="torch")
    _run_simpson_tests("torch")


def test_integrate_jax():
    """Test the integrate function in integration.Simpson with Torch"""
    set_log_level("INFO")
    set_precision("double", backend="jax")
    _run_simpson_tests("jax")


# Skip tensorflow since it does not yet support double as global precision


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()

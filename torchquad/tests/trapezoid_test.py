import sys

sys.path.append("../")

from integration.trapezoid import Trapezoid
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_trapezoid_tests(backend, _precision):
    """Test the integrate function in integration.Trapezoid for the given backend."""

    tp = Trapezoid()

    # 1D Tests
    N = 100000
    errors, funcs = compute_integration_test_errors(
        tp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Trapezoid Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # Polynomials up to degree 1 can be integrated almost exactly with Trapezoid.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 1 or err < 2e-11
    for error in errors:
        assert error < 1e-5

    N = 2  # integration points, here 2 for order check (2 points should lead to almost 0 err for low order polynomials)
    errors, funcs = compute_integration_test_errors(
        tp.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Trapezoid Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # All polynomials up to degree = 1 should be 0
    # If this breaks check if test functions in helper_functions changed.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 1 or err < 1e-15
    for error in errors[:2]:
        assert error < 1e-15

    # 3D Tests
    N = 1000000
    errors, funcs = compute_integration_test_errors(
        tp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True, backend=backend
    )
    print(f"3D Trapezoid Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 1 or err < 1e-12
    for error in errors:
        assert error < 6e-3

    # Tensorflow crashes with an Op:StridedSlice UnimplementedError with 10
    # dimensions
    if backend == "tensorflow":
        print("Skipping tensorflow 10D tests")
        return

    # 10D Tests
    N = 10000
    errors, funcs = compute_integration_test_errors(
        tp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True, backend=backend
    )
    print(f"10D Trapezoid Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 1 or err < 1e-11
    for error in errors:
        assert error < 7000


test_integrate_numpy = setup_test_for_backend(_run_trapezoid_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_trapezoid_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_trapezoid_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(
    _run_trapezoid_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

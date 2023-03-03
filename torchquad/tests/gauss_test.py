import sys

sys.path.append("../")

from integration.gaussian import GaussLegendre
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_gaussian_tests(backend, _precision):
    """Test the integrate function in integration.gaussian for the given backend."""

    integrators = [GaussLegendre()]

    # 1D Tests
    N = 60

    for integrator in integrators:
        ii = integrator
        errors, funcs = compute_integration_test_errors(
            ii.integrate,
            {"N": N, "dim": 1},
            integration_dim=1,
            use_complex=True,
            backend=backend,
        )
        print(
            f"1D {integrator} Test passed. N: {N}, backend: {backend}, Errors: {errors}"
        )
        # Polynomials up to degree 1 can be integrated almost exactly with gaussian.
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 1 or err < 2e-11
        for error in errors:
            assert error < 1e-5

    N = 2  # integration points, here 2 for order check (2 points should lead to almost 0 err for low order polynomials)
    for integrator in integrators:
        ii = integrator
        errors, funcs = compute_integration_test_errors(
            ii.integrate,
            {"N": N, "dim": 1},
            integration_dim=1,
            use_complex=True,
            backend=backend,
        )
        print(
            f"1D {integrator} Test passed. N: {N}, backend: {backend}, Errors: {errors}"
        )
        # All polynomials up to degree = 1 should be 0
        # If this breaks check if test functions in helper_functions changed.
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 1 or err < 1e-15
        for error in errors[:2]:
            assert error < 1e-15

    # 3D Tests
    N = 60**3
    for integrator in integrators:
        ii = integrator
        errors, funcs = compute_integration_test_errors(
            ii.integrate,
            {"N": N, "dim": 3},
            integration_dim=3,
            use_complex=True,
            backend=backend,
        )
        print(
            f"3D {integrator} Test passed. N: {N}, backend: {backend}, Errors: {errors}"
        )
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 1 or (
                err < 1e-12 if test_function.is_integrand_1d else err < 1e-11
            )
        for error in errors:
            assert error < 6e-3

    # Tensorflow crashes with an Op:StridedSlice UnimplementedError with 10
    # dimensions
    if backend == "tensorflow":
        print("Skipping tensorflow 10D tests")
        return

    # 10D Tests
    N = (60**3) * 3
    for integrator in integrators:
        ii = integrator
        errors, funcs = compute_integration_test_errors(
            ii.integrate,
            {"N": N, "dim": 10},
            integration_dim=10,
            use_complex=True,
            backend=backend,
        )
        print(
            f"10D {integrator} Test passed. N: {N}, backend: {backend}, Errors: {errors}"
        )
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 1 or err < 1e-11
        for error in errors:
            assert error < 7000


test_integrate_numpy = setup_test_for_backend(_run_gaussian_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_gaussian_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_gaussian_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(
    _run_gaussian_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

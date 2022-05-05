import sys

sys.path.append("../")

import warnings

from integration.boole import Boole
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_boole_tests(backend, _precision):
    """Test the integrate function in integration.Boole for the given backend.
    Note: For now the 10-D test is diabled due to lack of GPU memory on some computers."""

    bl = Boole()
    # 1D Tests
    N = 401

    errors, funcs = compute_integration_test_errors(
        bl.integrate, {"N": N, "dim": 1}, dim=1, use_complex=True, backend=backend
    )
    print(f"1D Boole Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # Polynomials up to degree 5 can be integrated almost exactly with Boole.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 5 or err < 6.33e-11
    for error in errors:
        assert error < 6.33e-11

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors, funcs = compute_integration_test_errors(
            bl.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True, backend=backend
        )
    print(f"3D Boole Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 5 or err < 2e-13
    for error in errors:
        assert error < 5e-6

    # 10D Tests
    # Have been disabled for now because it is too GPU-heavy
    # N = 5 ** 10
    # errors = compute_test_errors(bl.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True)
    # print("10D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    # for error in errors:
    # assert error < 5e-9


test_integrate_numpy = setup_test_for_backend(_run_boole_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_boole_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_boole_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(
    _run_boole_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

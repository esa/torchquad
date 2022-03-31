import sys

sys.path.append("../")

from integration.monte_carlo import MonteCarlo
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_monte_carlo_tests(backend, _precision):
    """Test the integrate function in integration.MonteCarlo for the given backend."""

    mc = MonteCarlo()

    # 1D Tests
    N = 100000  # integration points to use

    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 1, "seed": 0},
        dim=1,
        use_complex=True,
        backend=backend,
    )
    print(
        f"1D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}"
    )
    # Constant functions can be integrated exactly with MonteCarlo.
    # (at least our example functions)
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0

    # If this breaks check if test functions in helper_functions changed.
    for error in errors[:3]:
        assert error < 7e-3

    assert errors[3] < 0.5
    assert errors[4] < 32.0

    for error in errors[6:10]:
        assert error < 1e-2

    for error in errors[10:]:
        assert error < 28.03

    # 3D Tests
    N = 1000000
    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 3, "seed": 0},
        dim=3,
        use_complex=True,
        backend=backend,
    )
    print(
        f"3D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}"
    )
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0
    for error in errors:
        assert error < 1e-1

    # 10D Tests
    N = 10000
    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 10, "seed": 0},
        dim=10,
        use_complex=True,
        backend=backend,
    )
    print(
        f"10D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors:"
        f" {str(errors)}"
    )
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0
    for error in errors:
        assert error < 26


test_integrate_numpy = setup_test_for_backend(
    _run_monte_carlo_tests, "numpy", "float32"
)
test_integrate_torch = setup_test_for_backend(
    _run_monte_carlo_tests, "torch", "float32"
)
test_integrate_jax = setup_test_for_backend(_run_monte_carlo_tests, "jax", "float32")
test_integrate_tensorflow = setup_test_for_backend(
    _run_monte_carlo_tests, "tensorflow", "float32"
)


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

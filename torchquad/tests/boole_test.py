import sys

sys.path.append("../")

import warnings

from integration.boole import Boole
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level
from integration_test_utils import compute_integration_test_errors


def _run_boole_tests(backend):
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


def test_integrate_numpy():
    """Test the integrate function in integration.Boole with Numpy"""
    set_log_level("INFO")
    _run_boole_tests("numpy")


def test_integrate_torch():
    """Test the integrate function in integration.Boole with Torch"""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double", backend="torch")
    _run_boole_tests("torch")


def test_integrate_jax():
    """Test the integrate function in integration.Boole with Torch"""
    set_log_level("INFO")
    set_precision("double", backend="jax")
    _run_boole_tests("jax")


# Skip tensorflow since it does not yet support double as global precision


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()

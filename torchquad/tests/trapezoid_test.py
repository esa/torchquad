import sys

sys.path.append("../")


from integration.trapezoid import Trapezoid
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level
from integration_test_utils import compute_integration_test_errors

# ~ from tensorflow.python.ops.numpy_ops import np_config

# ~ np_config.enable_numpy_behavior()


def _run_trapezoid_tests(backend):
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
    # If this breaks check if test functions in integration_test_utils changed.
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


def test_integrate_numpy():
    """Test the integrate function in integration.Trapezoid with Numpy"""
    set_log_level("INFO")
    _run_trapezoid_tests("numpy")


def test_integrate_torch():
    """Test the integrate function in integration.Trapezoid with Torch"""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double", backend="torch")
    _run_trapezoid_tests("torch")


def test_integrate_jax():
    """Test the integrate function in integration.Trapezoid with Torch"""
    set_log_level("INFO")
    set_precision("double", backend="jax")
    _run_trapezoid_tests("jax")


# Skip tensorflow since it does not yet support double as global precision


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()

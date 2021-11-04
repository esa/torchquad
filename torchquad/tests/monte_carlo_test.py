import sys

sys.path.append("../")

from integration.monte_carlo import MonteCarlo
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level
from integration_test_utils import run_example_functions

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def _run_monte_carlo_tests(backend):
    """Test the integrate function in integration.MonteCarlo for the given backend."""

    mc = MonteCarlo()

    # 1D Tests
    N = 100000  # integration points to use

    errors, funcs = run_example_functions(
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

    # If this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 7e-3

    for error in errors[3:5]:
        assert error < 28.0

    for error in errors[6:10]:
        assert error < 1e-2

    for error in errors[10:]:
        assert error < 28.03

    # 3D Tests
    N = 1000000
    errors, funcs = run_example_functions(
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
    errors, funcs = run_example_functions(
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


def test_integrate_numpy():
    """Test the integrate function in integration.MonteCarlo with Numpy"""
    set_log_level("INFO")
    _run_monte_carlo_tests("numpy")


def test_integrate_torch():
    """Test the integrate function in integration.MonteCarlo with Torch"""
    set_log_level("INFO")
    enable_cuda()
    # 32 bit float precision suffices for Monte Carlo tests
    set_precision("float", backend="torch")
    _run_monte_carlo_tests("torch")


def test_integrate_jax():
    """Test the integrate function in integration.MonteCarlo with Torch"""
    set_log_level("INFO")
    set_precision("float", backend="jax")
    _run_monte_carlo_tests("jax")


def test_integrate_tensorflow():
    """Test the integrate function in integration.MonteCarlo with Tensorflow"""
    set_log_level("INFO")
    _run_monte_carlo_tests("tensorflow")


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

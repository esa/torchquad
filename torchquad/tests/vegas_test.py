import sys

sys.path.append("../")

import torch
import timeit

import cProfile

from integration.vegas import VEGAS
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision


def test_integrate():
    """Tests the integrate function in integration.Trapezoid"""
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    vegas = VEGAS()

    # 1D Tests
    N = 10000
    errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 1, "seed": 0})
    print("Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:3]:
        assert error < 5e-3

    for error in errors:
        assert error < 3.0

    for error in errors[6:]:
        assert error < 6e-3

    # 3D Tests
    N = 10000
    errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 3, "seed": 0}, dim=3)
    print("Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 0.61

    # 10D Tests
    N = 10000
    errors = compute_test_errors(
        vegas.integrate, {"N": N, "dim": 10, "seed": 0}, dim=10
    )
    print("Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 12.5


if __name__ == "__main__":
    # used to run this test individually
    start = timeit.default_timer()
    test_integrate()
    # cProfile.run("test_integrate()")
    stop = timeit.default_timer()
    print("Test ran for ", stop - start, " seconds.")

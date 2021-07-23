import sys

sys.path.append("../")

import warnings
import torch

from integration.boole import Boole
from integration.monte_carlo import MonteCarlo
from integration.vegas import VEGAS
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision


def test_integrate():
    """Tests the compatability with imaginary numbers for some different integrate functions."""
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_complex_test_errors

    # Boole
    bl = Boole()
    # 1D Tests
    N = 401
    errors = compute_complex_test_errors(bl.integrate, {"N": N, "dim": 1})
    print("Complex 1D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-11
    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = compute_complex_test_errors(bl.integrate, {"N": N, "dim": 3}, dim=3)
    print("Complex 1D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 2e-13

    # Monte Carlo
    mc = MonteCarlo()
    torch.manual_seed(0)  # we have to seed torch to get reproducible results
    # 1D Tests
    N = 100000  # integration points to use
    errors = compute_complex_test_errors(mc.integrate, {"N": N, "dim": 1})
    print("Complex 1D Monte Carlo Test: Passed N =", N, "\n", "Errors: ", errors)
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:1]:
        assert error < 7e-3

    for error in errors[2:]:
        assert error < 12.0
    # 3D Tests
    N = 1000000
    errors = compute_complex_test_errors(mc.integrate, {"N": N, "dim": 3}, dim=3)
    print("Complex 3D Monte Carlo Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-2

    # # VEGAS
    # vegas = VEGAS()
    # # 1D Tests
    # N = 10000
    # errors = compute_complex_test_errors(vegas.integrate, {"N": N, "dim": 1, "seed": 0})
    # print("N =", N, "\n", errors)
    # for error in errors[:3]:
    #     assert error < 5e-3
    # for error in errors:
    #     assert error < 3.0
    # for error in errors[6:]:
    #     assert error < 6e-3
    # # 3D Tests
    # N = 10000
    # errors = compute_complex_test_errors(
    #     vegas.integrate, {"N": N, "dim": 3, "seed": 0}, dim=3
    # )
    # print("N =", N, "\n", errors)
    # for error in errors:
    #     assert error < 0.61


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()
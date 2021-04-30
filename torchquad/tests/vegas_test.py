import sys

sys.path.append("../")

import torch
import timeit

import cProfile

from integration.vegas import VEGAS
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Trapezoid"""
    torch.set_default_tensor_type(torch.FloatTensor)
    vegas = VEGAS()

    # 1D Tests
    # N = 10000
    # errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 1, "seed": 0})
    # print("N =", N, "\n", errors)
    # for error in errors[:3]:
    #     assert error < 5e-3

    # for error in errors:
    #     assert error < 3.0

    # for error in errors[6:]:
    #     assert error < 6e-3

    # 3D Tests
    N = 10000
    errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 3, "seed": 0}, dim=3)
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 0.61


# used to run this test individually
start = timeit.default_timer()
cProfile.run("test_integrate()")
stop = timeit.default_timer()
print("Test ran for ", stop - start, " seconds.")

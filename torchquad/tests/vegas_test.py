import sys

sys.path.append("../")

import torch

from integration.vegas import VEGAS
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Trapezoid
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    vegas = VEGAS()

    N = 100000

    # TODO needs to be adjusted for vegas
    # 1D Tests
    # errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 1, "seed": 0})
    # print("N =", N, "\n", errors)
    # for error in errors:
    #     assert error < 1e-5

    # TODO needs to be adjusted for vegas
    # 3D Tests
    # N = 1000000
    # errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 3, "seed": 0}, dim=3)
    # print("N =", N, "\n", errors)
    # for error in errors[:2]:
    #     assert error < 1e-12
    # for error in errors:
    #     assert error < 6e-3


# used to run this test individually
test_integrate()

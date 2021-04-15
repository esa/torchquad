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

    # 1D Tests
    N = 10000
    errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 1, "seed": 0})
    print("N =", N, "\n", errors)
    for error in errors[:3]:
        assert error < 1e-2

    for error in errors[3:6]:
        assert error < 9.0

    for error in errors[6:]:
        assert error < 1e-2

    # 3D Tests
    N = 10000
    errors = compute_test_errors(vegas.integrate, {"N": N, "dim": 3, "seed": 0}, dim=3)
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1.1


# used to run this test individually
test_integrate()

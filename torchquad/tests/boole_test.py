import sys

sys.path.append("../")

import torch
import warnings

from integration.boole import Boole
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Boole"""
    torch.set_default_tensor_type(torch.DoubleTensor)
    bl = Boole()
    N = 401

    errors = compute_test_errors(bl.integrate, {"N": N, "dim": 1})
    print("Passed N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-11

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = compute_test_errors(bl.integrate, {"N": N, "dim": 3}, dim=3)
    print("Passed N =", N, "\n", errors)
    for error in errors[:3]:
        assert error < 2e-13
    for error in errors:
        assert error < 5e-6


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

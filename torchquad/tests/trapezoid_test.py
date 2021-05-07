import sys

sys.path.append("../")

import torch


from integration.trapezoid import Trapezoid
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Trapezoid"""
    torch.set_default_tensor_type(torch.DoubleTensor)
    tp = Trapezoid()
    N = 100000

    # 1D Tests
    errors = compute_test_errors(tp.integrate, {"N": N, "dim": 1})
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-5

    N = 2  # integration points, here 2 for order check (2 points should lead to almost 0 err for low order polynomials)
    errors = compute_test_errors(tp.integrate, {"N": N, "dim": 1})
    print("N =", N, "\n", errors)
    # all polynomials up to degree = 1 should be 0
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:2]:
        assert error < 1e-15

    # 3D Tests
    N = 1000000
    errors = compute_test_errors(tp.integrate, {"N": N, "dim": 3}, dim=3)
    print("N =", N, "\n", errors)
    for error in errors[:2]:
        assert error < 1e-12
    for error in errors:
        assert error < 6e-3


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

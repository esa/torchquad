import sys

sys.path.append("../")

import torch


from integration.simpson import Simpson
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Simpson
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    simp = Simpson()
    N = 100001

    errors = compute_test_errors(simp.integrate, {"N": N, "dim": 1})
    print("Passed N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-7

    N = 3  # integration points, here 3 for order check (3 points should lead to almost 0 err for low order polynomials)
    errors = compute_test_errors(simp.integrate, {"N": N, "dim": 1})
    print("Passed N =", N, "\n", errors)
    # all polynomials up to degree = 3 should be 0
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 1e-15

    # 3D Tests
    N = 1030340  # N = 101.001 per dim
    errors = compute_test_errors(simp.integrate, {"N": N, "dim": 3}, dim=3)
    print("Passed N =", N, "\n", errors)
    for error in errors[:3]:
        assert error < 1e-12
    for error in errors:
        assert error < 5e-6


# used to run this test individually
test_integrate()

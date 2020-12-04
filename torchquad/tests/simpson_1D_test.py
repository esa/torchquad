import sys

sys.path.append("../")

import torch


from integration.simpson_1D import Simpson1D
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Simpson
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    simp = Simpson1D()
    N = 1000001  # integration points

    errors = compute_test_errors(simp.integrate, {"N": N})
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-7

    N = 3  # integration points, here 3 for order check (3 points should lead to almost 0 err for low order polynomials)
    errors = compute_test_errors(simp.integrate, {"N": N})
    print("N =", N, "\n", errors)
    # all polynomials up to degree = 3 should be 0
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 1e-15


# used to run this test individually
test_integrate()

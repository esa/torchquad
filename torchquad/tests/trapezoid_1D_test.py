import sys

sys.path.append("../")

import torch


from integration.trapezoid_1D import Trapezoid1D
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    torch.set_default_tensor_type(torch.DoubleTensor)
    tp = Trapezoid1D()
    N = 10000000

    errors = compute_test_errors(tp.integrate, {"N": N})
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-6

    N = 2
    errors = compute_test_errors(tp.integrate, {"N": N})
    print("N =", N, "\n", errors)
    # all polynomials up to degree = 1 should be 0
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:2]:
        assert error < 1e-15


# used to run this test individually
test_integrate()

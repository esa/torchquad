import sys

sys.path.append("../")

import torch

from integration.monte_carlo import MonteCarlo
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.MonteCarlo
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    N = 100000000  # integration points to use
    torch.manual_seed(0)  # we have to seed torch to get reproducible results
    mc = MonteCarlo()

    errors = compute_test_errors(mc.integrate, {"N": N, "dim": 1})
    print(errors)
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 1e-4

    for error in errors[3:7]:
        assert error < 1.0

    for error in errors[7:]:
        assert error < 1e-4


# used to run this test individually
test_integrate()

import sys

sys.path.append("../")

import torch

from integration.monte_carlo import MonteCarlo
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision


def test_integrate():
    """Tests the integrate function in integration.MonteCarlo"""
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    N = 100000  # integration points to use
    torch.manual_seed(0)  # we have to seed torch to get reproducible results

    mc = MonteCarlo()

    errors = compute_test_errors(mc.integrate, {"N": N, "dim": 1})
    print(errors)
    # if this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 7e-3

    for error in errors[3:7]:
        assert error < 28.0

    for error in errors[7:]:
        assert error < 4e-3

    # 3D Tests
    N = 1000000
    errors = compute_test_errors(mc.integrate, {"N": N, "dim": 3}, dim=3)
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-1


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

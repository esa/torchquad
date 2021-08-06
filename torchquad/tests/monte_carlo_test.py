import sys

sys.path.append("../")

import torch

from integration.monte_carlo import MonteCarlo
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """Tests the integrate function in integration.MonteCarlo."""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    mc = MonteCarlo()

    # 1D Tests
    N = 100000  # integration points to use
    torch.manual_seed(0)  # we have to seed torch to get reproducible results

    errors = compute_test_errors(mc.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Monte Carlo Test: Passed N =", N, "\n", "Errors: ", errors)
    # If this breaks check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 7e-3

    for error in errors[3:5]:
        assert error < 28.0

    for error in errors[6:10]:
        assert error < 4e-3

    for error in errors[10:]:
        assert error < 28.0

    # 3D Tests
    N = 1000000
    errors = compute_test_errors(
        mc.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True
    )
    print("3D Monte Carlo Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-1

    # 10D Tests
    N = 10000
    errors = compute_test_errors(
        mc.integrate, {"N": N, "dim": 10, "seed": 0}, dim=10, use_complex=True
    )
    print("10D Monte Carlo Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 13


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

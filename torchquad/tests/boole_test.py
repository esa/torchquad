import sys

sys.path.append("../")

import warnings

from integration.boole import Boole
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """Tests the integrate function in integration.Boole.
    Note: For now the 10-D test is diabled due to lack of GPU memory on some computers."""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    bl = Boole()
    # 1D Tests
    N = 401

    errors = compute_test_errors(bl.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-11

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = compute_test_errors(
            bl.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True
        )
    print("3D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:3]:
        assert error < 2e-13
    for error in errors:
        assert error < 5e-6

    # 10D Tests
    # Have been disabled for now because it is too GPU-heavy
    # N = 5 ** 10
    # errors = compute_test_errors(bl.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True)
    # print("10D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    # for error in errors:
    # assert error < 5e-9


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

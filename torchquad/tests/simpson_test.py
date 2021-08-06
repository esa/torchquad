import sys

sys.path.append("../")

import warnings

from integration.simpson import Simpson
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """Tests the integrate function in integration.Simpson."""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    simp = Simpson()

    # 1D Tests
    N = 100001

    errors = compute_test_errors(simp.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Simpson Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-7

    N = 3  # integration points, here 3 for order check (3 points should lead to almost 0 err for low order polynomials)
    errors = compute_test_errors(simp.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Simpson Test: Passed N =", N, "\n", "Errors: ", errors)
    # All polynomials up to degree = 3 should be 0
    # If this breaks, check if test functions in integration_test_utils changed.
    for error in errors[:3]:
        assert error < 1e-15

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = compute_test_errors(
            simp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True
        )
    print("3D Simpson Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:3]:
        assert error < 1e-12
    for error in errors:
        assert error < 5e-6

    # 10D Tests
    N = 3 ** 10
    errors = compute_test_errors(
        simp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True
    )
    print("10D Simpson Test: N =", N, "\n", errors)
    for error in errors:
        assert error < 5e-9


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

import sys

sys.path.append("../")


from integration.trapezoid import Trapezoid
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """Tests the integrate function in integration.Trapezoid."""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    tp = Trapezoid()
    N = 100000

    # 1D Tests
    errors = compute_test_errors(tp.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Trapezoid Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-5

    N = 2  # integration points, here 2 for order check (2 points should lead to almost 0 err for low order polynomials)
    errors = compute_test_errors(tp.integrate, {"N": N, "dim": 1}, use_complex=True)
    print("1D Trapezoid Test: Passed N =", N, "\n", "Errors: ", errors)
    # All polynomials up to degree = 1 should be 0
    # If this breaks check if test functions in integration_test_utils changed.
    for error in errors[:2]:
        assert error < 1e-15

    # 3D Tests
    N = 1000000
    errors = compute_test_errors(
        tp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=True
    )
    print("3D Trapezoid Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:2]:
        assert error < 1e-12
    for error in errors:
        assert error < 6e-3

    # 10D Tests
    N = 10000
    errors = compute_test_errors(
        tp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True
    )
    print("10D Trapezoid Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 7000


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

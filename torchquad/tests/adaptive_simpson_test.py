import sys

sys.path.append("../")


from integration.adaptive_simpson import AdaptiveSimpson
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """Tests the integrate function in integration.AdaptiveSimpson."""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Needs to happen after precision / device settings to avoid having some tensors intialized on cpu and some on GPU
    from tests.integration_test_utils import compute_test_errors

    sp = AdaptiveSimpson()
    N = 100000

    # 1D Tests
    errors = compute_test_errors(sp.integrate, {"N": N, "dim": 1}, use_complex=False)
    print("1D AdaptiveSimpson Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 1e-9

    # 3D Tests
    N = 1000000
    errors = compute_test_errors(
        sp.integrate, {"N": N, "dim": 3}, dim=3, use_complex=False
    )
    print("3D AdaptiveSimpson Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:2]:
        assert error < 1e-13
    for error in errors:
        assert error < 1e-5

    # 10D Tests
    # N = 20000000
    # # Have been disabled for now because it is too memory-intense
    # errors = compute_test_errors(
    #     sp.integrate, {"N": N, "dim": 10}, dim=10, use_complex=True
    # )
    # print("10D AdaptiveSimpson Test: Passed N =", N, "\n", "Errors: ", errors)
    # for error in errors:
    #     assert error < 7000


if __name__ == "__main__":
    # used to run this test individually
    test_integrate()

import sys

sys.path.append("../")

import torch


from integration.boole import Boole
from tests.integration_test_utils import compute_test_errors


def test_integrate():
    """Tests the integrate function in integration.Boole
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    bl = Boole()
    N = 400

    errors = compute_test_errors(bl.integrate, {"N": N, "dim": 1})
    print("N =", N, "\n", errors)
    for error in errors:
        assert error < 1e-3

    # 3D Tests
    N = 1000000  # N = 369 per dim
    errors = compute_test_errors(bl.integrate, {"N": N, "dim": 3}, dim=3)
    print("N =", N, "\n", errors)
    for error in errors[:3]:
        assert error < 1e-12
    for error in errors:
        assert error < 5e-6


# used to run this test individually
test_integrate()

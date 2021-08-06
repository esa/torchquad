import sys

sys.path.append("../")

import torch

from integration.vegas import VEGAS
from integration.monte_carlo import MonteCarlo
from integration.trapezoid import Trapezoid
from integration.simpson import Simpson
from integration.boole import Boole

from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def some_function(x):
    """V shaped test function.
    Gradient in positive x should be 2,
    Gradient in negative x should be -2
    for -1 to 1 domain."""
    return 2 * torch.abs(x)


def test_gradients():
    """Tests that the implemented integrators
    maintain torch gradients and they are consistent and correct"""
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    N = 99997
    result_tolerence = 1e-2
    gradient_tolerance = 2e-2
    torch.manual_seed(0)  # we have to seed torch to get reproducible results

    # Define integrators
    integrators = [MonteCarlo(), Trapezoid(), Simpson(), Boole(), VEGAS()]
    for integrator in integrators:
        print("Running integrator...", integrator)

        # Compute integral
        domain = torch.tensor([[-1.0, 1.0]])
        domain.requires_grad = True
        result = integrator.integrate(
            some_function, dim=1, N=N, integration_domain=domain
        )

        # Check results are correct
        assert torch.abs(result - 2.0) < result_tolerence

        # Check for presence of gradient
        assert hasattr(result, "grad_fn")

        # Backprop gradient through integral
        result.backward()

        # Check that gradient is correct
        assert torch.abs(domain.grad[0, 0] + 2.0) < gradient_tolerance
        assert torch.abs(domain.grad[0, 1] - 2.0) < gradient_tolerance


if __name__ == "__main__":
    # used to run this test individually
    test_gradients()

from .integration_test_functions import Polynomial, Exponential, Sinusoid

import numpy as np

# Here we define a bunch of function that shall be used for testing
TEST_FUNCTIONS_1D = [
    Polynomial(4.0, [2.0]),  # y = 2
    Polynomial(0, [0, 1]),  # y = x
    Polynomial(2 / 3, [0, 0, 2], domain=[[0, 1]]),  # y = 2x^2
    Polynomial(27.75, [3, -1, 2, -3], domain=[[-2, 1]]),  # y=-3x^3+2x^2-x+3
    # y=7x^4-3x^3+2x^2-x+3
    Polynomial(44648.0 / 15.0, [3, -1, 2, -3, 7], domain=[[-4, 4]]),
    # # y=-x^5+7x^4-3x^3+2x^2-x+3
    Polynomial(8939.0 / 60.0, [3, -1, 2, -3, 7, -1], domain=[[2, 3]]),
    Exponential(np.exp(1) - np.exp(-2), domain=[[-2, 1]]),
    Exponential((np.exp(2) - 1.0) / np.exp(3), domain=[[-3, -1]]),
    Sinusoid(2 * np.sin(1) * np.sin(1), domain=[[0, 2]]),
]

TEST_FUNCTIONS_3D = [
    Polynomial(48.0, [2.0], dim=3),  # 2
    Polynomial(0, [0, 1], dim=3),  # y = x
    Polynomial(8.0, coeffs=[0, 0, 1], dim=3),  # x^2+y^2+z^2
    Exponential(
        27 * (np.exp(3) - 1) / np.exp(2), domain=[[-2, 1], [-2, 1], [-2, 1]]
    ),  # e^x+e^y+e^z
    Sinusoid(24 * np.sin(1) ** 2, domain=[[0, 2], [0, 2], [0, 2]]),
]

# Check if convergence order is met
def check_convergence_prder(integrator, errors, test_functions):
    """Checks for the passed integrator if all errors matched the expected convergence criteria

    Args:
        integrator (func): Used integrator
        errors (np.array): Computed error terms
        test_functions (torchquad.IntegrationTestFunction): utilized test functions
    """
    # TODO Implement this
    raise NotImplementedError("This is not yet implemented.")


def compute_test_errors(integrator, integrator_args, dim=1):
    """Computes errors on all test functions for given dimension and integrator

    Args:
        integrator (torchquad.base_integrator): Integrator to use
        integrator_args (dict): arguments for the integrator
        dim (int, optional): Dimensionality of test functions to use. Defaults to 1.

    Returns:
        list: absolute errors on all test functions
    """
    errors = []

    # get the test functions
    if dim == 1:
        test_functions = TEST_FUNCTIONS_1D
    elif dim == 3:
        test_functions = TEST_FUNCTIONS_3D
    else:
        raise ValueError("Not testing functions implemented for dim " + str(dim))

    # compute integration errors on all of them
    for test_function in test_functions:
        errors.append(
            np.abs(
                test_function.evaluate(integrator, integrator_args).cpu().numpy()
                - test_function.expected_result
            )
        )

    return errors

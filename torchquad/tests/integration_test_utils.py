from .integration_test_functions import Polynomial, Exponential, Sinusoid

import numpy as np

# Here we define a bunch of functions that will be used for testing
TEST_FUNCTIONS_1D = [
    Polynomial(4.0, [2.0]),  # y = 2
    Polynomial(0, [0, 1]),  # y = x
    Polynomial(2 / 3, [0, 0, 2], domain=[[0, 1]]),  # y = 2x^2
    Polynomial(27.75, [3, -1, 2, -3], domain=[[-2, 1]]),  # y = -3x^3+2x^2-x+3
    # y = 7x^4-3x^3+2x^2-x+3
    Polynomial(44648.0 / 15.0, [3, -1, 2, -3, 7], domain=[[-4, 4]]),
    # # y = -x^5+7x^4-3x^3+2x^2-x+3
    Polynomial(8939.0 / 60.0, [3, -1, 2, -3, 7, -1], domain=[[2, 3]]),
    Exponential(np.exp(1) - np.exp(-2), domain=[[-2, 1]]),
    Exponential((np.exp(2) - 1.0) / np.exp(3), domain=[[-3, -1]]),
    Sinusoid(2 * np.sin(1) * np.sin(1), domain=[[0, 2]]),
]

TEST_FUNCTIONS_3D = [
    Polynomial(48.0, [2.0], dim=3),  # f(x,y,z) = 2
    Polynomial(0, [0, 1], dim=3),  # f(x,y,z) = x + y + z
    Polynomial(8.0, coeffs=[0, 0, 1], dim=3),  # f(x,y,z) = x^2+y^2+z^2
    # e^x+e^y+e^z
    Exponential(27 * (np.exp(3) - 1) / np.exp(2), domain=[[-2, 1], [-2, 1], [-2, 1]]),
    Sinusoid(24 * np.sin(1) ** 2, domain=[[0, 2], [0, 2], [0, 2]]),
    # e^x+e^y+e^z
    Exponential(1.756, domain=[[-0.05, 0.1], [-0.25, 0.2], [-np.exp(1), np.exp(1)]]),
]

TEST_FUNCTIONS_10D = [
    # f(x_1, ..., x_10) = x_1^2+x_2^2+...
    Polynomial(3413.33333333, coeffs=[0, 0, 1], dim=10),
]


def compute_test_errors(integrator, integrator_args, dim=1):
    """Computes errors on all test functions for given dimension and integrator.

    Args:
        integrator (torchquad.base_integrator): Integrator to use.
        integrator_args (dict): Arguments for the integrator.
        dim (int, optional): Dimensionality of test functions to use. Defaults to 1.

    Returns:
        list: Absolute errors on all test functions
    """
    errors = []

    # Get the test functions
    if dim == 1:
        test_functions = TEST_FUNCTIONS_1D
    elif dim == 3:
        test_functions = TEST_FUNCTIONS_3D
    elif dim == 10:
        test_functions = TEST_FUNCTIONS_10D
    else:
        raise ValueError("Not testing functions implemented for dim " + str(dim))

    # Compute integration errors on all of them
    for test_function in test_functions:
        errors.append(
            np.abs(
                test_function.evaluate(integrator, integrator_args).cpu().numpy()
                - test_function.expected_result
            )
        )

    return errors


# Here we define a bunch of functions with complex numbers that will be used for testing
TEST_FUNCTIONS_COMPLEX_1D = [
    Polynomial(4.0j, [2.0j]),  # y = 2j
    Polynomial(0, [0, 1j]),  # y = xj
    # y=7x^4-3jx^3+2x^2-jx+3
    Polynomial(44648.0 / 15.0 + 0.0j, [3, -1j, 2, -3j, 7], domain=[[-4, 4]]),
]

TEST_FUNCTIONS_COMPLEX_3D = [
    Polynomial(48.0j, [2.0j], dim=3),  # f(x,y,z) = 2j
    Polynomial(0, [0, 1.0j], dim=3),  # f(x,y,z) = xj
    Polynomial(8.0j, coeffs=[0, 0, 1.0j], dim=3),  # j*x^2+j*y^2+j*z^2
]


def compute_complex_test_errors(integrator, integrator_args, dim=1):
    """Computes errors on all test functions for given dimension and integrator.
    Args:
        integrator (torchquad.base_integrator): Integrator to use.
        integrator_args (dict): Arguments for the integrator.
        dim (int, optional): Dimensionality of test functions to use. Defaults to 1.
    Returns:
        list: Absolute errors on all test functions
    """
    errors = []

    # Get the test functions
    if dim == 1:
        test_functions = TEST_FUNCTIONS_COMPLEX_1D
    elif dim == 3:
        test_functions = TEST_FUNCTIONS_COMPLEX_3D
    else:
        raise ValueError("Not testing functions implemented for dim " + str(dim))

    # Compute integration errors on all of them
    for test_function in test_functions:
        errors.append(
            np.abs(
                test_function.evaluate(integrator, integrator_args).cpu().numpy()
                - test_function.expected_result
            )
        )

    return errors

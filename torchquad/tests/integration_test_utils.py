import numpy as np

from .integration_test_functions import Polynomial, Exponential, Sinusoid


# Here we define a bunch of functions that will be used for testing
TEST_FUNCTIONS_1D = [
    # Real numbers
    Polynomial(4.0, [2.0], is_complex=False),  # y = 2
    Polynomial(0, [0, 1], is_complex=False),  # y = x
    Polynomial(2 / 3, [0, 0, 2], domain=[[0, 1]], is_complex=False),  # y = 2x^2
    # y = -3x^3+2x^2-x+3
    Polynomial(27.75, [3, -1, 2, -3], domain=[[-2, 1]], is_complex=False),
    # y = 7x^4-3x^3+2x^2-x+3
    Polynomial(44648.0 / 15.0, [3, -1, 2, -3, 7], domain=[[-4, 4]], is_complex=False),
    # # y = -x^5+7x^4-3x^3+2x^2-x+3
    Polynomial(8939.0 / 60.0, [3, -1, 2, -3, 7, -1], domain=[[2, 3]], is_complex=False),
    Exponential(np.exp(1) - np.exp(-2), domain=[[-2, 1]], is_complex=False),
    Exponential((np.exp(2) - 1.0) / np.exp(3), domain=[[-3, -1]], is_complex=False),
    Sinusoid(2 * np.sin(1) * np.sin(1), domain=[[0, 2]], is_complex=False),
    #
    # Complex numbers
    Polynomial(4.0j, [2.0j], is_complex=True),  # y = 2j
    Polynomial(0, [0, 1j], is_complex=True),  # y = xj
    # y=7x^4-3jx^3+2x^2-jx+3
    Polynomial(44648.0 / 15.0, [3, -1j, 2, -3j, 7], domain=[[-4, 4]], is_complex=True),
]

TEST_FUNCTIONS_3D = [
    # Real numbers
    Polynomial(48.0, [2.0], dim=3, is_complex=False),  # f(x,y,z) = 2
    Polynomial(0, [0, 1], dim=3, is_complex=False),  # f(x,y,z) = x + y + z
    # f(x,y,z) = x^2+y^2+z^2
    Polynomial(8.0, coeffs=[0, 0, 1], dim=3, is_complex=False),
    # e^x+e^y+e^z
    Exponential(
        27 * (np.exp(3) - 1) / np.exp(2),
        domain=[[-2, 1], [-2, 1], [-2, 1]],
        is_complex=False,
    ),
    Sinusoid(24 * np.sin(1) ** 2, domain=[[0, 2], [0, 2], [0, 2]], is_complex=False),
    # e^x+e^y+e^z
    Exponential(
        1.756,
        domain=[[-0.05, 0.1], [-0.25, 0.2], [-np.exp(1), np.exp(1)]],
        is_complex=False,
    ),
    #
    # Complex numbers
    Polynomial(48.0j, [2.0j], dim=3, is_complex=True),  # f(x,y,z) = 2j
    Polynomial(0, [0, 1.0j], dim=3, is_complex=True),  # f(x,y,z) = xj
    Polynomial(8.0j, coeffs=[0, 0, 1.0j], dim=3, is_complex=True),  # j*x^2+j*y^2+j*z^2
]

TEST_FUNCTIONS_10D = [
    # Real numbers
    # f(x_1, ..., x_10) = x_1^2+x_2^2+...
    Polynomial(3413.33333333, coeffs=[0, 0, 1], dim=10, is_complex=False),
    # Complex numbers
    # f(x_1, ..., x_10) = j*x_1^2+j*x_2^2+...
    Polynomial(3413.33333333j, coeffs=[0, 0, 1.0j], dim=10, is_complex=True),
]


def compute_test_errors(
    integrator,
    integrator_args,
    dim=1,
    use_complex=False,
):
    """Computes errors on all test functions for given dimension and integrator.

    Args:
        integrator (torchquad.base_integrator): Integrator to use.
        integrator_args (dict): Arguments for the integrator.
        dim (int, optional): Dimensionality of test functions to use. Defaults to 1.
        use_complex (Boolean, optional): If complex test functions should be used. Defaults to False.

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
        if not test_function.is_complex:
            errors.append(
                np.abs(
                    test_function.evaluate(integrator, integrator_args)
                    .cpu()
                    .detach()
                    .numpy()
                    - test_function.expected_result
                )
            )
        if test_function.is_complex and use_complex:
            errors.append(
                np.abs(
                    test_function.evaluate(integrator, integrator_args)
                    .cpu()
                    .detach()
                    .numpy()
                    - test_function.expected_result
                )
            )

    return errors

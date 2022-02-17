import numpy as np
import pytest

from integration_test_functions import Polynomial, Exponential, Sinusoid
from utils.set_up_backend import set_up_backend
from utils.set_log_level import set_log_level


def get_test_functions(dim, backend):
    """Here we define a bunch of functions that will be used for testing.

    Args:
        dim (int): Dimensionality of test functions to use.
        backend (string): Numerical backend used for the integration
    """
    if dim == 1:
        return [
            # Real numbers
            Polynomial(4.0, [2.0], is_complex=False, backend=backend),  # y = 2
            Polynomial(0, [0, 1], is_complex=False, backend=backend),  # y = x
            Polynomial(
                2 / 3, [0, 0, 2], domain=[[0, 1]], is_complex=False, backend=backend
            ),  # y = 2x^2
            # y = -3x^3+2x^2-x+3
            Polynomial(
                27.75,
                [3, -1, 2, -3],
                domain=[[-2, 1]],
                is_complex=False,
                backend=backend,
            ),
            # y = 7x^4-3x^3+2x^2-x+3
            Polynomial(
                44648.0 / 15.0,
                [3, -1, 2, -3, 7],
                domain=[[-4, 4]],
                is_complex=False,
                backend=backend,
            ),
            # # y = -x^5+7x^4-3x^3+2x^2-x+3
            Polynomial(
                8939.0 / 60.0,
                [3, -1, 2, -3, 7, -1],
                domain=[[2, 3]],
                is_complex=False,
                backend=backend,
            ),
            Exponential(
                np.exp(1) - np.exp(-2),
                domain=[[-2, 1]],
                is_complex=False,
                backend=backend,
            ),
            Exponential(
                (np.exp(2) - 1.0) / np.exp(3),
                domain=[[-3, -1]],
                is_complex=False,
                backend=backend,
            ),
            Sinusoid(
                2 * np.sin(1) * np.sin(1),
                domain=[[0, 2]],
                is_complex=False,
                backend=backend,
            ),
            #
            # Complex numbers
            Polynomial(4.0j, [2.0j], is_complex=True, backend=backend),  # y = 2j
            Polynomial(0, [0, 1j], is_complex=True, backend=backend),  # y = xj
            # y=7x^4-3jx^3+2x^2-jx+3
            Polynomial(
                44648.0 / 15.0,
                [3, -1j, 2, -3j, 7],
                domain=[[-4, 4]],
                is_complex=True,
                backend=backend,
            ),
        ]
    elif dim == 3:
        return [
            # Real numbers
            Polynomial(
                48.0, [2.0], dim=3, is_complex=False, backend=backend
            ),  # f(x,y,z) = 2
            Polynomial(
                0, [0, 1], dim=3, is_complex=False, backend=backend
            ),  # f(x,y,z) = x + y + z
            # f(x,y,z) = x^2+y^2+z^2
            Polynomial(8.0, coeffs=[0, 0, 1], dim=3, is_complex=False, backend=backend),
            # e^x+e^y+e^z
            Exponential(
                27 * (np.exp(3) - 1) / np.exp(2),
                dim=3,
                domain=[[-2, 1], [-2, 1], [-2, 1]],
                is_complex=False,
                backend=backend,
            ),
            Sinusoid(
                24 * np.sin(1) ** 2,
                dim=3,
                domain=[[0, 2], [0, 2], [0, 2]],
                is_complex=False,
                backend=backend,
            ),
            # e^x+e^y+e^z
            Exponential(
                1.756,
                dim=3,
                domain=[[-0.05, 0.1], [-0.25, 0.2], [-np.exp(1), np.exp(1)]],
                is_complex=False,
                backend=backend,
            ),
            #
            # Complex numbers
            Polynomial(
                48.0j, [2.0j], dim=3, is_complex=True, backend=backend
            ),  # f(x,y,z) = 2j
            Polynomial(
                0, [0, 1.0j], dim=3, is_complex=True, backend=backend
            ),  # f(x,y,z) = xj
            Polynomial(
                8.0j, coeffs=[0, 0, 1.0j], dim=3, is_complex=True, backend=backend
            ),  # j*x^2+j*y^2+j*z^2
        ]
    elif dim == 10:
        return [
            # Real numbers
            # f(x_1, ..., x_10) = x_1^2+x_2^2+...
            Polynomial(
                3413.33333333,
                coeffs=[0, 0, 1],
                dim=10,
                is_complex=False,
                backend=backend,
            ),
            # Complex numbers
            # f(x_1, ..., x_10) = j*x_1^2+j*x_2^2+...
            Polynomial(
                3413.33333333j,
                coeffs=[0, 0, 1.0j],
                dim=10,
                is_complex=True,
                backend=backend,
            ),
        ]
    else:
        raise ValueError("Not testing functions implemented for dim " + str(dim))


def compute_integration_test_errors(
    integrator,
    integrator_args,
    dim,
    use_complex,
    backend,
):
    """Computes errors on all test functions for given dimension and integrator.

    Args:
        integrator (torchquad.base_integrator): Integrator to use.
        integrator_args (dict): Arguments for the integrator.
        dim (int): Dimensionality of the example functions to choose.
        use_complex (Boolean): If True, skip complex example functions.
        backend (string): Numerical backend for the example functions.

    Returns:
        (list, list): Absolute errors on all example functions and the chosen
            example functions
    """
    errors = []
    chosen_functions = []

    # Compute integration errors on the chosen functions and remember those
    # functions
    for test_function in get_test_functions(dim, backend):
        if not use_complex and test_function.is_complex:
            continue
        if backend == "torch":
            errors.append(
                np.abs(
                    test_function.evaluate(integrator, integrator_args)
                    .cpu()
                    .detach()
                    .numpy()
                    - test_function.expected_result
                )
            )
        else:
            errors.append(
                np.abs(
                    test_function.evaluate(integrator, integrator_args)
                    - test_function.expected_result
                )
            )
        chosen_functions.append(test_function)

    return errors, chosen_functions


def setup_test_for_backend(test_func, backend, dtype_name):
    """
    Create a function to execute a test function with the given numerical backend.
    If the backend is not installed, skip the test.

    Args:
        test_func (function(backend, dtype_name)): The function which runs tests
        backend (string): The numerical backend
        dtype_name ("float32", "float64" or None): Floating point precision. If None, the global precision is not changed.

    Returns:
        function: A test function for Pytest
    """

    def func():
        pytest.importorskip(backend)
        set_log_level("INFO")
        set_up_backend(backend, dtype_name)
        if dtype_name is None:
            return test_func(backend)
        return test_func(backend, dtype_name)

    return func

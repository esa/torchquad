import numpy as np
import pytest
from autoray import numpy as anp
import autoray as ar

from integration_test_functions import Polynomial, Exponential, Sinusoid
from torchquad.utils.set_up_backend import set_up_backend
from torchquad.utils.set_log_level import set_log_level


def get_test_functions(integration_dim, backend, use_multi_dim_integrand):
    """Here we define a bunch of functions that will be used for testing.

    Args:
        integration_dim (int): Dimensionality of test functions to use.
        backend (string): Numerical backend used for the integration
        use_multi_dim_integrand (bool): Whether or not to allow for a multi-dimensional integrand i.e an array of integrands
    """
    if integration_dim == 1:
        res = [
            # Real numbers
            Polynomial(4.0, [2.0], is_complex=False, backend=backend, integrand_dims=1),  # y = 2
            Polynomial(0, [0, 1], is_complex=False, backend=backend, integrand_dims=1),  # y = x
            Polynomial(
                2 / 3,
                [0, 0, 2],
                domain=[[0, 1]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),  # y = 2x^2
            # y = -3x^3+2x^2-x+3
            Polynomial(
                27.75,
                [3, -1, 2, -3],
                domain=[[-2, 1]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            # y = 7x^4-3x^3+2x^2-x+3
            Polynomial(
                44648.0 / 15.0,
                [3, -1, 2, -3, 7],
                domain=[[-4, 4]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            # # y = -x^5+7x^4-3x^3+2x^2-x+3
            Polynomial(
                8939.0 / 60.0,
                [3, -1, 2, -3, 7, -1],
                domain=[[2, 3]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            Exponential(
                np.exp(1) - np.exp(-2),
                domain=[[-2, 1]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            Exponential(
                (np.exp(2) - 1.0) / np.exp(3),
                domain=[[-3, -1]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            Sinusoid(
                2 * np.sin(1) * np.sin(1),
                domain=[[0, 2]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            #
            # Complex numbers
            Polynomial(4.0j, [2.0j], is_complex=True, backend=backend, integrand_dims=1),  # y = 2j
            Polynomial(0, [0, 1j], is_complex=True, backend=backend, integrand_dims=1),  # y = xj
            # y=7x^4-3jx^3+2x^2-jx+3
            Polynomial(
                44648.0 / 15.0,
                [3, -1j, 2, -3j, 7],
                domain=[[-4, 4]],
                is_complex=True,
                backend=backend,
                integrand_dims=1,
            ),
        ]
        if use_multi_dim_integrand:
            res += [
                # Over 2 integrand dims, one of which is 1 or like 1
                Polynomial(
                    np.array([[0.0], [4.0]]),
                    [2.0],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 1],
                ),  # y = 2
                Polynomial(
                    np.array([0.0, 4.0]),
                    [2.0],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=(2,),
                ),  # y = 2
                # over 2 integrand dims
                Polynomial(
                    np.array([[0.0, 4.0], [8.0, 12.0]]),
                    [2.0],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2],
                ),  # y = 2
                Polynomial(
                    np.array([[0.0, 0.0], [0.0, 0.0]]),
                    [0, 1],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2],
                ),  # y = x
                # over 3 integrand dims
                Polynomial(
                    np.array([[[0.0, 4.0], [8.0, 12.0]], [[16.0, 20.0], [24.0, 28.0]]]),
                    [2.0],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2, 2],
                ),  # y = 2
                Polynomial(
                    np.array([[0.0, 0.0], [0.0, 0.0]]),
                    [0, 1],
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2, 2],
                ),  # y = x
            ]
        return res
    elif integration_dim == 3:
        res = [
            # Real numbers
            Polynomial(
                48.0,
                [2.0],
                integration_dim=3,
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),  # f(x,y,z) = 2
            Polynomial(
                0,
                [0, 1],
                integration_dim=3,
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),  # f(x,y,z) = x + y + z
            # f(x,y,z) = x^2+y^2+z^2
            Polynomial(
                8.0,
                coeffs=[0, 0, 1],
                integration_dim=3,
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            # e^x+e^y+e^z
            Exponential(
                27 * (np.exp(3) - 1) / np.exp(2),
                integration_dim=3,
                domain=[[-2, 1], [-2, 1], [-2, 1]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            Sinusoid(
                24 * np.sin(1) ** 2,
                integration_dim=3,
                domain=[[0, 2], [0, 2], [0, 2]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            # e^x+e^y+e^z
            Exponential(
                1.756,
                integration_dim=3,
                domain=[[-0.05, 0.1], [-0.25, 0.2], [-np.exp(1), np.exp(1)]],
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            #
            # Complex numbers
            Polynomial(
                48.0j,
                [2.0j],
                integration_dim=3,
                is_complex=True,
                backend=backend,
                integrand_dims=1,
            ),  # f(x,y,z) = 2j
            Polynomial(
                0,
                [0, 1.0j],
                integration_dim=3,
                is_complex=True,
                backend=backend,
                integrand_dims=1,
            ),  # f(x,y,z) = xj
            Polynomial(
                8.0j,
                coeffs=[0, 0, 1.0j],
                integration_dim=3,
                is_complex=True,
                backend=backend,
                integrand_dims=1,
            ),  # j*x^2+j*y^2+j*z^2
        ]
        if use_multi_dim_integrand:
            res += [
                # Over 2 integrand dims, one of which is 1 or like 1
                Polynomial(
                    np.array([[0.0], [48.0]]),
                    integration_dim=3,
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 1],
                ),  # f(x,y,z) = 2
                Polynomial(
                    np.array([0.0, 48.0]),
                    integration_dim=3,
                    is_complex=False,
                    backend=backend,
                    integrand_dims=(2,),
                ),  # f(x,y,z) = 2
                # Over 2 integrand dims
                Polynomial(
                    np.array([[0.0, 48.0], [96.0, 144.0]]),
                    integration_dim=3,
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2],
                ),  # f(x,y,z) = 2
                Polynomial(
                    np.array([[0.0, 0.0], [0.0, 0.0]]),
                    [0, 1],
                    integration_dim=3,
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2],
                ),  # f(x,y,z) = x + y + z
                # Over 3 integrand dims
                Polynomial(  # MC tests fail here with default float32 precision, so need float64
                    np.array([[[0.0, 48.0], [96.0, 144.0]], [[192.0, 240.0], [288.0, 336.0]]]),
                    integration_dim=3,
                    domain=anp.array(
                        [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                        like=backend,
                        dtype=ar.to_backend_dtype("float64", like=backend),
                    ),
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2, 2],
                ),  # f(x,y,z) = 2
                Polynomial(
                    np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    [0, 1],
                    integration_dim=3,
                    is_complex=False,
                    backend=backend,
                    integrand_dims=[2, 2, 2],
                ),  # f(x,y,z) = x + y + z
            ]
        return res
    elif integration_dim == 10:
        return [
            # Real numbers
            # f(x_1, ..., x_10) = x_1^2+x_2^2+...
            Polynomial(
                3413.33333333,
                coeffs=[0, 0, 1],
                integration_dim=10,
                is_complex=False,
                backend=backend,
                integrand_dims=1,
            ),
            # Complex numbers
            # f(x_1, ..., x_10) = j*x_1^2+j*x_2^2+...
            Polynomial(
                3413.33333333j,
                coeffs=[0, 0, 1.0j],
                integration_dim=10,
                is_complex=True,
                backend=backend,
                integrand_dims=1,
            ),
        ]
    else:
        raise ValueError(
            "Not testing functions implemented for integration_dim " + str(integration_dim)
        )


def compute_integration_test_errors(
    integrator,
    integrator_args,
    integration_dim,
    use_complex,
    backend,
    use_multi_dim_integrand=True,
    filter_test_functions=lambda x: x,
):
    """Computes errors on all test functions for given dimension and integrator.

    Args:
        integrator (torchquad.base_integrator): Integrator to use.
        integrator_args (dict): Arguments for the integrator.
        integration_dim (int): Dimensionality of the example functions to choose.
        use_complex (Boolean): If True, skip complex example functions.
        backend (string): Numerical backend for the example functions.
        use_multi_dim_integrand (bool, optional): Whether or not to allow for a multi-dimensional integrand i.e an array of integrands
        filter_test_functions (function, optional): function for filtering which test functions to run
    Returns:
        (list, list): Absolute errors on all example functions and the chosen
            example functions
    """
    errors = []
    chosen_functions = []

    # Compute integration errors on the chosen functions and remember those
    # functions
    for test_function in filter(
        filter_test_functions,
        get_test_functions(integration_dim, backend, use_multi_dim_integrand),
    ):
        if not use_complex and test_function.is_complex:
            continue
        if backend == "torch":
            diff = np.abs(
                test_function.evaluate(integrator, integrator_args).cpu().detach().numpy()
                - test_function.expected_result
            )
        else:
            diff = np.abs(
                test_function.evaluate(integrator, integrator_args) - test_function.expected_result
            )
        if test_function.is_integrand_1d:
            errors.append(diff)
        else:
            errors.append(np.sum(diff))
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

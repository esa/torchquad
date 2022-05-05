import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_numpy, to_backend_dtype, get_dtype_name
import numpy as np

from integration.vegas import VEGAS
from integration.monte_carlo import MonteCarlo
from integration.trapezoid import Trapezoid
from integration.simpson import Simpson
from integration.boole import Boole

from helper_functions import setup_test_for_backend


def _v_function(x):
    """
    V shaped test function 2 |x|.
    Gradient in positive x should be 2,
    Gradient in negative x should be -2
    for -1 to 1 domain.
    """
    return 2 * anp.abs(x)


def _polynomial_function(x):
    """
    2D test function 3 x_1 ^ 2 + 2 x_0 + 1.
    The four gradient components are integrals of the function over the
    integration domain rectangle's four sides multiplied by the factors
    -1, -1, 1 and 1 for the sides -X_1, -X_2, X_1 and X_2 respectively.
    For example, with integration_domain [[0.0, 1.0], [0.0, 2.0]],
    the gradient of the integral with respect to this domain is
    [[-10.0, 14.0], [-2.0, 14.0]].
    """
    return 1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 1] ** 2


def _polynomial_function_parameterized(x, coeffs):
    """
    2D test function coeffs_2 x_1 ^ 2 + coeffs_1 x_0 + coeffs_0.
    """
    return coeffs[0] + coeffs[1] * x[:, 0] + coeffs[2] * x[:, 1] ** 2


def _v_function_parameterized(x, c):
    """
    V shaped test function 2 |x + c|.
    """
    return 2 * anp.abs(x + c)


def _calculate_gradient(backend, param, func, dtype_name):
    """Backend-specific gradient calculation

    Args:
        backend (string): Numerical backend, e.g. "torch"
        param (list or float): Parameter value(s) for func. The gradient of func is calculated over param.
        func (function): A function which receives param and should be derived
        dtype_name (string): Floating point precision

    Returns:
        backend tensor: Gradient of func over param
        backend tensor: Value of func at param
    """
    if backend == "torch":
        import torch

        # Set up param for gradient calculation
        param = torch.tensor(param)
        param.requires_grad = True

        # Compute the value of func at param
        result = func(param)

        # Check for presence of gradient
        assert hasattr(result, "grad_fn")

        # Backpropagate to get the gradient of func over param
        result.backward()
        gradient = param.grad

    elif backend == "jax":
        import jax

        # Convert param to a JAX tensor
        param = anp.array(param, like="jax")

        # Calculate the value and gradient
        value_and_grad_func = jax.value_and_grad(func)
        result, gradient = value_and_grad_func(param)

    elif backend == "tensorflow":
        import tensorflow as tf

        # Set up param as Variable
        dtype = to_backend_dtype(dtype_name, like=backend)
        param = tf.Variable(param, dtype=dtype)

        # Calculate the value and gradient
        with tf.GradientTape() as tape:
            result = func(param)
        gradient = tape.gradient(result, param)

    else:
        raise ValueError(f"No gradient calculation for the backend {backend}")

    assert get_dtype_name(result) == dtype_name
    assert get_dtype_name(gradient) == dtype_name
    assert gradient.shape == param.shape
    return to_numpy(gradient), to_numpy(result)


def _calculate_gradient_over_domain(
    backend, integration_domain, integrate, integrate_kwargs, dtype_name
):
    """Backend-specific calculation of the gradient of integrate over integration_domain

    Args:
        backend (string): Numerical backend, e.g. "torch"
        integration_domain (list): Integration domain
        integrate (function): A integrator's integrate method
        integrate_kwargs (dict): Arguments for integrate except integration_domain
        dtype_name (string): Floating point precision

    Returns:
        backend tensor: Gradient with respect to integration_domain
        backend tensor: Integral result
    """
    return _calculate_gradient(
        backend,
        integration_domain,
        lambda dom: integrate(integration_domain=dom, **integrate_kwargs),
        dtype_name,
    )


def _calculate_gradient_over_param(
    backend, param, integrand_with_param, integrate, integrate_kwargs, dtype_name
):
    """Backend-specific calculation of the gradient of integrate over an integrand parameter

    Args:
        backend (string): Numerical backend, e.g. "torch"
        param (list or float): Parameter value(s) for the integrand. The gradient of integrate is calculated over param.
        integrand_with_param (function): An integrand function which receives sample points and param
        integrate (function): A integrator's integrate method
        integrate_kwargs (dict): Arguments for integrate except fn (the integrand)
        dtype_name (string): Floating point precision

    Returns:
        backend tensor: Gradient with respect to param
        backend tensor: Integral result
    """
    return _calculate_gradient(
        backend,
        param,
        lambda par: integrate(
            lambda x: integrand_with_param(x, par), **integrate_kwargs
        ),
        dtype_name,
    )


def _run_gradient_tests(backend, dtype_name):
    """
    Test if the implemented integrators
    maintain gradients and if the gradients are consistent and correct
    """
    # Define integrators and numbers of evaluation points
    integrators = [Trapezoid(), Simpson(), Boole(), MonteCarlo(), VEGAS()]
    Ns_1d = [149, 149, 149, 99997, 99997]
    Ns_2d = [549, 121, 81, 99997, 99997]
    for integrator, N_1d, N_2d in zip(integrators, Ns_1d, Ns_2d):
        integrator_name = type(integrator).__name__
        requires_seed = integrator_name in ["MonteCarlo", "VEGAS"]
        if backend != "torch" and integrator_name == "VEGAS":
            # Currently VEGAS supports only Torch.
            continue

        print(
            f"Calculating gradients; backend: {backend}, integrator: {integrator_name}"
        )

        print("Calculating gradients of the one-dimensional V-shaped function")
        integrate_kwargs = {"fn": _v_function, "dim": 1, "N": N_1d}
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient_over_domain(
            backend,
            [[-1.0, 1.0]],
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        # Check if the integral and gradient are accurate enough
        assert np.abs(integral - 2.0) < 1e-2
        assert np.all(np.abs(gradient - np.array([-2.0, 2.0])) < 2e-2)

        print("Calculating gradients of a 2D polynomial over the integration domain")
        integrate_kwargs = {"fn": _polynomial_function, "dim": 2, "N": N_2d}
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient_over_domain(
            backend,
            [[0.0, 1.0], [0.0, 2.0]],
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        # Check if the integral and gradient are accurate enough
        assert np.abs(integral - 12.0) < 8e-2
        assert np.all(np.abs(gradient - np.array([[-10.0, 14.0], [-2.0, 14.0]])) < 0.1)

        print("Calculating gradients of a 2D polynomial over polynomial coefficients")
        param = [1.0, 2.0, 3.0]
        integrate_kwargs = {
            "integration_domain": [[0.0, 1.0], [0.0, 2.0]],
            "dim": 2,
            "N": N_2d,
            "backend": backend,
        }
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient_over_param(
            backend,
            param,
            _polynomial_function_parameterized,
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        # Check if the integral and gradient are accurate enough
        assert np.abs(integral - 12.0) < 8e-2
        assert np.all(np.abs(gradient - np.array([2.0, 1.0, 8.0 / 3.0])) < 5e-2)

        print("Calculating gradients of a V-shaped function over an offset")
        param = 2.0
        integrate_kwargs = {
            "integration_domain": [[-5.0, 3.0]],
            "dim": 1,
            "N": N_1d,
            "backend": backend,
        }
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient_over_param(
            backend,
            param,
            _v_function_parameterized,
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        # Check if the integral and gradient are accurate enough
        assert np.abs(integral - 34.0) < 0.2
        assert np.abs(gradient - 4.0) < 0.1


test_gradients_torch = setup_test_for_backend(_run_gradient_tests, "torch", "float64")
test_gradients_jax = setup_test_for_backend(_run_gradient_tests, "jax", "float64")
test_gradients_tensorflow = setup_test_for_backend(
    _run_gradient_tests, "tensorflow", "float64"
)

if __name__ == "__main__":
    # used to run this test individually
    test_gradients_torch()
    test_gradients_jax()
    test_gradients_tensorflow()

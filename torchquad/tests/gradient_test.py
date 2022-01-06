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

from integration_test_utils import setup_test_for_backend


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


def _calculate_gradient(
    backend, integration_domain, integrate, integrate_kwargs, dtype_name
):
    """
    Backend-specific gradient calculation

    Args:
        backend (string): Numerical backend, e.g. "torch"
        integration_domain (list): Integration domain
        integrate (function): A integrator's integrate method
        integrate_kwargs (dict): Arguments for integrate except integration_domain
        dtype_name (string): Floating point precision

    Returns:
        backend tensor: Gradient with respect to integration_domain
        backend tensor or None: Integral result if available
    """
    if backend == "torch":
        import torch

        # Set up integration_domain for gradient calculation
        integration_domain = torch.tensor(integration_domain)
        integration_domain.requires_grad = True

        # Compute integral
        result = integrate(integration_domain=integration_domain, **integrate_kwargs)
        result_np = to_numpy(result)

        # Check for presence of gradient and correct dtype
        assert hasattr(result, "grad_fn")
        assert get_dtype_name(result) == dtype_name

        # Backprop gradient through integral and get the gradient
        result.backward()
        gradient = integration_domain.grad
        assert get_dtype_name(gradient) == dtype_name

        return to_numpy(gradient), result_np

    elif backend == "jax":
        import jax

        integration_domain = anp.array(integration_domain, like="jax")

        # Create a derivation of integrate with respect to integration_domain
        @jax.grad
        def grad_integrate(dom):
            return integrate(integration_domain=dom, **integrate_kwargs)

        # Calculate the gradient
        gradient = grad_integrate(integration_domain)
        assert get_dtype_name(gradient) == dtype_name
        return to_numpy(gradient), None

    elif backend == "tensorflow":
        import tensorflow as tf

        # Set up integration_domain as Variable
        dtype = to_backend_dtype(dtype_name, like=backend)
        integration_domain = tf.Variable(integration_domain, dtype=dtype)

        # Calculate the integral and gradient
        with tf.GradientTape() as tape:
            result = integrate(
                integration_domain=integration_domain, **integrate_kwargs
            )
        assert get_dtype_name(result) == dtype_name
        gradient = tape.gradient(result, integration_domain)
        assert get_dtype_name(gradient) == dtype_name
        return to_numpy(gradient), to_numpy(result)

    else:
        raise ValueError(f"No gradient calculation for the backend {backend}")


def _run_gradient_tests(backend, precision):
    """
    Test if the implemented integrators
    maintain gradients and if the gradients are consistent and correct
    """
    dtype_name = {"float": "float32", "double": "float64"}[precision]
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
            f"Calculating gradients with backend {backend} and integrator {integrator}"
        )

        # Test gradient calculation with the one-dimensional V-shaped function
        integrate_kwargs = {"fn": _v_function, "dim": 1, "N": N_1d}
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient(
            backend,
            [[-1.0, 1.0]],
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        if integral is not None:
            # Check if the integral is accurate enough
            true_integral = 2.0
            assert np.abs(integral - true_integral) < 1e-2
        # Check if the gradient is accurate enough
        true_gradient = np.array([-2.0, 2.0])
        assert gradient.shape == (1, 2)
        assert np.all(np.abs(gradient - true_gradient) < 2e-2)

        # Test gradient calculation with a two-dimensional polynomial
        integrate_kwargs = {"fn": _polynomial_function, "dim": 2, "N": N_2d}
        if requires_seed:
            integrate_kwargs["seed"] = 0
        gradient, integral = _calculate_gradient(
            backend,
            [[0.0, 1.0], [0.0, 2.0]],
            integrator.integrate,
            integrate_kwargs,
            dtype_name,
        )
        if integral is not None:
            # Check if the integral is accurate enough
            true_integral = 12.0
            assert np.abs(integral - true_integral) < 4e-2
        # Check if the gradient is accurate enough
        true_gradient = np.array([[-10.0, 14.0], [-2.0, 14.0]])
        assert gradient.shape == (2, 2)
        assert np.all(np.abs(gradient - true_gradient) < 5e-2)


test_gradients_torch = setup_test_for_backend(_run_gradient_tests, "torch", "double")
test_gradients_jax = setup_test_for_backend(_run_gradient_tests, "jax", "double")
test_gradients_tensorflow = setup_test_for_backend(
    _run_gradient_tests, "tensorflow", "double"
)

if __name__ == "__main__":
    # used to run this test individually
    test_gradients_torch()
    test_gradients_jax()
    test_gradients_tensorflow()

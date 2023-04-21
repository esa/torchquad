import warnings
from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from .utils import _check_integration_domain


class BaseIntegrator:
    """The (abstract) integrator that all other integrators inherit from. Provides no explicit definitions for methods."""

    # Function to evaluate
    _fn = None

    # Dimensionality of function to evaluate
    _dim = None

    # Integration domain
    _integration_domain = None

    # Number of function evaluations
    _nr_of_fevals = None

    def __init__(self):
        self._nr_of_fevals = 0

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def _eval(self, points, weights=None, args=None):
        """Call evaluate_integrand to evaluate self._fn function at the passed points and update self._nr_of_evals

        Args:
            points (backend tensor): Integration points
            weights (backend tensor, optional): Integration weights. Defaults to None.
            args (list or tuple, optional): Any arguments required by the function. Defaults to None.
        """
        result, num_points = self.evaluate_integrand(
            self._fn, points, weights=weights, args=args
        )
        self._nr_of_fevals += num_points
        return result

    @staticmethod
    def evaluate_integrand(fn, points, weights=None, args=None):
        """Evaluate the integrand function at the passed points

        Args:
            fn (function): Integrand function
            points (backend tensor): Integration points
            weights (backend tensor, optional): Integration weights. Defaults to None.
            args (list or tuple, optional): Any arguments required by the function. Defaults to None.

        Returns:
            backend tensor: Integrand function output
            int: Number of evaluated points
        """
        num_points = points.shape[0]

        if args is None:
            args = ()

        result = fn(points, *args)

        if infer_backend(result) != infer_backend(points):
            warnings.warn(
                "The passed function's return value has a different numerical backend than the passed points. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = anp.array(result, like=points)

        num_results = result.shape[0]
        if num_results != num_points:
            raise ValueError(
                f"The passed function was given {num_points} points but only returned {num_results} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        if weights is not None:
            if (
                len(result.shape) > 1
            ):  # if the the integrand is multi-dimensional, we need to reshape/repeat weights so they can be broadcast in the *=
                integrand_shape = anp.array(
                    result.shape[1:], like=infer_backend(points)
                )
                weights = anp.repeat(
                    anp.expand_dims(weights, axis=1), anp.prod(integrand_shape)
                ).reshape((weights.shape[0], *(integrand_shape)))
            result *= weights

        return result, num_points

    @staticmethod
    def _check_inputs(dim=None, N=None, integration_domain=None):
        """Used to check input validity
        Args:
            dim (int, optional): Dimensionality of function to integrate. Defaults to None.
            N (int, optional): Total number of integration points. Defaults to None.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.
        Raises:
            ValueError: if inputs are not compatible with each other.
        """
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
            dim_domain = _check_integration_domain(integration_domain)
            if dim is not None and dim != dim_domain:
                raise ValueError(
                    "The dimension of the integration domain must match the passed function dimensionality dim."
                )

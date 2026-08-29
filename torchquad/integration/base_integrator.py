import warnings
from autoray import numpy as anp
from autoray import get_dtype_name, infer_backend
from loguru import logger

from .utils import _check_integration_domain

# Mantissa width of each supported dtype. A complex type carries the precision of
# its real component, so complex128 is as precise as float64 and pairing the two
# loses nothing. Anything absent is treated as unknown and left alone rather than
# guessed at.
_DTYPE_PRECISION_BITS = {
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
    "float64": 64,
    "complex64": 32,
    "complex128": 64,
}


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
        raise (NotImplementedError("This is an abstract base class. Should not be called."))

    def _eval(self, points, weights=None, args=None):
        """Call evaluate_integrand to evaluate self._fn function at the passed points and update self._nr_of_evals

        Args:
            points (backend tensor): Integration points
            weights (backend tensor, optional): Integration weights. Defaults to None.
            args (list or tuple, optional): Any arguments required by the function. Defaults to None.

        Returns:
            backend tensor: Integrand function output
        """
        result, num_points = self.evaluate_integrand(self._fn, points, weights=weights, args=args)
        self._nr_of_fevals += num_points
        return result

    @staticmethod
    def evaluate_integrand(fn, points, weights=None, args=None):
        """Evaluate the integrand function at the passed points

        The tensor ``fn`` returns is never modified: the quadrature weights are
        applied out of place. Integrands may therefore cache and reuse a buffer,
        and autograd graphs survive even when an integrand's final operation needs
        its own output in the backward pass. When weights are applied, the returned
        dtype is the promotion of the integrand's and the weights' dtypes, and a
        warning is issued if the integrand's is the less precise of the two.

        Args:
            fn (function): Integrand function
            points (backend tensor): Integration points
            weights (backend tensor, optional): Integration weights. Defaults to None.
            args (list or tuple, optional): Any arguments required by the function. Defaults to None.

        Returns:
            backend tensor: Integrand function output
            int: Number of evaluated points

        Raises:
            ValueError: If the integrand does not return one value per integration point,
                i.e. if it is not vectorized along the first dimension.

        Warns:
            UserWarning: If the integrand's return value uses a different numerical
                backend than the points, or a less precise dtype than the weights.
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
            ):  # if the the integrand is multi-dimensional, we need to reshape/repeat weights so they can be broadcast against the result
                integrand_shape = anp.array(
                    [dim if isinstance(dim, int) else dim.as_list() for dim in result.shape[1:]],
                    like=infer_backend(points),
                )

                weights = anp.repeat(
                    anp.expand_dims(weights, axis=1), anp.prod(integrand_shape)
                ).reshape((weights.shape[0], *(integrand_shape)))
            result_bits = _DTYPE_PRECISION_BITS.get(get_dtype_name(result))
            weight_bits = _DTYPE_PRECISION_BITS.get(get_dtype_name(weights))
            if result_bits is not None and weight_bits is not None and result_bits < weight_bits:
                warnings.warn(
                    f"The passed function returned {get_dtype_name(result)} values while the "
                    f"quadrature weights are {get_dtype_name(weights)}. The result is promoted to "
                    f"{get_dtype_name(weights)}, but the integrand was only evaluated at "
                    f"{get_dtype_name(result)} precision, so the extra digits are not meaningful. "
                    "Return the same precision the integrator was set up with."
                )
            # Deliberately out-of-place: `result` is the tensor the user's integrand
            # returned and torchquad does not own it. An in-place `*=` mutates it,
            # which breaks PyTorch autograd whenever the integrand's last operation
            # needs its own output in the backward pass (exp, sqrt, tanh, sigmoid,
            # div, pow) and corrupts any tensor the integrand caches and reuses.
            result = result * weights

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

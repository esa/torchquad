import warnings
import torch
from loguru import logger


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

    def _eval(self, points):
        """Evaluates the function at the passed points and updates nr_of_evals

        Args:
            points (torch.tensor): Integration points
        """
        self._nr_of_fevals += len(points)
        result = self._fn(points)
        if type(result) != torch.Tensor:
            warnings.warn(
                "The passed function did not return a torch.tensor. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = torch.tensor(result)

        if len(result) != len(points):
            raise ValueError(
                f"The passed function was given {len(points)} points but only returned {len(result)} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        return result

    def _check_inputs(self, dim=None, N=None, integration_domain=None):
        """Used to check input validity

        Args:
            dim (int, optional): Dimensionality of function to integrate. Defaults to None.
            N (int, optional): Total number of integration points. Defaults to None.
            integration_domain (list, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.

        Raises:
            ValueError: if inputs are not compatible with each other.
        """
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

            if integration_domain is not None:
                if dim != len(integration_domain):
                    raise ValueError(
                        "Dimension of integration_domain needs to match the passed function dimensionality dim."
                    )

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
            for bounds in integration_domain:
                if len(bounds) != 2:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )
                if bounds[0] > bounds[1]:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )

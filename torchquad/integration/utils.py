"""This file contains various utility functions for the integrations methods."""

from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger


def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs.
    Args:
        start (backend tensor): Start point (inclusive).
        stop (backend tensor): End point (inclusive).
        N (backend tensor): Number of points.
        requires_grad (bool): Indicates if output should be recorded for backpropagation.
    Returns:
        backend tensor: Equally spaced 1D grid
    """
    if requires_grad:
        # Create 0 to 1 spaced grid
        grid = anp.linspace(0, 1, N, like=start)

        # Scale to desired range, thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        return anp.linspace(start, stop, N, like=start)


def _setup_integration_domain(dim, integration_domain, backend):
    """Sets up the integration domain if unspecified by the user.
    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
        backend (string): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain.
    Returns:
        backend tensor: Integration domain.
    """

    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain.")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]."
            )
        if infer_backend(integration_domain) == "builtins":
            return anp.array(integration_domain, like=backend)
        return integration_domain
    else:
        return anp.array([[-1, 1]] * dim, like=backend)

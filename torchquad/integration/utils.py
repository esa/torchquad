"""This file contains various utility functions for the integrations methods."""

import torch
from loguru import logger


def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs

    Args:
        start (torch.tensor): Start point (inclusive)
        stop (torch.tensor): End point (inclusive)
        N (torch.tensor): Number of points
        requires_grad (bool): Indicates if output should be
            recorded for backpropagation

    Returns:
        torch.tensor: Equally spaced 1D grid
    """
    if requires_grad:
        # Create 0 to 1 spaced grid
        grid = torch.linspace(0, 1, N)

        # Scale to desired range , thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        return torch.linspace(start, stop, N)


def _setup_integration_domain(dim, integration_domain):
    """Sets up the integration domain if unspecified by the user.

    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

    Returns:
        torch.tensor: Integration domain.
    """

    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain.")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]."
            )
        if type(integration_domain) == torch.Tensor:
            return integration_domain
        else:
            return torch.tensor(integration_domain)
    else:
        return torch.tensor([[-1, 1]] * dim)

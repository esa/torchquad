"""This file contains various utility functions for the integrations methods."""

import torch
import logging

logger = logging.getLogger(__name__)


def setup_integration_domain(dim, integration_domain):
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

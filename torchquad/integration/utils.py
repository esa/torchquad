"""This file contains various utility functions for the integrations methods"""

import torch
import logging

logger = logging.getLogger(__name__)


def setup_integration_domain(dim, integration_domain):
    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]"
            )
        return torch.tensor(integration_domain)
    else:
        return torch.tensor([[-1, 1]] * dim)


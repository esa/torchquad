from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid

import torch

import logging

logger = logging.getLogger(__name__)


class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration in torch. 
    """

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration

        Args:
            fn (func): The function to integrate over
            dim (int): dimensionality of the function to integrate
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            float: Integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "Monte Carlo integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points  over"
            + str(integration_domain),
        )

        self._dim = dim
        self._nr_of_fevals = 0
        self.fn = fn

        # Store integration_domain
        # If not specified, create [-1,1]^d bounds
        logger.debug("Setting up integration domain")
        if integration_domain is not None:
            if len(integration_domain) != dim:
                raise ValueError(
                    "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]"
                )
            self._integration_domain = (
                integration_domain
                if type(integration_domain) == torch.Tensor
                else torch.tensor(integration_domain)
            )
        else:
            self._integration_domain = torch.tensor([[-1, 1]] * dim)

        logger.debug("Picking random sampling points")
        # Pick sample points from integration domain
        sample_points = torch.zeros([N, dim])
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points[:, d] = torch.rand(N) * scale + offset

        logger.debug("Evaluating integrand")
        # Evaluate the function at all points
        function_values = fn(sample_points)

        logger.debug("Computing integration domain volume")
        # Compute domain volume
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = torch.prod(scales)

        # Integral = V / N * sum(func values)
        integral = volume * torch.sum(function_values) / N
        logger.info("Computed integral was " + str(integral))
        return integral


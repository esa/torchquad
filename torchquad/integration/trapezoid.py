from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch

import logging

logger = logging.getLogger(__name__)


class Trapezoid(BaseIntegrator):
    """Trapezoidal rule in 1 to 5-D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def integrate(self, fn, dim, N=2, integration_domain=None):
        """Integrates the passed function on the passed domain using the trapezoid rule

        Args:
            fn (func): The function to integrate over
            dim (int): dimensionality of the function to integrate
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: Integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "Using Trapezoid for integrating a fn with "
            + str(N)
            + " points over"
            + str(integration_domain)
        )

        self._dim = dim
        self._fn = fn
        self._integration_domain = setup_integration_domain(dim, integration_domain)

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid")
        function_values = self._eval(self._grid.points)
        # Reshape the output to instead of [dim*N] points
        # be [N,N,...] points
        function_values = function_values.reshape([self._grid._N] * dim)

        logger.debug("Computing trapezoid areas")

        # This will contain the trapezoid areas per dimension
        cur_dim_areas = None

        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                self._grid.h[dim]
                / 2.0
                * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)

        logger.info("Computed integral was " + str(cur_dim_areas))

        return integral
from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import setup_integration_domain

import torch

import logging

logger = logging.getLogger(__name__)


class Simpson(BaseIntegrator):
    """Simpsons' rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def __init__(self):
        super().__init__()
        self._convergence_order = 2  # quadratic approx

    def integrate(self, fn, dim, N=3, integration_domain=None):
        """Integrates the passed function on the passed domain using the simpson method

        Args:
            fn (func): The function to integrate over
            N (int, optional): Number of sample points to use for the integration. Has to be odd. Defaults to 3.
            integration_domain (list, optional): Integration domain. Defaults to [-1,1]^dim.

        Returns:
            float: Integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        self._integration_domain = setup_integration_domain(dim, integration_domain)

        logger.debug(
            "Using Simpson for integrating a fn with "
            + str(N)
            + " points over "
            + str(integration_domain)
        )

        self._dim = dim
        self._fn = fn

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        # Simpson requires odd N for correctness. There is a more complex rule
        # that works for even N as well but it is not implemented here.
        if self._grid._N % 2 != 1:
            raise (
                ValueError(
                    f"N was {self._grid._N}. N cannot be even due to necessary subdivisions."
                )
            )

        logger.debug("Evaluating integrand on the grid")
        function_values = self._eval(self._grid.points)

        # Reshape the output to instead of [dim*N] points
        # be [N,N,...] points
        function_values = function_values.reshape([self._grid._N] * dim)

        logger.debug("Computing areas")

        # This will contain the simpsons areas per dimension
        cur_dim_areas = function_values

        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                self._grid.h[cur_dim]
                / 3.0
                * (
                    cur_dim_areas[..., 0:-2][..., ::2]
                    + 4 * cur_dim_areas[..., 1:-1][..., ::2]
                    + cur_dim_areas[..., 2:][..., ::2]
                )
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)

        logger.info("Computed integral was " + str(cur_dim_areas))

        return cur_dim_areas

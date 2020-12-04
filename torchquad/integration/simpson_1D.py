from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch

import logging

logger = logging.getLogger(__name__)


class Simpson1D(BaseIntegrator):
    """Simpsons' rule in 1D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def __init__(self):
        super().__init__()
        self._convergence_order = 2  # quadratic approx

    def integrate(self, fn, N=3, integration_domain=[[-1, 1]]):
        """Integrates the passed function on the passed domain using the simpson method

        Args:
            fn (func): The function to integrate over
            N (int, optional): Number of sample points to use for the integration. Has to be odd. Defaults to 3.
            integration_domain (list, optional): Integration domain. Defaults to [-1,1]^dim.

        Returns:
            float: Integral value
        """
        # Simpson requires odd N for correctness. There is a more complex rule
        # that works for even N as well but it is not implemented here.
        if N % 2 != 1:
            raise (ValueError("N cannot be even due to necessary subdivisions."))

        self._check_inputs(dim=1, N=N, integration_domain=integration_domain)

        logger.debug(
            "Using Simpson1D for integrating a fn with "
            + str(N)
            + " points over"
            + str(integration_domain)
        )

        self._dim = 1
        self._fn = fn
        self._integration_domain = integration_domain

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid")
        function_values = self._eval(self._grid.points)

        logger.debug("Computing areas")
        f0 = function_values[0:-2][::2]
        f1 = function_values[1:-1][::2]
        f2 = function_values[2:][::2]
        areas = f0 + 4 * f1 + f2

        integral = self._grid.h / 3 * torch.sum(areas)
        logger.info("Computed integral was " + str(integral))
        return integral

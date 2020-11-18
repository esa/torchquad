from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch

import logging

logger = logging.getLogger(__name__)


class Trapezoid1D(BaseIntegrator):
    """Trapezoidal rule in 1D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def integrate(self, fn, N=2, integration_domain=[[-1, 1]]):
        """Integrates the passed function on the passed domain using the trapezoid method

        Args:
            fn (func): The function to integrate over
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain. Defaults to [-1,1]^dim.

        Returns:
            float: Integral value
        """
        self._check_inputs(dim=1, N=N, integration_domain=integration_domain)
        logger.debug(
            "Using Trapezoid1D for integrating a fn with "
            + str(N)
            + " points  over"
            + str(integration_domain)
        )

        self._dim = 1
        self._fn = fn
        self._integration_domain = integration_domain

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid")
        function_values = self._eval(self._grid._points)

        logger.debug("Computing trapezoid areas")
        f0 = function_values[0:-1]
        f1 = function_values[1:]
        areas = f0 + f1

        integral = self._grid._h / 2 * torch.sum(areas)
        logger.info("Computed integral was " + str(integral))
        return integral

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch


class Simpson1D(BaseIntegrator):
    """Simpsons' rule in 1D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def integrate(self, fn, N=3, integration_domain=[[-1, 1]], verbose=False):
        if N % 2 != 1:
            raise (ValueError("N cannot be even due to necessary subdivisions."))

        self._dim = 1
        self._fn = fn
        self._integration_domain = integration_domain

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        # Evaluate the function on the grid
        function_values = self._eval(self._grid._points)

        # Compute areas
        f0 = function_values[0:-2][::2]
        f1 = function_values[1:-1][::2]
        f2 = function_values[2:][::2]

        areas = f0 + 4 * f1 + f2

        return self._grid._h / 3 * torch.sum(areas)

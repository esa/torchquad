from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch


class Trapezoid1D(BaseIntegrator):
    """Trapezoidal rule in 1D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas . 
    """

    def integrate(self, fn, N=2, integration_domain=[[-1, 1]], verbose=False):
        self._dim = 1
        self._fn = fn
        self._integration_domain = integration_domain

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, integration_domain)

        # Evaluate the function on the grid
        function_values = self._eval(self._grid._points)

        # Compute trapezoid areas
        f0 = function_values[0:-1]
        f1 = function_values[1:]

        areas = f0 + f1

        return self._grid._h / 2 * torch.sum(areas)

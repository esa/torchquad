from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid


class Trapezoid_1D(BaseIntegrator):
    """Trapezoidal rule in 1D in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas .
    """

    def integrate(self, fn, N=2, integration_domain=[[-1, 1]], verbose=False):
        self.dim = 1
        self.nr_of_fevals = 0
        self.fn = fn
        self.integration_domain = integration_domain

        # Create grid and assemble evaluation points
        self.grid = IntegrationGrid(N)

        # Evaluate at all points
        function_values = fn(self.grid)
        f0 = function_values[0:-1]
        f1 = function_values[1:]
        areas = f0 + f1
        return self.h / 2 * torch.sum(areas)

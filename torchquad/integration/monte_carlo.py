from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
import torch


class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration in torch. 
    """

    def integrate(self, fn, dim, N=1000, integration_domain=None):
        self._dim = dim
        self._nr_of_fevals = 0
        self.fn = fn

        # Store integration_domain
        # If not specified, create [-1,1]^d bounds
        if integration_domain is not None:
            if len(integration_domain) != dim:
                raise ValueError(
                    "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]"
                )
            self._integration_domain = torch.tensor(integration_domain)
        else:
            self._integration_domain = torch.tensor([[-1, 1]] * dim)

        # Pick sample points from integration domain
        sample_points = torch.zeros([N, dim])
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points[:, d] = torch.rand(N) * scale + offset

        # Evaluate the function at all points
        function_values = fn(sample_points)

        # Compute domain volume
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = torch.prod(scales)

        # Integral = V / N * sum(func values)
        return volume * torch.sum(function_values) / N


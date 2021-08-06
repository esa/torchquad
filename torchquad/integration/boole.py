import torch
import warnings
from loguru import logger

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import _setup_integration_domain


class Boole(BaseIntegrator):

    """Boole's rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=None, integration_domain=None):
        """Integrates the passed function on the passed domain using Boole's rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. N has to be such that N^(1/dim) - 1 % 4 == 0. Defaults to 5 points per dimension if None is given.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
        """

        # If N is unspecified, set N to 5 points per dimension
        if N is None:
            N = 5 ** dim

        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)
        N = self._adjust_N(dim=dim, N=N)

        self._dim = dim
        self._fn = fn

        logger.debug(
            "Using Boole for integrating a fn with a total of "
            + str(N)
            + " points over "
            + str(self._integration_domain)
            + "."
        )

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, self._integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        function_values = self._eval(self._grid.points)

        # Reshape the output to be [N,N,...] points instead of [dim*N] points
        function_values = function_values.reshape([self._grid._N] * dim)

        logger.debug("Computing areas.")

        # This will contain the Simpson's areas per dimension
        cur_dim_areas = function_values

        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                self._grid.h[cur_dim]
                / 22.5
                * (
                    7 * cur_dim_areas[..., 0:-4][..., ::4]
                    + 32 * cur_dim_areas[..., 1:-3][..., ::4]
                    + 12 * cur_dim_areas[..., 2:-2][..., ::4]
                    + 32 * cur_dim_areas[..., 3:-1][..., ::4]
                    + 7 * cur_dim_areas[..., 4:][..., ::4]
                )
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)
        logger.info("Computed integral was " + str(cur_dim_areas) + ".")

        return cur_dim_areas

    def _adjust_N(self, dim, N):
        """Adjusts the total number of points to a valid number, i.e. N satisfies N^(1/dim) - 1 % 4 == 0.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int): Total number of sample points to use for the integration.

        Returns:
            int: An N satisfying N^(1/dim) - 1 % 4 == 0.
        """
        n_per_dim = int(N ** (1.0 / dim) + 1e-8)
        logger.debug(
            "Checking if N per dim is >=5 and N = 1 + 4n, where n is a positive integer."
        )

        # Boole's rule requires N per dim >=5 and N = 1 + 4n,
        # where n is a positive integer, for correctness.
        if n_per_dim < 5:
            warnings.warn(
                "N per dimension cannot be lower than 5. "
                "N per dim will now be changed to 5."
            )
            N = 5 ** dim
        elif (n_per_dim - 1) % 4 != 0:
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            warnings.warn(
                "N per dimension must be N = 1 + 4n with n a positive integer due to necessary subdivisions. "
                "N per dim will now be changed to the next lower N satisfying this, i.e. "
                f"{n_per_dim} -> {new_n_per_dim}."
            )
            N = (new_n_per_dim) ** (dim)
        return N

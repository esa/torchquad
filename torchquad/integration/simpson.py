import torch
from loguru import logger
import warnings

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import _setup_integration_domain


class Simpson(BaseIntegrator):

    """Simpson's rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=None, integration_domain=None):
        """Integrates the passed function on the passed domain using Simpson's rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. Should be odd. Defaults to 3 points per dimension if None is given.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
        """

        # If N is unspecified, set N to 3 points per dimension
        if N is None:
            N = 3 ** dim

        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)
        N = self._adjust_N(dim=dim, N=N)

        self._dim = dim
        self._fn = fn

        logger.debug(
            "Using Simpson for integrating a fn with a total of "
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
                / 3.0
                * (
                    cur_dim_areas[..., 0:-2][..., ::2]
                    + 4 * cur_dim_areas[..., 1:-1][..., ::2]
                    + cur_dim_areas[..., 2:][..., ::2]
                )
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)
        logger.info("Computed integral was " + str(cur_dim_areas) + ".")

        return cur_dim_areas

    def _adjust_N(self, dim, N):
        """Adjusts the current N to an odd integer >=3, if N is not that already.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int): Total number of sample points to use for the integration.

        Returns:
            int: An odd N >3.
        """
        n_per_dim = int(N ** (1.0 / dim) + 1e-8)
        logger.debug("Checking if N per dim is >=3 and odd.")

        # Simpson's rule requires odd N per dim >3 for correctness. There is a more
        # complex rule that works for even N as well but it is not implemented here.
        if n_per_dim < 3:
            warnings.warn(
                "N per dimension cannot be lower than 3. "
                "N per dim will now be changed to 3."
            )
            N = 3 ** dim
        elif n_per_dim % 2 != 1:
            warnings.warn(
                "N per dimension cannot be even due to necessary subdivisions. "
                "N per dim will now be changed to the next lower integer, i.e. "
                f"{n_per_dim} -> {n_per_dim - 1}."
            )
            N = (n_per_dim - 1) ** (dim)
        return N

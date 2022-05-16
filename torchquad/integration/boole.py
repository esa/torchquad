from autoray import numpy as anp
import warnings
from loguru import logger

from .newton_cotes import NewtonCotes


class Boole(NewtonCotes):

    """Boole's rule. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=None, integration_domain=None, backend=None):
        """Integrates the passed function on the passed domain using Boole's rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. N has to be such that N^(1/dim) - 1 % 4 == 0. Defaults to 5 points per dimension if None is given.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            backend-specific number: Integral value
        """
        return super().integrate(fn, dim, N, integration_domain, backend)

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs):
        """Apply composite Boole quadrature.
        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                hs[cur_dim]
                / 22.5
                * (
                    7 * cur_dim_areas[..., 0:-4][..., ::4]
                    + 32 * cur_dim_areas[..., 1:-3][..., ::4]
                    + 12 * cur_dim_areas[..., 2:-2][..., ::4]
                    + 32 * cur_dim_areas[..., 3:-1][..., ::4]
                    + 7 * cur_dim_areas[..., 4:][..., ::4]
                )
            )
            cur_dim_areas = anp.sum(cur_dim_areas, axis=dim - cur_dim - 1)
        return cur_dim_areas

    @staticmethod
    def _get_minimal_N(dim):
        """Get the minimal number of points N for the integrator rule"""
        return 5**dim

    @staticmethod
    def _adjust_N(dim, N):
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
            N = 5**dim
        elif (n_per_dim - 1) % 4 != 0:
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            warnings.warn(
                "N per dimension must be N = 1 + 4n with n a positive integer due to necessary subdivisions. "
                "N per dim will now be changed to the next lower N satisfying this, i.e. "
                f"{n_per_dim} -> {new_n_per_dim}."
            )
            N = (new_n_per_dim) ** (dim)
        return N

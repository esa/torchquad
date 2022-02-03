from loguru import logger
import torch

from .adaptive_newton_cotes import AdaptiveNewtonCotes


# TODO Once we merge from https://github.com/FHof/torchquad replace redundant methods
class AdaptiveBoole(AdaptiveNewtonCotes):
    """Adaptive Boole rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas .
    This method adaptively redefines high-variance regions of the integrand"""

    def __init__(self):
        super().__init__()

    def integrate(
        self,
        fn,
        dim,
        N=1000,
        subdomains_per_dim=2,
        max_refinement_level=4,
        integration_domain=None,
        complex_function=False,
        reuse_old_fvals=True,
    ):
        """Integrates the passed function on the passed domain using a Boole rule on an adaptively refined grid.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            complex_function(bool, optional): Describes if the integrand is complex. Defaults to False.
            reuse_old_fvals (bool): If True, will reuse already computed function values in refinement. Saves compute but costs memory writes.

        Returns:
            float: integral value
        """
        return super().integrate(
            fn,
            dim,
            N,
            subdomains_per_dim,
            max_refinement_level,
            integration_domain,
            complex_function,
            reuse_old_fvals,
        )

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
            cur_dim_areas = torch.sum(cur_dim_areas, axis=dim - cur_dim - 1)
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
            logger.warning(
                "N per dimension cannot be lower than 5. "
                "N per dim will now be changed to 5."
            )
            N = 5**dim
        elif (n_per_dim - 1) % 4 != 0:
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            logger.warning(
                "N per dimension must be N = 1 + 4n with n a positive integer due to necessary subdivisions. "
                "N per dim will now be changed to the next lower N satisfying this, i.e. "
                f"{n_per_dim} -> {new_n_per_dim}."
            )
            N = (new_n_per_dim) ** (dim)
        return N

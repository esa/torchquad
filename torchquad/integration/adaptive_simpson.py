from loguru import logger
import torch

from .adaptive_newton_cotes import AdaptiveNewtonCotes


# TODO Once we merge from https://github.com/FHof/torchquad replace redundant methods
class AdaptiveSimpson(AdaptiveNewtonCotes):
    """Adaptive Simpson rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas .
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
        """Integrates the passed function on the passed domain using a Simpson rule on an adaptively refined grid.

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
        """Apply composite Simpson quadrature.
        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                hs[cur_dim]
                / 3.0
                * (
                    cur_dim_areas[..., 0:-2][..., ::2]
                    + 4 * cur_dim_areas[..., 1:-1][..., ::2]
                    + cur_dim_areas[..., 2:][..., ::2]
                )
            )
            cur_dim_areas = torch.sum(cur_dim_areas, axis=dim - cur_dim - 1)
        return cur_dim_areas

    @staticmethod
    def _get_minimal_N(dim):
        """Get the minimal number of points N for the integrator rule"""
        return 3**dim

    @staticmethod
    def _adjust_N(dim, N):
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
            logger.warning(
                "N per dimension cannot be lower than 3. "
                "N per dim will now be changed to 3."
            )
            N = 3**dim
        elif n_per_dim % 2 != 1:
            logger.warning(
                "N per dimension cannot be even due to necessary subdivisions. "
                "N per dim will now be changed to the next lower integer, i.e. "
                f"{n_per_dim} -> {n_per_dim - 1}."
            )
            N = (n_per_dim - 1) ** (dim)
        return N

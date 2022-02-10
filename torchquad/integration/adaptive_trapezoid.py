import torch

from .adaptive_newton_cotes import AdaptiveNewtonCotes


class AdaptiveTrapezoid(AdaptiveNewtonCotes):
    """Adaptive Trapezoidal rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas .
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
        """Integrates the passed function on the passed domain using a trapezoid rule on an adaptively refined grid.

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

    # TODO replace this once we merge from https://github.com/FHof/torchquad
    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs):
        """Apply composite Trapezoid quadrature.
        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                hs[cur_dim] / 2.0 * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
            )
            cur_dim_areas = torch.sum(cur_dim_areas, axis=dim - cur_dim - 1)
        return cur_dim_areas

    @staticmethod
    def _adjust_N(dim, N):
        # Nothing to do for Trapezoid
        return N

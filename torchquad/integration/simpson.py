import warnings

from autoray import numpy as anp
from loguru import logger

from .newton_cotes import NewtonCotes


class Simpson(NewtonCotes):
    """Simpson's rule. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=None, integration_domain=None, backend=None, args=None):
        """Integrates the passed function on the passed domain using Simpson's rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int or sequence of int, optional): Total number of sample points, or the number of points in each dimension. Should be odd. Defaults to 3 points per dimension if None is given.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It can also determine the numerical backend.
            backend (string, optional): Numerical backend. Defaults to integration_domain's backend if it is a tensor and otherwise to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.
            args (list or tuple, optional): Extra arguments passed to the integrand as ``fn(points, *args)``. Defaults to None.

        Returns:
            backend-specific number: Integral value
        """
        return super().integrate(fn, dim, N, integration_domain, backend, args=args)

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
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
            cur_dim_areas = anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
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
            int or tuple of int: An odd N >3.
        """
        logger.debug("Checking if N per dim is >=3 and odd.")

        def adjust_count(count):
            if count < 3:
                warnings.warn(
                    "N per dimension cannot be lower than 3. N per dim will now be changed to 3."
                )
                return 3
            if count % 2 != 1:
                warnings.warn(
                    "N per dimension cannot be even due to necessary subdivisions. "
                    "N per dim will now be changed to the next lower integer, i.e. "
                    f"{count} -> {count - 1}."
                )
                return count - 1
            return count

        if isinstance(N, tuple):
            return tuple(adjust_count(count) for count in N)

        n_per_dim = int(N ** (1.0 / dim) + 1e-8)
        adjusted_count = adjust_count(n_per_dim)
        return adjusted_count**dim

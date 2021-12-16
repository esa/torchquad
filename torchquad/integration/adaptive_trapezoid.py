import torch
from loguru import logger

from .base_integrator import BaseIntegrator
from .adaptive_grid import AdaptiveGrid
from .utils import _setup_integration_domain


class AdaptiveTrapezoid(BaseIntegrator):
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
    ):
        """Integrates the passed function on the passed domain using a trapezoid rule on an adaptively refined grid.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
        """
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)

        logger.debug(
            "Using AdaptiveTrapezoid for integrating a fn with "
            + str(N)
            + " points over "
            + str(self._integration_domain)
            + "."
        )

        self._dim = dim
        self._fn = fn

        # TODO determine a smarter initial N
        initial_N = N // 20

        # Initialize the adaptive grid
        self._grid = AdaptiveGrid(
            initial_N,
            self._integration_domain,
            subdomains_per_dim,
            max_refinement_level,
        )

        hit_maximum_evals = False
        while not hit_maximum_evals:
            eval_points, chunksizes = self._grid.get_next_eval_points()

            logger.debug("Evaluating integrand on the grid.")
            function_values = self._eval(eval_points)

            self._grid.set_fvals(function_values, chunksizes)

            logger.debug("Computing trapezoid areas for subdomains.")
            # Compute integral for each subdomain
            for subdomain in self._grid.subdomains:

                # Skip up-to-date subdomains
                if not subdomain.requires_integral_value:
                    logger.trace("Skipping up-to-date subdomain.")
                    continue

                function_values = subdomain.fval

                # Reshape the output to be [N,N,...] points
                # instead of [dim*N] points
                function_values = function_values.reshape([subdomain.N_per_dim] * dim)

                # This will contain the trapezoid areas per dimension
                cur_dim_areas = function_values

                # We collapse dimension by dimension
                for cur_dim in range(dim):
                    cur_dim_areas = (
                        subdomain.h[cur_dim]
                        / 2.0
                        * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
                    )
                    cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)

                logger.debug(
                    "Computed subdomain integral was " + str(cur_dim_areas) + "."
                )
                subdomain.set_integral(cur_dim_areas)

            hit_maximum_evals = self._nr_of_fevals < N

            if not hit_maximum_evals:
                # Refine the grid
                self._grid.refine()

        return self._grid.get_integral()
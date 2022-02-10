import torch
from loguru import logger

from .base_integrator import BaseIntegrator
from .adaptive_grid import AdaptiveGrid
from .utils import _setup_integration_domain


class AdaptiveNewtonCotes(BaseIntegrator):
    """Abstract integrator that adaptive Newton Cotes integrators inherit from"""

    _adjust_N = None

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

        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)

        logger.debug(
            "Using AdaptiveNewton Cotes for integrating a fn with "
            + str(N)
            + " points over "
            + str(self._integration_domain)
            + "."
        )

        self._dim = dim
        self._fn = fn

        # Assuming we want to end up with N points, and one subdomain may be refined up to max level, that subdomain will have
        # (initialN * 2**(max_refinement_level-1))**dim points. So we should start of with N / (2**(max_refinement_level-1))**dim
        initial_N = int((N // 2 ** max_refinement_level) ** (1 / dim))

        logger.debug("initial_N based on refinement and dim is " + str(initial_N))

        # However we need a minimum number of points so each subdomain has at least enough
        # points to compute the respective integration rule.
        if initial_N < self._get_minimal_N(dim) * (subdomains_per_dim ** dim):
            initial_N = self._get_minimal_N(dim) * (subdomains_per_dim ** dim)
            minimum_number_of_total_points = (
                2 ** max_refinement_level
            ) ** dim + self._get_minimal_N(dim) * ((subdomains_per_dim ** dim) - 1)
            logger.warning(
                "The chosen N is too small for the desired refinement / subdomains (too few points).  Requires at least N="
                + str(minimum_number_of_total_points)
                + "."
            )
            logger.debug(
                "After accounting for min points per subdomain became initial_N="
                + str(initial_N)
            )

        # Initialize the adaptive grid
        self._grid = AdaptiveGrid(
            N=initial_N,
            integration_domain=self._integration_domain,
            subdomains_per_dim=subdomains_per_dim,
            max_refinement_level=max_refinement_level,
            complex_function=complex_function,
            reuse_old_fvals=reuse_old_fvals,
            N_adjustment_function=lambda x: self._adjust_N(dim=dim, N=x),
        )

        hit_maximum_evals = False
        while not hit_maximum_evals:
            eval_points, chunksizes = self._grid.get_next_eval_points()

            logger.debug("Evaluating integrand on the grid.")
            logger.trace(f"Points are {eval_points}")
            function_values = self._eval(eval_points)

            self._grid.set_fvals(function_values, chunksizes)

            logger.debug("Computing areas for subdomains.")
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

                result = self._apply_composite_rule(cur_dim_areas, dim, subdomain.h)

                logger.debug("Computed subdomain integral was " + str(result) + ".")
                subdomain.set_integral(result)

            hit_maximum_evals = self._nr_of_fevals >= N

            if not hit_maximum_evals:
                # Refine the grid
                self._grid.refine()
            else:
                logger.debug(
                    "Hit "
                    + str(self._nr_of_fevals)
                    + " evaluations. Exiting refinement loop."
                )

        return self._grid.get_integral()

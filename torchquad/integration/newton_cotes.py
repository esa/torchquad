from loguru import logger

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import _setup_integration_domain


class NewtonCotes(BaseIntegrator):
    """The abstract integrator that Composite Newton Cotes integrators inherit from"""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N, integration_domain, backend):
        """Integrate the passed function on the passed domain using a Composite Newton Cotes rule.
        The argument meanings are explained in the sub-classes.

        Returns:
            float: integral value
        """
        # If N is None, use the minimal required number of points per dimension
        if N is None:
            N = self._get_minimal_N(dim)

        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)

        grid_points, hs, n_per_dim = self.calculate_grid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        function_values, num_points = self.evaluate_integrand(fn, grid_points)
        self._nr_of_fevals = num_points

        return self.calculate_result(function_values, dim, n_per_dim, hs)

    def calculate_grid(self, N, integration_domain):
        """Calculate grid points, widths and N per dim

        Args:
            N (int): Number of points
            integration_domain (backend tensor): Integration domain

        Returns:
            backend tensor: Grid points
            backend tensor: Grid widths
            int: Number of grid slices per dimension
        """
        N = self._adjust_N(dim=integration_domain.shape[0], N=N)

        # Log with lazy to avoid redundant synchronisations with certain
        # backends
        logger.opt(lazy=True).debug(
            "Creating a grid for {name} to integrate a function with {N} points over {d}.",
            name=lambda: type(self).__name__,
            N=lambda: str(N),
            d=lambda: str(integration_domain),
        )

        # Create grid and assemble evaluation points
        grid = IntegrationGrid(N, integration_domain)

        return grid.points, grid.h, grid._N

    def calculate_result(self, function_values, dim, n_per_dim, hs):
        """Apply the Composite Newton Cotes rule to calculate a result from the evaluated integrand.

        Args:
            function_values (backend tensor): Output of the integrand
            dim (int): Dimensionality
            n_per_dim (int): Number of grid slices per dimension
            hs (backend tensor): Distances between grid slices for each dimension

        Returns:
            backend tensor: Quadrature result
        """
        # Reshape the output to be [N,N,...] points instead of [dim*N] points
        function_values = function_values.reshape([n_per_dim] * dim)

        logger.debug("Computing areas.")

        result = self._apply_composite_rule(function_values, dim, hs)

        logger.opt(lazy=True).info(
            "Computed integral: {result}", result=lambda: str(result)
        )
        return result

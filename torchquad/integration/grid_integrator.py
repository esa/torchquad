from loguru import logger
from autoray import numpy as anp

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import _linspace_with_grads, expand_func_values_and_squeeze_intergal, _setup_integration_domain

class GridIntegrator(BaseIntegrator):
    """The abstract integrator that grid-like integrators (Newton-Cotes and Gaussian) integrators inherit from"""

    def __init__(self):
        super().__init__()

    @property
    def _grid_func(self):
        def f(a, b, N, requires_grad=False, backend=None):
            return _linspace_with_grads(a, b, N, requires_grad=requires_grad)
        return f
    

    def _weights(self, N, dim, backend):
        return None

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
        function_values, num_points = self.evaluate_integrand(fn, grid_points, weights=self._weights(n_per_dim, dim, backend))
        self._nr_of_fevals = num_points

        return self.calculate_result(function_values, dim, n_per_dim, hs, integration_domain)

    @expand_func_values_and_squeeze_intergal
    def calculate_result(self, function_values, dim, n_per_dim, hs, integration_domain):
        """Apply the "composite rule" to calculate a result from the evaluated integrand.

        Args:
            function_values (backend tensor): Output of the integrand
            dim (int): Dimensionality
            n_per_dim (int): Number of grid slices per dimension
            hs (backend tensor): Distances between grid slices for each dimension

        Returns:
            backend tensor: Quadrature result
        """
        # Reshape the output to be [integrand_dim,N,N,...] points instead of [integrand_dim,dim*N] points
        integrand_shape = function_values.shape[1:]
        dim_shape = [n_per_dim] * dim
        new_shape = [*integrand_shape, *dim_shape]
        # We need to use einsum instead of just reshape here because reshape does not move the axis - it only reshapes.
        # So the first line generates a character string for einsum, followed by moving the first dimension i.e `dim*N`
        # to the end.  Finally we reshape the resulting object so that instead of the last dimension being `dim*N`, it is
        # `N,N,...` as desired.
        einsum = "".join([chr(i + 65) for i in range(len(function_values.shape))])
        reshaped_function_values = anp.einsum(f'{einsum}->{einsum[1:]}{einsum[0]}', function_values)
        reshaped_function_values = reshaped_function_values.reshape(new_shape)
        assert new_shape == list(reshaped_function_values.shape), f"reshaping produced shape {reshaped_function_values.shape}, expected shape was {new_shape}"
        logger.debug("Computing areas.")

        result = self._apply_composite_rule(reshaped_function_values, dim, hs, integration_domain)

        logger.opt(lazy=True).info(
            "Computed integral: {result}", result=lambda: str(result)
        )
        return result

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
        grid = IntegrationGrid(N, integration_domain, self._grid_func)

        return grid.points, grid.h, grid._N

    @staticmethod
    def _adjust_N(dim, N):
        # Nothing to do by default
        return N
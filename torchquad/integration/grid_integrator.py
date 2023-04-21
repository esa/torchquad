from loguru import logger
from autoray import numpy as anp, infer_backend

from .base_integrator import BaseIntegrator
from .integration_grid import IntegrationGrid
from .utils import (
    _linspace_with_grads,
    expand_func_values_and_squeeze_integral,
    _setup_integration_domain,
)


class GridIntegrator(BaseIntegrator):
    """The abstract integrator that grid-like integrators (Newton-Cotes and Gaussian) integrators inherit from"""

    def __init__(self):
        super().__init__()

    @property
    def _grid_func(self):
        def f(integration_domain, N, requires_grad=False, backend=None):
            a = integration_domain[0]
            b = integration_domain[1]
            return _linspace_with_grads(a, b, N, requires_grad=requires_grad)

        return f

    def _weights(self, N, dim, backend, requires_grad=False):
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
        backend = infer_backend(integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)

        grid_points, hs, n_per_dim = self.calculate_grid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        function_values, num_points = self.evaluate_integrand(
            fn, grid_points, weights=self._weights(n_per_dim, dim, backend)
        )
        self._nr_of_fevals = num_points

        return self.calculate_result(
            function_values, dim, n_per_dim, hs, integration_domain
        )

    @expand_func_values_and_squeeze_integral
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
        einsum = "".join(
            [chr(i + 65) for i in range(len(function_values.shape))]
        )  # chr(i + 65) generates an alphabetical character
        reshaped_function_values = anp.einsum(
            f"{einsum}->{einsum[1:]}{einsum[0]}", function_values
        )
        reshaped_function_values = reshaped_function_values.reshape(new_shape)
        assert new_shape == list(
            reshaped_function_values.shape
        ), f"reshaping produced shape {reshaped_function_values.shape}, expected shape was {new_shape}"
        logger.debug("Computing areas.")

        result = self._apply_composite_rule(
            reshaped_function_values, dim, hs, integration_domain
        )

        logger.opt(lazy=True).info(
            "Computed integral: {result}", result=lambda: str(result)
        )
        return result

    def calculate_grid(
        self,
        N,
        integration_domain,
        disable_integration_domain_check=False,
    ):
        """Calculate grid points, widths and N per dim

        Args:
            N (int): Number of points
            integration_domain (backend tensor): Integration domain
            disable_integration_domain_check (bool): Disbaling integration domain checks (default False)

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
        grid = IntegrationGrid(
            N, integration_domain, self._grid_func, disable_integration_domain_check
        )

        return grid.points, grid.h, grid._N

    @staticmethod
    def _adjust_N(dim, N):
        # Nothing to do by default
        return N

    def get_jit_compiled_integrate(
        self, dim, N=None, integration_domain=None, backend=None
    ):
        """Create an integrate function where the performance-relevant steps except the integrand evaluation are JIT compiled.
        Use this method only if the integrand cannot be compiled.
        The compilation happens when the function is executed the first time.
        With PyTorch, return values of different integrands passed to the compiled function must all have the same format, e.g. precision.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. See the integrate method documentation for more details.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It can also determine the numerical backend.
            backend (string, optional): Numerical backend. Defaults to integration_domain's backend if it is a tensor and otherwise to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            function(fn, integration_domain): JIT compiled integrate function where all parameters except the integrand and domain are fixed
        """
        # If N is None, use the minimal required number of points per dimension
        if N is None:
            N = self._get_minimal_N(dim)

        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        backend = infer_backend(integration_domain)
        if backend in ["tensorflow", "jax"]:
            # Tensorflow and JAX automatically recompile functions if
            # the parameters change
            if backend == "tensorflow":
                if not hasattr(self, "_tf_jit_calculate_grid"):
                    import tensorflow as tf

                    self._tf_jit_calculate_grid = tf.function(
                        self.calculate_grid, jit_compile=True
                    )
                    self._tf_jit_calculate_result = tf.function(
                        self.calculate_result, jit_compile=True
                    )
                jit_calculate_grid = self._tf_jit_calculate_grid
                jit_calculate_result = self._tf_jit_calculate_result
            elif backend == "jax":
                if not hasattr(self, "_jax_jit_calculate_grid"):
                    import jax

                    self._jax_jit_calculate_grid = jax.jit(
                        self.calculate_grid, static_argnames=["N"]
                    )
                    self._jax_jit_calculate_result = jax.jit(
                        self.calculate_result,
                        static_argnums=(
                            1,
                            2,
                        ),  # dim and n_per_dim
                    )
                jit_calculate_grid = self._jax_jit_calculate_grid
                jit_calculate_result = self._jax_jit_calculate_result

            def compiled_integrate(fn, integration_domain):
                grid_points, hs, n_per_dim = jit_calculate_grid(N, integration_domain)
                function_values, _ = self.evaluate_integrand(
                    fn, grid_points, weights=self._weights(n_per_dim, dim, backend)
                )
                return jit_calculate_result(
                    function_values, dim, int(n_per_dim), hs, integration_domain
                )

            return compiled_integrate

        elif backend == "torch":
            # Torch requires explicit tracing with example inputs.
            def do_compile(example_integrand):
                import torch

                # Define traceable first and third steps
                def step1(integration_domain):
                    grid_points, hs, n_per_dim = self.calculate_grid(
                        N, integration_domain
                    )
                    return (
                        grid_points,
                        hs,
                        torch.Tensor([n_per_dim]),
                    )  # n_per_dim is constant

                dim = int(integration_domain.shape[0])

                def step3(function_values, hs, integration_domain):
                    return self.calculate_result(
                        function_values, dim, n_per_dim, hs, integration_domain
                    )

                # Trace the first step
                step1 = torch.jit.trace(step1, (integration_domain,))

                # Get example input for the third step
                grid_points, hs, n_per_dim = step1(integration_domain)
                n_per_dim = int(n_per_dim)
                function_values, _ = self.evaluate_integrand(
                    example_integrand,
                    grid_points,
                    weights=self._weights(n_per_dim, dim, backend),
                )

                # Trace the third step
                # Avoid the warnings about a .grad attribute access of a
                # non-leaf Tensor
                if hs.requires_grad:
                    hs = hs.detach()
                    hs.requires_grad = True
                if function_values.requires_grad:
                    function_values = function_values.detach()
                    function_values.requires_grad = True
                step3 = torch.jit.trace(
                    step3, (function_values, hs, integration_domain)
                )

                # Define a compiled integrate function
                def compiled_integrate(fn, integration_domain):
                    grid_points, hs, _ = step1(integration_domain)
                    function_values, _ = self.evaluate_integrand(
                        fn, grid_points, weights=self._weights(n_per_dim, dim, backend)
                    )
                    result = step3(function_values, hs, integration_domain)
                    return result

                return compiled_integrate

            # Do the compilation when the returned function is executed the
            # first time
            compiled_func = [None]

            def lazy_compiled_integrate(fn, integration_domain):
                if compiled_func[0] is None:
                    compiled_func[0] = do_compile(fn)
                return compiled_func[0](fn, integration_domain)

            return lazy_compiled_integrate

        raise ValueError(f"Compilation not implemented for backend {backend}")

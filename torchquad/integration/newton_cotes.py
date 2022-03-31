from loguru import logger
from autoray import infer_backend

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
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

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
                        self.calculate_result, static_argnames=["dim", "n_per_dim"]
                    )
                jit_calculate_grid = self._jax_jit_calculate_grid
                jit_calculate_result = self._jax_jit_calculate_result

            def compiled_integrate(fn, integration_domain):
                grid_points, hs, n_per_dim = jit_calculate_grid(N, integration_domain)
                function_values, _ = self.evaluate_integrand(fn, grid_points)
                return jit_calculate_result(function_values, dim, int(n_per_dim), hs)

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

                def step3(function_values, hs):
                    return self.calculate_result(function_values, dim, n_per_dim, hs)

                # Trace the first step
                step1 = torch.jit.trace(step1, (integration_domain,))

                # Get example input for the third step
                grid_points, hs, n_per_dim = step1(integration_domain)
                n_per_dim = int(n_per_dim)
                function_values, _ = self.evaluate_integrand(
                    example_integrand, grid_points
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
                step3 = torch.jit.trace(step3, (function_values, hs))

                # Define a compiled integrate function
                def compiled_integrate(fn, integration_domain):
                    grid_points, hs, _ = step1(integration_domain)
                    function_values, _ = self.evaluate_integrand(fn, grid_points)
                    result = step3(function_values, hs)
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

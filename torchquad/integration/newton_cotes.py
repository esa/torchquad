from loguru import logger
from autoray import infer_backend
from autoray import numpy as anp

from .grid_integrator import GridIntegrator
from .integration_grid import IntegrationGrid
from .utils import _setup_integration_domain

class NewtonCotes(GridIntegrator):
    """The abstract integrator that Composite Newton Cotes integrators inherit from"""

    def __init__(self):
        super().__init__()

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
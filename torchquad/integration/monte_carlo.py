from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain
from .rng import RNG


class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration"""

    def __init__(self):
        super().__init__()

    def integrate(
        self,
        fn,
        dim,
        N=1000,
        integration_domain=None,
        seed=None,
        rng=None,
        backend=None,
    ):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.
            rng (RNG, optional): An initialised RNG; this can be used when compiling the function for Tensorflow
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            backend-specific number: Integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.opt(lazy=True).debug(
            "Monte Carlo integrating a {dim}-dimensional fn with {N} points over {dom}",
            dim=lambda: dim,
            N=lambda: N,
            dom=lambda: integration_domain,
        )
        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        sample_points = self.calculate_sample_points(N, integration_domain, seed, rng)
        logger.debug("Evaluating integrand")
        function_values, self._nr_of_fevals = self.evaluate_integrand(fn, sample_points)
        return self.calculate_result(function_values, integration_domain)

    def calculate_result(self, function_values, integration_domain):
        """Calculate an integral result from the function evaluations

        Args:
            function_values (backend tensor): Output of the integrand
            integration_domain (backend tensor): Integration domain

        Returns:
            backend tensor: Quadrature result
        """
        logger.debug("Computing integration domain volume")
        scales = integration_domain[:, 1] - integration_domain[:, 0]
        volume = anp.prod(scales)

        # Integral = V / N * sum(func values)
        N = function_values.shape[0]
        integral = volume * anp.sum(function_values) / N
        # NumPy automatically casts to float64 when dividing by N
        if (
            infer_backend(integration_domain) == "numpy"
            and function_values.dtype != integral.dtype
        ):
            integral = integral.astype(function_values.dtype)
        logger.opt(lazy=True).info(
            "Computed integral: {result}", result=lambda: str(integral)
        )
        return integral

    def calculate_sample_points(self, N, integration_domain, seed=None, rng=None):
        """Calculate random points for the integrand evaluation

        Args:
            N (int): Number of points
            integration_domain (backend tensor): Integration domain
            seed (int, optional): Random number generation seed for the sampling point creation, only set if provided. Defaults to None.
            rng (RNG, optional): An initialised RNG; this can be used when compiling the function for Tensorflow

        Returns:
            backend tensor: Sample points
        """
        if rng is None:
            rng = RNG(backend=infer_backend(integration_domain), seed=seed)
        elif seed is not None:
            raise ValueError("seed and rng cannot both be passed")

        logger.debug("Picking random sampling points")
        dim = integration_domain.shape[0]
        domain_starts = integration_domain[:, 0]
        domain_sizes = integration_domain[:, 1] - domain_starts
        # Scale and translate random numbers via broadcasting
        return (
            rng.uniform(size=[N, dim], dtype=domain_sizes.dtype) * domain_sizes
            + domain_starts
        )

    def get_jit_compiled_integrate(
        self, dim, N=1000, integration_domain=None, seed=None, backend=None
    ):
        """Create an integrate function where the performance-relevant steps except the integrand evaluation are JIT compiled.
        Use this method only if the integrand cannot be compiled.
        The compilation happens when the function is executed the first time.
        With PyTorch, return values of different integrands passed to the compiled function must all have the same format, e.g. precision.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            seed (int, optional): Random number generation seed for the sequence of sampling point calculations, only set if provided. The returned integrate function calculates different points in each invocation with and without specified seed. Defaults to None.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            function(fn, integration_domain): JIT compiled integrate function where all parameters except the integrand and domain are fixed
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        backend = infer_backend(integration_domain)
        # autoray's autojit function does JIT compilation, too.
        # We don't use it here for the following reasons:
        # * The way random number generators have to be included or excluded
        #   from compilation differs between backends.
        # * autojit is not yet included in the latest autoray release
        #   (which is version 0.2.5).
        # * It uses the deprecated experimental_compile argument with Tensorflow.
        # * Additional detach() calls against warnings with PyTorch are not yet
        #   included in autojit.
        if backend == "tensorflow":
            if not hasattr(self, "_tf_jit_calculate_sample_points"):
                import tensorflow as tf

                self._tf_jit_calculate_sample_points = tf.function(
                    self.calculate_sample_points, jit_compile=True
                )
                self._tf_jit_calculate_result = tf.function(
                    self.calculate_result, jit_compile=True
                )
            jit_calculate_sample_points = self._tf_jit_calculate_sample_points
            jit_calculate_result = self._tf_jit_calculate_result
            rng = RNG(backend="tensorflow", seed=seed)

            def compiled_integrate(fn, integration_domain):
                sample_points = jit_calculate_sample_points(
                    N, integration_domain, rng=rng
                )
                function_values, _ = self.evaluate_integrand(fn, sample_points)
                return jit_calculate_result(function_values, integration_domain)

            return compiled_integrate
        elif backend == "jax":
            import jax

            rng = RNG(backend="jax", seed=seed)
            rng_key = rng.jax_get_key()

            @jax.jit
            def jit_calc_sample_points(integration_domain, rng_key):
                rng.jax_set_key(rng_key)
                sample_points = self.calculate_sample_points(
                    N, integration_domain, seed=None, rng=rng
                )
                return sample_points, rng.jax_get_key()

            if not hasattr(self, "_jax_jit_calculate_result"):
                self._jax_jit_calculate_result = jax.jit(
                    self.calculate_result, static_argnames=["dim", "n_per_dim"]
                )

            jit_calculate_result = self._jax_jit_calculate_result

            def compiled_integrate(fn, integration_domain):
                nonlocal rng_key
                sample_points, rng_key = jit_calc_sample_points(
                    integration_domain, rng_key
                )
                function_values, _ = self.evaluate_integrand(fn, sample_points)
                return jit_calculate_result(function_values, integration_domain)

            return compiled_integrate

        elif backend == "torch":
            # Torch requires explicit tracing with example inputs.
            def do_compile(example_integrand):
                import torch

                # Define traceable first and third steps
                def step1(integration_domain):
                    return self.calculate_sample_points(
                        N, integration_domain, seed=seed
                    )

                step3 = self.calculate_result

                # Trace the first step (which is non-deterministic)
                step1 = torch.jit.trace(step1, (integration_domain,), check_trace=False)

                # Get example input for the third step
                sample_points = step1(integration_domain)
                function_values, _ = self.evaluate_integrand(
                    example_integrand, sample_points
                )

                # Trace the third step
                if function_values.requires_grad:
                    # Avoid the warning about a .grad attribute access of a
                    # non-leaf Tensor
                    function_values = function_values.detach()
                    function_values.requires_grad = True
                step3 = torch.jit.trace(step3, (function_values, integration_domain))

                # Define a compiled integrate function
                def compiled_integrate(fn, integration_domain):
                    sample_points = step1(integration_domain)
                    function_values, _ = self.evaluate_integrand(fn, sample_points)
                    return step3(function_values, integration_domain)

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

from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain, RNG


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
        backend="torch",
    ):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.
            rng (RNG, optional): An initialised RNG; this can be used when compiling the function for Tensorflow
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to "torch".

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            torch.Tensor: integral value
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
        sample_points = []
        for d in range(dim):
            scale = integration_domain[d, 1] - integration_domain[d, 0]
            offset = integration_domain[d, 0]
            sample_points.append(
                rng.uniform(size=[N], dtype=scale.dtype) * scale + offset
            )
        return anp.stack(sample_points, axis=1, like=integration_domain)

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
        # Numpy automatically casts to float64 when dividing by N
        if (
            infer_backend(integration_domain) == "numpy"
            and function_values.dtype != integral.dtype
        ):
            integral = integral.astype(function_values.dtype)
        logger.opt(lazy=True).info(
            "Computed integral: {result}", result=lambda: str(integral)
        )
        return integral

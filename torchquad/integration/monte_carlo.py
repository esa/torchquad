from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain, _RNG


class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration"""

    def __init__(self):
        super().__init__()

    def integrate(
        self, fn, dim, N=1000, integration_domain=None, seed=None, backend="torch"
    ):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to "torch".

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            torch.Tensor: integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "Monte Carlo integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points over "
            + str(integration_domain),
        )

        self._dim = dim
        self._nr_of_fevals = 0
        self.fn = fn
        self._integration_domain = _setup_integration_domain(
            dim, integration_domain, backend
        )
        backend = infer_backend(self._integration_domain)
        rng = _RNG(backend=backend, seed=seed)

        logger.debug("Picking random sampling points")
        sample_points = []
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points.append(
                rng.uniform(size=[N], dtype=scale.dtype) * scale + offset
            )
        # FIXME: Is there a performance difference when initializing it
        # with zero instead of stacking it?
        sample_points = anp.stack(sample_points, axis=1, like=self._integration_domain)

        logger.debug("Evaluating integrand")
        function_values = fn(sample_points)

        logger.debug("Computing integration domain volume")
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = anp.prod(scales)

        # Integral = V / N * sum(func values)
        integral = volume * anp.sum(function_values) / N
        # Numpy automatically casts to float64 when dividing by N
        if backend == "numpy" and function_values.dtype != integral.dtype:
            integral = integral.astype(function_values.dtype)
        logger.info("Computed integral was " + str(integral))
        return integral

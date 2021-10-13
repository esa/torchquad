from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger
import warnings

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain


class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration"""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None, seed=None):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend (if it is a list, the backend is "torch").
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.

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
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        if seed is not None:
            if infer_backend(self._integration_domain) == "torch":
                anp.random.manual_seed(seed, like="torch")
            else:
                warnings.warn(
                    "The seed argument is not supported for the given backend."
                )

        logger.debug("Picking random sampling points")
        sample_points = anp.zeros([N, dim], like=self._integration_domain)
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points[:, d] = (
                anp.random.uniform(size=[N], like=self._integration_domain) * scale
                + offset
            )

        logger.debug("Evaluating integrand")
        function_values = fn(sample_points)

        logger.debug("Computing integration domain volume")
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = anp.prod(scales)

        # Integral = V / N * sum(func values)
        integral = volume * anp.sum(function_values) / N
        logger.info("Computed integral was " + str(integral))
        return integral

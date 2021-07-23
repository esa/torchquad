import torch

import logging

logger = logging.getLogger(__name__)


class IntegrationTestFunction:
    """Wrapper class for test functions"""

    domain = None  # Domain that is integrated over
    dim = None  # Expected input dimension of the function
    expected_result = None  # What the true integral solution is
    # Order of the function if applicable, can be used to infer expected convergence order
    order = None
    f = None  # Function to evaluate

    def __init__(self, expected_result, dim=1, domain=None):
        """Initializes domain and stores vars.

        Args:
            expected_result (float): Expected integration result.
            dim (int, optional): Dimensionality of investigated function. Defaults to 1.
            domain (list, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.
        """
        self.dim = dim
        self.expected_result = expected_result
        # Init domain to [-1,1]^dim if not passed
        if domain is None:
            self.domain = torch.tensor([[-1, 1]] * self.dim)
        else:
            self.domain = domain
        logging.debug("Initialized Test function with ")
        logging.debug(
            "dim="
            + str(self.dim)
            + "| domain="
            + str(self.domain)
            + "| expected_result="
            + str(expected_result)
        )

    def evaluate(self, integrator, integration_args):
        """Evaluates the passed integration functions with args.

        Args:
            integrator (func): Integration function to call.
            integration_args (list): Arguments to pass to integrator.

        Returns:
            float: Integration result
        """
        return integrator(fn=self.f, integration_domain=self.domain, **integration_args)


class Polynomial(IntegrationTestFunction):
    def __init__(self, expected_result=None, coeffs=[2], dim=1, domain=None):
        """N-dimensional, degree K poylnomial test functions.

        Args:
            expected_result (torch.tensor): Expected result. Required to compute errors.
            coeffs (list, optional): Polynomial coefficients. Are the same for each dim. Defaults to [2].
            dim (int, optional): Polynomial dimensionality. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
        """
        super().__init__(expected_result, dim, domain)
        self.coeffs = torch.tensor(coeffs)
        self.order = len(coeffs) - 1  # polynomial order is defined by the coeffs
        self.f = self._poly

    def _poly(self, x):
        # compute x^k
        exponentials = torch.zeros([x.shape[0], x.shape[1], self.order + 1])
        for o in range(self.order + 1):
            exponentials[:, :, o] = torch.pow(x, o)

        # multiply by coefficients
        exponentials = torch.multiply(exponentials, self.coeffs)

        # Collapse dimensions
        exponentials = torch.sum(exponentials, dim=2)

        # sum all values for each dim
        return torch.sum(exponentials, dim=1)


class Exponential(IntegrationTestFunction):
    def __init__(self, expected_result=None, dim=1, domain=None):
        """Creates an n-dimensional exponential test function.

        Args:
            expected_result (torch.tensor): Expected result. Required to compute errors.
            dim (int, optional): Input dimension. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
        """
        super().__init__(expected_result, dim, domain)
        self.f = self._exp

    def _exp(self, x):
        # compute e^x
        return torch.sum(torch.exp(x), dim=1)


class Sinusoid(IntegrationTestFunction):
    def __init__(self, expected_result=None, dim=1, domain=None):
        """Creates an n-dimensional sinusoidal test function.

        Args:
            expected_result (torch.tensor): Expected result. Required to compute errors.
            dim (int, optional): Input dimension. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
        """
        super().__init__(expected_result, dim, domain)
        self.f = self._sinusoid

    def _sinusoid(self, x):
        return torch.sum(torch.sin(x), dim=1)

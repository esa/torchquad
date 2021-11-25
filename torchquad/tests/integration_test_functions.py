import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend
from numpy import inf
from loguru import logger

from integration.utils import _setup_integration_domain


class IntegrationTestFunction:
    """Wrapper class for test functions."""

    domain = None  # Domain that is integrated over
    dim = None  # Expected input dimension of the function
    expected_result = None  # What the true integral solution is
    # Order of the function if applicable, can be used to infer expected convergence order
    order = None
    f = None  # Function to evaluate
    is_complex = False  # If the test function contains complex numbers

    def __init__(
        self, expected_result, dim=1, domain=None, is_complex=False, backend="torch"
    ):
        """Initializes domain and stores variables.

        Args:
            expected_result (float): Expected integration result.
            dim (int, optional): Dimensionality of investigated function. Defaults to 1.
            domain (list, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from domain. Defaults to "torch".
        """
        self.dim = dim
        self.expected_result = expected_result

        self.is_complex = is_complex
        self.domain = _setup_integration_domain(dim, domain, backend)
        logger.debug("Initialized Test function with ")
        logger.debug(
            "dim="
            + str(self.dim)
            + "| domain="
            + str(self.domain)
            + "| expected_result="
            + str(expected_result)
        )

    def evaluate(self, integrator, integration_args):
        """Evaluates the passed integration functions with arguments.

        Args:
            integrator (func): Integration function to call.
            integration_args (list): Arguments to pass to integrator.

        Returns:
            float: Integration result
        """

        def integrand(x):
            assert infer_backend(self.domain) == infer_backend(x), (
                "Integration domain and points have a different backend:"
                f" {infer_backend(self.domain)} and {infer_backend(x)}"
            )
            assert self.domain.dtype == x.dtype, (
                "Integration domain and points have a different dtype:"
                f" {self.domain.dtype} and {x.dtype}"
            )
            return self.f(x)

        return integrator(
            fn=integrand, integration_domain=self.domain, **integration_args
        )

    def get_order(self):
        """Get the order (polynomial degree) of the function

        Returns:
            float: Order of the function or infinity if it is not a finite polynomial
        """
        return inf if self.order is None else self.order


class Polynomial(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        coeffs=[2],
        dim=1,
        domain=None,
        is_complex=False,
        backend="torch",
    ):
        """Creates an n-dimensional, degree-K poylnomial test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.
            coeffs (list, optional): Polynomial coefficients. Are the same for each dim. Defaults to [2].
            dim (int, optional): Polynomial dimensionality. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from domain. Defaults to "torch".
        """
        super().__init__(expected_result, dim, domain, is_complex, backend)
        if backend == "tensorflow":
            # Ensure than all coefficients are either Python3 float or Python3
            # complex since tensorflow requires this.
            if is_complex:
                coeffs = list(map(complex, coeffs))
            else:
                coeffs = list(map(float, coeffs))
        if not is_complex:
            self.coeffs = anp.array(coeffs, like=self.domain, dtype=self.domain.dtype)
        else:
            self.coeffs = anp.array(coeffs, like=self.domain)
        self.order = len(coeffs) - 1  # polynomial order is defined by the coeffs
        self.f = self._poly

    def _poly(self, x):
        # Compute all relevant x^k
        # The shape of exponentials is (dim, N, order+1)
        if infer_backend(x) != "tensorflow":
            exponentials = x.reshape(x.shape + (1,)) ** anp.linspace(
                0, self.order, self.order + 1, like=x, dtype=x.dtype
            )
            assert exponentials.dtype == x.dtype
        else:
            # Tensorflow's exponentiation gives float64 values if x are float32
            # and the exponent are integer
            ks = anp.array(range(self.order + 1), dtype=x.dtype, like=x)
            exponentials = x.reshape(x.shape + (1,)) ** ks
            assert exponentials.dtype == x.dtype
            if exponentials.dtype != self.coeffs.dtype:
                # Tensorflow does not automatically cast float32 to complex128,
                # so we do it here explicitly.
                assert self.is_complex
                exponentials = anp.cast(exponentials, self.coeffs.dtype)

        # multiply by coefficients
        exponentials = anp.multiply(exponentials, self.coeffs)

        # Collapse dimensions
        exponentials = anp.sum(exponentials, axis=2)

        # sum all values for each dim
        return anp.sum(exponentials, axis=1)


class Exponential(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        dim=1,
        domain=None,
        is_complex=False,
        backend="torch",
    ):
        """Creates an n-dimensional exponential test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.
            dim (int, optional): Input dimension. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from domain. Defaults to "torch".
        """
        super().__init__(expected_result, dim, domain, is_complex, backend)
        self.f = self._exp

    def _exp(self, x):
        # compute e^x
        return anp.sum(anp.exp(x), axis=1)


class Sinusoid(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        dim=1,
        domain=None,
        is_complex=False,
        backend="torch",
    ):
        """Creates an n-dimensional sinusoidal test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.
            dim (int, optional): Input dimension. Defaults to 1.
            domain (list, optional): Integration domain. Defaults to [-1.0, 1.0]^dim.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from domain. Defaults to "torch".
        """
        super().__init__(expected_result, dim, domain, is_complex, backend)
        self.f = self._sinusoid

    def _sinusoid(self, x):
        return anp.sum(anp.sin(x), axis=1)

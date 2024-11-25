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
    integration_dim = None  # Expected input dimension of the function
    expected_result = None  # What the true integral solution is
    # Order of the function if applicable, can be used to infer expected convergence order
    order = None
    f = None  # Function to evaluate
    is_complex = False  # If the test function contains complex numbers
    integrand_dims = None  # What the dimensions of the integrand should be

    def __init__(
        self,
        expected_result,
        integration_dim=1,
        domain=None,
        is_complex=False,
        backend=None,
        integrand_dims=1,
    ):
        """Initializes domain and stores variables.

        Args:
            expected_result (float): Expected integration result.
            integration_dim (int, optional): Dimensionality of investigated function. Defaults to 1.
            domain (list or backend tensor, optional): Integration domain passed to _setup_integration_domain.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend passed to _setup_integration_domain.
            integrand_dims (Union[int, tuple], optional): Defaults to 1.  Should either be 1 or a tuple.  Determines how the integrand will be evaluated,
            whether once or over a matrix/vector of scaling factors.
        """
        self.integration_dim = integration_dim
        self.expected_result = expected_result
        if type(integrand_dims) is int or hasattr(integrand_dims, "__len__"):
            self.integrand_dims = integrand_dims
        else:
            ValueError(
                "Integrand dims should either be either an int or something that can be used to size an ndarray"
            )

        self.is_complex = is_complex
        self.domain = _setup_integration_domain(integration_dim, domain, backend)
        logger.debug("Initialized Test function with ")
        logger.debug(
            "integration_dim="
            + str(self.integration_dim)
            + "| integrand_dimensions"
            + str(self.integrand_dims)
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
            return self.integrand_scaling(self.f(x))

        return integrator(
            fn=integrand, integration_domain=self.domain, **integration_args
        )

    def get_order(self):
        """Get the order (polynomial degree) of the function

        Returns:
            float: Order of the function or infinity if it is not a finite polynomial
        """
        return inf if self.order is None else self.order

    def integrand_scaling(self, integrand):
        """Applies the scaling to the integrand, which is an `arange`-ed tensor in the shape of the
        desired integrand (by `self.integrand_dims`) giving a scaled version of the integrand repeated.

        Args:
            integrand (backend tensor): the integrand to be multiplied by the scaling grid

        Returns:
            Union[int, anp.ndarray]: The scaled integrand
        """
        integrand_scaling = self._integrand_scaling
        if self.is_integrand_1d:
            return integrand_scaling * integrand
        if self._is_integrand_tensor:
            scaling_einsum = "".join(
                [chr(i + 65) for i in range(len(self.integrand_dims))]
            )
            return anp.einsum(
                f"i,{scaling_einsum}->i{scaling_einsum}", integrand, integrand_scaling
            )

    @property
    def _integrand_scaling(self):
        """Get the integrand scaling factors dependent on integrand_dims

        Returns:
            Union[int, anp.ndarray]: The scaling factors
        """
        if self.is_integrand_1d:
            return 1
        if self._is_integrand_tensor:
            backend = infer_backend(self.domain)
            return anp.arange(
                anp.prod(self.integrand_dims), like=backend, dtype=self.domain.dtype
            ).reshape(self.integrand_dims)
        raise NotImplementedError(
            f"Integrand testing not implemented for dimensions {str(self.integrand_dims)}"
        )

    @property
    def is_integrand_1d(self):
        return self.integrand_dims == 1 or (
            len(self.integrand_dims) == 1 and self.integrand_dims[0] == 1
        )

    @property
    def _is_integrand_tensor(self):
        return len(self.integrand_dims) > 1 or (
            len(self.integrand_dims) == 1 and self.integrand_dims[0] > 1
        )


class Polynomial(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        coeffs=[2],
        integration_dim=1,
        domain=None,
        is_complex=False,
        backend=None,
        integrand_dims=1,
    ):
        """Creates an n-dimensional, degree-K poylnomial test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.
            coeffs (list, optional): Polynomial coefficients. Are the same for each dim. Defaults to [2].
            integration_dim (int, optional): Polynomial dimensionality. Defaults to 1.
            domain (list or backend tensor, optional): Integration domain passed to _setup_integration_domain.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend.
            integrand_dims (Union[int, tuple], optional): Defaults to 1.  Should either be 1 or a tuple.  Determines how the integrand will be evaluated,
            whether once or over a matrix/vector of scaling factors.
        """
        super().__init__(
            expected_result,
            integration_dim,
            domain,
            is_complex,
            backend,
            integrand_dims,
        )
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
        # The shape of exponentials is (integration_dim, N, order+1)
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
                exponentials = exponentials.astype(self.coeffs.dtype)

        # multiply by coefficients
        exponentials = anp.multiply(exponentials, self.coeffs)

        # Collapse dimensions
        exponentials = anp.sum(exponentials, axis=2)

        # sum all values for each integration_dim
        return anp.sum(exponentials, axis=1)


class Exponential(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        integration_dim=1,
        domain=None,
        is_complex=False,
        backend=None,
        integrand_dims=1,
    ):
        """Creates an n-dimensional exponential test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.            integration_dim (int, optional): Input dimension. Defaults to 1.
            domain (list or backend tensor, optional): Integration domain passed to _setup_integration_domain.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend passed to _setup_integration_domain.
            integrand_dims (Union[int, tuple], optional): Defaults to 1.  Should either be 1 or a tuple.  Determines how the integrand will be evaluated,
            whether once or over a matrix/vector of scaling factors.
        """
        super().__init__(
            expected_result,
            integration_dim,
            domain,
            is_complex,
            backend,
            integrand_dims,
        )
        self.f = self._exp

    def _exp(self, x):
        # compute e^x
        return anp.sum(anp.exp(x), axis=1)


class Sinusoid(IntegrationTestFunction):
    def __init__(
        self,
        expected_result=None,
        integration_dim=1,
        domain=None,
        is_complex=False,
        backend=None,
        integrand_dims=1,
    ):
        """Creates an n-dimensional sinusoidal test function.

        Args:
            expected_result (backend tensor): Expected result. Required to compute errors.            integration_dim (int, optional): Input dimension. Defaults to 1.
            domain (list or backend tensor, optional): Integration domain passed to _setup_integration_domain.
            is_complex (Boolean): If the test function contains complex numbers. Defaults to False.
            backend (string, optional): Numerical backend passed to _setup_integration_domain.
            integrand_dims (Union[int, tuple], optional): Defaults to 1.  Should either be 1 or a tuple.  Determines how the integrand will be evaluated,
            whether once or over a matrix/vector of scaling factors.
        """
        super().__init__(
            expected_result,
            integration_dim,
            domain,
            is_complex,
            backend,
            integrand_dims,
        )
        self.f = self._sinusoid

    def _sinusoid(self, x):
        return anp.sum(anp.sin(x), axis=1)

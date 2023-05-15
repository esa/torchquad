import numpy
from autoray import numpy as anp
from .grid_integrator import GridIntegrator


class Gaussian(GridIntegrator):
    """
    Base method for Gaussian Quadrature.  Different Gaussian methods should inherit from this class, and override as necessary methods.
    Default behaviour is Gauss-Legendre quadrature on [-1,1] (i.e., this "parent" class should __not__ be used directly with other integration domains, and for this parent class `integration_domain` as an argument to `integrate` is ignored internally).

    For an example of how to properly override the behavior to acheive different Gaussian Integration methods, please see the `Custom Integrators` section of the Tutorial or the implementation of `GaussLegendre`.

    The primary methods/attributes of interest to override are `_root_fn` (for different polynomials, like `numpy.polynomial.legendre.leggauss`), `_apply_composite_rule` (as in other integration methods), and `_resize_roots` (for handling different integration domains).

    Attributes:
        name  (str): A human-readable name for the integral.
        _root_fn (function): A function that returns roots and weights like `numpy.polynomial.legendre.leggauss`.
        _root_args (tuple): a way of adding information to be passed into `_root_fn` as needed.  This is then used when caching roots/weights to potentially distinguish different calls to `_root_fn` based on arguments.
        _cache (dict): a cache for roots and weights, used internally.
    """

    def __init__(self):
        super().__init__()
        self.name = "Gauss-Legendre"
        self._root_fn = numpy.polynomial.legendre.leggauss
        self._root_args = ()
        self._cache = {}

    def integrate(self, fn, dim, N=8, integration_domain=None, backend=None):
        """Integrates the passed function on the passed domain using a Gaussian rule (Gauss-Legendre on [-1,1] as a default).

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. Should be odd. Defaults to 3 points per dimension if None is given.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.   It also determines the numerical backend if possible.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            backend-specific number: Integral value
        """
        return super().integrate(fn, dim, N, integration_domain, backend)

    def _weights(self, N, dim, backend, requires_grad=False):
        """return the weights, broadcast across the dimensions, generated from the polynomial of choice

        Args:
            N (int): number of nodes
            dim (int): number of dimensions
            backend (string): which backend array to return

        Returns:
            backend tensor: the weights
        """
        weights = anp.array(self._cached_points_and_weights(N)[1], like=backend)
        if backend == "torch":
            weights.requires_grad = requires_grad
            return anp.prod(
                anp.array(
                    anp.stack(
                        list(anp.meshgrid(*([weights] * dim))), like=backend, dim=0
                    )
                ),
                axis=0,
            ).ravel()
        else:
            return anp.prod(
                anp.meshgrid(*([weights] * dim), like=backend), axis=0
            ).ravel()

    def _roots(self, N, backend, requires_grad=False):
        """return the roots generated from the polynomial of choice

        Args:
            N (int): number of nodes
            backend (string): which backend array to return

        Returns:
            backend tensor: the roots
        """
        roots = anp.array(self._cached_points_and_weights(N)[0], like=backend)
        if requires_grad:
            roots.requires_grad = True
        return roots

    @property
    def _grid_func(self):
        """
        function for generating a grid to be integrated over i.e., the polynomial roots, resized to the domain.
        """

        def f(integration_domain, N, requires_grad, backend=None):
            return self._resize_roots(
                integration_domain, self._roots(N, backend, requires_grad)
            )

        return f

    def _resize_roots(self, integration_domain, roots):  # scale from [-1,1] to [a,b]
        """Resize the roots based on domain of [a,b].  Default behavior is to simply return the roots, unsized by `integraton_domain`.

        Args:
            integration_domain (backend tensor): domain
            roots (backend tensor): polynomial nodes

        Returns:
            backend tensor: rescaled roots
        """
        return roots

    # credit for the idea https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/integrate/_quadrature.py#L72
    def _cached_points_and_weights(self, N):
        """wrap the calls to get weights/roots in a cache

        Args:
            N (int): number of nodes to return
            backend (string): which backend to use

        Returns:
            tuple: nodes and weights
        """
        _root_args = (N, *self._root_args)
        if not isinstance(N, int):
            if hasattr(N, "item"):
                _root_args = (N.item(), *self._root_args)
            else:
                raise NotImplementedError(
                    f"N {N} is not an int and lacks an `item` method"
                )
        if _root_args in self._cache:
            return self._cache[_root_args]
        self._cache[_root_args] = self._root_fn(*_root_args)
        return self._cache[_root_args]

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                0.5
                * (domain[cur_dim][1] - domain[cur_dim][0])
                * anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
            )
        return cur_dim_areas


class GaussLegendre(Gaussian):
    """Gauss Legendre quadrature rule in torch for any domain [a,b]. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature.

    Examples
    --------
    >>> gl=torchquad.GaussLegendre()
    >>> integral = gl.integrate(lambda x:np.sin(x), dim=1, N=101, integration_domain=[[0,5]]) #integral from 0 to 5 of np.sin(x)
    |TQ-INFO| Computed integral was 0.7163378000259399 #analytic result = 1-np.cos(5)"""

    def __init__(self):
        super().__init__()

    def _resize_roots(self, integration_domain, roots):  # scale from [-1,1] to [a,b]
        a = integration_domain[0]
        b = integration_domain[1]
        return ((b - a) / 2) * roots + ((a + b) / 2)

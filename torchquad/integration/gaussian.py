import torch
import numpy
import scipy
from scipy import special
from loguru import logger
from autoray import numpy as anp
from autoray import do
from .grid_integrator import GridIntegrator
from .utils import _setup_integration_domain

class Gaussian(GridIntegrator):
    """Gaussian quadrature methods inherit from this. Default behaviour is Gauss-Legendre quadrature on [-1,1]."""

    def __init__(self):
        super().__init__()
        self.name = "Gauss-Legendre"
        self.root_fn = numpy.polynomial.legendre.leggauss
        self.root_args = ()
        self.default_integration_domain = [[-1, 1]]
        self.transform_interval = True
        self._cache = {}
    
    def integrate(self, fn, dim, N=8, integration_domain=None, backend=None):
        """Integrates the passed function on the passed domain using Simpson's rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. Should be odd. Defaults to 3 points per dimension if None is given.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
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
            return anp.prod(anp.array(anp.stack(list(anp.meshgrid(*([weights] * dim))), like=backend, dim=0)), axis=0).ravel()
        else:
            return anp.prod(anp.meshgrid(*([weights] * dim), like=backend), axis=0).ravel()
    
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
        def f(integration_domain, N, requires_grad, backend=None):
            return self._resize_roots(integration_domain, self._roots(N, backend, requires_grad))
        return f

    def _resize_roots(self, integration_domain, roots):  # scale from [-1,1] to [a,b]
        """resize the roots based on domain of [a,b]

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
        root_args = (N, *self.root_args)
        if root_args in self._cache:
            return self._cache[root_args]

        self._cache[root_args] = self.root_fn(*root_args)
        return self._cache[root_args]
    
    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = .5 * (domain[cur_dim][1] - domain[cur_dim][0]) * anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
        return cur_dim_areas


class GaussLegendre(Gaussian):
    """Gauss Legendre quadrature rule in torch. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature.

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
        return ((b-a) / 2) * roots + ((a+b) / 2)


class GaussJacobi(Gaussian):
    """Gauss-Jacobi quadrature rule in torch, for integrals of the form :math:`\\int_{a}^{b} f(x) (1-x)^{\alpha} (1+x)^{\beta} dx`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature.

    Examples
    --------
    >>> gj=torchquad.GaussJacobi(2,3)
    >>> integral = gj.integrate(lambda x:x, dim=1, N=101, integration_domain=[[0,5]]) #integral from 0 to 5 of x * (1-x)**2 * (1+x)**3
    |TQ-INFO| Computed integral was 7.61904761904762 #analytic result = 1346/105 #wrong?
    """

    def __init__(self, alpha, beta):
        super().__init__()
        self.name = "Gauss-Jacobi"
        self.root_fn = scipy.special.roots_jacobi
        self.root_args = (alpha, beta)

    def _resize_roots(self, a, b, roots):  # scale from [-1,1] to [a,b]
        return ((b-a) / 2) * roots + ((a+b) / 2)


class GaussLaguerre(Gaussian):
    """Gauss Laguerre quadrature rule in torch, for integrals of the form :math:`\\int_0^{\\infty} e^{-x} f(x) dx`. It will correctly integrate polynomials of degree :math:`2n - 1` or less
    over the interval :math:`[0, \\infty]` with weight function
    :math:`f(x) = x^{\alpha} e^{-x}`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature.

    Examples
    --------
    >>> gl=torchquad.GaussLaguerre()
    >>> integral=gl.integrate(lambda x,a: np.sin(a*x),dim=1,N=20,args=(1,)) #integral from 0 to inf of np.exp(-x)*np.sin(x)
    |TQ-INFO| Computed integral was 0.49999999999998246. #analytic result = 0.5"""

    def __init__(self):
        super().__init__()
        self.name = "Gauss-Laguerre"
        self.root_fn = scipy.special.roots_laguerre
        self.default_integration_domain = [[0, numpy.inf]]
        self.wrapper_func = None

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for _ in range(dim):
            cur_dim_areas = anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
        return cur_dim_areas


class GaussHermite(Gaussian):
    """Gauss Hermite quadrature rule in torch, for integrals of the form :math:`\\int_{-\\infty}^{+\\infty} e^{-x^{2}} f(x) dx`. It will correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\\infty, \\infty]` with weight function :math:`f(x) = e^{-x^2}`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature

    Examples
    --------
    >>> gh=torchquad.GaussHermite()
    >>> integral=gh.integrate(lambda x: 1-x,dim=1,N=200) #integral from -inf to inf of np.exp(-(x**2))*(1-x)
    |TQ-INFO| Computed integral was 1.7724538509055168. #analytic result = sqrt(pi)
    """

    def __init__(self):
        super().__init__()
        self.name = "Gauss-Hermite"
        self.root_fn = scipy.special.roots_hermite
        self.default_integration_domain = [[-1 * numpy.inf, numpy.inf]]
        self.wrapper_func = None

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for _ in range(dim):
            cur_dim_areas = anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
        return cur_dim_areas

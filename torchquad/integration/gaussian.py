import torch
import numpy
import scipy
from loguru import logger
from autoray import numpy as anp
from autoray import do

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain

class Gaussian(BaseIntegrator):
    """Gaussian quadrature methods inherit from this. Default behaviour is Gauss-Legendre quadrature."""

    def __init__(self):
        super().__init__()
        self.name="Gauss-Legendre" #default
        self.root_fn=numpy.polynomial.legendre.leggauss
        self.root_args=()
        self.default_integration_domain=[[-1,1]]
        self.transform_interval=True
        self.wrapper_func=None
        
    def _points_and_weights(self,root_fn,root_args,wrapper_func=None):
        """Returns points and weights for integration.
        Adjusts from interval [-1,1] to integration limits [a,b]
        
        Args:
            root_fn (func): function to use for computing sample points and weights via the roots of appropriate polynomials
            root_args (tuple): arguments required for root-finding function, most commonly the degree N
            wrapper_func (func,optional): function that performs any additional calculations required to get the proper points and weights, eg. for use in Gauss-Lobatto quadrature. Default is None.
        
        Returns:
            tuple(points, weights)
            """
        a,b=self._integration_domain.T
        xi, wi = root_fn(*root_args) #can autoray work with scipy functions?
        if wrapper_func is not None:
            xi,wi=wrapper_func(xi,wi)
        
        if self.transform_interval: #scale from [-1,1] to [a,b] e.g. https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
            ## doesn't work correctly yet for Gauss-Jacobi!! why?
            n=xi.shape[0]
            xm=0.5*(b+a)
            xl=0.5*(b-a)
            if isinstance(xm,torch.Tensor):
                #for now... figure out a better solution later
                aa=torch.zeros(n)
                xi=aa.new(xi)
                wi=aa.new(wi)

            if xm.device !='cpu': #xi and xm need to be on same device
                xi=do("repeat",xm,n,like="numpy").reshape(self._dim,n)+anp.outer(xl,xi)
            else:
                xi=do("repeat",xm.cpu(),n,like="numpy").reshape(self._dim,n).to(torch.cuda.current_device())+anp.outer(xl,xi) #what if backend isn't torch?
            wi=anp.outer(wi,xl).T
        
        return xi,wi
    
    def integrate(self, fn, dim, args=None, N=8, integration_domain=None):
        """Integrates the passed function on the passed domain using fixed-point Gaussian quadrature.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            args (iterable object, optional): Additional arguments ``t0, ..., tn``, required by `fn`.
            N (int, optional): Degree to use for computing sample points and weights. Defaults to 8.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value

        """
        if not integration_domain or self.name in ["Gauss-Laguerre","Gauss-Hermite"]:
            integration_domain=self.default_integration_domain*dim
            if self.name in ["Gauss-Laguerre","Gauss-Hermite"]:
                logger.info(f"{self.name} integration only allowed over the interval {self.default_integration_domain[0]}!")
                self.transform_interval=False
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)

        logger.debug(f"Using {self.name} for integrating a fn with {N} points over {self._integration_domain}")

        self._dim = dim
        self._fn = fn
        
        root_args=(N,)+self.root_args
        
        xi, wi = self._points_and_weights(self.root_fn,root_args,wrapper_func=self.wrapper_func)
        integral= anp.sum(self._eval(xi,args=args,weights=wi)) #what if there is a sum in the function? then wi*self._eval() will have dimension mismatch
        logger.info(f"Computed integral was {integral}.")

        return integral


class GaussLegendre(Gaussian):
    """Gauss Legendre quadrature rule in torch. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature.
    
    Examples
    --------
    >>> gl=torchquad.GaussLegendre()
    >>> integral = gl.integrate(lambda x:np.sin(x), dim=1, N=101, integration_domain=[[0,5]]) #integral from 0 to 5 of np.sin(x)
    |TQ-INFO| Computed integral was 0.7163378000259399 #analytic result = 1-np.cos(5)"""

    def __init__(self):
        super().__init__()
        
        
class GaussJacobi(Gaussian):
    """Gauss-Jacobi quadrature rule in torch, for integrals of the form :math:`\int_{a}^{b} f(x) (1-x)^{\alpha} (1+x)^{\beta} dx`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature.
    
    Examples
    --------
    >>> gj=torchquad.GaussJacobi(2,3)
    >>> integral = gj.integrate(lambda x:x, dim=1, N=101, integration_domain=[[0,5]]) #integral from 0 to 5 of x * (1-x)**2 * (1+x)**3
    |TQ-INFO| Computed integral was 0.7163378000259399 #analytic result = 1346/105
    """

    def __init__(self, alpha,beta):
        super().__init__()
        self.name="Gauss-Jacobi"
        self.root_fn=scipy.special.roots_jacobi
        self.root_args=(alpha,beta)
        
        
class GaussLaguerre(Gaussian):
    """Gauss Laguerre quadrature rule in torch, for integrals of the form :math:`\int_0^{\infty} e^{-x} f(x) dx`. It will correctly integrate polynomials of degree :math:`2n - 1` or less
    over the interval :math:`[0, \infty]` with weight function
    :math:`f(x) = x^{\alpha} e^{-x}`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature.
    
    Examples
    --------
    >>> gl=torchquad.GaussLaguerre()
    >>> integral=gl.integrate(lambda x,a: np.sin(a*x),dim=1,N=20,args=(1,)) #integral from 0 to inf of np.exp(-x)*np.sin(x)
    |TQ-INFO| Computed integral was 0.49999999999998246. #analytic result = 0.5"""

    def __init__(self):
        super().__init__()
        self.name="Gauss-Laguerre"
        self.root_fn=scipy.special.roots_laguerre
        self.default_integration_domain=[[0,anp.inf]]
        
        
class GaussHermite(Gaussian):
    """Gauss Hermite quadrature rule in torch, for integrals of the form :math:`\int_{-\infty}^{+\infty} e^{-x^{2}} f(x) dx`. It will correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\infty, \infty]` with weight function :math:`f(x) = e^{-x^2}`. See https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    
    Examples
    --------
    >>> gh=torchquad.GaussHermite()
    >>> integral=gh.integrate(lambda x: 1-x,dim=1,N=200) #integral from -inf to inf of np.exp(-(x**2))*(1-x)
    |TQ-INFO| Computed integral was 1.7724538509055168. #analytic result = sqrt(pi)
    """

    def __init__(self):
        super().__init__()
        self.name="Gauss-Hermite"
        self.root_fn=scipy.special.roots_hermite
        self.default_integration_domain=[[-1*anp.inf,anp.inf]]
                
                
#class GaussLobatto(Gaussian):
#    """Gauss Lobatto quadrature rule in torch. See """
#
#    def __init__(self):
#        super().__init__()
#        self.name="Gauss-Lobatto"
#        self.root_fn=scipy.special.roots_jacobi #need additional function to operate on xi,wi to get the correct weights and points from the Jacobi polynomial roots...
#
#    def wrapper_func(self,xi,wi): #sould override default None


#class GaussRadau(Gaussian):
#    """Gauss Radau quadrature rule in torch. See """
#
#    def __init__(self):
#        super().__init__()
#        self.name="Gauss-Lobatto"
#        self.root_fn=scipy.special.roots_jacobi #need additional function to operate on xi,wi to get the correct weights and points from the Jacobi polynomial roots...


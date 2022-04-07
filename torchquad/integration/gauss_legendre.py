import torch
from loguru import logger

from .base_integrator import BaseIntegrator
#from .integration_grid import GaussLegendre#IntegrationGrid
from .utils import _setup_integration_domain


class GaussLegendre(BaseIntegrator):
    """Gauss Legendre quadrature rule in torch. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature."""

    def __init__(self):
        super().__init__()
        
    def _gauss_legendre(self,n):
        '''returns Gauss-Legendre points and weights for degree n and dimension self._dim'''
        x,w=np.polynomial.legendre.leggauss(n)
        xi=np.repeat(x,self._dim).reshape((n,self._dim)).T
        wi=np.repeat(w,self._dim).reshape((n,self._dim)).T
        return xi,wy

    def integrate(self, fn, dim, start_npoints=4, eps_rel=None,
    eps_abs=None, max_npoints=12,integration_domain=None):
        """Integrates the passed function on the passed domain using the trapezoid rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 8.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
            
        Basic example (works):
            def cosfn(x): return np.cos(x)
            gl=GaussLegendre()
            integral=gl.integrate(cosfn,dim=10,eps_rel=1e-10)
        """
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=start_npoints, integration_domain=self._integration_domain) #might need to check more

        #logger.debug(f"Using Trapezoid for integrating a fn with {npoints} points over {self._integration_domain}")

        self._dim = dim
        self._fn = fn

        for ires in range(int(np.log2(start_npoints)), max_npoints + 1): #if starting at npoints=8
            npoints = 2 ** ires #is this standard?
            if npoints > 2**max_npoints:
                raise ValueError(f"Integral did not satisfy the conditions eps_abs={eps_abs} or eps_rel={eps_rel} using the maximum number of points {2**max_npoints}") #or a different error?
                break

            # generate positions and weights
            xi, wi = self._gauss_legendre(npoints)
            #TO DO: need to scale from [-1,1] to [a,b]
            
            logger.debug("Evaluating integrand for {xi}.")
            if self._nr_of_fevals > 0:
                lastsum = np.array(integral)
                integral[i] = torch.sum(self._eval(xi)*wi,axis=1) #integral from -1 to 1 f(x) ≈ sum (w_i*f(x_i))
            else:
                integral = torch.sum(self._eval(xi)*wi,axis=1) #integral from -1 to 1 f(x) ≈ sum (w_i*f(x_i))

            print(npoints,integral)
            # Convergence criterion
            if self._nr_of_fevals//start_npoints > 1:
                l1 = np.abs(integral - lastsum)
                if eps_abs is not None:
                    i = np.where(l1 > eps_abs)[0] #does this work in higher dimensions?
                if eps_rel is not None:
                    l2 = eps_rel * np.abs(integral)
                    i = np.where(l1 > l2)[0] #does this work in higher dimensions?
            else:
                i= np.arange(self._dim) #indices of integral

            # If all point have reached criterion return value
            if i.size == 0:
                break

        logger.info(f"Computed integral was {integral}.")

        return integral

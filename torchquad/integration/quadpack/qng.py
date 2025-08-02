"""QNG - Non-adaptive Gauss-Kronrod quadrature."""

import warnings
from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from .base_quadpack import BaseQuadpack
from .gauss_kronrod import GaussKronrodRule


class QNG(BaseQuadpack):
    """QNG - Non-adaptive Gauss-Kronrod quadrature.
    
    QNG is a simple non-adaptive automatic integrator, based on
    a sequence of rules with increasing degree of algebraic precision
    (Patterson, 1968). It applies the Patterson sequence: 10, 21, 43, 87 points.
    
    This is a direct translation of the Fortran QUADPACK QNG routine.
    """

    def __init__(self):
        super().__init__()

    def _integrate_1d(self, domain_1d, epsabs, epsrel, max_fevals, **kwargs):
        """Perform 1D QNG integration.
        
        Args:
            domain_1d (backend tensor): Integration bounds [a, b]
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            max_fevals (int, optional): Maximum number of function evaluations
            
        Returns:
            backend tensor: Integral approximation
            
        Raises:
            RuntimeError: If algorithm fails to converge
        """
        a, b = domain_1d[0], domain_1d[1]
        backend = infer_backend(a)
        
        # Validate domain
        if a >= b:
            raise ValueError(f"Invalid integration domain: a={a} >= b={b}")
        
        logger.debug(f"QNG 1D integration: [{a}, {b}], epsabs={epsabs}, epsrel={epsrel}")
        
        # If domain has zero length, return zero
        if anp.abs(b - a) < self._epmach:
            return anp.array(0.0, like=a)
        
        # Create 1D wrapper for the function
        def f_1d(x_array):
            """Wrapper to evaluate function at 1D points."""
            # Convert to proper shape for self._fn
            if len(x_array.shape) == 1:
                points = anp.expand_dims(x_array, axis=-1)  # Shape: (n_points, 1)
            else:
                points = x_array
            result = self._fn(points)
            return result.flatten() if len(result.shape) > 1 else result
        
        # Use the corrected QNG algorithm
        try:
            result, abserr, neval, ier = GaussKronrodRule.qng(f_1d, a, b, epsabs, epsrel, backend)
            self._nr_of_fevals += neval
            
            # Check max_fevals limit (for QNG this happens after evaluation since it's non-adaptive)
            if self._check_max_fevals():
                logger.warning(f"QNG: Maximum function evaluations ({max_fevals}) exceeded with {neval} evaluations")
                # Still return the result since QNG completes the evaluation
            
            if ier != 0:
                logger.warning(f"QNG failed to converge: result={result}, abserr={abserr}, epsabs={epsabs}, epsrel={epsrel}")
                # Handle CUDA tensors and complex numbers for warning message
                if backend == "torch":
                    abserr_val = float(abserr)
                    # Handle complex numbers
                    if hasattr(result, 'dtype') and result.dtype.is_complex:
                        result_abs = float(anp.abs(result))
                        warnings.warn(f"QNG did not achieve requested tolerance. "
                                     f"Estimated error: {abserr_val}, result magnitude: {result_abs}")
                    else:
                        result_val = float(result)
                        requested_val = max(float(epsabs), float(epsrel) * abs(result_val))
                        warnings.warn(f"QNG did not achieve requested tolerance. "
                                     f"Estimated error: {abserr_val}, requested: {requested_val}")
                else:
                    warnings.warn(f"QNG did not achieve requested tolerance. "
                                 f"Estimated error: {abserr}, requested: {anp.maximum(epsabs, epsrel * anp.abs(result))}")
            else:
                logger.debug(f"QNG converged: result={result}, error={abserr}")
                
            return result
            
        except Exception as e:
            logger.error(f"QNG algorithm failed: {e}")
            raise RuntimeError(f"QNG failed: {e}")

    def integrate(self, fn, dim, integration_domain=None, backend=None,
                  epsabs=1.49e-8, epsrel=1.49e-8, max_fevals=None, **kwargs):
        """Integrate function using QNG algorithm.
        
        Args:
            fn (callable): Function to integrate
            dim (int): Dimensionality of integration domain
            integration_domain (list, optional): Integration bounds
            backend (str, optional): Numerical backend
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            max_fevals (int, optional): Maximum number of function evaluations
            
        Returns:
            backend tensor: Integral approximation
        """
        # Filter out unsupported kwargs for QNG
        supported_kwargs = {}
        if kwargs:
            logger.debug(f"QNG ignoring unsupported parameters: {list(kwargs.keys())}")
        
        return super().integrate(fn, dim, integration_domain, backend, 
                               epsabs, epsrel, max_fevals, **supported_kwargs)
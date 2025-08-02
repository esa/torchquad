"""QAGI - Adaptive quadrature for infinite intervals."""

import warnings
from loguru import logger

from .base_quadpack import BaseQuadpack
from .qags import QAGS


class QAGI(BaseQuadpack):
    """QAGI - Adaptive quadrature for infinite intervals.
    
    QAGI handles integration over infinite intervals. The infinite range
    is mapped onto a finite interval and then the same strategy as in QAGS
    is applied.
    
    Supported interval types:
    - [a, +∞): Semi-infinite, unbounded above  
    - (-∞, b]: Semi-infinite, unbounded below
    - (-∞, +∞): Infinite interval
    
    The algorithm uses transformation:
    - For [a, ∞): x = a + (1-t)/t, dx = dt/t²
    - For (-∞, b]: x = b - (1-t)/t, dx = dt/t²  
    - For (-∞, ∞): x = (1-t)/t, dx = dt/t²
    
    Note: This is currently a placeholder implementation.
    Full infinite interval transformations will be implemented in a later phase.
    """

    def __init__(self):
        super().__init__()
        self._qags_fallback = QAGS()  # Use QAGS as fallback

    def _integrate_1d(self, domain_1d, epsabs, epsrel, **kwargs):
        """Perform 1D QAGI integration.
        
        Args:
            domain_1d (backend tensor): Integration bounds [a, b] 
                                       where a and/or b may be infinite
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            
        Returns:
            backend tensor: Integral approximation
        """
        from autoray import numpy as anp
        
        a, b = domain_1d[0], domain_1d[1]
        
        # Check for infinite bounds
        a_inf = anp.isinf(a)
        b_inf = anp.isinf(b)
        
        if not (a_inf or b_inf):
            # Finite interval - use QAGS
            logger.debug("QAGI: finite interval, delegating to QAGS")
            return self._qags_fallback._integrate_1d(domain_1d, epsabs, epsrel, **kwargs)
        
        # TODO: Implement infinite interval transformations
        # For now, we'll raise an error for infinite intervals
        if a_inf and b_inf:
            interval_type = "(-∞, +∞)"
        elif a_inf:
            interval_type = f"(-∞, {b}]"
        else:
            interval_type = f"[{a}, +∞)"
            
        raise NotImplementedError(f"QAGI infinite interval integration not yet implemented for {interval_type}. "
                                 "This will be added in a future phase.")

    def integrate(self, fn, dim, integration_domain=None, backend=None,
                  epsabs=1.49e-8, epsrel=1.49e-8, **kwargs):
        """Integrate function using QAGI algorithm.
        
        Args:
            fn (callable): Function to integrate
            dim (int): Dimensionality of integration domain
            integration_domain (list, optional): Integration bounds (may contain ±∞)
            backend (str, optional): Numerical backend
            epsabs (float): Absolute error tolerance  
            epsrel (float): Relative error tolerance
            
        Returns:
            backend tensor: Integral approximation
        """
        if dim > 1:
            warnings.warn("QAGI is designed for 1D infinite intervals. "
                         "Multi-dimensional infinite domains are not well-supported.")
        
        return super().integrate(fn, dim, integration_domain, backend,
                               epsabs, epsrel, **kwargs)
"""
QUADPACK - Advanced adaptive quadrature methods for torchquad

This module provides Python implementations of classic QUADPACK algorithms,
extended to work with torchquad's multi-backend architecture and n-dimensional problems.

Available integrators:
- QNG: Non-adaptive Gauss-Kronrod quadrature
- QAG: Adaptive Gauss-Kronrod quadrature  
- QAGS: Adaptive quadrature with extrapolation
- QAGI: Adaptive quadrature for infinite intervals
- QAWC: Adaptive quadrature for Cauchy principal values

Example usage:
    >>> import torchquad
    >>> from torchquad.integration.quadpack import QAGS
    >>> 
    >>> integrator = QAGS()
    >>> result = integrator.integrate(lambda x: x**2, dim=1, 
    ...                              integration_domain=[[0, 1]],
    ...                              epsabs=1e-10, epsrel=1e-10)
"""

from .qng import QNG
from .qag import QAG  
from .qags import QAGS
from .qagi import QAGI
from .qawc import QAWC

__all__ = [
    "QNG",
    "QAG", 
    "QAGS",
    "QAGI",
    "QAWC",
]
"""QAGI - Adaptive quadrature for infinite intervals."""

import warnings
from autoray import numpy as anp
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
    """

    def __init__(self):
        super().__init__()
        self._qags = QAGS()  # Use QAGS for transformed integral

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
        a, b = domain_1d[0], domain_1d[1]
        
        # Check for infinite bounds
        a_inf = anp.isinf(a)
        b_inf = anp.isinf(b)
        
        if not (a_inf or b_inf):
            # Finite interval - use QAGS directly
            logger.debug("QAGI: finite interval, delegating to QAGS")
            # Set up QAGS with same function and backend
            self._qags._fn = self._fn
            self._qags._dim = self._dim
            self._qags._integration_domain = self._integration_domain
            self._qags._backend = self._backend
            self._qags._setup_machine_constants(self._backend)
            return self._qags._integrate_1d(domain_1d, epsabs, epsrel, **kwargs)
        
        # Store original function
        original_fn = self._fn
        
        if a_inf and b_inf:
            # Case 1: (-∞, +∞)
            # Transform: x = (1-t)/t for t in (0,1), but split at 0
            # Split integral at 0: ∫_{-∞}^{+∞} = ∫_{-∞}^0 + ∫_0^{+∞}
            logger.debug("QAGI: integrating over (-∞, +∞), splitting at 0")
            
            # First integral: (-∞, 0]
            def transformed_fn_neg(t_batch):
                # x = -(1-t)/t, dx = dt/t²
                # t_batch has shape (n_points, 1)
                t = t_batch[:, 0]
                # Avoid division by zero at t=0
                mask = t > self._epmach
                x = anp.where(mask, -(1-t)/t, -self._oflow)
                jacobian = anp.where(mask, 1/(t*t), 0)
                
                # Evaluate original function
                x_expanded = anp.expand_dims(x, axis=-1)
                f_vals = original_fn(x_expanded)
                if len(f_vals.shape) > 1:
                    f_vals = f_vals[:, 0]
                
                return f_vals * jacobian
            
            # Second integral: [0, +∞)
            def transformed_fn_pos(t_batch):
                # x = (1-t)/t, dx = dt/t²
                t = t_batch[:, 0]
                mask = t > self._epmach
                x = anp.where(mask, (1-t)/t, self._oflow)
                jacobian = anp.where(mask, 1/(t*t), 0)
                
                x_expanded = anp.expand_dims(x, axis=-1)
                f_vals = original_fn(x_expanded)
                if len(f_vals.shape) > 1:
                    f_vals = f_vals[:, 0]
                
                return f_vals * jacobian
            
            # Integrate both parts
            self._qags._fn = transformed_fn_neg
            self._qags._backend = self._backend
            self._qags._setup_machine_constants(self._backend)
            result_neg = self._qags._integrate_1d(
                anp.array([self._epmach, 1.0], like=a), 
                epsabs/2, epsrel, **kwargs
            )
            
            self._qags._fn = transformed_fn_pos
            result_pos = self._qags._integrate_1d(
                anp.array([self._epmach, 1.0], like=a), 
                epsabs/2, epsrel, **kwargs
            )
            
            return result_neg + result_pos
            
        elif a_inf:
            # Case 2: (-∞, b]
            # Transform: x = b - (1-t)/t, dx = dt/t²
            logger.debug(f"QAGI: integrating over (-∞, {b}]")
            
            def transformed_fn(t_batch):
                t = t_batch[:, 0]
                mask = t > self._epmach
                x = anp.where(mask, b - (1-t)/t, -self._oflow)
                jacobian = anp.where(mask, 1/(t*t), 0)
                
                x_expanded = anp.expand_dims(x, axis=-1)
                f_vals = original_fn(x_expanded)
                if len(f_vals.shape) > 1:
                    f_vals = f_vals[:, 0]
                
                return f_vals * jacobian
            
            self._qags._fn = transformed_fn
            self._qags._backend = self._backend
            self._qags._setup_machine_constants(self._backend)
            return self._qags._integrate_1d(
                anp.array([self._epmach, 1.0], like=a), 
                epsabs, epsrel, **kwargs
            )
            
        else:
            # Case 3: [a, +∞)
            # Transform: x = a + (1-t)/t, dx = dt/t²
            logger.debug(f"QAGI: integrating over [{a}, +∞)")
            
            def transformed_fn(t_batch):
                t = t_batch[:, 0]
                mask = t > self._epmach
                x = anp.where(mask, a + (1-t)/t, self._oflow)
                jacobian = anp.where(mask, 1/(t*t), 0)
                
                x_expanded = anp.expand_dims(x, axis=-1)
                f_vals = original_fn(x_expanded)
                if len(f_vals.shape) > 1:
                    f_vals = f_vals[:, 0]
                
                return f_vals * jacobian
            
            self._qags._fn = transformed_fn
            self._qags._backend = self._backend
            self._qags._setup_machine_constants(self._backend)
            return self._qags._integrate_1d(
                anp.array([self._epmach, 1.0], like=a), 
                epsabs, epsrel, **kwargs
            )

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
"""QAWC - Adaptive quadrature for Cauchy principal values."""

import warnings
from autoray import numpy as anp
from loguru import logger

from .base_quadpack import BaseQuadpack
from .qags import QAGS


class QAWC(BaseQuadpack):
    """QAWC - Adaptive quadrature for Cauchy principal values.
    
    QAWC computes the Cauchy Principal Value of f(x)/(x-c) over a finite
    interval (a,b) and for user-determined c.
    
    The Cauchy principal value is defined as:
    P.V. ∫[a,b] f(x)/(x-c) dx = lim[ε→0] [∫[a,c-ε] f(x)/(x-c) dx + ∫[c+ε,b] f(x)/(x-c) dx]
    
    The algorithm:
    - Uses globally adaptive subdivision strategy
    - Applies modified Clenshaw-Curtis integration on subranges containing x = c
    - Special handling of the singularity at x = c
    
    Requirements:
    - a < c < b (singularity must be interior to interval)
    - f(x) should be well-behaved except at x = c
    """

    def __init__(self):
        super().__init__()
        self._qags = QAGS()  # Use QAGS for subintervals

    def _integrate_1d(self, domain_1d, epsabs, epsrel, c=None, **kwargs):
        """Perform 1D QAWC integration.
        
        Args:
            domain_1d (backend tensor): Integration bounds [a, b]
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            c (float): Location of Cauchy singularity (must satisfy a < c < b)
            
        Returns:
            backend tensor: Cauchy principal value
        """
        a, b = domain_1d[0], domain_1d[1]
        
        if c is None:
            raise ValueError("QAWC requires singularity location 'c' parameter")
        
        # Validate singularity location
        if not (a < c < b):
            raise ValueError(f"Singularity c={c} must be in interval ({a}, {b})")
        
        logger.debug(f"QAWC: Computing Cauchy principal value at c={c}")
        
        # For Cauchy principal value, we use the fact that:
        # P.V. ∫[a,b] f(x)/(x-c) dx = ∫[a,b] [f(x) - f(c)]/(x-c) dx + f(c) * log|(b-c)/(c-a)|
        #
        # The first integral is regular (no singularity) and the second is analytical.
        
        # Store original function and evaluate at c
        original_fn = self._fn
        c_expanded = anp.expand_dims(anp.array([c], like=a), axis=-1)
        f_c = original_fn(c_expanded)[0]
        
        # Create regularized function: [f(x) - f(c)]/(x-c)
        def regularized_fn(x_batch):
            x = x_batch[:, 0]
            f_x = original_fn(x_batch)
            if len(f_x.shape) > 1:
                f_x = f_x[:, 0]
            
            # Compute [f(x) - f(c)]/(x-c), handling x near c
            diff = x - c
            mask = anp.abs(diff) > self._epmach
            
            # For points away from c: [f(x) - f(c)]/(x-c)
            result = anp.where(mask, (f_x - f_c) / diff, 0.0)
            
            # For points very close to c, use derivative approximation
            # f'(c) ≈ [f(c+h) - f(c-h)]/(2h)
            h = anp.sqrt(self._epmach) * max(anp.abs(c), 1.0)
            close_mask = ~mask
            
            if anp.any(close_mask):
                # Estimate derivative at c
                c_plus_h = anp.expand_dims(anp.array([c + h], like=a), axis=-1)
                c_minus_h = anp.expand_dims(anp.array([c - h], like=a), axis=-1)
                f_plus = original_fn(c_plus_h)[0]
                f_minus = original_fn(c_minus_h)[0]
                f_prime_c = (f_plus - f_minus) / (2 * h)
                
                # For x very close to c: f'(c)
                result = anp.where(close_mask, f_prime_c, result)
            
            return result
        
        # Split integral at c to avoid the singularity
        # ∫[a,b] = ∫[a,c-δ] + ∫[c+δ,b] where δ is small
        delta = anp.sqrt(self._epmach) * max(b - a, 1.0)
        
        # Set up QAGS with regularized function
        self._qags._fn = regularized_fn
        self._qags._dim = self._dim
        self._qags._backend = self._backend
        self._qags._setup_machine_constants(self._backend)
        
        # Integrate left part [a, c-δ]
        if c - a > delta:
            result_left = self._qags._integrate_1d(
                anp.array([a, c - delta], like=a), 
                epsabs/2, epsrel, **kwargs
            )
        else:
            result_left = anp.array(0.0, like=a)
        
        # Integrate right part [c+δ, b]
        if b - c > delta:
            result_right = self._qags._integrate_1d(
                anp.array([c + delta, b], like=a), 
                epsabs/2, epsrel, **kwargs
            )
        else:
            result_right = anp.array(0.0, like=a)
        
        # Add analytical part: f(c) * log|(b-c)/(c-a)|
        analytical_part = f_c * anp.log(anp.abs((b - c) / (c - a)))
        
        # Total Cauchy principal value
        result = result_left + result_right + analytical_part
        
        logger.debug(f"QAWC result: {result} (left: {result_left}, right: {result_right}, analytical: {analytical_part})")
        
        return result

    def integrate(self, fn, dim, integration_domain=None, backend=None,
                  epsabs=1.49e-8, epsrel=1.49e-8, c=None, **kwargs):
        """Integrate function using QAWC algorithm.
        
        Args:
            fn (callable): Function to integrate (should be f(x) where the integral
                          is P.V. ∫ f(x)/(x-c) dx)
            dim (int): Dimensionality of integration domain (should be 1)
            integration_domain (list, optional): Integration bounds [a, b]
            backend (str, optional): Numerical backend
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            c (float): Location of Cauchy singularity
            
        Returns:
            backend tensor: Cauchy principal value
        """
        if dim != 1:
            raise ValueError("QAWC is only applicable to 1D integration problems")
        
        if c is None:
            raise ValueError("QAWC requires singularity location 'c' parameter")
        
        # Pass through QAWC-specific parameters
        qawc_kwargs = {"c": c}
        qawc_kwargs.update(kwargs)
        
        return super().integrate(fn, dim, integration_domain, backend,
                               epsabs, epsrel, **qawc_kwargs)
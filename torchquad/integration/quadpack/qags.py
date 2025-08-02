"""QAGS - Adaptive Gauss-Kronrod quadrature with extrapolation."""

import warnings
from autoray import numpy as anp
from loguru import logger

from .base_quadpack import BaseQuadpack
from .gauss_kronrod import GaussKronrodRule
from .subdivision import AdaptiveSubdivision
from .extrapolation import EpsilonExtrapolation


class QAGS(BaseQuadpack):
    """QAGS - Adaptive Gauss-Kronrod quadrature with extrapolation.
    
    QAGS is an integrator based on globally adaptive interval subdivision
    in connection with extrapolation (de Doncker, 1978) by the Epsilon algorithm
    (Wynn, 1956). This is the most robust QUADPACK algorithm.
    
    Features:
    - Adaptive subdivision based on error estimates
    - Epsilon algorithm extrapolation for convergence acceleration  
    - Excellent handling of difficult integrands
    - Automatic detection of integration difficulties
    - Most robust QUADPACK algorithm
    
    The algorithm:
    1. Subdivides intervals where error is largest
    2. Applies Gauss-Kronrod rules on each subinterval
    3. Uses epsilon extrapolation to accelerate convergence
    4. Monitors for roundoff error and other issues
    """

    def __init__(self):
        super().__init__()

    def _integrate_1d(self, domain_1d, epsabs, epsrel, limit=50, **kwargs):
        """Perform 1D QAGS integration.
        
        Args:
            domain_1d (backend tensor): Integration bounds [a, b]
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance  
            limit (int): Maximum number of subdivisions
            
        Returns:
            backend tensor: Integral approximation
        """
        a, b = domain_1d[0], domain_1d[1]
        
        # Validate domain
        if a >= b:
            raise ValueError(f"Invalid integration domain: a={a} >= b={b}")
        
        # Validate limit parameter
        if limit <= 0:
            raise ValueError(f"Invalid limit: {limit}. Must be positive integer.")
        
        logger.debug(f"QAGS 1D integration: [{a}, {b}], epsabs={epsabs}, epsrel={epsrel}, limit={limit}")
        
        # If domain has zero length, return zero
        if anp.abs(b - a) < self._epmach:
            return anp.array(0.0, like=a)
        
        # Create 1D wrapper for the function
        def f_1d(x_array):
            """Wrapper to evaluate function at 1D points."""
            if len(x_array.shape) == 1:
                points = anp.expand_dims(x_array, axis=-1)  # Shape: (n_points, 1)
            else:
                points = x_array
            result = self._fn(points)
            return result.flatten() if len(result.shape) > 1 else result
        
        # Initialize subdivision and extrapolation
        subdivision = AdaptiveSubdivision(limit)
        extrapolation = EpsilonExtrapolation()
        
        # Initial integration over [a, b]
        try:
            result, abserr, _, _, neval = GaussKronrodRule.evaluate_gk21(f_1d, a, b, self._backend)
            subdivision.neval += neval
            self._nr_of_fevals += neval
            
            # Add initial interval
            subdivision.add_interval(a, b, result, abserr)
            
        except Exception as e:
            logger.error(f"QAGS initial evaluation failed: {e}")
            raise RuntimeError(f"QAGS failed: {e}")
        
        # Check initial convergence
        if subdivision.converged(epsabs, epsrel):
            logger.debug(f"QAGS converged immediately: result={result}, error={abserr}")
            return result
        
        # Add initial result to extrapolation
        extrap_result, extrap_error, accept_extrap = extrapolation.add_result(result)
        
        # Adaptive subdivision loop
        iteration = 0
        last_result = result
        
        while subdivision.should_continue():
            # Get interval with largest error
            interval_data = subdivision.get_largest_error_interval()
            if interval_data is None:
                break
                
            idx, a_i, b_i, res_i, err_i = interval_data
            
            # Split the interval at midpoint
            c = 0.5 * (a_i + b_i)
            
            try:
                # Integrate left subinterval [a_i, c]
                res_left, err_left, _, _, neval_left = GaussKronrodRule.evaluate_gk21(f_1d, a_i, c, self._backend)
                subdivision.neval += neval_left
                self._nr_of_fevals += neval_left
                
                # Integrate right subinterval [c, b_i]
                res_right, err_right, _, _, neval_right = GaussKronrodRule.evaluate_gk21(f_1d, c, b_i, self._backend)
                subdivision.neval += neval_right
                self._nr_of_fevals += neval_right
                
            except Exception as e:
                logger.warning(f"QAGS subdivision failed at iteration {iteration}: {e}")
                break
            
            # Replace the interval with two subintervals
            subdivision.replace_interval(idx, a_i, c, res_left, err_left, 
                                       c, b_i, res_right, err_right)
            
            # Get current total and add to extrapolation
            current_result, current_error = subdivision.get_total_result()
            extrap_result, extrap_error, accept_extrap = extrapolation.add_result(current_result)
            
            # Check convergence with extrapolation
            if accept_extrap:
                extrap_tolerance = anp.maximum(epsabs, epsrel * anp.abs(extrap_result))
                if extrap_error <= extrap_tolerance:
                    logger.debug(f"QAGS converged with extrapolation after {iteration+1} subdivisions: result={extrap_result}, error={extrap_error}")
                    return extrap_result
            
            # Check regular convergence
            if subdivision.converged(epsabs, epsrel):
                total_result, total_error = subdivision.get_total_result()
                logger.debug(f"QAGS converged after {iteration+1} subdivisions: result={total_result}, error={total_error}")
                return total_result
            
            # Check for lack of progress
            if iteration > 10:
                progress = abs(current_result - last_result) / max(abs(current_result), abs(last_result), 1e-15)
                if progress < 1e-12:
                    logger.debug(f"QAGS: Limited progress detected, stopping early")
                    break
            
            last_result = current_result
            iteration += 1
            
            # Safety check
            if iteration > limit * 2:
                logger.warning(f"QAGS: Too many iterations ({iteration}), stopping")
                break
        
        # Final result - prefer extrapolated if available and reasonable
        total_result, total_error = subdivision.get_total_result()
        
        if accept_extrap and extrap_error < total_error:
            final_result = extrap_result
            final_error = extrap_error
        else:
            final_result = total_result
            final_error = total_error
        
        # Check if we achieved tolerance
        tolerance = anp.maximum(epsabs, epsrel * anp.abs(final_result))
        if final_error > tolerance:
            logger.warning(f"QAGS failed to converge: result={final_result}, error={final_error}, tolerance={tolerance}")
            warnings.warn(f"QAGS did not achieve requested tolerance. "
                         f"Estimated error: {final_error}, requested: {tolerance}")
        else:
            logger.debug(f"QAGS converged: result={final_result}, error={final_error}")
        
        return final_result

    def integrate(self, fn, dim, integration_domain=None, backend=None,
                  epsabs=1.49e-8, epsrel=1.49e-8, limit=50, **kwargs):
        """Integrate function using QAGS algorithm.
        
        Args:
            fn (callable): Function to integrate
            dim (int): Dimensionality of integration domain
            integration_domain (list, optional): Integration bounds
            backend (str, optional): Numerical backend  
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            limit (int): Maximum number of subdivisions
            
        Returns:
            backend tensor: Integral approximation
        """
        # Pass through QAGS-specific parameters
        qags_kwargs = {"limit": limit}
        qags_kwargs.update(kwargs)
        
        return super().integrate(fn, dim, integration_domain, backend,
                               epsabs, epsrel, **qags_kwargs)
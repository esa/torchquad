"""QAG - Adaptive Gauss-Kronrod quadrature."""

import warnings
from autoray import numpy as anp
from loguru import logger

from .base_quadpack import BaseQuadpack
from .gauss_kronrod import GaussKronrodRule
from .subdivision import AdaptiveSubdivision


class QAG(BaseQuadpack):
    """QAG - Adaptive Gauss-Kronrod quadrature.

    QAG is a globally adaptive integrator using the strategy of Aind (Piessens, 1973).
    It uses adaptive subdivision with Gauss-Kronrod quadrature rules for local error
    estimation.

    Features:
    - Adaptive interval subdivision
    - Multiple Gauss-Kronrod rule options
    - Error-driven refinement
    - Suitable for moderately difficult integrands
    """

    def __init__(self):
        super().__init__()

    def _integrate_1d(self, domain_1d, epsabs, epsrel, max_fevals, limit=50, key=2, **kwargs):
        """Perform 1D QAG integration.

        Args:
            domain_1d (backend tensor): Integration bounds [a, b]
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            max_fevals (int, optional): Maximum number of function evaluations
            limit (int): Maximum number of subdivisions
            key (int): Choice of Gauss-Kronrod rule (1-6)

        Returns:
            backend tensor: Integral approximation
        """
        a, b = domain_1d[0], domain_1d[1]

        # Validate domain
        if a >= b:
            raise ValueError(f"Invalid integration domain: a={a} >= b={b}")

        logger.debug(
            f"QAG 1D integration: [{a}, {b}], epsabs={epsabs}, epsrel={epsrel}, limit={limit}"
        )

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

        # Initialize subdivision
        subdivision = AdaptiveSubdivision(limit)

        # Initial integration over [a, b]
        try:
            result, abserr, _, _, neval = GaussKronrodRule.evaluate_gk21(f_1d, a, b, "numpy")
            subdivision.neval += neval
            self._nr_of_fevals += neval

            # Add initial interval
            subdivision.add_interval(a, b, result, abserr)

        except Exception as e:
            logger.error(f"QAG initial evaluation failed: {e}")
            raise RuntimeError(f"QAG failed: {e}")

        # Check initial convergence
        if subdivision.converged(epsabs, epsrel):
            logger.debug(f"QAG converged immediately: result={result}, error={abserr}")
            return result
        
        # Check max_fevals after initial evaluation
        if self._check_max_fevals():
            logger.warning(f"QAG: Maximum function evaluations ({max_fevals}) exceeded after initial evaluation")
            return result

        # Adaptive subdivision loop
        iteration = 0
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
                res_left, err_left, _, _, neval_left = GaussKronrodRule.evaluate_gk21(
                    f_1d, a_i, c, "numpy"
                )
                subdivision.neval += neval_left
                self._nr_of_fevals += neval_left

                # Integrate right subinterval [c, b_i]
                res_right, err_right, _, _, neval_right = GaussKronrodRule.evaluate_gk21(
                    f_1d, c, b_i, "numpy"
                )
                subdivision.neval += neval_right
                self._nr_of_fevals += neval_right

            except Exception as e:
                logger.warning(f"QAG subdivision failed at iteration {iteration}: {e}")
                break

            # Replace the interval with two subintervals
            subdivision.replace_interval(
                idx, a_i, c, res_left, err_left, c, b_i, res_right, err_right
            )

            # Check convergence
            if subdivision.converged(epsabs, epsrel):
                total_result, total_error = subdivision.get_total_result()
                logger.debug(
                    f"QAG converged after {iteration+1} subdivisions: result={total_result}, error={total_error}"
                )
                return total_result
            
            # Check max_fevals before next iteration
            if self._check_max_fevals():
                logger.warning(f"QAG: Maximum function evaluations ({max_fevals}) exceeded after {iteration+1} subdivisions")
                break

            iteration += 1

            # Safety check
            if iteration > limit * 2:
                logger.warning(f"QAG: Too many iterations ({iteration}), stopping")
                break

        # Final result
        total_result, total_error = subdivision.get_total_result()

        # Check if we achieved tolerance - use scalar comparison to avoid CUDA tensor issues
        tolerance = anp.maximum(
            anp.array(epsabs, like=total_result),
            anp.array(epsrel, like=total_result) * anp.abs(total_result),
        )
        total_error_float = float(total_error) if hasattr(total_error, "item") else total_error
        tolerance_float = float(tolerance) if hasattr(tolerance, "item") else tolerance

        if total_error_float > tolerance_float:
            logger.warning(
                f"QAG failed to converge: result={total_result}, error={total_error}, tolerance={tolerance}"
            )
            warnings.warn(
                f"QAG did not achieve requested tolerance. "
                f"Estimated error: {total_error}, requested: {tolerance}"
            )
        else:
            logger.debug(f"QAG converged: result={total_result}, error={total_error}")

        return total_result

    def integrate(
        self,
        fn,
        dim,
        integration_domain=None,
        backend=None,
        epsabs=1.49e-8,
        epsrel=1.49e-8,
        max_fevals=None,
        limit=50,
        key=2,
        **kwargs,
    ):
        """Integrate function using QAG algorithm.

        Args:
            fn (callable): Function to integrate
            dim (int): Dimensionality of integration domain
            integration_domain (list, optional): Integration bounds
            backend (str, optional): Numerical backend
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            max_fevals (int, optional): Maximum number of function evaluations
            limit (int): Maximum number of subdivisions
            key (int): Choice of Gauss-Kronrod rule

        Returns:
            backend tensor: Integral approximation
        """
        # Pass through QAG-specific parameters
        qag_kwargs = {"limit": limit, "key": key}
        qag_kwargs.update(kwargs)

        return super().integrate(fn, dim, integration_domain, backend, epsabs, epsrel, max_fevals, **qag_kwargs)

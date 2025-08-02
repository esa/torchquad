"""Base class for QUADPACK integrators."""

import warnings
from abc import abstractmethod
from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger

from ..base_integrator import BaseIntegrator
from ..utils import _setup_integration_domain


class BaseQuadpack(BaseIntegrator):
    """Base class for all QUADPACK integrators.
    
    Provides common functionality for QUADPACK algorithms including:
    - Parameter validation for epsabs/epsrel tolerances
    - Multi-dimensional extension via tensor products
    - Backend-agnostic implementations
    - Error handling and convergence detection
    """

    def __init__(self):
        super().__init__()
        
        # QUADPACK-specific machine constants
        self._epmach = None  # Will be set based on backend
        self._uflow = None   # Underflow threshold
        self._oflow = None   # Overflow threshold
        
        # Function evaluation counter
        self._nr_of_fevals = 0

    def integrate(self, fn, dim, integration_domain=None, backend=None, 
                  epsabs=1.49e-8, epsrel=1.49e-8, max_fevals=None, **kwargs):
        """Integrate function using QUADPACK algorithm.
        
        Args:
            fn (callable): Function to integrate. Should accept array of shape (n_points, dim)
                          and return array of shape (n_points,) or (n_points, output_dim)
            dim (int): Dimensionality of the integration domain
            integration_domain (list or tensor, optional): Integration bounds [[a1,b1], [a2,b2], ...]
                                                          Defaults to [-1,1]^dim
            backend (str, optional): Numerical backend ('torch', 'jax', 'tensorflow', 'numpy')
            epsabs (float): Absolute error tolerance (default: ~1.5e-8)
            epsrel (float): Relative error tolerance (default: ~1.5e-8)
            max_fevals (int, optional): Maximum number of function evaluations. Defaults to None (no limit)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            backend tensor: Integral approximation
            
        Raises:
            ValueError: If tolerances or domain are invalid
            RuntimeError: If algorithm fails to converge
        """
        # Validate inputs
        self._check_inputs(dim=dim, integration_domain=integration_domain)
        
        # Validate max_fevals parameter
        if max_fevals is not None and max_fevals <= 0:
            raise ValueError("max_fevals must be positive if specified")
        
        # Reset function evaluation counter
        self._nr_of_fevals = 0
        self._max_fevals = max_fevals
        
        # Setup domain and backend first
        integration_domain = _setup_integration_domain(dim, integration_domain, backend)
        backend = infer_backend(integration_domain)
        self._backend = backend  # Store for later use
        self._setup_machine_constants(backend)
        
        # Now validate tolerances (needs machine constants)
        self._validate_tolerances(epsabs, epsrel)
        
        # Store function and domain for use in algorithms
        self._fn = fn
        self._dim = dim
        self._integration_domain = integration_domain
        
        if dim == 1:
            # Direct 1D integration
            return self._integrate_1d(integration_domain[0], epsabs, epsrel, max_fevals, **kwargs)
        else:
            # Multi-dimensional via tensor product
            return self._integrate_nd_tensor_product(integration_domain, epsabs, epsrel, max_fevals, **kwargs)
    
    def _validate_tolerances(self, epsabs, epsrel):
        """Validate error tolerances."""
        if epsabs < 0:
            raise ValueError("epsabs must be non-negative")
        if epsrel < 0:
            raise ValueError("epsrel must be non-negative")
        if epsabs <= 0 and epsrel <= 0:
            raise ValueError("At least one of epsabs or epsrel must be positive")
        
        # Check if tolerances are achievable
        if epsrel > 0 and epsrel < 50 * self._epmach:
            warnings.warn(f"epsrel ({epsrel}) may be too small relative to machine precision ({self._epmach})")
    
    def _setup_machine_constants(self, backend):
        """Setup machine constants based on backend."""
        if backend == "torch":
            import torch
            self._epmach = float(torch.finfo(torch.float64).eps)
            self._uflow = float(torch.finfo(torch.float64).tiny)
            self._oflow = float(torch.finfo(torch.float64).max)
        elif backend == "jax":
            import jax.numpy as jnp
            self._epmach = float(jnp.finfo(jnp.float64).eps)
            self._uflow = float(jnp.finfo(jnp.float64).tiny)
            self._oflow = float(jnp.finfo(jnp.float64).max)
        elif backend == "tensorflow":
            import tensorflow as tf
            self._epmach = float(tf.experimental.numpy.finfo(tf.float64).eps)
            self._uflow = float(tf.experimental.numpy.finfo(tf.float64).tiny)
            self._oflow = float(tf.experimental.numpy.finfo(tf.float64).max)
        else:  # numpy
            import numpy as np
            self._epmach = float(np.finfo(np.float64).eps)
            self._uflow = float(np.finfo(np.float64).tiny)
            self._oflow = float(np.finfo(np.float64).max)
    
    def _check_max_fevals(self):
        """Check if maximum function evaluations exceeded.
        
        Returns:
            bool: True if max_fevals exceeded, False otherwise
        """
        if self._max_fevals is not None and self._nr_of_fevals >= self._max_fevals:
            return True
        return False
    
    @abstractmethod
    def _integrate_1d(self, domain_1d, epsabs, epsrel, max_fevals, **kwargs):
        """Perform 1D integration on interval [a, b].
        
        Args:
            domain_1d (backend tensor): Integration bounds [a, b]
            epsabs (float): Absolute error tolerance
            epsrel (float): Relative error tolerance
            max_fevals (int, optional): Maximum number of function evaluations
            **kwargs: Algorithm-specific parameters
            
        Returns:
            backend tensor: Integral approximation
        """
        pass
    
    def _integrate_nd_tensor_product(self, integration_domain, epsabs, epsrel, max_fevals, **kwargs):
        """Proper tensor product integration for n-D problems.
        
        Implements the mathematical tensor product formula:
        ∫∫...∫ f(x1,...,xn) dx1...dxn = ∫ (∫ (...(∫ f(x1,...,xn) dx1)...) dxn-1) dxn
        
        This performs sequential 1D integrations along each dimension.
        """
        logger.debug(f"Starting {self._dim}D tensor product integration")
        
        if self._dim > 5:
            warnings.warn(f"High-dimensional integration (dim={self._dim}) may be slow and inaccurate. "
                         "Consider using Monte Carlo methods for dimensions > 5.")
        
        # For TensorFlow, disable multi-D due to tensor assignment issues
        if self._backend == "tensorflow":
            raise NotImplementedError("QUADPACK multi-dimensional integration is not supported with TensorFlow backend. "
                                    "Use dim=1 or switch to numpy/torch backend for multi-dimensional problems.")
        
        # Store original function for nested integration
        original_fn = self._fn
        
        def recursive_integrate(dim_idx, fixed_vars):
            """Recursively integrate over remaining dimensions."""
            if dim_idx == self._dim:
                # Base case: evaluate function at fixed point
                # Build point from fixed variables
                point = anp.stack([anp.array([fixed_vars[i]], like=integration_domain[0][0]) 
                                  for i in range(self._dim)], axis=1)
                result = original_fn(point)
                # Return scalar value
                if hasattr(result, 'shape') and len(result.shape) > 0:
                    return result[0]
                return result
            
            # Create 1D function for current dimension
            def integrand_1d(x):
                # x has shape (n_points, 1)
                n_points = x.shape[0]
                results = anp.zeros(n_points, like=x)
                
                for i in range(n_points):
                    # Create new fixed_vars with current x value
                    new_fixed = fixed_vars.copy()
                    new_fixed[dim_idx] = float(x[i, 0])
                    # Recursively integrate over remaining dimensions
                    results = results.at[i].set(recursive_integrate(dim_idx + 1, new_fixed)) if hasattr(results, 'at') else results
                    if not hasattr(results, 'at'):
                        results[i] = recursive_integrate(dim_idx + 1, new_fixed)
                
                return results
            
            # Temporarily set up for 1D integration
            current_domain = integration_domain[dim_idx]
            old_fn = self._fn
            self._fn = integrand_1d
            
            try:
                result = self._integrate_1d(current_domain, epsabs, epsrel, max_fevals, **kwargs)
                return result
            finally:
                self._fn = old_fn
        
        # Start recursive integration
        try:
            result = recursive_integrate(0, {})
            return result
        finally:
            self._fn = original_fn
    

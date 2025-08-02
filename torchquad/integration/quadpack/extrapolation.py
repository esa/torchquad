"""Epsilon extrapolation algorithm for QAGS."""

import numpy as np
from autoray import numpy as anp
from loguru import logger


class EpsilonExtrapolation:
    """Epsilon algorithm for convergence acceleration.
    
    Implements the Wynn epsilon algorithm used in QAGS to accelerate
    convergence of the integration sequence.
    """

    def __init__(self, maxelts=50):
        """Initialize epsilon extrapolation.
        
        Args:
            maxelts (int): Maximum number of elements in epsilon table
        """
        self.maxelts = maxelts
        self.reset()
        
    def reset(self):
        """Reset extrapolation state."""
        self.epstab = []  # Epsilon table
        self.res3la = []  # Last 3 results
        self.nres = 0     # Number of results stored
        
    def add_result(self, result):
        """Add a new integration result to the sequence.
        
        Args:
            result: New integration result
            
        Returns:
            tuple: (extrapolated_result, absolute_error, should_accept)
        """
        epmach = np.finfo(np.float64).eps
        oflow = np.finfo(np.float64).max
        
        self.nres += 1
        abserr = oflow
        accept_result = False
        
        # Store in res3la (last 3 results)
        if len(self.res3la) >= 3:
            self.res3la.pop(0)
        self.res3la.append(result)
        
        # Need at least 3 results for extrapolation
        if self.nres < 3:
            return result, abserr, False
            
        # Epsilon algorithm
        n = self.nres - 1
        
        # Initialize or extend epsilon table
        if len(self.epstab) <= n:
            self.epstab.extend([0.0] * (n + 1 - len(self.epstab)))
            
        self.epstab[n] = result
        
        if n >= 2:
            # Apply epsilon algorithm
            for i in range(n, 1, -1):
                if i >= len(self.epstab):
                    break
                    
                # Compute epsilon[i-1] from epsilon[i] and epsilon[i-1]
                if abs(self.epstab[i] - self.epstab[i-1]) < 1e-15:
                    # Avoid division by zero
                    continue
                    
                aux = 1.0 / (self.epstab[i] - self.epstab[i-1])
                if i >= 2:
                    aux += self.epstab[i-2]
                if i - 1 < len(self.epstab):
                    self.epstab[i-1] = aux
                    
        # Check for convergence acceleration
        if n >= 2 and n < len(self.epstab):
            # Estimate error
            if n >= 4:
                error1 = abs(self.epstab[n] - self.epstab[n-2])
                error2 = abs(self.epstab[n-1] - self.epstab[n-3])
                abserr = min(error1, error2)
            else:
                abserr = abs(self.epstab[n] - self.epstab[n-1])
                
            # Check if we should accept the extrapolated result
            if n >= 4:
                # More stringent test for acceptance
                if abserr <= max(1e-5 * abs(self.epstab[n]), epmach * abs(self.epstab[n])):
                    accept_result = True
            elif abserr <= 1e-3 * abs(self.epstab[n]):
                accept_result = True
                
            result = self.epstab[n]
            
        logger.debug(f"Epsilon extrapolation: n={n}, result={result}, error={abserr}, accept={accept_result}")
        
        return result, abserr, accept_result
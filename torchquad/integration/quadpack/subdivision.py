"""Adaptive subdivision algorithms for QUADPACK."""

import numpy as np
from autoray import numpy as anp
from loguru import logger


class AdaptiveSubdivision:
    """Adaptive subdivision for QAG and QAGS algorithms.
    
    Implements the interval subdivision strategy used in QUADPACK algorithms.
    """

    def __init__(self, limit=50):
        """Initialize subdivision.
        
        Args:
            limit (int): Maximum number of subdivisions
        """
        self.limit = limit
        self.reset()
        
    def reset(self):
        """Reset subdivision state."""
        self.neval = 0
        self.last = 0
        self.alist = []  # Left endpoints
        self.blist = []  # Right endpoints  
        self.rlist = []  # Integral estimates
        self.elist = []  # Error estimates
        self.iord = []   # Error ordering
        
    def add_interval(self, a, b, result, error):
        """Add an interval with its result and error estimate.
        
        Args:
            a, b: Interval endpoints
            result: Integral estimate over [a,b]
            error: Error estimate
        """
        self.alist.append(a)
        self.blist.append(b)
        self.rlist.append(result)
        self.elist.append(error)
        self.last += 1
        
        # Insert into ordered list by error size
        self._insert_ordered(len(self.elist) - 1, error)
        
    def _insert_ordered(self, idx, error):
        """Insert interval index into error-ordered list."""
        # Find position to insert (errors in decreasing order)
        pos = 0
        while pos < len(self.iord) and self.elist[self.iord[pos]] > error:
            pos += 1
        self.iord.insert(pos, idx)
        
    def get_largest_error_interval(self):
        """Get the interval with the largest error estimate.
        
        Returns:
            tuple: (index, a, b, result, error)
        """
        if not self.iord:
            return None
        
        idx = self.iord[0]  # Largest error
        return idx, self.alist[idx], self.blist[idx], self.rlist[idx], self.elist[idx]
        
    def replace_interval(self, idx, a1, b1, r1, e1, a2, b2, r2, e2):
        """Replace interval at idx with two subintervals.
        
        Args:
            idx: Index of interval to replace
            a1, b1, r1, e1: Left subinterval data
            a2, b2, r2, e2: Right subinterval data
        """
        # Remove old interval from ordered list
        self.iord.remove(idx)
        
        # Replace with left subinterval
        self.alist[idx] = a1
        self.blist[idx] = b1
        self.rlist[idx] = r1
        self.elist[idx] = e1
        
        # Add right subinterval
        self.add_interval(a2, b2, r2, e2)
        
        # Re-insert left subinterval in ordered position
        self._insert_ordered(idx, e1)
        
    def get_total_result(self):
        """Get total integral estimate and error.
        
        Returns:
            tuple: (result, error)
        """
        total_result = sum(self.rlist[:self.last])
        total_error = sum(self.elist[:self.last])
        return total_result, total_error
        
    def converged(self, epsabs, epsrel):
        """Check if convergence criteria are satisfied.
        
        Args:
            epsabs: Absolute tolerance
            epsrel: Relative tolerance
            
        Returns:
            bool: True if converged
        """
        result, error = self.get_total_result()
        tolerance = max(epsabs, epsrel * abs(result))
        return error <= tolerance
        
    def should_continue(self):
        """Check if subdivision should continue.
        
        Returns:
            bool: True if should continue subdivision
        """
        return self.last < self.limit and len(self.iord) > 0
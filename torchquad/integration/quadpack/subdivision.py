"""Adaptive subdivision algorithms for QUADPACK."""

import numpy as np
from autoray import numpy as anp
from loguru import logger


class AdaptiveSubdivision:
    """Adaptive subdivision for QAG and QAGS algorithms.
    
    Implements the interval subdivision strategy used in QUADPACK algorithms.
    
    CONSERVATIVE OPTIMIZATION: Only caches sum calculations to avoid expensive 
    repeated summations, while keeping original algorithm logic intact.
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
        
        # OPTIMIZATION: Cache total results to avoid repeated sum calculations
        self._cached_result = 0.0
        self._cached_error = 0.0
        self._cache_valid = True
        
    def add_interval(self, a, b, result, error):
        """Add an interval with its result and error estimate.
        
        CONSERVATIVE OPTIMIZATION: Only caches sums, keeps original ordering logic.
        """
        self.alist.append(a)
        self.blist.append(b)
        self.rlist.append(result)
        self.elist.append(error)
        self.last += 1
        
        # Insert into ordered list by error size (original algorithm)
        self._insert_ordered(len(self.elist) - 1, error)
        
        # OPTIMIZATION: Update cached totals incrementally
        if self._cache_valid:
            self._cached_result += float(result)
            self._cached_error += float(error)
        
    def _insert_ordered(self, idx, error):
        """Insert interval index into error-ordered list."""
        # Find position to insert (errors in decreasing order)
        pos = 0
        while pos < len(self.iord) and self.elist[self.iord[pos]] > error:
            pos += 1
        self.iord.insert(pos, idx)
        
    def get_largest_error_interval(self):
        """Get the interval with the largest error estimate."""
        if not self.iord:
            return None
        
        idx = self.iord[0]  # Largest error
        return idx, self.alist[idx], self.blist[idx], self.rlist[idx], self.elist[idx]
        
    def replace_interval(self, idx, a1, b1, r1, e1, a2, b2, r2, e2):
        """Replace interval at idx with two subintervals.
        
        CONSERVATIVE OPTIMIZATION: Updates cached totals, keeps original list logic.
        """
        # OPTIMIZATION: Update cached totals incrementally before replacement
        if self._cache_valid:
            old_result = float(self.rlist[idx])
            old_error = float(self.elist[idx])
            self._cached_result += float(r1) + float(r2) - old_result
            self._cached_error += float(e1) + float(e2) - old_error
        
        # Remove old interval from ordered list (original algorithm)
        self.iord.remove(idx)
        
        # Replace with left subinterval
        self.alist[idx] = a1
        self.blist[idx] = b1
        self.rlist[idx] = r1
        self.elist[idx] = e1
        
        # Add right subinterval
        self.add_interval(a2, b2, r2, e2)
        
        # Re-insert left subinterval in ordered position (original algorithm)
        self._insert_ordered(idx, e1)
        
    def get_total_result(self):
        """Get total integral estimate and error.
        
        OPTIMIZED: Uses cached values when valid, falls back to computation when needed.
        """
        if self._cache_valid:
            return self._cached_result, self._cached_error
        
        # Fallback: Recompute if cache is invalid
        total_result = sum(float(r) for r in self.rlist[:self.last])
        total_error = sum(float(e) for e in self.elist[:self.last])
        
        # Update cache
        self._cached_result = total_result
        self._cached_error = total_error
        self._cache_valid = True
        
        return total_result, total_error
        
    def converged(self, epsabs, epsrel):
        """Check if convergence criteria are satisfied.
        
        OPTIMIZED: Uses cached totals, avoids repeated float conversions.
        """
        result, error = self.get_total_result()
        tolerance = max(float(epsabs), float(epsrel) * abs(result))
        return error <= tolerance
        
    def should_continue(self):
        """Check if subdivision should continue."""
        return self.last < self.limit and len(self.iord) > 0
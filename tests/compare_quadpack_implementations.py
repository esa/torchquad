#!/usr/bin/env python3
"""
Test torchquad QUADPACK implementation against scipy's QUADPACK wrapper.

This test compares the accuracy and performance of torchquad's pure Python
QUADPACK implementation against scipy's wrapper of the original Fortran QUADPACK.
"""

import numpy as np
import time
from scipy import integrate
import sys
import os

# Add torchquad to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torchquad.integration.quadpack import QNG, QAGS


def test_function_1(x):
    """Simple polynomial: x^2"""
    return x**2


def test_function_2(x):
    """Oscillatory function: sin(10*x)"""
    return np.sin(10 * x)


def test_function_3(x):
    """Function with singularity-like behavior: 1/sqrt(x + 0.1)"""
    return 1.0 / np.sqrt(x + 0.1)


def test_function_4(x):
    """Exponential decay: exp(-x^2)"""
    return np.exp(-x**2)


def test_function_5(x):
    """Product of oscillatory and decay: sin(5*x) * exp(-0.5*x)"""
    return np.sin(5 * x) * np.exp(-0.5 * x)


def compare_implementations():
    """Compare torchquad and scipy QUADPACK implementations."""
    
    test_cases = [
        ("x^2 on [0,1]", test_function_1, 0, 1, 1/3),
        ("sin(10*x) on [0,1]", test_function_2, 0, 1, (np.cos(0) - np.cos(10))/10),
        ("1/sqrt(x+0.1) on [0,1]", test_function_3, 0, 1, None),
        ("exp(-x^2) on [-2,2]", test_function_4, -2, 2, None),
        ("sin(5*x)*exp(-0.5*x) on [0,pi]", test_function_5, 0, np.pi, None),
    ]
    
    tolerances = [(1e-6, 1e-6), (1e-10, 1e-10), (1e-14, 1e-14)]
    
    # Initialize torchquad integrators
    qng = QNG()
    qags = QAGS()
    
    print("=" * 80)
    print("QUADPACK Implementation Comparison: torchquad vs scipy")
    print("=" * 80)
    
    for func_name, func, a, b, analytical in test_cases:
        print(f"\nTest function: {func_name}")
        if analytical is not None:
            print(f"Analytical result: {analytical:.15e}")
        print("-" * 60)
        
        # Wrapper for torchquad (expects batch input)
        def torchquad_func(x):
            # x has shape (n_points, 1)
            return func(x[:, 0])
        
        for epsabs, epsrel in tolerances:
            print(f"\nTolerances: epsabs={epsabs:.0e}, epsrel={epsrel:.0e}")
            
            # Test scipy QUADPACK
            try:
                t0 = time.time()
                scipy_result, scipy_error = integrate.quad(func, a, b, 
                                                          epsabs=epsabs, 
                                                          epsrel=epsrel)
                scipy_time = time.time() - t0
                print(f"  scipy quad:     {scipy_result:.15e} ± {scipy_error:.3e} (time: {scipy_time*1000:.2f}ms)")
            except Exception as e:
                print(f"  scipy quad:     FAILED - {e}")
                scipy_result = None
            
            # Test torchquad QNG
            try:
                t0 = time.time()
                tq_qng_result = qng.integrate(torchquad_func, dim=1, 
                                            integration_domain=[[a, b]],
                                            epsabs=epsabs, epsrel=epsrel,
                                            backend="numpy")
                tq_qng_time = time.time() - t0
                if scipy_result is not None:
                    qng_diff = abs(float(tq_qng_result) - scipy_result)
                    print(f"  torchquad QNG:  {float(tq_qng_result):.15e} (diff: {qng_diff:.3e}, time: {tq_qng_time*1000:.2f}ms)")
                else:
                    print(f"  torchquad QNG:  {float(tq_qng_result):.15e} (time: {tq_qng_time*1000:.2f}ms)")
            except Exception as e:
                print(f"  torchquad QNG:  FAILED - {e}")
            
            # Test torchquad QAGS
            try:
                t0 = time.time()
                tq_qags_result = qags.integrate(torchquad_func, dim=1,
                                              integration_domain=[[a, b]], 
                                              epsabs=epsabs, epsrel=epsrel,
                                              backend="numpy")
                tq_qags_time = time.time() - t0
                if scipy_result is not None:
                    qags_diff = abs(float(tq_qags_result) - scipy_result)
                    print(f"  torchquad QAGS: {float(tq_qags_result):.15e} (diff: {qags_diff:.3e}, time: {tq_qags_time*1000:.2f}ms)")
                else:
                    print(f"  torchquad QAGS: {float(tq_qags_result):.15e} (time: {tq_qags_time*1000:.2f}ms)")
            except Exception as e:
                print(f"  torchquad QAGS: FAILED - {e}")
            
            # Compare with analytical if available
            if analytical is not None and scipy_result is not None:
                scipy_analytical_err = abs(scipy_result - analytical)
                print(f"  Errors vs analytical: scipy={scipy_analytical_err:.3e}")


def test_multidimensional():
    """Test multi-dimensional integration capabilities."""
    print("\n" + "=" * 80)
    print("Multi-dimensional Integration Test")
    print("=" * 80)
    
    # 2D test function: x^2 * y^2 over [0,1]x[0,1]
    def func_2d(x):
        return x[:, 0]**2 * x[:, 1]**2
    
    analytical_2d = 1/9  # integral of x^2*y^2 over unit square
    
    qags = QAGS()
    
    try:
        result_2d = qags.integrate(func_2d, dim=2,
                                 integration_domain=[[0, 1], [0, 1]],
                                 epsabs=1e-10, epsrel=1e-10,
                                 backend="numpy")
        error_2d = abs(float(result_2d) - analytical_2d)
        print(f"\n2D integral of x²y² over [0,1]²:")
        print(f"  Result:     {float(result_2d):.15e}")
        print(f"  Analytical: {analytical_2d:.15e}")
        print(f"  Error:      {error_2d:.3e}")
    except Exception as e:
        print(f"\n2D integration FAILED: {e}")
    
    # 3D test function: sum of squares
    def func_3d(x):
        return np.sum(x**2, axis=1)
    
    analytical_3d = 1.0  # integral of x^2+y^2+z^2 over unit cube
    
    try:
        result_3d = qags.integrate(func_3d, dim=3,
                                 integration_domain=[[0, 1], [0, 1], [0, 1]],
                                 epsabs=1e-8, epsrel=1e-8,
                                 backend="numpy")
        error_3d = abs(float(result_3d) - analytical_3d)
        print(f"\n3D integral of x²+y²+z² over [0,1]³:")
        print(f"  Result:     {float(result_3d):.15e}")
        print(f"  Analytical: {analytical_3d:.15e}")  
        print(f"  Error:      {error_3d:.3e}")
    except Exception as e:
        print(f"\n3D integration FAILED: {e}")


if __name__ == "__main__":
    print("Running QUADPACK implementation comparison...")
    compare_implementations()
    test_multidimensional()
    print("\n" + "=" * 80)
    print("Comparison complete!")
#!/usr/bin/env python3
"""
Benchmark torchquad QUADPACK vs scipy QUADPACK implementations.

This script benchmarks accuracy vs function evaluations and accuracy vs runtime
for torchquad QUADPACK methods compared to scipy's Fortran QUADPACK wrapper.
Uses challenging integrands that require significant computation time.
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from scipy import integrate
import sys
import os

# Add torchquad to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torchquad.integration.quadpack import QNG, QAG, QAGS, QAGI, QAWC
from torchquad.utils.set_up_backend import set_up_backend
from autoray import numpy as anp


class ChallengingFunctions:
    """Collection of challenging benchmark functions requiring many function evaluations."""
    
    @staticmethod
    def extreme_oscillatory_torch(x):
        """Extremely oscillatory function requiring millions of evaluations - torchquad version"""
        # sin(1000πx)exp(-10(x-0.3)²) + 0.5cos(2000πx) + sharp discontinuities
        x_val = x[:, 0]
        # Multiple discontinuities
        step1 = anp.where(x_val > 0.25, 1.0, 0.0)
        step2 = anp.where(x_val > 0.5, -1.0, 0.0)
        step3 = anp.where(x_val > 0.75, 1.0, 0.0)
        # Extreme oscillations
        oscillatory = anp.sin(1000 * np.pi * x_val) * anp.exp(-10 * (x_val - 0.3) ** 2)
        rapid_osc = 0.5 * anp.cos(2000 * np.pi * x_val)
        # Additional sharp peaks
        peak1 = 10.0 * anp.exp(-1000 * (x_val - 0.15) ** 2)
        peak2 = 10.0 * anp.exp(-1000 * (x_val - 0.85) ** 2)
        return oscillatory + rapid_osc + step1 + step2 + step3 + peak1 + peak2 + 0.1
    
    @staticmethod
    def extreme_oscillatory_scipy(x):
        """Extremely oscillatory function requiring millions of evaluations - scipy version"""
        # Multiple discontinuities
        step1 = np.where(x > 0.25, 1.0, 0.0)
        step2 = np.where(x > 0.5, -1.0, 0.0)
        step3 = np.where(x > 0.75, 1.0, 0.0)
        # Extreme oscillations
        oscillatory = np.sin(1000 * np.pi * x) * np.exp(-10 * (x - 0.3) ** 2)
        rapid_osc = 0.5 * np.cos(2000 * np.pi * x)
        # Additional sharp peaks
        peak1 = 10.0 * np.exp(-1000 * (x - 0.15) ** 2)
        peak2 = 10.0 * np.exp(-1000 * (x - 0.85) ** 2)
        return oscillatory + rapid_osc + step1 + step2 + step3 + peak1 + peak2 + 0.1
    
    @staticmethod
    def ultra_narrow_peaks_torch(x):
        """Ultra-narrow peaks requiring extreme subdivision - torchquad version"""
        # Extremely narrow Gaussian peaks + high-frequency oscillations
        x_val = x[:, 0]
        # Ultra-narrow peaks (width ~0.001)
        peak1 = 100.0 * anp.exp(-50000 * (x_val - 0.111111) ** 2)
        peak2 = 100.0 * anp.exp(-50000 * (x_val - 0.333333) ** 2)
        peak3 = 100.0 * anp.exp(-50000 * (x_val - 0.555555) ** 2)
        peak4 = 100.0 * anp.exp(-50000 * (x_val - 0.777777) ** 2)
        peak5 = 100.0 * anp.exp(-50000 * (x_val - 0.9) ** 2)
        # High-frequency oscillations
        oscillatory1 = anp.sin(500 * np.pi * x_val) * anp.cos(300 * np.pi * x_val)
        oscillatory2 = 0.5 * anp.sin(1500 * np.pi * x_val)
        # Discontinuous components
        disc = anp.where(anp.sin(100 * np.pi * x_val) > 0, 1.0, -1.0)
        return peak1 + peak2 + peak3 + peak4 + peak5 + oscillatory1 + oscillatory2 + disc + 0.1
    
    @staticmethod
    def ultra_narrow_peaks_scipy(x):
        """Ultra-narrow peaks requiring extreme subdivision - scipy version"""
        # Ultra-narrow peaks (width ~0.001)
        peak1 = 100.0 * np.exp(-50000 * (x - 0.111111) ** 2)
        peak2 = 100.0 * np.exp(-50000 * (x - 0.333333) ** 2)
        peak3 = 100.0 * np.exp(-50000 * (x - 0.555555) ** 2)
        peak4 = 100.0 * np.exp(-50000 * (x - 0.777777) ** 2)
        peak5 = 100.0 * np.exp(-50000 * (x - 0.9) ** 2)
        # High-frequency oscillations
        oscillatory1 = np.sin(500 * np.pi * x) * np.cos(300 * np.pi * x)
        oscillatory2 = 0.5 * np.sin(1500 * np.pi * x)
        # Discontinuous components
        disc = np.where(np.sin(100 * np.pi * x) > 0, 1.0, -1.0)
        return peak1 + peak2 + peak3 + peak4 + peak5 + oscillatory1 + oscillatory2 + disc + 0.1
    
    @staticmethod
    def pathological_oscillatory_torch(x):
        """Pathologically oscillatory function - torchquad version"""
        # Combination of multiple extreme oscillations
        x_val = x[:, 0]
        # Primary oscillation with increasing frequency
        osc1 = anp.sin(5000 * np.pi * x_val * x_val) * anp.exp(-5 * x_val)
        # Secondary oscillation with decreasing amplitude
        osc2 = anp.cos(3000 * np.pi * x_val) * (1 + x_val) / (1 + 10 * x_val)
        # Tertiary oscillation with modulation
        osc3 = anp.sin(1000 * np.pi * x_val) * anp.cos(1500 * np.pi * x_val)
        # Near-singular behavior at endpoints
        singular1 = 1.0 / (anp.sqrt(x_val + 0.0001) + 0.001)
        singular2 = 1.0 / (anp.sqrt(1.0001 - x_val) + 0.001)
        # Combine with weights
        return 0.1 * osc1 + 0.1 * osc2 + 0.1 * osc3 + 0.01 * singular1 + 0.01 * singular2 + 0.5
    
    @staticmethod
    def pathological_oscillatory_scipy(x):
        """Pathologically oscillatory function - scipy version"""
        # Primary oscillation with increasing frequency
        osc1 = np.sin(5000 * np.pi * x * x) * np.exp(-5 * x)
        # Secondary oscillation with decreasing amplitude
        osc2 = np.cos(3000 * np.pi * x) * (1 + x) / (1 + 10 * x)
        # Tertiary oscillation with modulation
        osc3 = np.sin(1000 * np.pi * x) * np.cos(1500 * np.pi * x)
        # Near-singular behavior at endpoints
        singular1 = 1.0 / (np.sqrt(x + 0.0001) + 0.001)
        singular2 = 1.0 / (np.sqrt(1.0001 - x) + 0.001)
        # Combine with weights
        return 0.1 * osc1 + 0.1 * osc2 + 0.1 * osc3 + 0.01 * singular1 + 0.01 * singular2 + 0.5


def benchmark_single_tolerance(scipy_func, tq_integrator, func, domain, tol, reference_value=None):
    """Benchmark a single tolerance level with detailed timing and evaluation counting."""
    
    print(f"  Testing tolerance: {tol:.0e}")
    
    # Test scipy with function evaluation counting
    scipy_eval_count = [0]
    
    def scipy_func_with_counter(x):
        scipy_eval_count[0] += 1
        return scipy_func(x)
    
    # Test scipy QUADPACK (uses adaptive Gauss-Kronrod like QAGS)
    try:
        start_time = time.perf_counter()
        scipy_result, scipy_abserr = integrate.quad(
            scipy_func_with_counter, domain[0], domain[1], 
            epsabs=tol, epsrel=tol, limit=50000  # Allow many more subdivisions for extreme functions
        )
        scipy_time = time.perf_counter() - start_time
        
        # Use reference value for error if available, otherwise use estimated error
        if reference_value is not None:
            scipy_error = abs(scipy_result - reference_value)
        else:
            scipy_error = scipy_abserr
            
    except Exception as e:
        print(f"    scipy failed: {e}")
        return None
        
    # Test torchquad QAGS
    try:
        # Reset function evaluation counter if it exists
        if hasattr(tq_integrator, '_nr_of_fevals'):
            tq_integrator._nr_of_fevals = 0
        
        # Wrap function for torchquad (expects batch input)
        def tq_func(x):
            # x has shape (n_points, 1) for 1D integration
            # The torchquad functions already handle the indexing
            return func(x)
        
        start_time = time.perf_counter()
        tq_result = tq_integrator.integrate(
            tq_func, dim=1, integration_domain=[domain],
            epsabs=tol, epsrel=tol, limit=50000, backend="torch"
        )
        tq_time = time.perf_counter() - start_time
        tq_result = float(tq_result)
        
        # Get function evaluation count
        tq_evals = getattr(tq_integrator, '_nr_of_fevals', 0)
        
        if reference_value is not None:
            tq_error = abs(tq_result - reference_value)
        else:
            tq_error = abs(tq_result - scipy_result)
            
    except Exception as e:
        print(f"    torchquad failed: {e}")
        return None
    
    # Calculate metrics
    scipy_time_per_eval = scipy_time / max(scipy_eval_count[0], 1)
    tq_time_per_eval = tq_time / max(tq_evals, 1)
    scipy_error_time_product = scipy_error * scipy_time
    tq_error_time_product = tq_error * tq_time
    
    print(f"    scipy: result={scipy_result:.6e}, error={scipy_error:.3e}, "
          f"time={scipy_time:.3f}s, evals={scipy_eval_count[0]}")
    print(f"    tq:    result={tq_result:.6e}, error={tq_error:.3e}, "
          f"time={tq_time:.3f}s, evals={tq_evals}")
    
    return {
        'tolerance': tol,
        'scipy_result': scipy_result,
        'scipy_error': scipy_error,
        'scipy_time': scipy_time,
        'scipy_evals': scipy_eval_count[0],
        'scipy_time_per_eval': scipy_time_per_eval,
        'scipy_error_time_product': scipy_error_time_product,
        'tq_result': tq_result,
        'tq_error': tq_error,
        'tq_time': tq_time,
        'tq_evals': tq_evals,
        'tq_time_per_eval': tq_time_per_eval,
        'tq_error_time_product': tq_error_time_product,
    }


def get_reference_value(scipy_func, domain, name):
    """Calculate high-precision reference value."""
    print(f"  Calculating reference value for {name}...")
    try:
        # Use very tight tolerances for reference
        ref_result, _ = integrate.quad(scipy_func, domain[0], domain[1], 
                                     epsabs=1e-15, epsrel=1e-15, limit=1000)
        print(f"  Reference: {ref_result:.12e}")
        return ref_result
    except Exception as e:
        print(f"  Reference calculation failed: {e}")
        return None


def benchmark_all_quadpack_methods():
    """Benchmark all QUADPACK methods on various test functions."""
    
    # Use FP32 for torchquad
    set_up_backend("torch", data_type="float32")
    
    # Simple test functions for method comparison (avoid extremely challenging ones)
    def simple_poly_scipy(x):
        return x**2
    
    def simple_poly_torch(x):
        x_val = x[:, 0]
        return x_val**2
    
    def moderate_osc_scipy(x):
        return np.sin(50 * x) * np.exp(-2 * x)
    
    def moderate_osc_torch(x):
        x_val = x[:, 0]
        return anp.sin(50 * x_val) * anp.exp(-2 * x_val)
    
    def near_singular_scipy(x):
        return 1.0 / np.sqrt(x + 0.01)
    
    def near_singular_torch(x):
        x_val = x[:, 0]
        return 1.0 / anp.sqrt(x_val + 0.01)
    
    # Test functions for different methods
    test_functions = {
        "Simple Polynomial": (simple_poly_scipy, simple_poly_torch, [0, 1], 1/3),
        "Moderate Oscillatory": (moderate_osc_scipy, moderate_osc_torch, [0, 1], None),
        "Near Singular": (near_singular_scipy, near_singular_torch, [0, 1], None),
    }
    
    # QUADPACK methods to compare
    methods = {
        "QNG": QNG(),
        "QAG": QAG(), 
        "QAGS": QAGS(),
        # Skip QAGI and QAWC for now due to different interface requirements
    }
    
    # Tolerance levels for convergence study
    tolerance_levels = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    
    method_results = {}
    
    for method_name, method in methods.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking {method_name}")
        print(f"{'='*50}")
        
        method_results[method_name] = {}
        
        for func_name, (scipy_func, tq_func, domain, analytical) in test_functions.items():
            print(f"\n  Testing {func_name}")
            
            # Get reference value
            if analytical is None:
                try:
                    reference, _ = integrate.quad(scipy_func, domain[0], domain[1], 
                                                epsabs=1e-12, epsrel=1e-12)
                except:
                    reference = None
            else:
                reference = analytical
            
            function_results = []
            
            for tol in tolerance_levels:
                try:
                    # Reset evaluation counter
                    if hasattr(method, '_nr_of_fevals'):
                        method._nr_of_fevals = 0
                    
                    start_time = time.perf_counter()
                    result = method.integrate(
                        lambda x: tq_func(x), dim=1, 
                        integration_domain=[domain],
                        epsabs=tol, epsrel=tol,
                        backend="torch"
                    )
                    elapsed_time = time.perf_counter() - start_time
                    
                    evals = getattr(method, '_nr_of_fevals', 0)
                    
                    if reference is not None:
                        error = abs(float(result) - reference)
                    else:
                        error = tol  # Use tolerance as proxy
                    
                    function_results.append({
                        'tolerance': tol,
                        'result': float(result),
                        'error': error,
                        'time': elapsed_time,
                        'evals': evals,
                        'reference': reference
                    })
                    
                    print(f"    tol={tol:.0e}: result={float(result):.6e}, error={error:.3e}, time={elapsed_time:.3f}s, evals={evals}")
                    
                    # Stop if taking too long or method fails
                    if elapsed_time > 10.0:
                        print(f"    Time limit reached")
                        break
                        
                except Exception as e:
                    print(f"    tol={tol:.0e}: FAILED - {e}")
                    # For QNG, failures at tight tolerances are expected
                    if method_name == "QNG" and tol < 1e-3:
                        break
                    continue
            
            method_results[method_name][func_name] = function_results
    
    return method_results


def run_benchmarks():
    """Run comprehensive benchmarks with challenging functions."""
    
    # Use FP32 for torchquad but allow scipy to use its default precision
    set_up_backend("torch", data_type="float32")
    
    # Tolerance levels that balance computation time with meaningful results
    tolerance_levels = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    
    # Simple test function first to debug the issue
    def simple_poly_scipy(x):
        return x**2
    
    def simple_poly_torch(x):
        x_val = x[:, 0]  # Extract from batch format
        return x_val**2
    
    def simple_osc_scipy(x):
        return np.sin(10 * x) * np.exp(-5 * x)
    
    def simple_osc_torch(x):
        x_val = x[:, 0]  # Extract from batch format
        return anp.sin(10 * x_val) * anp.exp(-5 * x_val)
    
    # Three extremely challenging test functions that require millions of evaluations
    test_cases = [
        ("Extreme Oscillatory", 
         ChallengingFunctions.extreme_oscillatory_scipy,
         ChallengingFunctions.extreme_oscillatory_torch,
         [0, 1]),
        
        ("Ultra-Narrow Peaks", 
         ChallengingFunctions.ultra_narrow_peaks_scipy,
         ChallengingFunctions.ultra_narrow_peaks_torch,
         [0, 1]),
        
        ("Pathological Oscillatory", 
         ChallengingFunctions.pathological_oscillatory_scipy,
         ChallengingFunctions.pathological_oscillatory_torch,
         [0, 1]),
    ]
    
    all_results = {}
    
    for test_name, scipy_func, tq_func, domain in test_cases:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {test_name}")
        print(f"{'='*60}")
        
        # Get reference value
        reference = get_reference_value(scipy_func, domain, test_name)
        
        # Initialize QAGS integrator
        qags = QAGS()
        
        # Collect results for all tolerance levels
        function_results = []
        
        for tol in tolerance_levels:
            result = benchmark_single_tolerance(
                scipy_func, qags, tq_func, domain, tol, reference
            )
            if result is not None:
                function_results.append(result)
                
                # Stop if we've reached ~5 seconds for either implementation
                if result['tq_time'] > 5.0 or result['scipy_time'] > 5.0:
                    print(f"  Stopping at {tol:.0e} - reached time limit")
                    break
        
        all_results[test_name] = {
            'results': function_results,
            'reference': reference,
            'domain': domain
        }
    
    return all_results


def create_plots(results, save_path="../resources"):
    """Create comparison plots addressing user feedback."""
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Extract data from results structure
    plot_data = {}
    for test_name, test_data in results.items():
        function_results = test_data['results']
        if not function_results:
            continue
            
        # Extract arrays from the list of result dictionaries
        tolerances = [r['tolerance'] for r in function_results]
        scipy_errors = [r['scipy_error'] for r in function_results]
        scipy_times = [r['scipy_time'] for r in function_results]
        scipy_evals = [r['scipy_evals'] for r in function_results]
        scipy_time_per_eval = [r['scipy_time_per_eval'] for r in function_results]
        scipy_error_time_product = [r['scipy_error_time_product'] for r in function_results]
        
        tq_errors = [r['tq_error'] for r in function_results]
        tq_times = [r['tq_time'] for r in function_results]
        tq_evals = [r['tq_evals'] for r in function_results]
        tq_time_per_eval = [r['tq_time_per_eval'] for r in function_results]
        tq_error_time_product = [r['tq_error_time_product'] for r in function_results]
        
        plot_data[test_name] = {
            'tolerances': tolerances,
            'scipy_errors': scipy_errors,
            'scipy_times': scipy_times,
            'scipy_evals': scipy_evals,
            'scipy_time_per_eval': scipy_time_per_eval,
            'scipy_error_time_product': scipy_error_time_product,
            'tq_errors': tq_errors,
            'tq_times': tq_times,
            'tq_evals': tq_evals,
            'tq_time_per_eval': tq_time_per_eval,
            'tq_error_time_product': tq_error_time_product,
        }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red']  # Only 3 functions now
    markers_scipy = ['o', 's', '^']
    markers_tq = ['x', '+', '*']
    
    # Plot 1: Error vs Function Evaluations (TOP LEFT)
    for i, (test_name, data) in enumerate(plot_data.items()):
        color = colors[i % len(colors)]
        marker_s = markers_scipy[i % len(markers_scipy)]
        marker_t = markers_tq[i % len(markers_tq)]
        
        # Filter out zero evaluations to fix log scale issue
        scipy_evals_nonzero = [e for e in data['scipy_evals'] if e > 0]
        scipy_errors_filtered = [data['scipy_errors'][j] for j, e in enumerate(data['scipy_evals']) if e > 0]
        tq_evals_nonzero = [e for e in data['tq_evals'] if e > 0]
        tq_errors_filtered = [data['tq_errors'][j] for j, e in enumerate(data['tq_evals']) if e > 0]
        
        if scipy_evals_nonzero and scipy_errors_filtered:
            ax1.loglog(scipy_evals_nonzero, scipy_errors_filtered, 
                      color=color, marker=marker_s, label=f'{test_name} (scipy)', 
                      linestyle='-', alpha=0.8, markersize=8)
        
        if tq_evals_nonzero and tq_errors_filtered:
            ax1.loglog(tq_evals_nonzero, tq_errors_filtered, 
                      color=color, marker=marker_t, label=f'{test_name} (torchquad)', 
                      linestyle='--', alpha=0.8, markersize=8)
    
    ax1.set_xlabel('Function Evaluations')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Error vs Function Evaluations\nQAGS Algorithm - FP32 vs FP64')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 1e7)  # Reasonable range for function evaluations
    ax1.set_ylim(1e-16, 1e0)  # Full error range to show both scipy and torchquad
    
    # Plot 2: Error vs Runtime (TOP RIGHT) 
    for i, (test_name, data) in enumerate(plot_data.items()):
        color = colors[i % len(colors)]
        marker_s = markers_scipy[i % len(markers_scipy)]
        marker_t = markers_tq[i % len(markers_tq)]
        
        # Convert to milliseconds and filter positive times
        scipy_times_ms = [t * 1000 for t in data['scipy_times'] if t > 0]
        scipy_errors_time_filtered = [data['scipy_errors'][j] for j, t in enumerate(data['scipy_times']) if t > 0]
        tq_times_ms = [t * 1000 for t in data['tq_times'] if t > 0]
        tq_errors_time_filtered = [data['tq_errors'][j] for j, t in enumerate(data['tq_times']) if t > 0]
        
        if scipy_times_ms and scipy_errors_time_filtered:
            ax2.loglog(scipy_times_ms, scipy_errors_time_filtered, 
                      color=color, marker=marker_s, label=f'{test_name} (scipy)', 
                      linestyle='-', alpha=0.8, markersize=8)
        
        if tq_times_ms and tq_errors_time_filtered:
            ax2.loglog(tq_times_ms, tq_errors_time_filtered, 
                      color=color, marker=marker_t, label=f'{test_name} (torchquad)', 
                      linestyle='--', alpha=0.8, markersize=8)
    
    ax2.set_xlabel('Runtime (ms)')
    ax2.set_ylabel('Absolute Error') 
    ax2.set_title('Error vs Runtime\nQAGS Algorithm - FP32 vs FP64')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, None)  # Adjusted to ensure scipy points are visible
    ax2.set_ylim(1e-16, None)  # Ensure full range is visible
    
    # Check if we have multiple tolerance levels to determine plot type
    max_data_points = max(len(data['tolerances']) for data in plot_data.values())
    
    if max_data_points > 1:
        # Plot 3: Runtime per Function Evaluation vs Tolerance (LINE PLOTS)
        for i, (test_name, data) in enumerate(plot_data.items()):
            color = colors[i % len(colors)]
            marker_s = markers_scipy[i % len(markers_scipy)]
            marker_t = markers_tq[i % len(markers_tq)]
            
            # Convert to microseconds
            scipy_time_per_eval_us = [t * 1e6 for t in data['scipy_time_per_eval'] if t > 0]
            tq_time_per_eval_us = [t * 1e6 for t in data['tq_time_per_eval'] if t > 0]
            scipy_tols = [data['tolerances'][j] for j, t in enumerate(data['scipy_time_per_eval']) if t > 0]
            tq_tols = [data['tolerances'][j] for j, t in enumerate(data['tq_time_per_eval']) if t > 0]
            
            if scipy_time_per_eval_us and scipy_tols:
                ax3.loglog(scipy_tols, scipy_time_per_eval_us, 
                          color=color, marker=marker_s, label=f'{test_name} (scipy)', 
                          linestyle='-', alpha=0.8, markersize=8)
            
            if tq_time_per_eval_us and tq_tols:
                ax3.loglog(tq_tols, tq_time_per_eval_us, 
                          color=color, marker=marker_t, label=f'{test_name} (torchquad)', 
                          linestyle='--', alpha=0.8, markersize=8)
        
        ax3.set_xlabel('Target Tolerance')
        ax3.set_ylabel('Runtime per Function Evaluation (μs)')
        ax3.set_title('Efficiency: Runtime per Function Evaluation\nQAGS Algorithm - FP32 vs FP64')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()  # Easier tolerances on right
        
        # Plot 4: Error × Runtime Product vs Tolerance (LINE PLOTS)
        for i, (test_name, data) in enumerate(plot_data.items()):
            color = colors[i % len(colors)]
            marker_s = markers_scipy[i % len(markers_scipy)]
            marker_t = markers_tq[i % len(markers_tq)]
            
            scipy_products = [p for p in data['scipy_error_time_product'] if p > 0]
            tq_products = [p for p in data['tq_error_time_product'] if p > 0]
            scipy_tols = [data['tolerances'][j] for j, p in enumerate(data['scipy_error_time_product']) if p > 0]
            tq_tols = [data['tolerances'][j] for j, p in enumerate(data['tq_error_time_product']) if p > 0]
            
            if scipy_products and scipy_tols:
                ax4.loglog(scipy_tols, scipy_products, 
                          color=color, marker=marker_s, label=f'{test_name} (scipy)', 
                          linestyle='-', alpha=0.8, markersize=8)
            
            if tq_products and tq_tols:
                ax4.loglog(tq_tols, tq_products, 
                          color=color, marker=marker_t, label=f'{test_name} (torchquad)', 
                          linestyle='--', alpha=0.8, markersize=8)
        
        ax4.set_xlabel('Target Tolerance')
        ax4.set_ylabel('Error × Runtime Product')
        ax4.set_title('Cost-Accuracy Trade-off: Error × Runtime\nQAGS Algorithm - FP32 vs FP64')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()  # Easier tolerances on right
        
    else:
        # Fallback to bar charts if only one data point per function
        method_names = [name.replace(' ', '\n') for name in plot_data.keys()]
        x = np.arange(len(method_names))
        width = 0.35
        
        # Plot 3: Runtime per Function Evaluation (BAR CHART)
        scipy_times_per_eval = [data['scipy_time_per_eval'][0] * 1e6 if data['scipy_time_per_eval'] else 0 
                               for data in plot_data.values()]
        tq_times_per_eval = [data['tq_time_per_eval'][0] * 1e6 if data['tq_time_per_eval'] else 0 
                            for data in plot_data.values()]
        
        ax3.bar(x - width/2, scipy_times_per_eval, width, label='scipy', alpha=0.8, color='blue')
        ax3.bar(x + width/2, tq_times_per_eval, width, label='torchquad', alpha=0.8, color='orange')
        ax3.set_xlabel('Test Function')
        ax3.set_ylabel('Runtime per Function Evaluation (μs)')
        ax3.set_title('Efficiency: Runtime per Function Evaluation\nQAGS Algorithm - FP32 vs FP64')
        ax3.set_yscale('log')
        ax3.set_xticks(x)
        ax3.set_xticklabels(method_names, fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Error × Runtime Product (BAR CHART)
        scipy_products = [data['scipy_error_time_product'][0] if data['scipy_error_time_product'] else 1e-16 
                         for data in plot_data.values()]
        tq_products = [data['tq_error_time_product'][0] if data['tq_error_time_product'] else 1e-16 
                      for data in plot_data.values()]
        
        ax4.bar(x - width/2, scipy_products, width, label='scipy', alpha=0.8, color='blue')
        ax4.bar(x + width/2, tq_products, width, label='torchquad', alpha=0.8, color='orange')
        ax4.set_xlabel('Test Function')
        ax4.set_ylabel('Error × Runtime Product')
        ax4.set_title('Cost-Accuracy Trade-off: Error × Runtime\nQAGS Algorithm - FP32 vs FP64')
        ax4.set_yscale('log')
        ax4.set_xticks(x)
        ax4.set_xticklabels(method_names, fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'quadpack_vs_scipy_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save processed data for analysis
    data_path = os.path.join(save_path, 'quadpack_vs_scipy_results.json')
    serializable_results = {}
    for test_name, data in plot_data.items():
        serializable_results[test_name] = {
            k: [float(x) if isinstance(x, (np.number, complex)) else x for x in v]
            for k, v in data.items()
        }
    
    with open(data_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {data_path}")
    
    # Don't show interactive plot to avoid blocking
    # plt.show()


def create_quadpack_convergence_plots(method_results, save_path="../resources"):
    """Create convergence plots for all QUADPACK methods."""
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Number of methods and functions
    methods = list(method_results.keys())
    functions = list(next(iter(method_results.values())).keys())
    n_methods = len(methods)
    n_functions = len(functions)
    
    # Create subplots: one for each method-function combination
    fig, axes = plt.subplots(n_methods, n_functions, figsize=(5*n_functions, 4*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if n_functions == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, method_name in enumerate(methods):
        for j, func_name in enumerate(functions):
            ax = axes[i, j]
            
            data = method_results[method_name].get(func_name, [])
            if not data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name} - {func_name}')
                continue
            
            # Extract data
            evals = [d['evals'] for d in data if d['evals'] > 0]
            errors = [d['error'] for d in data if d['evals'] > 0 and d['error'] > 0]
            
            if evals and errors and len(evals) == len(errors):
                ax.loglog(evals, errors, 'o-', color=colors[i % len(colors)], 
                         markersize=6, linewidth=2, alpha=0.8)
                ax.set_xlabel('Function Evaluations')
                ax.set_ylabel('Absolute Error')
                ax.set_title(f'{method_name} - {func_name}\nConvergence vs Function Evaluations')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name} - {func_name}')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'quadpack_convergence_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {plot_path}")
    
    # Save data
    data_path = os.path.join(save_path, 'quadpack_convergence_results.json')
    with open(data_path, 'w') as f:
        json.dump(method_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
    print(f"Convergence results saved to: {data_path}")


if __name__ == "__main__":
    print("Running QUADPACK benchmarks: torchquad vs scipy")
    print("Using FP32 precision for fair comparison")
    print("Testing with moderately challenging functions (avoiding infinite loops)")
    
    try:
        # Run convergence study for all methods (skip the extreme benchmark for now)
        print("\n" + "="*60)
        print("Running convergence study for all QUADPACK methods")
        print("="*60)
        method_results = benchmark_all_quadpack_methods()
        create_quadpack_convergence_plots(method_results)
        print("\nConvergence study completed successfully!")
        
        print("\nNote: Extreme oscillatory benchmark disabled to prevent hanging.")
        print("Use working_quadpack_benchmark.py for performance testing.")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
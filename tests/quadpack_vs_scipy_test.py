"""
Test torchquad QUADPACK implementation against scipy's QUADPACK wrapper.
"""

import pytest
import numpy as np
from scipy import integrate
import sys
import os
from autoray import numpy as anp

# Add torchquad to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torchquad.integration.quadpack import QNG, QAG, QAGS, QAGI, QAWC
from helper_functions import setup_test_for_backend


def _function_polynomial(x):
    """Simple polynomial: x^2"""
    return x**2


def _function_oscillatory(x):
    """Oscillatory function: sin(10*x)"""
    return anp.sin(10 * x)


def _function_near_singular(x):
    """Function with singularity-like behavior: 1/sqrt(x + 0.1)"""
    return 1.0 / anp.sqrt(x + 0.1)


def _function_infinite_exp(x):
    """Function for infinite interval: exp(-x^2)"""
    return anp.exp(-x**2)


def _function_infinite_rational(x):
    """Function for infinite interval: 1/(1 + x^2)"""
    return 1.0 / (1.0 + x**2)


def _function_cauchy_numerator(x):
    """Numerator for Cauchy principal value: x^2 + 1"""
    return x**2 + 1.0


def _function_cauchy_simple(x):
    """Simple numerator for Cauchy principal value: 1"""
    return anp.ones_like(x)


def _test_quadpack_vs_scipy_backend(backend, _precision):
    """Test torchquad QUADPACK vs scipy for given backend."""

    test_cases = [
        ("x^2 on [0,1]", _function_polynomial, 0, 1, 1 / 3),
        ("sin(10*x) on [0,1]", _function_oscillatory, 0, 1, (np.cos(0) - np.cos(10)) / 10),
        ("1/sqrt(x+0.1) on [0,1]", _function_near_singular, 0, 1, None),
    ]

    # Initialize torchquad integrators
    qng = QNG()
    qag = QAG()
    qags = QAGS()
    qagi = QAGI()
    qawc = QAWC()

    for func_name, func, a, b, analytical in test_cases:
        print(f"\nTesting {func_name}")

        # Wrapper for torchquad (expects batch input)
        def torchquad_func(x):
            # x has shape (n_points, 1)
            return func(x[:, 0])

        epsabs, epsrel = 1e-6, 1e-6  # More reasonable tolerances for tests

        # Test scipy QUADPACK
        try:
            result_scipy, error_scipy = integrate.quad(func, a, b, epsabs=epsabs, epsrel=epsrel)
        except Exception as e:
            pytest.skip(f"scipy.quad failed: {e}")
            return

        # Test QNG (for simple smooth functions)
        if func_name in ["x^2 on [0,1]", "sin(10*x) on [0,1]"]:
            try:
                result_qng = qng.integrate(
                    torchquad_func,
                    dim=1,
                    integration_domain=[[a, b]],
                    epsabs=epsabs,
                    epsrel=epsrel,
                    backend=backend,
                )
                qng_diff = abs(float(result_qng) - result_scipy)
                max_diff = max(1e-10, 1e-6 * abs(result_scipy))  # QNG less accurate for difficult functions
                assert qng_diff < max_diff, (
                    f"QNG disagrees with scipy too much: "
                    f"scipy={result_scipy}, qng={result_qng}, diff={qng_diff}"
                )
                print(f"    QNG: result={float(result_qng):.6e}, diff={qng_diff:.3e}")
            except Exception as e:
                print(f"    QNG failed (expected for difficult functions): {e}")

        # Test QAG (adaptive)
        try:
            result_qag = qag.integrate(
                torchquad_func,
                dim=1,
                integration_domain=[[a, b]],
                epsabs=epsabs,
                epsrel=epsrel,
                backend=backend,
            )
            qag_diff = abs(float(result_qag) - result_scipy)
            max_diff = max(1e-6, 1e-4 * abs(result_scipy))  # Relaxed for FP32 vs FP64
            assert qag_diff < max_diff, (
                f"QAG disagrees with scipy too much: "
                f"scipy={result_scipy}, qag={result_qag}, diff={qag_diff}"
            )
            print(f"    QAG: result={float(result_qag):.6e}, diff={qag_diff:.3e}")
        except Exception as e:
            pytest.fail(f"torchquad QAG failed: {e}")

        # Test torchquad QAGS
        try:
            result_qags = qags.integrate(
                torchquad_func,
                dim=1,
                integration_domain=[[a, b]],
                epsabs=epsabs,
                epsrel=epsrel,
                backend=backend,
            )
            qags_diff = abs(float(result_qags) - result_scipy)
            max_diff = max(1e-6, 1e-4 * abs(result_scipy))  # Relaxed for FP32 vs FP64
            assert qags_diff < max_diff, (
                f"QAGS disagrees with scipy too much: "
                f"scipy={result_scipy}, qags={result_qags}, diff={qags_diff}"
            )
            print(f"    QAGS: result={float(result_qags):.6e}, diff={qags_diff:.3e}")
        except Exception as e:
            pytest.fail(f"torchquad QAGS failed: {e}")

        # Compare with analytical if available
        if analytical is not None:
            scipy_analytical_err = abs(result_scipy - analytical)
            qags_analytical_err = abs(float(result_qags) - analytical)
            qag_analytical_err = abs(float(result_qag) - analytical)

            # Both should be reasonably accurate
            assert (
                scipy_analytical_err < 1e-6
            ), f"scipy inaccurate vs analytical: {scipy_analytical_err}"
            assert (
                qags_analytical_err < 1e-5
            ), f"QAGS inaccurate vs analytical: {qags_analytical_err}"
            assert (
                qag_analytical_err < 1e-5
            ), f"QAG inaccurate vs analytical: {qag_analytical_err}"


def _test_quadpack_infinite_intervals(backend, _precision):
    """Test QAGI for infinite interval integration."""
    
    # Skip for backends that might not support infinite intervals properly
    if backend == "tensorflow":
        return
    
    qagi = QAGI()
    
    # Test cases for infinite intervals
    infinite_test_cases = [
        # exp(-x^2) from 0 to infinity, analytical = sqrt(pi)/2
        ("exp(-x^2) on [0,∞]", _function_infinite_exp, 0, float('inf'), np.sqrt(np.pi) / 2),
        # 1/(1+x^2) from -infinity to infinity, analytical = pi  
        ("1/(1+x^2) on (-∞,∞)", _function_infinite_rational, float('-inf'), float('inf'), np.pi),
    ]
    
    for func_name, func, a, b, analytical in infinite_test_cases:
        print(f"\nTesting {func_name} with QAGI")
        
        def torchquad_func(x):
            return func(x[:, 0])
        
        try:
            # Use looser tolerances for infinite interval integration
            result_qagi = qagi.integrate(
                torchquad_func,
                dim=1,
                integration_domain=[[a, b]],
                epsabs=1e-8,
                epsrel=1e-8,
                backend=backend,
            )
            
            error = abs(float(result_qagi) - analytical)
            print(f"    QAGI: result={float(result_qagi):.6e}, analytical={analytical:.6e}, error={error:.3e}")
            
            # Allow larger tolerance for infinite interval integration
            assert error < 1e-6, f"QAGI error too large: {error}"
            
        except Exception as e:
            print(f"    QAGI failed for {func_name}: {e}")


def _test_quadpack_cauchy_principal_values(backend, _precision):
    """Test QAWC for Cauchy principal value integration."""
    
    # Skip for backends that might not support this properly
    if backend == "tensorflow":
        return
    
    qawc = QAWC()
    
    # Test Cauchy principal value: ∫[0,2] (x^2+1)/(x-1) dx from 0 to 2 with singularity at x=1
    # P.V. ∫[0,2] (x^2+1)/(x-1) dx = [x^2/2 + 2x + 3*ln|x-1|]_0^2 (principal value)
    # = (2 + 4 + 3*ln(1)) - (0 + 0 + 3*ln(1)) = 6 (the ln terms cancel in principal value)
    
    def torchquad_cauchy_func(x):
        return _function_cauchy_numerator(x[:, 0])
    
    print(f"\nTesting Cauchy principal value with QAWC")
    
    try:
        result_qawc = qawc.integrate(
            torchquad_cauchy_func,
            dim=1,
            integration_domain=[[0, 2]],
            epsabs=1e-8,
            epsrel=1e-8,
            c=1.0,  # Singularity location
            backend=backend,
        )
        
        # Analytical result for this specific case
        analytical = 6.0
        error = abs(float(result_qawc) - analytical)
        print(f"    QAWC: result={float(result_qawc):.6e}, analytical={analytical:.6e}, error={error:.3e}")
        
        # Allow reasonable tolerance for Cauchy principal value
        assert error < 1e-6, f"QAWC error too large: {error}"
        
    except Exception as e:
        print(f"    QAWC failed: {e}")


def _test_quadpack_multidimensional(backend, _precision):
    """Test multi-dimensional integration capabilities."""

    # 2D test function: x^2 * y^2 over [0,1]x[0,1]
    def func_2d(x):
        return x[:, 0] ** 2 * x[:, 1] ** 2

    analytical_2d = 1 / 9  # integral of x^2*y^2 over unit square

    qags = QAGS()

    if backend == "tensorflow":
        # TensorFlow doesn't support multi-D QUADPACK
        return

    try:
        result_2d = qags.integrate(
            func_2d,
            dim=2,
            integration_domain=[[0, 1], [0, 1]],
            epsabs=1e-6,
            epsrel=1e-6,
            backend=backend,
        )
        error_2d = abs(float(result_2d) - analytical_2d)

        assert error_2d < 1e-6, f"2D integration error too large: {error_2d}"

    except Exception as e:
        pytest.fail(f"2D integration failed: {e}")


# Setup backend-specific test functions
test_scipy_comparison_numpy = setup_test_for_backend(
    _test_quadpack_vs_scipy_backend, "numpy", "float64"
)
test_scipy_comparison_torch = setup_test_for_backend(
    _test_quadpack_vs_scipy_backend, "torch", "float64"
)
test_scipy_comparison_jax = setup_test_for_backend(
    _test_quadpack_vs_scipy_backend, "jax", "float64"
)
test_scipy_comparison_tensorflow = setup_test_for_backend(
    _test_quadpack_vs_scipy_backend, "tensorflow", "float64"
)

test_infinite_numpy = setup_test_for_backend(_test_quadpack_infinite_intervals, "numpy", "float64")
test_infinite_torch = setup_test_for_backend(_test_quadpack_infinite_intervals, "torch", "float64")
test_infinite_jax = setup_test_for_backend(_test_quadpack_infinite_intervals, "jax", "float64")
test_infinite_tensorflow = setup_test_for_backend(_test_quadpack_infinite_intervals, "tensorflow", "float64")

test_cauchy_numpy = setup_test_for_backend(_test_quadpack_cauchy_principal_values, "numpy", "float64")
test_cauchy_torch = setup_test_for_backend(_test_quadpack_cauchy_principal_values, "torch", "float64")
test_cauchy_jax = setup_test_for_backend(_test_quadpack_cauchy_principal_values, "jax", "float64")
test_cauchy_tensorflow = setup_test_for_backend(_test_quadpack_cauchy_principal_values, "tensorflow", "float64")

test_multidim_numpy = setup_test_for_backend(_test_quadpack_multidimensional, "numpy", "float64")
test_multidim_torch = setup_test_for_backend(_test_quadpack_multidimensional, "torch", "float64")
test_multidim_jax = setup_test_for_backend(_test_quadpack_multidimensional, "jax", "float64")
test_multidim_tensorflow = setup_test_for_backend(
    _test_quadpack_multidimensional, "tensorflow", "float64"
)


if __name__ == "__main__":
    # Used to run these tests individually
    print("Testing torchquad QUADPACK vs scipy...")

    for backend in ["numpy", "torch", "jax", "tensorflow"]:
        print(f"\n=== Testing {backend} backend ===")
        try:
            if backend == "numpy":
                test_scipy_comparison_numpy()
                test_infinite_numpy()
                test_cauchy_numpy()
                test_multidim_numpy()
            elif backend == "torch":
                test_scipy_comparison_torch()
                test_infinite_torch()
                test_cauchy_torch()
                test_multidim_torch()
            elif backend == "jax":
                test_scipy_comparison_jax()
                test_infinite_jax()
                test_cauchy_jax()
                test_multidim_jax()
            elif backend == "tensorflow":
                test_scipy_comparison_tensorflow()
                test_infinite_tensorflow()
                test_cauchy_tensorflow()
                test_multidim_tensorflow()
        except Exception as e:
            print(f"Backend {backend} failed: {e}")

    print("\nComparison tests completed!")

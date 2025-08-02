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

from torchquad.integration.quadpack import QNG, QAGS
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


def _test_quadpack_vs_scipy_backend(backend, _precision):
    """Test torchquad QUADPACK vs scipy for given backend."""

    test_cases = [
        ("x^2 on [0,1]", _function_polynomial, 0, 1, 1 / 3),
        ("sin(10*x) on [0,1]", _function_oscillatory, 0, 1, (np.cos(0) - np.cos(10)) / 10),
        ("1/sqrt(x+0.1) on [0,1]", _function_near_singular, 0, 1, None),
    ]

    # Initialize torchquad integrators
    qng = QNG()
    qags = QAGS()

    for func_name, func, a, b, analytical in test_cases:
        print(f"\nTesting {func_name}")

        # Wrapper for torchquad (expects batch input)
        def torchquad_func(x):
            # x has shape (n_points, 1)
            return func(x[:, 0])

        epsabs, epsrel = 1e-10, 1e-10

        # Test scipy QUADPACK
        try:
            result_scipy, error_scipy = integrate.quad(func, a, b, epsabs=epsabs, epsrel=epsrel)
        except Exception as e:
            pytest.skip(f"scipy.quad failed: {e}")
            return

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

            # Allow some tolerance between implementations
            # QUADPACK can have slight numerical differences between implementations
            max_diff = max(1e-10, 1e-8 * abs(result_scipy))
            assert qags_diff < max_diff, (
                f"QAGS disagrees with scipy too much: "
                f"scipy={result_scipy}, qags={result_qags}, diff={qags_diff}"
            )

        except Exception as e:
            pytest.fail(f"torchquad QAGS failed: {e}")

        # Compare with analytical if available
        if analytical is not None:
            scipy_analytical_err = abs(result_scipy - analytical)
            qags_analytical_err = abs(float(result_qags) - analytical)

            # Both should be reasonably accurate
            assert (
                scipy_analytical_err < 1e-8
            ), f"scipy inaccurate vs analytical: {scipy_analytical_err}"
            assert (
                qags_analytical_err < 1e-8
            ), f"QAGS inaccurate vs analytical: {qags_analytical_err}"


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
            epsabs=1e-10,
            epsrel=1e-10,
            backend=backend,
        )
        error_2d = abs(float(result_2d) - analytical_2d)

        assert error_2d < 1e-8, f"2D integration error too large: {error_2d}"

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
                test_multidim_numpy()
            elif backend == "torch":
                test_scipy_comparison_torch()
                test_multidim_torch()
            elif backend == "jax":
                test_scipy_comparison_jax()
                test_multidim_jax()
            elif backend == "tensorflow":
                test_scipy_comparison_tensorflow()
                test_multidim_tensorflow()
        except Exception as e:
            print(f"Backend {backend} failed: {e}")

    print("\nComparison tests completed!")

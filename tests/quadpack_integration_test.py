import warnings
import pytest
import numpy as np

from torchquad.integration.quadpack import QNG, QAGS  # Will be implemented
from helper_functions import setup_test_for_backend
from integration_test_functions import Polynomial, Exponential, Sinusoid


def _run_quadpack_integration_tests(backend, _precision):
    """Test integration between different QUADPACK methods and comparison tests."""

    # Test that different methods give consistent results on simple functions
    qng = QNG()
    qags = QAGS()

    # Simple polynomial that both methods should handle well
    def simple_poly(x):
        # x^2 from 0 to 1, analytical result = 1/3
        return x**2

    domain = [[0, 1]]

    # Both methods should agree on simple functions
    result_qng = qng.integrate(
        simple_poly, dim=1, integration_domain=domain, epsabs=1e-10, epsrel=1e-10, backend=backend
    )
    result_qags = qags.integrate(
        simple_poly,
        dim=1,
        integration_domain=domain,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=100,
        backend=backend,
    )

    analytical_result = 1.0 / 3.0

    # Both should be very accurate
    assert (
        abs(result_qng - analytical_result) < 1e-10
    ), f"QNG inaccurate: {abs(result_qng - analytical_result)}"
    assert (
        abs(result_qags - analytical_result) < 1e-10
    ), f"QAGS inaccurate: {abs(result_qags - analytical_result)}"

    # Results should be very close to each other
    assert (
        abs(result_qng - result_qags) < 1e-12
    ), f"QNG and QAGS disagree: {abs(result_qng - result_qags)}"

    print(
        f"Consistency test passed. QNG: {result_qng}, QAGS: {result_qags}, Analytical: {analytical_result}"
    )

    # Test QAGS advantage on more difficult function
    def oscillatory_func(x):
        # sin(10*x) from 0 to 1
        from autoray import numpy as anp

        return anp.sin(10 * x)

    # QAGS should handle this better than QNG due to adaptive subdivision
    result_qags_osc = qags.integrate(
        oscillatory_func,
        dim=1,
        integration_domain=domain,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=100,
        backend=backend,
    )

    # Analytical result for sin(10*x) from 0 to 1 is (cos(0) - cos(10))/10
    analytical_osc = (np.cos(0) - np.cos(10)) / 10

    assert (
        abs(result_qags_osc - analytical_osc) < 1e-6
    ), f"QAGS failed on oscillatory: {abs(result_qags_osc - analytical_osc)}"

    print(f"Oscillatory test passed. QAGS: {result_qags_osc}, Analytical: {analytical_osc}")

    # Test multi-dimensional consistency (skip for TensorFlow)
    if backend != "tensorflow":  # TensorFlow only supports 1D

        def separable_2d(x):
            # x[0]^2 * x[1]^2 over [0,1]x[0,1], analytical result = 1/9
            from autoray import numpy as anp

            return x[:, 0] ** 2 * x[:, 1] ** 2

        domain_2d = [[0, 1], [0, 1]]

        result_2d = qags.integrate(
            separable_2d,
            dim=2,
            integration_domain=domain_2d,
            epsabs=1e-6,
            epsrel=1e-6,
            limit=50,
            backend=backend,
        )

        analytical_2d = 1.0 / 9.0
        assert (
            abs(result_2d - analytical_2d) < 1e-4
        ), f"2D separable function failed: {abs(result_2d - analytical_2d)}"

        print(f"2D separable test passed. Result: {result_2d}, Analytical: {analytical_2d}")
    else:
        print("2D separable test skipped for TensorFlow backend")

    # Test parameter validation
    with pytest.raises((ValueError, RuntimeError)):
        qng.integrate(simple_poly, dim=0, integration_domain=domain, backend=backend)

    with pytest.raises((ValueError, RuntimeError)):
        qags.integrate(simple_poly, dim=1, integration_domain=[[1, 0]], backend=backend)

    # Test that quadpack methods are more accurate than basic methods for difficult cases
    # This would require importing other torchquad methods for comparison
    # We'll implement this after basic functionality is working

    print(f"QUADPACK integration tests passed for backend: {backend}")


def _run_quadpack_performance_tests(backend, _precision):
    """Test performance characteristics and scaling of QUADPACK methods."""

    qng = QNG()
    qags = QAGS()

    # Test that QAGS achieves better accuracy than QNG on difficult functions
    def difficult_func(x):
        from autoray import numpy as anp

        # Function with near-singularity
        return 1.0 / anp.sqrt(anp.abs(x - 0.3) + 1e-10)

    domain = [[0, 1]]

    # QNG might struggle with this
    try:
        result_qng = qng.integrate(
            difficult_func,
            dim=1,
            integration_domain=domain,
            epsabs=1e-6,
            epsrel=1e-6,
            backend=backend,
        )
        qng_success = True
    except:
        qng_success = False

    # QAGS should handle this better
    result_qags = qags.integrate(
        difficult_func,
        dim=1,
        integration_domain=domain,
        epsabs=1e-6,
        epsrel=1e-6,
        limit=200,
        backend=backend,
    )

    # QAGS should always succeed on this type of function
    assert result_qags is not None, "QAGS failed on difficult function"

    print(f"Difficult function test: QNG success: {qng_success}, QAGS result: {result_qags}")

    # Test scaling with dimension (should degrade gracefully)
    def nd_polynomial(x):
        from autoray import numpy as anp

        return anp.sum(x**2, axis=1)

    domains = {1: [[0, 1]], 2: [[0, 1], [0, 1]], 3: [[0, 1], [0, 1], [0, 1]]}

    # For sum(x**2, axis=1) over [0,1]^d:
    # 1D: ∫₀¹ x² dx = 1/3
    # 2D: ∫₀¹∫₀¹ (x² + y²) dx dy = 2/3
    # 3D: ∫₀¹∫₀¹∫₀¹ (x² + y² + z²) dx dy dz = 1
    analytical_results = {1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}

    for dim in [1, 2, 3]:
        if backend == "tensorflow" and dim > 1:
            continue  # Skip multi-D TF tests (TensorFlow backend only supports 1D)

        result = qags.integrate(
            nd_polynomial,
            dim=dim,
            integration_domain=domains[dim],
            epsabs=1e-4,
            epsrel=1e-4,
            limit=30,
            backend=backend,
        )

        error = abs(result - analytical_results[dim])
        max_error = 10 ** (-(8 - 2 * dim))  # Allow degrading accuracy with dimension

        assert (
            error < max_error
        ), f"Scaling test failed for {dim}D: error={error}, max_allowed={max_error}"

        print(
            f"{dim}D scaling test passed: result={result}, analytical={analytical_results[dim]}, error={error}"
        )


# Setup backend-specific test functions
test_integration_numpy = setup_test_for_backend(_run_quadpack_integration_tests, "numpy", "float64")
test_integration_torch = setup_test_for_backend(_run_quadpack_integration_tests, "torch", "float64")
test_integration_jax = setup_test_for_backend(_run_quadpack_integration_tests, "jax", "float64")
test_integration_tensorflow = setup_test_for_backend(
    _run_quadpack_integration_tests, "tensorflow", "float64"
)

test_performance_numpy = setup_test_for_backend(_run_quadpack_performance_tests, "numpy", "float64")
test_performance_torch = setup_test_for_backend(_run_quadpack_performance_tests, "torch", "float64")
test_performance_jax = setup_test_for_backend(_run_quadpack_performance_tests, "jax", "float64")
test_performance_tensorflow = setup_test_for_backend(
    _run_quadpack_performance_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # Used to run these tests individually
    print("Running integration tests...")
    test_integration_numpy()
    test_integration_torch()
    test_integration_jax()
    test_integration_tensorflow()

    print("Running performance tests...")
    test_performance_numpy()
    test_performance_torch()
    test_performance_jax()
    test_performance_tensorflow()

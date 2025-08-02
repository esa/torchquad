import warnings
import numpy as np

from torchquad.integration.quadpack.qagi import QAGI
from helper_functions import setup_test_for_backend


def _run_qagi_tests(backend, _precision):
    """Test the integrate function in integration.quadpack.QAGI for the given backend."""
    
    qagi = QAGI()
    
    # Test 1: Exponential decay on [0, ∞)
    # ∫[0,∞) exp(-x) dx = 1
    def exp_decay(x):
        from autoray import numpy as anp
        return anp.exp(-x[:, 0])
    
    result = qagi.integrate(
        exp_decay,
        dim=1,
        integration_domain=[[0, np.inf]],
        epsabs=1e-10,
        epsrel=1e-10,
        backend=backend
    )
    error = abs(float(result) - 1.0)
    print(f"Exponential decay [0,inf): result={result}, error={error}")
    assert error < 1e-8, f"QAGI failed on exponential decay: error={error}"
    
    # Test 2: Gaussian on (-inf, inf)
    # ∫[-inf,inf) exp(-x²) dx = sqrt(pi)
    def gaussian(x):
        from autoray import numpy as anp
        return anp.exp(-x[:, 0]**2)
    
    result = qagi.integrate(
        gaussian,
        dim=1,
        integration_domain=[[-np.inf, np.inf]],
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    expected = np.sqrt(np.pi)
    error = abs(float(result) - expected)
    print(f"Gaussian (-inf,inf): result={result}, expected={expected}, error={error}")
    assert error < 1e-6, f"QAGI failed on Gaussian: error={error}"
    
    # Test 3: Rational function on (-inf, 0]
    # ∫[-inf,0] 1/(1+x²) dx = pi/2
    def rational(x):
        from autoray import numpy as anp
        return 1.0 / (1.0 + x[:, 0]**2)
    
    result = qagi.integrate(
        rational,
        dim=1,
        integration_domain=[[-np.inf, 0]],
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    expected = np.pi / 2
    error = abs(float(result) - expected)
    print(f"Rational function (-inf,0]: result={result}, expected={expected}, error={error}")
    assert error < 1e-6, f"QAGI failed on rational function: error={error}"
    
    # Test 4: Oscillatory decay on [0, inf)
    # ∫[0,inf) sin(x)*exp(-x) dx = 1/2
    def osc_decay(x):
        from autoray import numpy as anp
        return anp.sin(x[:, 0]) * anp.exp(-x[:, 0])
    
    result = qagi.integrate(
        osc_decay,
        dim=1,
        integration_domain=[[0, np.inf]],
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    expected = 0.5
    error = abs(float(result) - expected)
    print(f"Oscillatory decay [0,inf): result={result}, expected={expected}, error={error}")
    assert error < 1e-6, f"QAGI failed on oscillatory decay: error={error}"
    
    # Test 5: Finite interval (should delegate to QAGS)
    def polynomial(x):
        return x[:, 0]**2
    
    result = qagi.integrate(
        polynomial,
        dim=1,
        integration_domain=[[0, 1]],
        epsabs=1e-10,
        epsrel=1e-10,
        backend=backend
    )
    expected = 1.0 / 3.0
    error = abs(float(result) - expected)
    print(f"Polynomial [0,1] (finite): result={result}, expected={expected}, error={error}")
    assert error < 1e-10, f"QAGI failed on finite interval: error={error}"
    
    # Test 6: Error handling
    try:
        # Test invalid domain (a > b)
        qagi.integrate(
            exp_decay,
            dim=1,
            integration_domain=[[np.inf, 0]],
            backend=backend
        )
        assert False, "Should have raised error for invalid domain"
    except ValueError:
        pass  # Expected
    
    # Test warning for multi-dimensional (skip for TensorFlow)
    if backend != "tensorflow":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qagi.integrate(
                lambda x: x[:, 0],
                dim=2,
                integration_domain=[[0, np.inf], [0, 1]],
                backend=backend
            )
            assert len(w) > 0, "Should have warned about multi-dimensional infinite domains"
    
    print(f"QAGI tests passed for backend: {backend}")


# Setup backend-specific test functions
test_integrate_numpy = setup_test_for_backend(_run_qagi_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_qagi_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_qagi_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(_run_qagi_tests, "tensorflow", "float64")


if __name__ == "__main__":
    # Used to run this test individually
    print("Testing QAGI (infinite interval integration)...")
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()
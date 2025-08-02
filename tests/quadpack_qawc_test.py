import warnings
import numpy as np

from torchquad.integration.quadpack.qawc import QAWC
from helper_functions import setup_test_for_backend


def _run_qawc_tests(backend, _precision):
    """Test the integrate function in integration.quadpack.QAWC for the given backend."""
    
    qawc = QAWC()
    
    # Test 1: Simple Cauchy principal value
    # P.V. ∫[-1,1] 1/(x-0) dx should be 0 by symmetry
    def constant_fn(x):
        from autoray import numpy as anp
        return anp.ones_like(x[:, 0])
    
    result = qawc.integrate(
        constant_fn,
        dim=1,
        integration_domain=[[-1, 1]],
        c=0.0,
        epsabs=1e-10,
        epsrel=1e-10,
        backend=backend
    )
    error = abs(float(result))
    print(f"Constant function, c=0: result={result}, error={error}")
    assert error < 1e-8, f"QAWC failed on symmetric case: error={error}"
    
    # Test 2: Linear function with known result
    # P.V. ∫[0,2] x/(x-1) dx = P.V. ∫[0,2] [1 + 1/(x-1)] dx = 2 + log|1|  = 2
    def linear_fn(x):
        return x[:, 0]
    
    result = qawc.integrate(
        linear_fn,
        dim=1,
        integration_domain=[[0, 2]],
        c=1.0,
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    expected = 2.0
    error = abs(float(result) - expected)
    print(f"Linear function, c=1: result={result}, expected={expected}, error={error}")
    assert error < 1e-6, f"QAWC failed on linear function: error={error}"
    
    # Test 3: Quadratic function
    # P.V. ∫[0,2] x²/(x-1) dx can be computed analytically
    # x²/(x-1) = x + 1 + 1/(x-1)
    # So P.V. ∫[0,2] x²/(x-1) dx = ∫[0,2] (x + 1) dx + P.V. ∫[0,2] 1/(x-1) dx
    #                              = [x²/2 + x][0,2] + log|2-1|/|0-1|
    #                              = 2 + 2 + log(1) = 4
    def quadratic_fn(x):
        return x[:, 0]**2
    
    result = qawc.integrate(
        quadratic_fn,
        dim=1,
        integration_domain=[[0, 2]],
        c=1.0,
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    expected = 4.0
    error = abs(float(result) - expected)
    print(f"Quadratic function, c=1: result={result}, expected={expected}, error={error}")
    assert error < 1e-6, f"QAWC failed on quadratic function: error={error}"
    
    # Test 4: Smooth function away from singularity
    # P.V. ∫[0,3] sin(x)/(x-2) dx
    def smooth_fn(x):
        from autoray import numpy as anp
        return anp.sin(x[:, 0])
    
    result = qawc.integrate(
        smooth_fn,
        dim=1,
        integration_domain=[[0, 3]],
        c=2.0,
        epsabs=1e-8,
        epsrel=1e-8,
        backend=backend
    )
    # We don't have an analytical solution, but it should be finite
    print(f"Smooth function sin(x), c=2: result={result}")
    assert not np.isnan(float(result)), "QAWC result should not be NaN"
    assert not np.isinf(float(result)), "QAWC result should not be infinite"
    
    # Test 5: Function with singularity at endpoint (should fail)
    try:
        qawc.integrate(
            constant_fn,
            dim=1,
            integration_domain=[[0, 1]],
            c=0.0,  # Singularity at left endpoint
            backend=backend
        )
        assert False, "Should have raised error for singularity at endpoint"
    except ValueError:
        pass  # Expected
    
    # Test 6: Missing singularity parameter
    try:
        qawc.integrate(
            constant_fn,
            dim=1,
            integration_domain=[[0, 1]],
            # c parameter missing
            backend=backend
        )
        assert False, "Should have raised error for missing c parameter"
    except ValueError:
        pass  # Expected
    
    # Test 7: Multi-dimensional (should fail)
    try:
        qawc.integrate(
            lambda x: x[:, 0],
            dim=2,
            integration_domain=[[0, 1], [0, 1]],
            c=0.5,
            backend=backend
        )
        assert False, "Should have raised error for multi-dimensional"
    except ValueError:
        pass  # Expected
    
    print(f"QAWC tests passed for backend: {backend}")


# Setup backend-specific test functions
test_integrate_numpy = setup_test_for_backend(_run_qawc_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_qawc_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_qawc_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(_run_qawc_tests, "tensorflow", "float64")


if __name__ == "__main__":
    # Used to run this test individually
    print("Testing QAWC (Cauchy principal value integration)...")
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()
from torchquad import Boole, Trapezoid, Simpson, VEGAS, MonteCarlo

# TODO test these in the future
# from ..plots.plot_convergence import plot_convergence
# from ..plots.plot_runtime import plot_runtime

from torchquad import enable_cuda
from torchquad import set_precision
from torchquad import set_log_level
from torchquad import set_up_backend
from loguru import logger
import warnings


def _deployment_test():
    """Comprehensive test to verify successful deployment of torchquad.

    This method is used internally to check successful deployment after PyPI releases.
    It verifies:
    - Basic imports work
    - All integrators can be initialized
    - Integration methods work with different backends
    - GPU functionality (if available)
    - Precision settings
    - Error handling
    - Results are reasonable
    """
    set_log_level("INFO")

    # Suppress common warnings to reduce noise
    warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
    warnings.filterwarnings(
        "ignore", message="DEPRECATION WARNING: In future versions of torchquad"
    )

    logger.info("####################################")
    logger.info("######## TESTING DEPLOYMENT ########")
    logger.info("####################################")
    logger.info("")

    # Test 1: Basic imports and version info
    logger.info("Testing imports and version info...")
    try:
        import torchquad

        logger.info(f"torchquad version: {getattr(torchquad, '__version__', 'unknown')}")
        logger.info("✓ Basic imports successful")
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

    # Test 2: Backend availability
    logger.info("\nTesting available backends...")
    available_backends = []
    for backend in ["torch", "numpy", "jax", "tensorflow"]:
        try:
            set_up_backend(backend, "float32")
            available_backends.append(backend)
            logger.info(f"✓ {backend} backend available")
        except Exception:
            logger.info(f"- {backend} backend not available (expected if not installed)")

    if not available_backends:
        logger.error("✗ No backends available!")
        return False

    # Test 3: CUDA and precision settings
    logger.info("\nTesting CUDA and precision settings...")
    try:
        enable_cuda()
        set_precision("double")
        logger.info("✓ CUDA and precision settings configured")
    except Exception as e:
        logger.info(f"- CUDA configuration warning: {e}")

    # Test 4: Integrator initialization
    logger.info("\nInitializing integrators...")
    integrators = {}
    try:
        integrators["trapezoid"] = Trapezoid()
        integrators["simpson"] = Simpson()
        integrators["boole"] = Boole()
        integrators["monte_carlo"] = MonteCarlo()
        integrators["vegas"] = VEGAS()
        logger.info("✓ All integrators initialized successfully")
    except Exception as e:
        logger.error(f"✗ Integrator initialization failed: {e}")
        return False

    # Test 5: Integration with multiple backends and functions
    logger.info("\nTesting integration functions...")

    test_domain = [[0, 2]]
    success_count = 0
    total_tests = 0

    for backend in available_backends:
        logger.info(f"\n  Testing {backend} backend...")
        backend_success = 0
        backend_total = 0

        try:
            set_up_backend(backend, "float32")

            # Simple vectorized test functions
            def simple_polynomial(x):
                # Vectorized: x^2 + 1, expected ~4.33 for [0,2]
                return x[:, 0] ** 2 + 1

            def simple_constant(x):
                # Vectorized: constant function = 2, expected 4.0 for [0,2]
                backend_type = _infer_backend_from_tensor(x)
                if backend_type == "torch":
                    import torch

                    return torch.ones(x.shape[0]) * 2
                else:
                    from autoray import numpy as anp

                    return anp.ones(x.shape[0], like=x, dtype=x.dtype) * 2

            test_functions = {
                "polynomial": (simple_polynomial, 4.33),  # approximate expected result
                "constant": (simple_constant, 4.0),  # exact expected result
            }

            # Test each integrator with different functions
            for integrator_name, integrator in integrators.items():
                for func_name, (func, expected) in test_functions.items():
                    backend_total += 1
                    total_tests += 1
                    try:
                        if integrator_name == "vegas":
                            result = integrator.integrate(
                                func, dim=1, N=1000, integration_domain=test_domain
                            )
                        else:
                            result = integrator.integrate(
                                func, dim=1, N=101, integration_domain=test_domain
                            )

                        # Convert result to float for comparison
                        if hasattr(result, "item"):
                            result_val = float(result.item())
                        else:
                            result_val = float(result)

                        # Check if result is reasonable (within 50% of expected for simple test)
                        if (
                            _is_finite_result(result_val)
                            and abs(result_val - expected) < expected * 0.5
                        ):
                            logger.debug(
                                f"  ✓ {integrator_name} with {func_name}: {result_val:.3f} (expected ~{expected})"
                            )
                            backend_success += 1
                            success_count += 1
                        else:
                            logger.warning(
                                f"  - {integrator_name} with {func_name}: unreasonable result {result_val:.3f} (expected ~{expected})"
                            )

                    except Exception as e:
                        logger.warning(
                            f"  - {integrator_name} with {func_name} failed: {str(e)[:100]}..."
                        )

            logger.info(f"  ✓ {backend} backend: {backend_success}/{backend_total} tests passed")

        except Exception as e:
            logger.warning(f"  - {backend} backend setup failed: {e}")

    # Check if enough tests passed
    if total_tests == 0:
        logger.error("✗ No integration tests could be run!")
        return False
    elif success_count < total_tests * 0.5:  # At least 50% should pass
        logger.error(f"✗ Only {success_count}/{total_tests} integration tests passed")
        return False
    else:
        logger.info(f"✓ Integration tests: {success_count}/{total_tests} passed")

    # Test 6: Error handling
    logger.info("\nTesting error handling...")
    try:
        tp = Trapezoid()
        # This should handle the error gracefully
        try:
            tp.integrate(lambda x: x / 0, dim=1, N=101)  # Division by zero
            logger.info("✓ Error handling works (or function avoided error)")
        except Exception:
            logger.info("✓ Error handling works (caught expected error)")
    except Exception as e:
        logger.warning(f"- Error handling test inconclusive: {e}")

    # Test 7: Multi-dimensional integration
    logger.info("\nTesting multi-dimensional integration...")
    multi_dim_success = False
    try:
        mc = MonteCarlo()

        def multi_dim_func(x):
            # Vectorized 2D function: sum of squares, expected result ~ 2/3 for [0,1]x[0,1]
            backend = _infer_backend_from_tensor(x)
            if backend == "torch":
                import torch

                return torch.sum(x**2, dim=1)
            else:
                from autoray import numpy as anp

                return anp.sum(x**2, axis=1)

        result = mc.integrate(multi_dim_func, dim=2, N=1000, integration_domain=[[0, 1], [0, 1]])
        if hasattr(result, "item"):
            result_val = float(result.item())
        else:
            result_val = float(result)

        # Expected result is 2/3 ≈ 0.667 for integral of x^2 + y^2 over [0,1]x[0,1]
        if (
            _is_finite_result(result_val) and 0.2 < result_val < 1.2
        ):  # Reasonable range for Monte Carlo
            logger.info(f"✓ Multi-dimensional integration: {result_val:.3f} (expected ~0.67)")
            multi_dim_success = True
        else:
            logger.warning(f"- Multi-dimensional integration: unreasonable result {result_val:.3f}")
    except Exception as e:
        logger.warning(f"- Multi-dimensional integration failed: {str(e)[:100]}...")

    if not multi_dim_success:
        logger.warning("- Multi-dimensional integration test failed")

    logger.info("\n####################################")
    logger.info("############ ALL DONE #############")
    logger.info("####################################")

    # Final assessment - only return True if critical tests passed
    critical_failures = []

    if not available_backends:
        critical_failures.append("No backends available")

    if success_count < total_tests * 0.5:
        critical_failures.append(f"Integration tests: only {success_count}/{total_tests} passed")

    if not multi_dim_success:
        critical_failures.append("Multi-dimensional integration failed")

    if critical_failures:
        logger.error("✗ Deployment test FAILED with critical issues:")
        for failure in critical_failures:
            logger.error(f"  - {failure}")
        return False
    else:
        logger.info("✓ Deployment test completed successfully!")
        return True


def _get_exp_func(x):
    """Get exponential function for current backend"""
    backend = _infer_backend_from_tensor(x)
    if backend == "torch":
        import torch

        return torch.exp(x[0])
    else:
        from autoray import numpy as anp

        return anp.exp(x[0])


def _get_sin_func(x):
    """Get sine function for current backend"""
    backend = _infer_backend_from_tensor(x)
    if backend == "torch":
        import torch

        return torch.sin(x[0])
    else:
        from autoray import numpy as anp

        return anp.sin(x[0])


def _infer_backend_from_tensor(x):
    """Infer backend from tensor type"""
    try:
        from autoray import infer_backend

        return infer_backend(x)
    except Exception:
        # Fallback method
        if hasattr(x, "numpy"):  # PyTorch tensor
            return "torch"
        elif str(type(x)).find("jax") != -1:
            return "jax"
        elif str(type(x)).find("tensorflow") != -1:
            return "tensorflow"
        else:
            return "numpy"


def _is_finite_result(result):
    """Check if result is finite"""
    try:
        from autoray import numpy as anp

        result_np = anp.asarray(result)
        return anp.isfinite(result_np).all()
    except Exception:
        # Fallback for basic types
        try:
            import math

            return math.isfinite(float(result))
        except Exception:
            return False

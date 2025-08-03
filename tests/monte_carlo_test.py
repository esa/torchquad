import torch
import pytest

from torchquad.integration.monte_carlo import MonteCarlo
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_monte_carlo_tests(backend, _precision):
    """Test the integrate function in integration.MonteCarlo for the given backend."""

    mc = MonteCarlo()

    # 1D Tests
    N = 100000  # integration points to use

    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 1, "seed": 0},
        integration_dim=1,
        use_complex=True,
        backend=backend,
    )
    print(f"1D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}")
    # Constant functions can be integrated exactly with MonteCarlo.
    # (at least our example functions)
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0

    # If this breaks check if test functions in helper_functions changed.
    for error in errors[:3]:
        assert error < 7e-3

    assert errors[3] < 0.5
    assert errors[4] < 32.0

    for error in errors[6:10]:
        assert error < 1.1e-2

    for error in errors[10:]:
        assert error < 28.03

    # 3D Tests
    N = 1000000
    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 3, "seed": 0},
        integration_dim=3,
        use_complex=True,
        backend=backend,
    )
    print(f"3D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors: {str(errors)}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0
    for error, test_function in zip(errors, funcs):
        assert (
            error < 1e-1 if test_function.is_integrand_1d else error < 0.33
        )  # errors add up if the integrand is higher dimensional

    # 10D Tests
    N = 10000
    errors, funcs = compute_integration_test_errors(
        mc.integrate,
        {"N": N, "dim": 10, "seed": 0},
        integration_dim=10,
        use_complex=True,
        backend=backend,
    )
    print(f"10D Monte Carlo Test passed. N: {N}, backend: {backend}, Errors:" f" {str(errors)}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 0 or err == 0.0
    for error in errors:
        assert error < 26

    # JIT Tests
    if backend != "numpy":
        N = 100000
        jit_integrate = None

        def integrate(*args, **kwargs):
            # this function initializes the jit_integrate variable with a jit'ed integrate function
            # which is then re-used on all other integrations (as is the point of JIT).
            nonlocal jit_integrate
            if jit_integrate is None:
                jit_integrate = mc.get_jit_compiled_integrate(dim=1, N=N, backend=backend)
            return jit_integrate(*args, **kwargs)

        errors, funcs = compute_integration_test_errors(
            integrate,
            {},
            integration_dim=1,
            use_complex=True,
            backend=backend,
            filter_test_functions=lambda x: x.is_integrand_1d,
        )
        print(
            f"1D MC JIT Test passed for 1D integrands. N: {N}, backend: {backend}, Errors: {errors}"
        )

        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 0 or err < 1e-14

        # If this breaks check if test functions in helper_functions changed.
        for error in errors[:3]:
            assert error < 1e-2

        assert errors[3] < 0.5
        assert errors[4] < 43.0

        for error in errors[6:10]:
            assert error < 2e-2

        for error in errors[10:]:
            assert error < 35.0

        jit_integrate = None  # set to None again so can be re-used with new integrand shape

        errors, funcs = compute_integration_test_errors(
            integrate,
            {},
            integration_dim=1,
            use_complex=True,
            backend=backend,
            filter_test_functions=lambda x: x.integrand_dims == [2, 2, 2],
        )
        print(
            f"1D MC JIT Test passed for [2, 2, 2] dimensional integrands. N: {N}, backend: {backend}, Errors: {errors}"
        )

        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 0 or err < 1e-14


def test_monte_carlo_calculate_result_kwargs():
    """Test that MonteCarlo().calculate_result() works correctly with keyword arguments."""

    def integrand1(x):
        return torch.rand(x.shape, device=x.device)

    # Use the same device as the current default to avoid device mismatch
    integration_domain = torch.tensor([[0.0, 1.0]])
    N = 1000

    integrator = MonteCarlo()
    sample_points = integrator.calculate_sample_points(N, integration_domain, seed=42)
    function_values, _ = integrator.evaluate_integrand(integrand1, sample_points)

    # Test with positional arguments (should work as before)
    integral1 = integrator.calculate_result(function_values, integration_domain)

    # Test with keyword arguments (this was failing before the fix)
    integral2 = integrator.calculate_result(
        function_values=function_values, integration_domain=integration_domain
    )

    # Test with mixed positional and keyword arguments
    integral3 = integrator.calculate_result(function_values, integration_domain=integration_domain)

    # All results should be approximately equal
    assert torch.allclose(integral1, integral2, rtol=1e-10)
    assert torch.allclose(integral1, integral3, rtol=1e-10)


def test_monte_carlo_calculate_result_error_handling():
    """Test that MonteCarlo().calculate_result() gives meaningful error messages for invalid inputs."""

    integrator = MonteCarlo()

    # Test missing function_values argument
    with pytest.raises(ValueError) as exc_info:
        integrator.calculate_result(integration_domain=torch.tensor([[0.0, 1.0]]))

    assert "function_values argument not found" in str(exc_info.value)
    assert "Please provide function_values" in str(exc_info.value)

    # Test with only self argument (no function_values)
    with pytest.raises(ValueError) as exc_info:
        integrator.calculate_result()

    assert "function_values argument not found" in str(exc_info.value)


test_integrate_numpy = setup_test_for_backend(_run_monte_carlo_tests, "numpy", "float32")
test_integrate_torch = setup_test_for_backend(_run_monte_carlo_tests, "torch", "float32")
test_integrate_jax = setup_test_for_backend(_run_monte_carlo_tests, "jax", "float32")
test_integrate_tensorflow = setup_test_for_backend(_run_monte_carlo_tests, "tensorflow", "float32")


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

    # Test the new keyword argument functionality
    test_monte_carlo_calculate_result_kwargs()
    test_monte_carlo_calculate_result_error_handling()
    print("All MonteCarlo keyword argument tests passed!")

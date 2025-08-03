import warnings
import torch
import pytest

from torchquad.integration.boole import Boole
from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_boole_tests(backend, _precision):
    """Test the integrate function in integration.Boole for the given backend.
    Note: For now the 10-D test is diabled due to lack of GPU memory on some computers.
    """

    bl = Boole()
    # 1D Tests
    N = 401

    errors, funcs = compute_integration_test_errors(
        bl.integrate,
        {"N": N, "dim": 1},
        integration_dim=1,
        use_complex=True,
        backend=backend,
    )
    print(f"1D Boole Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    # Polynomials up to degree 5 can be integrated almost exactly with Boole.
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 5 or err < 6.33e-11
    for error in errors:
        assert error < 6.33e-11

    # 3D Tests
    N = 1076890  # N = 102.5 per dim (will change to 101 if all works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors, funcs = compute_integration_test_errors(
            bl.integrate,
            {"N": N, "dim": 3},
            integration_dim=3,
            use_complex=True,
            backend=backend,
        )
    print(f"3D Boole Test passed. N: {N}, backend: {backend}, Errors: {errors}")
    for err, test_function in zip(errors, funcs):
        assert test_function.get_order() > 5 or (
            err < 2e-13 if test_function.is_integrand_1d else err < 2e-11
        )
    for error in errors:
        assert error < 5e-6

    # 10D Tests
    # Have been disabled for now because it is too GPU-heavy
    # N = 5 ** 10
    # errors = compute_test_errors(bl.integrate, {"N": N, "dim": 10}, integration_dim=10, use_complex=True)
    # print("10D Boole Test: Passed N =", N, "\n", "Errors: ", errors)
    # for error in errors:
    # assert error < 5e-9

    # JIT Tests
    if backend != "numpy":
        N = 401
        jit_integrate = None

        def integrate(*args, **kwargs):
            # this function initializes the jit_integrate variable with a jit'ed integrate function
            # which is then re-used on all other integrations (as is the point of JIT).
            nonlocal jit_integrate
            if jit_integrate is None:
                jit_integrate = bl.get_jit_compiled_integrate(dim=1, N=N, backend=backend)
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
            f"1D Boole JIT Test passed for 1D integrands. N: {N}, backend: {backend}, Errors: {errors}"
        )
        # Polynomials up to degree 5 can be integrated almost exactly with Boole.
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 5 or err < 6.33e-11
        for error in errors:
            assert error < 6.33e-11

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
            f"1D Boole JIT Test passed for [2, 2, 2] dimensional integrands. N: {N}, backend: {backend}, Errors: {errors}"
        )
        # Polynomials up to degree 5 can be integrated almost exactly with Boole.
        for err, test_function in zip(errors, funcs):
            assert test_function.get_order() > 5 or err < 6.33e-11
        for error in errors:
            assert error < 6.33e-11


def test_boole_calculate_result_kwargs():
    """Test that Boole().calculate_result() works correctly with keyword arguments."""

    def integrand1(x):
        return torch.rand(x.shape, device=x.device)

    integration_domain = torch.tensor([[0.0, 1.0]])
    dim = 1
    N = 125  # Must be 5^n for Boole's rule

    integrator = Boole()
    grid_points, hs, n_per_dim = integrator.calculate_grid(N, integration_domain)
    function_values, _ = integrator.evaluate_integrand(integrand1, grid_points)

    # Test with positional arguments (should work as before)
    integral1 = integrator.calculate_result(function_values, dim, n_per_dim, hs, integration_domain)

    # Test with keyword arguments (this was failing before the fix)
    integral2 = integrator.calculate_result(
        function_values=function_values,
        dim=dim,
        n_per_dim=n_per_dim,
        hs=hs,
        integration_domain=integration_domain,
    )

    # Test with mixed positional and keyword arguments
    integral3 = integrator.calculate_result(
        function_values, dim=dim, n_per_dim=n_per_dim, hs=hs, integration_domain=integration_domain
    )

    # All results should be approximately equal
    assert torch.allclose(integral1, integral2, rtol=1e-10)
    assert torch.allclose(integral1, integral3, rtol=1e-10)


def test_boole_calculate_result_error_handling():
    """Test that Boole().calculate_result() gives meaningful error messages for invalid inputs."""

    integrator = Boole()

    # Test missing function_values argument
    with pytest.raises(ValueError) as exc_info:
        integrator.calculate_result(
            dim=1,
            n_per_dim=5,
            hs=torch.tensor([0.25]),
            integration_domain=torch.tensor([[0.0, 1.0]]),
        )

    assert "function_values argument not found" in str(exc_info.value)
    assert "Please provide function_values" in str(exc_info.value)

    # Test with only self argument (no function_values)
    with pytest.raises(ValueError) as exc_info:
        integrator.calculate_result()

    assert "function_values argument not found" in str(exc_info.value)


test_integrate_numpy = setup_test_for_backend(_run_boole_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_boole_tests, "torch", "float64")
test_integrate_jax = setup_test_for_backend(_run_boole_tests, "jax", "float64")
test_integrate_tensorflow = setup_test_for_backend(_run_boole_tests, "tensorflow", "float64")


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()
    test_integrate_torch()
    test_integrate_jax()
    test_integrate_tensorflow()

    # Test the new keyword argument functionality
    test_boole_calculate_result_kwargs()
    test_boole_calculate_result_error_handling()
    print("All Boole keyword argument tests passed!")

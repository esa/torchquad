import sys
import pytest


sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype
import autoray as ar
from integration.integration_grid import IntegrationGrid
from integration.grid_integrator import GridIntegrator
from integration.utils import _linspace_with_grads
from helper_functions import setup_test_for_backend


class MockIntegrator(GridIntegrator):
    def __init__(self, disable_integration_domain_check, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_integration_domain_check = disable_integration_domain_check

    def integrate(self, fn, dim, N, integration_domain, backend, grid_check):
        grid_points, _, _ = self.calculate_grid(
            N,
            integration_domain,
            disable_integration_domain_check=self.disable_integration_domain_check,
        )
        grid_points = grid_points.reshape(N, -1)
        assert grid_check(grid_points)

    @property  # need to override in order to handle the grid so that we return a multiple 1d grids for each domain in the customized integration_domain
    def _grid_func(self):
        def f(integration_domain, N, requires_grad=False, backend=None):
            b = integration_domain[:, 1]
            a = integration_domain[:, 0]
            grid = anp.stack(
                [
                    _linspace_with_grads(a[ind], b[ind], N, requires_grad=requires_grad)
                    for ind in range(len(a))
                ]
            ).T
            return anp.reshape(
                grid, [-1]
            )  # flatten, but it works with TF as well which has no flatten

        return f


def _check_grid_validity(grid, integration_domain, N, eps):
    """Check if a specific grid object contains illegal values"""
    assert grid._N == int(
        N ** (1 / len(integration_domain)) + 1e-8
    ), "Incorrect number of points per dimension"
    assert grid.points.shape == (
        int(N),
        integration_domain.shape[0],
    ), "Incorrect number of calculated points"
    assert (
        grid.points.dtype == integration_domain.dtype
    ), "Grid points have an incorrect dtype"
    assert (
        grid.h.dtype == integration_domain.dtype
    ), "Mesh widths have an incorrect dtype"
    for dim in range(len(integration_domain)):
        domain_width = integration_domain[dim][1] - integration_domain[dim][0]
        assert (
            anp.abs(grid.h[dim] - domain_width / (grid._N - 1)) < eps
        ), "Incorrect mesh width"
        assert (
            anp.min(grid.points[:, dim]) >= integration_domain[dim][0]
        ), "Points are outside of the integration domain"
        assert (
            anp.max(grid.points[:, dim]) <= integration_domain[dim][1]
        ), "Points are outside of the integration domain"


def _run_integration_grid_tests(backend, dtype_name):
    """
    Test IntegrationGrid in integration.integration_grid for illegal values with various input arguments
    """
    if backend == "torch":
        import torch

        torch.set_printoptions(10)

    # Generate a grid in different dimensions with different N on different domains
    eps = 2e-8  # error bound
    dtype = to_backend_dtype(dtype_name, like=backend)

    # Test 1: N is float, 1-D
    # Test 2: N is int, 3-D
    # Test 3: N is float, 3-D
    Ns = [10.0, 4**3, 4.0**3]
    domains = [
        [[0, 1]],  # integer domain to check correct treatment
        [[0.0, 2.0], [-2.0, 1.0], [0.5, 1.0]],
        [[0.0, 2.0], [-2.0, 1.0], [0.5, 1.0]],
    ]
    for N, dom in zip(Ns, domains):
        integration_domain = anp.array(dom, dtype=dtype, like=backend)
        grid = IntegrationGrid(N, integration_domain)
        _check_grid_validity(grid, integration_domain, N, eps)

    mock_integrator_no_check = MockIntegrator(disable_integration_domain_check=True)
    mock_integrator_check = MockIntegrator(disable_integration_domain_check=False)

    # Bypassing check, the output grid should be shape (N, 3) for 3 different 1d domains.
    # Our custom _grid_func treats the integration_domain as a list of 1d domains
    # That is why the domain shape is (1, 3, 2) so that the IntegrationGrid recognizes it as a 1d integral but our
    # custom handler does the rest without the check, and fails with the check.
    N = 500
    dim = 1

    def grid_check(x):
        has_right_shape = x.shape == (N, 3)
        has_right_vals = anp.all(ar.to_numpy(x[0, :]) == 0) and anp.all(
            ar.to_numpy(x[-1, :]) == 1
        )
        return has_right_shape and has_right_vals

    mock_integrator_no_check.integrate(
        lambda x: x,
        dim,
        N,
        anp.array([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], like=backend),
        backend,
        grid_check,
    )
    # Without bypassing check, the error raised should be that the input domain is not compatible with the requested dimensions
    with pytest.raises(ValueError) as excinfo:
        mock_integrator_check.integrate(
            lambda x: x,
            dim,
            49,
            anp.array([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], like=backend),
            backend,
            grid_check,
        )
    assert "The integration_domain tensor has an invalid shape" == str(excinfo.value)


test_integration_grid_numpy = setup_test_for_backend(
    _run_integration_grid_tests, "numpy", "float64"
)
test_integration_grid_torch = setup_test_for_backend(
    _run_integration_grid_tests, "torch", "float64"
)
test_integration_grid_jax = setup_test_for_backend(
    _run_integration_grid_tests, "jax", "float64"
)
test_integration_grid_tensorflow = setup_test_for_backend(
    _run_integration_grid_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    test_integration_grid_numpy()
    test_integration_grid_torch()
    test_integration_grid_jax()
    test_integration_grid_tensorflow()

import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.integration_grid import IntegrationGrid
from helper_functions import setup_test_for_backend


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
        [[0.0, 1.0]],
        [[0.0, 2.0], [-2.0, 1.0], [0.5, 1.0]],
        [[0.0, 2.0], [-2.0, 1.0], [0.5, 1.0]],
    ]
    for N, dom in zip(Ns, domains):
        integration_domain = anp.array(dom, dtype=dtype, like=backend)
        grid = IntegrationGrid(N, integration_domain)
        _check_grid_validity(grid, integration_domain, N, eps)


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

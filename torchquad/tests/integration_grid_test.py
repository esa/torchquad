import sys

sys.path.append("../")

from integration.integration_grid import IntegrationGrid
import torch


def test_integration_grid():
    """Tests the integration grid in integration.integration_grid 
    """

    # Generate a grid in different dimension with different N on different domains
    N = 10
    eps = 1e-8
    integration_domain = [[0, 1]]
    grid = IntegrationGrid(N, integration_domain)

    # test if  numbers of points is correct
    assert grid._N == N
    assert len(grid.points) == N
    for dim in range(len(integration_domain)):
        # test if mesh width is correct
        assert torch.abs(grid.h[dim] - 1 / (N - 1)) < eps
        # test if all points are inside
        assert torch.all(grid.points[:, dim] >= integration_domain[dim][0])
        assert torch.all(grid.points[:, dim] <= integration_domain[dim][1])

    N = 27
    integration_domain = [[0, 2], [-2, 1], [0.5, 1]]
    grid = IntegrationGrid(N, integration_domain)

    # test if  numbers of points is correct
    assert grid._N == int(N ** (1 / len(integration_domain)))
    assert len(grid.points) == N
    for dim in range(len(integration_domain)):
        domain_width = integration_domain[dim][1] - integration_domain[dim][0]
        # test if mesh width is correct
        assert torch.abs(grid.h[dim] - domain_width / (grid._N - 1)) < eps
        # test if all points are inside
        assert torch.all(grid.points[:, dim] >= integration_domain[dim][0])
        assert torch.all(grid.points[:, dim] <= integration_domain[dim][1])


test_integration_grid()


import sys

sys.path.append("../")

from integration.integration_grid import IntegrationGrid


def test_integration_grid():
    """TODO
    """
    # Generate a grid in different dimension with different N
    # on different domains and check if
    # all points are inside
    # mesh width is correct
    # numbers of points is correct

    N = 10
    integration_domain = [[0, 1]]
    grid = IntegrationGrid(N, integration_grid)

    assert grid._N == N
    assert grid.h == 0.1
    assert len(grid.points) == N
    assert grid.point[:, 0] <= 1.0
    assert grid.point[:, 0] >= 0.0


test_integration_grid()


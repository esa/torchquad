import sys

sys.path.append("../")

from integration.integration_grid import IntegrationGrid


def test_integration_grid():
    """Tests the integration grid 
    """#TODO add more

    # Generate a grid in different dimension with different N
    # on different domains and check if
    N = 10
    l_bound, r_bound = 0, 1
    integration_domain = [[l_bound, r_bound]]
    grid = IntegrationGrid(N, integration_grid)

    assert grid._N == N
    # mesh width is correct
    assert grid.h == 1/N
    # numbers of points is correct
    assert len(grid.points) == N
    # all points are inside
    assert grid.point[:, 0] >= l_bound
    assert grid.point[:, 0] <= r_bound

test_integration_grid()


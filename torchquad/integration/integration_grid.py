import torch


class IntegrationGrid:
    grid = None
    h = None
    N = None

    def __init__(self, target_nr_of_points):
        # TODO expand to more than one dim
        grid_1d = torch.linspace(integration_domain[0][0], integration_domain[0][1], N)
        self.h = grid_1d[1] - grid_1d[0]
        self.grid = torch.tensor([x for x in grid_1d])
        self.N = target_nr_of_points


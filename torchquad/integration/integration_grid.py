import torch


class IntegrationGrid:
    _points = None
    _h = None
    _N = None

    def __init__(self, N, integration_domain):
        # TODO expand to more than one dim
        grid_1d = torch.linspace(integration_domain[0][0], integration_domain[0][1], N)
        self._h = grid_1d[1] - grid_1d[0]
        self._points = torch.tensor([x for x in grid_1d])
        self._N = N


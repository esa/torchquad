import sys

sys.path.append("../")

import torch

from integration.monte_carlo import MonteCarlo


def _f1(x):
    return torch.pow(x, 2)


def _f2(x):
    return torch.sin(x[:, 0]) + torch.exp(x[:, 1])


def _f3(x):
    return torch.exp(x[:, 2]) + torch.exp(x[:, 1]) + torch.exp(x[:, 0])


def _eval(f, dim, integration_domain, N=10000000):
    mc = MonteCarlo()
    return mc.integrate(f, dim, N=N, integration_domain=integration_domain)


def test_integrate():
    torch.manual_seed(0)  # we have to seed torch to get reproducible results
    torch.set_default_tensor_type(torch.DoubleTensor)

    # F1 , x^2
    integration_domain = [[-2, 2]]
    ground_truth = 16 / 3
    assert torch.abs(_eval(_f1, 1, integration_domain) - ground_truth) < 2e-3

    # F2 , sin(x) + exp(y)
    integration_domain = [[0, 5], [1, 3]]
    ground_truth = 88.26895110271665999890793257
    assert torch.abs(_eval(_f2, 2, integration_domain) - ground_truth) < 1e-2

    # F3 , exp(x)
    integration_domain = [[0, 5], [1, 3], [-2, 2]]
    ground_truth = 1599.18758287212565283365376
    assert torch.abs(_eval(_f3, 3, integration_domain) - ground_truth) < 0.32

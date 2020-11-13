import sys

sys.path.append("../")

import torch


from integration.simpson_1D import Simpson1D


def _f1(x):
    return torch.pow(x, 2)


def _f2(x):
    return torch.sin(x)


def _f3(x):
    return torch.exp(x)


def _eval(f, integration_domain, N=50001):
    # Init a trapezoid
    simp = Simpson1D()
    return simp.integrate(f, N=N, integration_domain=integration_domain)


_max_err = 1e-6


def test_integrate():
    torch.set_default_tensor_type(torch.DoubleTensor)

    # F1 , x^2
    integration_domain = [[-2, 2]]
    ground_truth = 16 / 3
    assert torch.abs(_eval(_f1, integration_domain) - ground_truth) < _max_err

    # F2 , sin(x)
    integration_domain = [[-10, -1]]
    ground_truth = -1.37937383494459
    assert torch.abs(_eval(_f2, integration_domain) - ground_truth) < _max_err

    # F3 , exp(x)
    integration_domain = [[0, 5]]
    ground_truth = 147.4131599102577
    assert torch.abs(_eval(_f3, integration_domain) - ground_truth) < _max_err

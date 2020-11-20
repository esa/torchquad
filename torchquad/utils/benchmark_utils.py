import numpy as np
import torch
from time import perf_counter


def integrate_N_points(method, f_n, N, scipy_based=True):
    """Performing integration of f_n on the integration interval [-1,1] by using N points. 

    Args:
        method (function): scipy-based or torchquad integration method.
        f_n (function): function to integrate.
        N (integer): number of points in the [-1, 1] interval. Defaults to [10, 100, 1000].
        scipy_based (bool, optional): if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
    """
    if scipy_based:
        sp = torch.linspace(-1, 1, int(N)).cpu()
        method(y=f_n(sp), x=sp)
    else:
        method(fn=f_n, N=N)


def runtime_measure(method, f_n, scipy_based=True, N=[10, 100, 1000], N_average=1):
    """[summary]

    Args:
        method (function): scipy-based or torchquad integration method.
        f_n (function): function to integrate.
        scipy_based (bool, optional):  if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
        N (list, optional): list containing different number of points in the [-1, 1] interval that should be used for testing the integration method. Defaults to [10, 100, 1000].
        N_average (int, optional): Number of average measurement for every number of integration points. Defaults to 1.

    Returns:
        [numpy]: runtime measurements for all the different number of integration points. Runtime measurements include sleep period of cpus.
    """
    runtime_measurement = []
    for n in N:
        measure = 0
        for m in range(N_average):
            start_time = perf_counter()
            integrate_N_points(method, f_n, n, scipy_based)
            stop_time = perf_counter()
            measure = measure + stop_time - start_time
        runtime_measurement.append(measure / N_average)

    return np.array(runtime_measurement)

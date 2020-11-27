import numpy as np
import torch
from time import perf_counter


def integrate_N_points(method, fn, N, scipy_based=True):
    """Performing integration of fn on the integration interval [-1,1] by using N points. 

    Args:
        method (function): scipy-based or torchquad integration method.
        fn (function): function to integrate.
        N (integer): number of points in the [-1, 1] interval. Defaults to [10, 100, 1000].
        scipy_based (bool, optional): if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
    """
    if scipy_based:
        sp = torch.linspace(-1, 1, int(N)).cpu()
        method(y=fn(sp), x=sp)
    else:
        method(fn=fn, N=N)


def runtime_measure(method, fn, scipy_based=True, N=[10, 100, 1000], iterations=1):
    """Performing runtime measurement of integration of fn over the interval [-1, 1] through the method ""method"" by using ""N"" points. For each value ""N"", a number of iterations ""iterations"" is used to calculate the average.

    Args:
        method (function): scipy-based or torchquad integration method.
        fn (function): function to integrate.
        scipy_based (bool, optional):  if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
        N (list, optional): list containing different number of points in the [-1, 1] interval that should be used for testing the integration method. Defaults to [10, 100, 1000].
        iterations (int, optional): Number of average measurement for every number of integration points. Defaults to 1.

    Returns:
        [numpy]: runtime measurements for all the different number of integration points. Runtime measurements include sleep period of cpus.
    """
    runtime_measurement = []
    for n in N:
        measure = 0
        for m in range(iterations):
            start_time = perf_counter()
            integrate_N_points(method, fn, n, scipy_based)
            stop_time = perf_counter()
            measure = measure + stop_time - start_time
        runtime_measurement.append(measure / iterations)

    return np.array(runtime_measurement)

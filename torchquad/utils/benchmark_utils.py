import numpy as np
import torch
from time import perf_counter
from scipy.integrate import nquad


def _N_point_grid(N, dim=3, domain=[[-1, 1], [-1, 1], [-1, 1]]):
    """Creating a meshgrid for the dim-dimensional domain by using N points. 

    Args:
        N (integer): Number of points in the meshgrid.
        dim (integer): dimension of the domain. Defaults to 3.
        domain (list): domain. Defaults to [[-1, 1], [-1, 1], [-1, 1]].

    Returns:
        [torch]: N-points and dim-dimensional meshgrid.
    """

    N_dim = int(N ** (1.0 / dim))  # convert to points per dim
    h = torch.zeros([dim])

    grid_1d = []
    # Determine for each dimension grid points and mesh width
    for n in range(dim):
        grid_1d.append(torch.linspace(domain[n][0], domain[n][1], N_dim))
        h[n] = grid_1d[n][1] - grid_1d[n][0]

    # Get grid points
    points = torch.meshgrid(*grid_1d)

    # Flatten to 1D
    points = [p.flatten() for p in points]

    return torch.stack((tuple(points))).transpose(0, 1)


def _integrate_N_points(method, fn, dim, N, scipy_based=True):
    """Performing the 1-dimensional integration of fn on the integration interval [-1,1] by using N points. 

    Args:
        method (function): scipy-based or torchquad integration method.
        dim (integer): dimensionality of the integral
        fn (function): function to integrate.
        N (integer): number of points in the [-1, 1] interval. Defaults to [10, 100, 1000].
        scipy_based (bool, optional): if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
    """
    if dim == 1:
        if scipy_based:
            sp = torch.linspace(-1, 1, N).unsqueeze(1).cpu()
            return method(y=fn(sp).cpu(), x=sp[:, 0])
        else:
            return method(fn=fn, N=N, dim=1, integration_domain=[[-1, 1]])
    else:
        if scipy_based:
            integration_domain = [[-1, 1], [-1, 1], [-1, 1]]
            func = lambda x, y, z: fn(torch.tensor([[x, y, z]]))
            opts = {"limit": N}
            return nquad(func, integration_domain, opts=opts, full_output=True)
        else:
            return method(
                fn=fn, N=N, dim=3, integration_domain=[[-1, 1], [-1, 1], [-1, 1]]
            )


def _runtime_measure(
    method, fn, dim, scipy_based=True, N=[10, 100, 1000], iterations=1
):
    """Performing runtime measurement of integration of fn over the interval [-1, 1] through the method ""method"" by using ""N"" points. For each value ""N"", a number of iterations ""iterations"" is used to calculate the average.

    Args:
        method (function): scipy-based or torchquad integration method.
        fn (function): function to integrate.
        dim (integer): integral dimensionality.
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
            _integrate_N_points(method, fn, dim, n, scipy_based)
            stop_time = perf_counter()
            measure = measure + stop_time - start_time
        runtime_measurement.append(measure / iterations)

    return np.array(runtime_measurement)


def _get_integral(method, fn, dim, scipy_based=True, N=[10, 100, 1000]):
    """Calculating the integral value of fn over the interval [-1, 1] through the method ""method"" by using ""N"" points.

    Args:
        method (function): scipy-based or torchquad integration method.
        fn (function): function to integrate.
        dim (integer): dimensionality of integration.
        scipy_based (bool, optional):  if True, a scipy-based method is used; otherwise, a torchquad method is used. Defaults to True.
        N (list, optional): list containing different number of points in the [-1, 1] interval that should be used for testing the integration method. Defaults to [10, 100, 1000].

    Returns:
        [numpy]: integral values.
    """
    integral_values = []

    for n in N:
        if scipy_based:
            integral_values.append(_integrate_N_points(method, fn, dim, n, scipy_based))
        else:
            integral_values.append(
                _integrate_N_points(method, fn, dim, n, scipy_based).cpu()
            )

    return np.array(integral_values)

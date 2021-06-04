from torchquad import Boole, Trapezoid, Simpson, VEGAS, MonteCarlo

# TODO test these in the future
# from ..plots.plot_convergence import plot_convergence
# from ..plots.plot_runtime import plot_runtime

from torchquad import enable_cuda
from torchquad import set_precision

import torch


def _deployment_test():
    """This method is used to check successful deployment of torch.
    It should not be used by users. We use it internally to check
    successful deployment of torchquad.
    """
    """[summary]
    """
    print()
    print()
    print()
    print("######## TESTING DEPLOYMENT ########")
    print()

    print("Testing CUDA init... ", end="")
    # Test inititialization on GPUs if available
    enable_cuda()
    set_precision("double")
    print("Done.")

    print("Initializing integrators... ", end="")
    tp = Trapezoid()
    sp = Simpson()
    boole = Boole()
    mc = MonteCarlo()
    vegas = VEGAS()
    print("Done.")

    def some_test_function(x):
        return torch.exp(x) * torch.pow(x, 2)

    print("Testing integrate functions... ", end="")
    tp.integrate(some_test_function, dim=1, N=101)
    sp.integrate(some_test_function, dim=1, N=101)
    boole.integrate(some_test_function, dim=1, N=101)
    mc.integrate(some_test_function, dim=1, N=101)
    vegas.integrate(some_test_function, dim=1, N=101)
    print("Done.")

    print()
    print()
    print()
    print("######## ALL DONE. ########")
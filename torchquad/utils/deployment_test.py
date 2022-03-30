from torchquad import Boole, Trapezoid, Simpson, VEGAS, MonteCarlo

# TODO test these in the future
# from ..plots.plot_convergence import plot_convergence
# from ..plots.plot_runtime import plot_runtime

from torchquad import enable_cuda
from torchquad import set_precision
from torchquad import set_log_level
from loguru import logger


def _deployment_test():
    """This method is used to check successful deployment of torch.
    It should not be used by users. We use it internally to check
    successful deployment of torchquad.
    """
    """[summary]
    """
    import torch

    set_log_level("INFO")
    logger.info("####################################")
    logger.info("######## TESTING DEPLOYMENT ########")
    logger.info("####################################")
    logger.info("")

    logger.info("Testing CUDA init... ")
    # Test inititialization on GPUs if available
    enable_cuda()
    set_precision("double")
    logger.info("Done.")

    logger.info("")
    logger.info("####################################")

    logger.info("Initializing integrators... ")
    tp = Trapezoid()
    sp = Simpson()
    boole = Boole()
    mc = MonteCarlo()
    vegas = VEGAS()
    logger.info("Done.")

    def some_test_function(x):
        return torch.exp(x) * torch.pow(x, 2)

    logger.info("")
    logger.info("####################################")

    logger.info("Testing integrate functions... ")
    tp.integrate(some_test_function, dim=1, N=101)
    sp.integrate(some_test_function, dim=1, N=101)
    boole.integrate(some_test_function, dim=1, N=101)
    mc.integrate(some_test_function, dim=1, N=101)
    vegas.integrate(some_test_function, dim=1, N=300)
    logger.info("Done.")
    logger.info("")

    logger.info("####################################")
    logger.info("############ ALL DONE. #############")
    logger.info("####################################")

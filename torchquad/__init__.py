import os
import logging

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

# TODO: Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.integration_grid import IntegrationGrid
from .integration.monte_carlo import MonteCarlo
from .integration.trapezoid import Trapezoid
from .integration.simpson import Simpson
from .integration.boole import Boole
from .integration.vegas import VEGAS

from .plots.plot_convergence import plot_convergence
from .plots.plot_runtime import plot_runtime

from .utils.enable_cuda import enable_cuda
from .utils.set_precision import set_precision
from .utils.deployment_test import _deployment_test

__all__ = [
    "IntegrationGrid",
    "MonteCarlo",
    "Trapezoid",
    "Simpson",
    "Boole",
    "VEGAS",
    "plot_convergence",
    "plot_runtime",
    "enable_cuda",
    "set_precision",
    "set_log_level",
    "_deployment_test",
]

# Initialize logger
logger = logging.getLogger(__name__)


def set_log_level(level=logging.WARN):
    """Allows setting global log level for the application.

    Args:
        level (logging.level, optional): Level to set, available are (logging.DEBUG,logging.INFO,logging.WARN,logging.ERROR). Defaults to logging.WARN.
    """
    logger.setLevel(level)
    if level == 10:
        logger.info("Log level set to debug")
    elif level == 20:
        logger.info("Log level set to info")
    # we still store it in case we might write some logfile or sth later
    elif level == 30:
        logger.info("Log level set to warn")
    elif level == 40:
        logger.info("Log level set to error")

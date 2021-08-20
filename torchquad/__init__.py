import os
from loguru import logger

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

from .utils.set_log_level import set_log_level
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

set_log_level("INFO")
logger.info("Initializing torchquad.")

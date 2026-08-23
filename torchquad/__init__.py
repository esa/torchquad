import os

from loguru import logger

__version__ = "0.5.0"

# Disable torchquad's own log records by default so importing the library never
# adds output to a host application's loguru configuration (issue #184).
# set_log_level() re-enables them when the user opts into logging.
logger.disable("torchquad")

# TODO: Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.integration_grid import IntegrationGrid
from .integration.monte_carlo import MonteCarlo
from .integration.trapezoid import Trapezoid
from .integration.simpson import Simpson
from .integration.boole import Boole
from .integration.vegas import VEGAS
from .integration.vegas_result import VEGASResult
from .integration.gaussian import GaussLegendre
from .integration.gaussian import Gaussian
from .integration.grid_integrator import GridIntegrator
from .integration.base_integrator import BaseIntegrator

from .integration.rng import RNG
from .integration.qmc import Sobol


from .utils.set_log_level import set_log_level
from .utils.enable_cuda import enable_cuda
from .utils.set_precision import set_precision
from .utils.set_up_backend import set_up_backend

# _deployment_test is an internal post-release self-test. It is deliberately kept
# out of __all__ (and keeps its leading underscore) so it does not appear in the
# public API or user autocomplete, while staying reachable as torchquad._deployment_test
# for the wheel-smoke CI job and the install self-check documented in the README.
# The redundant `as` marks it as an intentional re-export so it is not flagged as unused.
from .utils.deployment_test import _deployment_test as _deployment_test

__all__ = [
    "__version__",
    "GridIntegrator",
    "BaseIntegrator",
    "IntegrationGrid",
    "MonteCarlo",
    "Trapezoid",
    "Simpson",
    "Boole",
    "VEGAS",
    "VEGASResult",
    "GaussLegendre",
    "Gaussian",
    "RNG",
    "Sobol",
    "enable_cuda",
    "set_precision",
    "set_log_level",
    "set_up_backend",
]

# Opt in to logging from the environment. Only an explicitly set
# TORCHQUAD_LOG_LEVEL turns torchquad's records on, so the library stays silent
# by default; set_log_level() does the same thing at runtime.
if "TORCHQUAD_LOG_LEVEL" in os.environ:
    set_log_level(os.environ["TORCHQUAD_LOG_LEVEL"])
    logger.info("Initializing torchquad.")

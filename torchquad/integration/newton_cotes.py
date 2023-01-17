from loguru import logger
from autoray import infer_backend
from autoray import numpy as anp

from .grid_integrator import GridIntegrator
from .integration_grid import IntegrationGrid
from .utils import _setup_integration_domain

class NewtonCotes(GridIntegrator):
    """The abstract integrator that Composite Newton Cotes integrators inherit from"""

    def __init__(self):
        super().__init__()
from .grid_integrator import GridIntegrator


class NewtonCotes(GridIntegrator):
    """The abstract integrator that Composite Newton Cotes integrators inherit from"""

    def __init__(self):
        super().__init__()

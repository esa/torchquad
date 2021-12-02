from autoray import numpy as anp
from autoray import infer_backend
from time import perf_counter
from loguru import logger

from .utils import (
    _linspace_with_grads,
    _check_integration_domain,
    _setup_integration_domain,
)


class IntegrationGrid:
    """This class is used to store the integration grid for methods like Trapezoid or Simpsons, which require a grid."""

    points = None  # integration points
    h = None  # mesh width
    _N = None  # number of mesh points
    _dim = None  # dimensionality of the grid
    _runtime = None  # runtime for the creation of the integration grid

    def __init__(self, N, integration_domain):
        """Creates an integration grid of N points in the passed domain. Dimension will be len(integration_domain)

        Args:
            N (int): Total desired number of points in the grid (will take next lower root depending on dim)
            integration_domain (list or backend tensor): Domain to choose points in, e.g. [[-1,1],[0,1]]. It also determines the numerical backend (if it is a list, the backend is "torch").
        """
        start = perf_counter()
        self._check_inputs(N, integration_domain)
        if infer_backend(integration_domain) == "builtins":
            integration_domain = _setup_integration_domain(
                len(integration_domain), integration_domain, backend="torch"
            )
        self._dim = integration_domain.shape[0]

        # TODO Add that N can be different for each dimension
        # A rounding error occurs for certain numbers with certain powers,
        # e.g. (4**3)**(1/3) = 3.99999... Because int() floors the number,
        # i.e. int(3.99999...) -> 3, a little error term is useful
        self._N = int(N ** (1.0 / self._dim) + 1e-8)  # convert to points per dim

        logger.opt(lazy=True).debug(
            "Creating {dim}-dimensional integration grid with {N} points over {dom}",
            dim=lambda: str(self._dim),
            N=lambda: str(N),
            dom=lambda: str(integration_domain),
        )

        # Check if domain requires gradient
        if hasattr(integration_domain, "requires_grad"):
            requires_grad = integration_domain.requires_grad
        else:
            requires_grad = False

        grid_1d = []
        # Determine for each dimension grid points and mesh width
        for dim in range(self._dim):
            grid_1d.append(
                _linspace_with_grads(
                    integration_domain[dim][0],
                    integration_domain[dim][1],
                    self._N,
                    requires_grad=requires_grad,
                )
            )
        self.h = anp.stack(
            [grid_1d[dim][1] - grid_1d[dim][0] for dim in range(self._dim)],
            like=integration_domain,
        )

        logger.opt(lazy=True).debug("Grid mesh width is {h}", h=lambda: str(self.h))

        # Get grid points
        points = anp.meshgrid(*grid_1d)
        self.points = anp.stack(
            [mg.ravel() for mg in points], axis=1, like=integration_domain
        )

        logger.info("Integration grid created.")

        self._runtime = perf_counter() - start

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to IntegrationGrid.")
        dim = _check_integration_domain(integration_domain)

        if N < 2:
            raise ValueError("N has to be > 1.")

        if N ** (1.0 / dim) < 2:
            raise ValueError(
                "Cannot create a ",
                dim,
                "-dimensional grid with ",
                N,
                " points. Too few points per dimension.",
            )

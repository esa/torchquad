import torch
from time import perf_counter
from loguru import logger

from .utils import _linspace_with_grads


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
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
        """
        start = perf_counter()
        self._check_inputs(N, integration_domain)
        self._dim = len(integration_domain)

        # TODO Add that N can be different for each dimension
        # A rounding error occurs for certain numbers with certain powers,
        # e.g. (4**3)**(1/3) = 3.99999... Because int() floors the number,
        # i.e. int(3.99999...) -> 3, a little error term is useful
        self._N = int(N ** (1.0 / self._dim) + 1e-8)  # convert to points per dim

        self.h = torch.zeros([self._dim])

        logger.debug(
            "Creating "
            + str(self._dim)
            + "-dimensional integration grid with "
            + str(N)
            + " points over"
            + str(integration_domain),
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
            self.h[dim] = grid_1d[dim][1] - grid_1d[dim][0]

        logger.debug("Grid mesh width is " + str(self.h))

        # Get grid points
        points = torch.meshgrid(*grid_1d)

        # Flatten to 1D
        points = [p.flatten() for p in points]

        self.points = torch.stack((tuple(points))).transpose(0, 1)

        logger.info("Integration grid created.")

        self._runtime = perf_counter() - start

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to IntegrationGrid.")
        dim = len(integration_domain)

        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

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

        for bounds in integration_domain:
            if len(bounds) != 2:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
            if bounds[0] > bounds[1]:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )

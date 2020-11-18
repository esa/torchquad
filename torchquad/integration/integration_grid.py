import torch
import logging

logger = logging.getLogger(__name__)


class IntegrationGrid:
    """This class is used to store the integration grid for methods like Trapezoid or Simpsons which require a grid
    """

    _points = None  # integration points
    _h = None  # mesh width
    _N = None  # number of mesh points
    _dim = None  # dimensionality of the grid

    def __init__(self, N, integration_domain):
        """Creates an integration grid of N points in the passed domain. Dimension will be len(integration_domain)

        Args:
            N (int): Number of points in the grid per dimension
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
        """
        self._check_inputs(N, integration_domain)
        self._dim = len(integration_domain)
        self._N = int(N ** (1.0 / self._dim))

        logger.debug(
            "Creating "
            + str(self._dim)
            + "-dimensional integration grid with "
            + str(N)
            + " points  over"
            + str(integration_domain),
        )

        # TODO expand to more than one dim
        grid_1d = torch.linspace(integration_domain[0][0], integration_domain[0][1], N)

        self._h = grid_1d[1] - grid_1d[0]

        logger.debug("Grid mesh width is " + str(self._h))

        self._points = torch.tensor([x for x in grid_1d])

        logger.info("Integration grid created.")

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to IntegrationGrid.")
        dim = len(integration_domain)

        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

        if N ** (1.0 / dim) < 1:
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

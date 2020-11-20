import logging

logger = logging.getLogger(__name__)


class BaseIntegrator:
    """The (abstract) integrator that all other integrators inherit from. Provides no explicit definitions for methods.
    """

    # Function to evaluate
    _fn = None

    # Dimensionality of function to evaluate
    _dim = None

    # Integration domain
    _integration_domain = None

    # Number of function evaluations
    _nr_of_fevals = None

    # Convergence order
    _convergence_order = None

    def __init__(self):
        self._nr_of_fevals = 0
        self._convergence_order = -1

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def _eval(self, points):
        """Evaluates the function at the passed points and updates nr_of_evals

        Args:
            points (torch.tensor): integration points
        """
        self._nr_of_fevals += len(points)
        return self._fn(points)

    def _check_inputs(self, dim=None, N=None, integration_domain=None):
        """Used to check input validity"""
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

            if integration_domain is not None:
                if dim != len(integration_domain):
                    raise ValueError(
                        "Dimension of integration_domain needs to be match passed dim."
                    )

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
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


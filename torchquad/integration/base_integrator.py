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

    def __init__(self):
        pass

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )


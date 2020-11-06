class BaseIntegrator:
    """The (abstract) integrator that all other integrators inherit from. Provides no explicit definitions for methods.
    """

    # Function to evaluate
    fn = None

    # Dimensionality of function to evaluate
    dim = None

    # Integration domain
    integration_domain = None

    # Number of function evaluations
    nr_of_fevals = None

    def __init__(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )


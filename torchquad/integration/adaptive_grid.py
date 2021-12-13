import torch
from time import perf_counter
from loguru import logger
import numpy as np

from .utils import _linspace_with_grads


class Subdomain:
    """A subdomain of the adaptiveGrid. Contains the points, fvals at those, etc."""

    integration_domain = None  # Spatial dimensions of the subdomain
    refinement_level = None  # Refinement level of the subdomain, 1 - initially, incremented on refinement
    points = None  # Points of the subdomain
    fval = None  # Function values at the points
    N_per_dim = None  # Number of points in each dimension
    h = None  # Mesh width
    requires_integral_value = True
    _integral_value = (
        None  # Integral value in this domain, has to be set for refinement
    )

    def __init__(self, N, integration_domain):
        """Initialize a subdomain.

        Args:
            N (int): Number of points to use for this subdomain. (will take next lower root depending on dim)
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
        """
        self._dim = len(integration_domain)
        self.integration_domain = integration_domain
        self.refinement_level = 1
        self.N_per_dim = int(N ** (1.0 / self._dim) + 1e-8)  # convert to points per dim
        self.h = torch.zeros([self._dim])

        logger.debug(
            "Creating "
            + str(self._dim)
            + "-dimensional subdomain with "
            + str(N)
            + " points over"
            + str(integration_domain),
        )
        self._create_grid()
        logger.debug("Subdomain created.")

    def _create_grid(self):
        # Check if domain requires gradient
        if hasattr(self.integration_domain, "requires_grad"):
            requires_grad = self.integration_domain.requires_grad
        else:
            requires_grad = False

        grid_1d = []
        # Determine for each dimension grid points and mesh width
        for dim in range(self._dim):
            grid_1d.append(
                _linspace_with_grads(
                    self.integration_domain[dim][0],
                    self.integration_domain[dim][1],
                    self.N_per_dim,
                    requires_grad=requires_grad,
                )
            )
            self.h[dim] = grid_1d[dim][1] - grid_1d[dim][0]

        logger.debug("Grid mesh width is " + str(self.h))

        # Get grid points
        points = torch.meshgrid(*grid_1d)

        # TODO We could already allocate larger tensors here to avoid later reallocations
        self.points = torch.stack(list(map(torch.ravel, points)), dim=1)
        self.fval = torch.zeros_like(self.points)

    def get_points_to_eval(self):
        """Get points to evaluate the function at.

        Returns:
            [torch.Tensor]: Points to evaluate the function at.
        """
        # TODO Implement only giving the points that were not evaluated before

        return self.points

    def set_fvals(self, vals):
        """Set the function values at the points."""
        # TODO Set only new points, maybe add some sanity check here?

        self.fval = vals

    def set_integral(self, val):
        """Set the integral value

        Args:
            val ([type]): [description]
        """
        self._integral_value = val
        self.requires_integral_value = False

    def refine(self):
        """Refines the subdomain by doubling the number of points in it"""
        if self.integral_value is None or self.requires_integral_value:
            raise RuntimeError("Integral value has to be set for subdomain refinement.")

        # Double the number of grid points
        self.N_per_dim *= 2
        self.refinement_level += 1
        self.h = self.h / 2.0

        # Recreate grid
        # TODO find a smarter way to do without reallocating all the tensors
        self._create_grid()

        # After refinement, integral value is outdated
        self._integral_value = None
        self.requires_integral_value = True


class AdaptiveGrid:
    """This class is used to store the integration grid for methods like AdaptiveTrapezoid, which require an adaptive grid."""

    _dim = None  # dimensionality of the grid
    _runtime = None  # runtime for the creation of the adaptive grid
    subdomains = None  # List of subdomains
    _subdomains_per_dim = None  # Number of subdomains in each dimension
    _max_refinement_level = None  # Maximum refinements called on a subdomain

    def __init__(self, N, integration_domain, subdomains_per_dim, max_refinement_level):
        """Creates an integration grid of N points in the passed domain. Dimension will be len(integration_domain)

        Args:
            N (int): Total desired number of points in the grid (will take next lower root depending on dim)
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
            subdomains_per_dim (int): Number of subdomains in each dimension.
            max_refinement_level (int): Maximum refinement level for the subdomains.
        """
        start = perf_counter()
        self._check_inputs(N, integration_domain)
        self._dim = len(integration_domain)
        self._subdomains_per_dim = subdomains_per_dim
        self._max_refinement_level = max_refinement_level

        # Check if domain requires gradient
        if hasattr(integration_domain, "requires_grad"):
            requires_grad = integration_domain.requires_grad
        else:
            requires_grad = False

        logger.debug("Initializing subdomains")
        self._create_subdomains(requires_grad)

        self._runtime += perf_counter() - start

    def _create_subdomains(self, requires_grad, N_per_subdomain):
        """Create all subdomains."""
        self.subdomains = []

        # Compute total number of subdomains
        total_subdomains = self._subdomains_per_dim ** self._dim

        # We create the subdomains by first identifying all lower and upper bounds
        # For this we first create a meshgrid for the points at them
        lower_bounds = []
        upper_bounds = []
        # Determine for each dimension subdomain points and mesh width
        for dim in range(self._dim):
            offset = (
                self.integration_domain[dim][1] - self.integration_domain[dim][0]
            ) / self._subdomains_per_dim
            lower_bounds.append(
                _linspace_with_grads(
                    self.integration_domain[dim][0],
                    self.integration_domain[dim][1] - offset,
                    self._subdomains_per_dim,
                    requires_grad=requires_grad,
                )
            )
            upper_bounds.append(
                _linspace_with_grads(
                    self.integration_domain[dim][0] + offset,
                    self.integration_domain[dim][1],
                    self._subdomains_per_dim,
                    requires_grad=requires_grad,
                )
            )

        # Store them in a single tensor for convenience
        bounds = []
        for d in range(self._dim):
            bounds.append(torch.stack([lower_bounds[d], upper_bounds[d]], dim=1))

        # Convert to a 1D single tensor to make it work in abritrary dimensions
        # Otherwise we would have to do n for loops which is programmatically
        # rather challenging...
        bounds = torch.cat(bounds)

        # Compute the indices of all permutations from the bounds
        permutations_per_dim = []

        # For each dimension we compute the permutations
        for d in range(self._dim):
            # Get the permutations for this dimension (alternating values depending on current dim)
            # e.g. in the last one we want [0,1,...,n_subdomains]
            values = np.array(range(self._subdomains_per_dim)) + (
                (self._dim - d - 1) * self._subdomains_per_dim
            )

            # Compute how often we need to repeat the values
            # and do it
            values = list(np.repeat(values, self._subdomains_per_dim ** d))
            repetitions = int(total_subdomains / len(values))
            indices = list(values) * repetitions

            permutations_per_dim.append(np.array(indices))

        # Reverse order of dimensions for compatibility with bounds tensor
        permutations_per_dim.reverse()
        permutations = np.dstack(permutations_per_dim).squeeze()

        # Now we get the actual bounds for each of them
        for i in range(total_subdomains):
            subdomain_bound = []
            for d in range(self._dim):
                index = permutations[i, d]
                subdomain_bound.append(bounds[index])
            self.subdomains.append(Subdomain(N_per_subdomain, subdomain_bound))

    def _compute_refinement_criterion(self, subdomain):
        """Computes the refinement criterion for each subdomain"""
        raise NotImplementedError()

    def refine(self):
        """Refines the grid by doubling the number of points in the subdomain with largest variance."""
        criterion_values = []
        for subdomain in self.subdomains:
            raise NotImplementedError()
            # TODO compute criterion

        domain_to_refine = self.subdomains[np.argmax(criterion_values)]
        domain_to_refine.refine()

    def get_next_eval_points(self):
        """Returns the next evaluation points for the adaptive grid."""
        raise NotImplementedError()

    def set_fvals(self, fvals):
        """Sets the fvals in the matching subdomains"""
        raise NotImplementedError()

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to AdaptiveGrid.")
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

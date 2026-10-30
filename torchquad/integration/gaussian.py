import numpy
from autoray import numpy as anp

from .grid_integrator import GridIntegrator

#: Largest number of nodes *per dimension* a Gaussian rule will build.
#:
#: The nodes come from the eigenvalues of an ``n x n`` companion matrix, so the
#: cost is quadratic in memory and cubic in time: 2 000 nodes take 0.16 s and
#: 32 MB, 10 000 take about 20 s and 0.75 GiB, and 200 000 raise a bare
#: ``MemoryError: Unable to allocate 298. GiB``. Without this limit a caller who
#: asks for ``GaussLegendre().integrate(..., dim=1, N=10**6)`` gets that NumPy
#: error, which names neither torchquad nor a way forward, after an unbounded
#: wait. Passing the limit is nearly always a mistake rather than a real need:
#: Gauss-Legendre converges exponentially on smooth integrands and reaches
#: double precision within a few hundred nodes, so more nodes buy nothing.
MAX_NODES_PER_DIMENSION = 10_000


class Gaussian(GridIntegrator):
    """
    Base method for Gaussian Quadrature.  Different Gaussian methods should inherit from this class, and override as necessary methods.
    Default behaviour is Gauss-Legendre quadrature on [-1,1] (i.e., this "parent" class should __not__ be used directly with other integration domains, and for this parent class `integration_domain` as an argument to `integrate` is ignored internally).

    For an example of how to properly override the behavior to acheive different Gaussian Integration methods, please see the `Custom Integrators` section of the Tutorial or the implementation of `GaussLegendre`.

    The primary methods/attributes of interest to override are `_root_fn` (for different polynomials, like `numpy.polynomial.legendre.leggauss`), `_apply_composite_rule` (as in other integration methods), and `_resize_roots` (for handling different integration domains).

    Attributes:
        _root_fn (function): A function that returns roots and weights like `numpy.polynomial.legendre.leggauss`.
        _root_args (tuple): a way of adding information to be passed into `_root_fn` as needed.  This is then used when caching roots/weights to potentially distinguish different calls to `_root_fn` based on arguments.
        _cache (dict): a cache for roots and weights, used internally.
    """

    def __init__(self):
        super().__init__()
        self._root_fn = numpy.polynomial.legendre.leggauss
        self._root_args = ()
        self._cache = {}

    def integrate(self, fn, dim, N=8, integration_domain=None, backend=None, args=None):
        """Integrates the passed function on the passed domain using a Gaussian rule (Gauss-Legendre on [-1,1] as a default).

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int or sequence of int, optional): Total number of sample points, or the number of points in each dimension. Should be odd. Defaults to 3 points per dimension if None is given.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.   It also determines the numerical backend if possible.
            backend (string, optional): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain. Defaults to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.
            args (list or tuple, optional): Extra arguments passed to the integrand as ``fn(points, *args)``. Defaults to None.

        Returns:
            backend-specific number: Integral value
        """
        return super().integrate(fn, dim, N, integration_domain, backend, args=args)

    def _weights(self, N, dim, backend, requires_grad=False):
        """return the weights, broadcast across the dimensions, generated from the polynomial of choice

        Args:
            N (int or sequence of int): Number of nodes per dimension.
            dim (int): number of dimensions
            backend (string): which backend array to return
            requires_grad (bool, optional): whether the returned weights should track gradients (torch only). Defaults to False.

        Returns:
            backend tensor: the weights
        """
        counts = [N] * dim if isinstance(N, int) else list(N)
        weights = [
            anp.array(self._cached_points_and_weights(count)[1], like=backend) for count in counts
        ]
        if backend == "torch":
            for weight in weights:
                weight.requires_grad = requires_grad
        weight_grids = anp.meshgrid(*weights, indexing="ij")
        return anp.prod(anp.stack(list(weight_grids), like=backend, axis=0), axis=0).ravel()

    def _roots(self, N, backend, requires_grad=False):
        """return the roots generated from the polynomial of choice

        Args:
            N (int): number of nodes
            backend (string): which backend array to return
            requires_grad (bool, optional): whether the returned roots should track gradients (torch only). Defaults to False.

        Returns:
            backend tensor: the roots
        """
        roots = anp.array(self._cached_points_and_weights(N)[0], like=backend)
        if requires_grad:
            roots.requires_grad = True
        return roots

    @property
    def _grid_func(self):
        """
        function for generating a grid to be integrated over i.e., the polynomial roots, resized to the domain.
        """

        def f(integration_domain, N, requires_grad, backend=None):
            return self._resize_roots(integration_domain, self._roots(N, backend, requires_grad))

        return f

    def _resize_roots(self, integration_domain, roots):  # scale from [-1,1] to [a,b]
        """Resize the roots based on domain of [a,b].  Default behavior is to simply return the roots, unsized by `integraton_domain`.

        Args:
            integration_domain (backend tensor): domain
            roots (backend tensor): polynomial nodes

        Returns:
            backend tensor: rescaled roots
        """
        return roots

    # credit for the idea https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/integrate/_quadrature.py#L72
    def _cached_points_and_weights(self, N):
        """wrap the calls to get weights/roots in a cache

        Args:
            N (int): number of nodes to return

        Returns:
            tuple: nodes and weights

        Raises:
            NotImplementedError: If N is not an int and has no ``item`` method to convert it to one.
            ValueError: If N exceeds :data:`MAX_NODES_PER_DIMENSION`, since building
                that many nodes needs a quadratically large intermediate matrix.
        """
        _root_args = (N, *self._root_args)
        if not isinstance(N, int):
            if hasattr(N, "item"):
                _root_args = (N.item(), *self._root_args)
            else:
                raise NotImplementedError(f"N {N} is not an int and lacks an `item` method")
        if _root_args in self._cache:
            return self._cache[_root_args]

        # Check before calling _root_fn: past a few tens of thousands of nodes it
        # raises a bare NumPy MemoryError about an n x n array, which says
        # nothing about which argument caused it or what to do instead.
        nodes_per_dimension = _root_args[0]
        if nodes_per_dimension > MAX_NODES_PER_DIMENSION:
            required_gib = 8 * nodes_per_dimension**2 / 1024**3
            raise ValueError(
                f"Gaussian quadrature needs {nodes_per_dimension} nodes per dimension, above "
                f"the limit of {MAX_NODES_PER_DIMENSION}. The nodes are the eigenvalues of an "
                f"n x n matrix, so this one would need about {required_gib:.1f} GiB and "
                "correspondingly long to diagonalize. Note that N is divided across the "
                "dimensions, so this is N**(1/dim) rather than N itself. Gauss-Legendre "
                "converges exponentially on smooth integrands and reaches double precision "
                "within a few hundred nodes, so a lower N is very likely to be just as "
                "accurate; for a genuinely large number of points, use a Newton-Cotes rule "
                "(Trapezoid, Simpson, Boole) or MonteCarlo instead."
            )

        self._cache[_root_args] = self._root_fn(*_root_args)
        return self._cache[_root_args]

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                0.5
                * (domain[cur_dim][1] - domain[cur_dim][0])
                * anp.sum(cur_dim_areas, axis=len(cur_dim_areas.shape) - 1)
            )
        return cur_dim_areas


class GaussLegendre(Gaussian):
    """Gauss Legendre quadrature rule in torch for any domain [a,b]. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature.

    Examples
    --------
    >>> gl=torchquad.GaussLegendre()
    >>> integral = gl.integrate(lambda x:np.sin(x), dim=1, N=101, integration_domain=[[0,5]]) #integral from 0 to 5 of np.sin(x)
    |TQ-INFO| Computed integral was 0.7163378000259399 #analytic result = 1-np.cos(5)"""

    def __init__(self):
        super().__init__()

    def _resize_roots(self, integration_domain, roots):  # scale from [-1,1] to [a,b]
        a = integration_domain[0]
        b = integration_domain[1]
        return ((b - a) / 2) * roots + ((a + b) / 2)

"""The Genz test-function family, with closed-form integrals in any dimension.

``tests/integration_test_functions.py`` exists to check correctness, and it is
built for that: its integrands are mostly low-order polynomials, almost all
additively separable, and only dimensions 1, 3 and 10 are defined -- the 10-D
set containing a single function. Those properties make it a poor basis for a
convergence plot. Polynomials are exact for Gauss-Legendre at every N, and
separable integrands flatter quasi-Monte Carlo well beyond what a user should
expect on their own problem.

The Genz family (Genz 1984) is the field's standard alternative: integrand
shapes parameterised so difficulty can be held fixed as the dimension grows,
each with a closed-form integral over the unit hypercube. That makes the family
usable both as a convergence benchmark and as a dimension-scaling benchmark.

All functions are defined on ``[0, 1]^d``. ``a`` controls difficulty and ``u``
shifts the feature; both are per-dimension and may be given as a scalar to
broadcast. Integrands evaluate through ``autoray``, so they run on any torchquad
backend, while every reference value is computed in NumPy float64.
"""

import itertools
import math

import numpy as np
from autoray import numpy as anp
from scipy.special import erf


def _as_array(value, dim):
    """Broadcast a scalar or sequence to a length-``dim`` float64 array.

    Args:
        value (float or sequence): Per-dimension parameter, or a scalar to
            broadcast across all dimensions.
        dim (int): Number of dimensions.

    Returns:
        np.ndarray: Array of shape ``(dim,)``.

    Raises:
        ValueError: If a sequence is given whose length is neither 1 nor ``dim``.
    """
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.size == 1:
        return np.full(dim, float(array[0]))
    if array.size != dim:
        raise ValueError(f"Expected 1 or {dim} parameters, got {array.size}.")
    return array


class GenzFunction:
    """One Genz integrand together with its exact integral over ``[0, 1]^d``.

    Attributes:
        name (str): Human-readable name, used for plot labels.
        dim (int): Dimensionality.
        a (np.ndarray): Per-dimension difficulty parameters.
        u (np.ndarray): Per-dimension shift parameters.
        exact (float): Closed-form integral over the unit hypercube.
    """

    def __init__(self, name, dim, a, u, evaluate, exact):
        """Initialize a Genz integrand.

        Args:
            name (str): Human-readable name.
            dim (int): Dimensionality.
            a (np.ndarray): Per-dimension difficulty parameters.
            u (np.ndarray): Per-dimension shift parameters.
            evaluate (callable): Maps an ``(N, dim)`` point array to ``(N,)``
                integrand values.
            exact (float): Closed-form integral over the unit hypercube.
        """
        self.name = name
        self.dim = dim
        self.a = a
        self.u = u
        self.exact = exact
        self._evaluate = evaluate

    def __call__(self, points):
        """Evaluate the integrand on a batch of points.

        Args:
            points (backend tensor): ``(N, dim)`` points in ``[0, 1]^d``.

        Returns:
            backend tensor: ``(N,)`` integrand values.
        """
        return self._evaluate(points)

    @property
    def integration_domain(self):
        """list: The unit hypercube, in the form torchquad expects."""
        return [[0.0, 1.0]] * self.dim

    def relative_error(self, value):
        """Relative error of an estimate against the closed-form integral.

        Args:
            value (float): Estimated value of the integral.

        Returns:
            float: ``|value - exact| / |exact|``.
        """
        return abs(float(value) - self.exact) / abs(self.exact)


def oscillatory(dim, a=1.0, u=0.5):
    """Build the oscillatory integrand ``cos(2*pi*u_0 + sum(a_i x_i))``.

    Difficulty grows with ``sum(a)``: the integrand oscillates more often across
    the cube, which defeats low-order rules and rewards high-order ones.

    Args:
        dim (int): Dimensionality.
        a (float or sequence, optional): Difficulty parameters. Defaults to 1.0.
        u (float or sequence, optional): Phase shift; only ``u[0]`` is used.
            Defaults to 0.5.

    Returns:
        GenzFunction: The integrand and its exact integral.
    """
    a = _as_array(a, dim)
    u = _as_array(u, dim)
    offset = 2.0 * np.pi * u[0]

    def evaluate(points):
        coefficients = anp.array(a, like=points, dtype=points.dtype)
        return anp.cos(offset + anp.sum(points * coefficients, axis=1))

    # int_0^1 exp(i a x) dx = exp(i a / 2) sin(a / 2) / (a / 2); take the real part.
    exact = np.cos(offset + 0.5 * np.sum(a)) * np.prod(np.sin(a / 2.0) / (a / 2.0))
    return GenzFunction("Oscillatory", dim, a, u, evaluate, float(exact))


def product_peak(dim, a=5.0, u=0.5):
    """Build the product peak ``prod(1 / (a_i^-2 + (x_i - u_i)^2))``.

    A sharp peak in every coordinate at once, narrowing as ``a`` grows.

    Args:
        dim (int): Dimensionality.
        a (float or sequence, optional): Inverse peak widths. Defaults to 5.0.
        u (float or sequence, optional): Peak locations. Defaults to 0.5.

    Returns:
        GenzFunction: The integrand and its exact integral.
    """
    a = _as_array(a, dim)
    u = _as_array(u, dim)

    def evaluate(points):
        widths = anp.array(a**-2.0, like=points, dtype=points.dtype)
        centers = anp.array(u, like=points, dtype=points.dtype)
        return anp.prod(1.0 / (widths + (points - centers) ** 2), axis=1)

    exact = np.prod(a * (np.arctan(a * (1.0 - u)) + np.arctan(a * u)))
    return GenzFunction("Product peak", dim, a, u, evaluate, float(exact))


def corner_peak(dim, a=1.0, u=0.5):
    """Build the corner peak ``(1 + sum(a_i x_i))^-(d+1)``.

    Mass concentrates in one corner of the cube, which punishes methods that
    spread their points uniformly.

    The closed form is an inclusion-exclusion sum over the cube's ``2^d``
    vertices, so this becomes impractical much beyond ``dim`` of about 20.

    Args:
        dim (int): Dimensionality.
        a (float or sequence, optional): Difficulty parameters. Defaults to 1.0.
        u (float or sequence, optional): Unused; accepted so every builder in
            the family shares one signature. Defaults to 0.5.

    Returns:
        GenzFunction: The integrand and its exact integral.
    """
    a = _as_array(a, dim)
    u = _as_array(u, dim)

    def evaluate(points):
        coefficients = anp.array(a, like=points, dtype=points.dtype)
        return (1.0 + anp.sum(points * coefficients, axis=1)) ** (-(dim + 1.0))

    # Integrating once per coordinate leaves an alternating sum over the vertices.
    # The sign is (-1)^|v|, not (-1)^(d-|v|): those differ by a factor (-1)^d, so
    # the wrong one is right in even dimensions and negates the result in odd ones.
    total = 0.0
    for vertex in itertools.product((0.0, 1.0), repeat=dim):
        vertex = np.asarray(vertex)
        total += (-1.0) ** vertex.sum() / (1.0 + np.dot(a, vertex))
    exact = total / (math.factorial(dim) * np.prod(a))
    return GenzFunction("Corner peak", dim, a, u, evaluate, float(exact))


def gaussian(dim, a=5.0, u=0.5):
    """Build the Gaussian ``exp(-sum(a_i^2 (x_i - u_i)^2))``.

    Smooth, but increasingly localized as ``a`` grows.

    Args:
        dim (int): Dimensionality.
        a (float or sequence, optional): Inverse widths. Defaults to 5.0.
        u (float or sequence, optional): Peak locations. Defaults to 0.5.

    Returns:
        GenzFunction: The integrand and its exact integral.
    """
    a = _as_array(a, dim)
    u = _as_array(u, dim)

    def evaluate(points):
        widths = anp.array(a, like=points, dtype=points.dtype)
        centers = anp.array(u, like=points, dtype=points.dtype)
        return anp.exp(-anp.sum((widths * (points - centers)) ** 2, axis=1))

    exact = np.prod(np.sqrt(np.pi) / (2.0 * a) * (erf(a * (1.0 - u)) + erf(a * u)))
    return GenzFunction("Gaussian", dim, a, u, evaluate, float(exact))


def c0_continuous(dim, a=2.0, u=0.5):
    """Build the C0 integrand ``exp(-sum(a_i |x_i - u_i|))``.

    Continuous but not differentiable at the peak. The kink caps the order any
    quadrature rule can achieve, which is what separates the deterministic rules
    from the stochastic ones on a convergence plot.

    Args:
        dim (int): Dimensionality.
        a (float or sequence, optional): Decay rates. Defaults to 2.0.
        u (float or sequence, optional): Kink locations. Defaults to 0.5.

    Returns:
        GenzFunction: The integrand and its exact integral.
    """
    a = _as_array(a, dim)
    u = _as_array(u, dim)

    def evaluate(points):
        rates = anp.array(a, like=points, dtype=points.dtype)
        centers = anp.array(u, like=points, dtype=points.dtype)
        return anp.exp(-anp.sum(rates * anp.abs(points - centers), axis=1))

    exact = np.prod((2.0 - np.exp(-a * u) - np.exp(-a * (1.0 - u))) / a)
    return GenzFunction("C0 continuous", dim, a, u, evaluate, float(exact))


#: The family in plot order, keyed by short name.
GENZ_FAMILY = {
    "oscillatory": oscillatory,
    "product_peak": product_peak,
    "corner_peak": corner_peak,
    "gaussian": gaussian,
    "c0_continuous": c0_continuous,
}

"""Validation of the Genz benchmark family's closed-form integrals.

These closed forms are the ground truth for every convergence plot in the
README, so an error in one would silently misrepresent a method's accuracy
rather than fail loudly. Each is checked against a numerical reference that
shares no code with the formula under test: adaptive cubature for the integrands
that do not factor, and a product of one-dimensional quadratures for those that
do. Both bugs these tests caught during development were real -- a sign error in
the corner peak that only showed in odd dimensions, and an unconverged reference
that made a correct C0 closed form look wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarking"))

from genz_functions import GENZ_FAMILY, _as_array  # noqa: E402

from scipy.integrate import quad  # noqa: E402

# scipy.integrate.cubature landed in SciPy 1.15; without it the non-separable
# integrands have no independent reference and their tests are skipped.
try:
    from scipy.integrate import cubature
except ImportError:
    cubature = None


# Integrands that factor into a product of one-dimensional integrals. For these
# the reference is a product of 1-D quadratures, which is both more accurate than
# adaptive cubature in several dimensions and able to be told where a kink is.
_SEPARABLE = {"product_peak", "gaussian", "c0_continuous"}


def _reference(name, function):
    """Integrate a Genz function numerically, independently of its closed form.

    Adaptive cubature is used for the integrands that do not factor. It is not
    used for the C0 integrand, where it reports ``not_converged`` on the kink and
    returns a value wrong in the eighth digit -- accurate enough to look like a
    disagreement with a correct closed form.

    Args:
        name (str): Key of the function in ``GENZ_FAMILY``.
        function (GenzFunction): The integrand to integrate.

    Returns:
        float: The reference value of the integral.
    """
    if name in _SEPARABLE:
        # For f(x) = prod_i g_i(x_i), fix every axis but one at c = 0.5 and
        # integrate the free axis. That gives F_i = (int g_i) * P / g_i(c), where
        # P = f(c, ..., c). Multiplying over i, the unknown g_i(c) cancel:
        #     prod_i F_i = I * P^(d-1)   =>   I = prod_i F_i / P^(d-1)
        # so the exact integral follows from d one-dimensional quadratures
        # without ever needing the individual factors.
        anchor = 0.5
        at_anchor = float(np.asarray(function(np.full((1, function.dim), anchor)))[0])
        assert at_anchor != 0.0, f"{name} vanishes at the anchor point; cannot normalize"

        product = 1.0
        for axis in range(function.dim):

            def slice_along(x, axis=axis):
                points = np.full((1, function.dim), anchor)
                points[0, axis] = x
                # quad calls this with a scalar and warns if handed back an
                # array, so unwrap rather than returning the (1,) batch.
                return float(np.asarray(function(points))[0])

            # points= tells quad where the C0 integrand's derivative jumps.
            value, _ = quad(slice_along, 0.0, 1.0, points=[float(function.u[axis])], limit=400)
            product *= value

        return product / at_anchor ** (function.dim - 1)

    result = cubature(function, np.zeros(function.dim), np.ones(function.dim), rtol=1e-11)
    assert result.status == "converged", (
        f"the reference integrator did not converge for {name}; its value cannot "
        f"be used to judge the closed form (status={result.status!r})"
    )
    return float(np.asarray(result.estimate).reshape(()))


@pytest.mark.skipif(cubature is None, reason="requires scipy>=1.15 for cubature")
@pytest.mark.parametrize("name", sorted(GENZ_FAMILY))
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_closed_form_matches_numerical_reference(name, dim):
    """Every closed form must agree with the numerical reference to near machine precision."""
    function = GENZ_FAMILY[name](dim)
    reference = _reference(name, function)

    assert function.exact == pytest.approx(reference, rel=1e-8), (
        f"{name} at dim={dim}: closed form {function.exact!r} disagrees with the "
        f"numerical reference {reference!r}"
    )


@pytest.mark.skipif(cubature is None, reason="requires scipy>=1.15 for cubature")
@pytest.mark.parametrize("name", sorted(GENZ_FAMILY))
def test_closed_form_tracks_non_default_parameters(name):
    """The formulas must follow a and u, not just happen to work at the defaults.

    Defaults are uniform and symmetric (u = 0.5), which hides whole classes of
    error -- a dropped shift term, or a parameter used for the wrong axis.
    """
    dim = 3
    a = [0.7, 1.3, 2.1]
    u = [0.2, 0.5, 0.9]
    function = GENZ_FAMILY[name](dim, a=a, u=u)
    reference = _reference(name, function)

    assert function.exact == pytest.approx(reference, rel=1e-8), (
        f"{name} with non-default a/u: closed form {function.exact!r} disagrees "
        f"with the numerical reference {reference!r}"
    )


@pytest.mark.parametrize("name", sorted(GENZ_FAMILY))
def test_integrand_is_vectorized_over_the_batch(name):
    """Evaluating N points at once must equal evaluating them one at a time.

    torchquad calls integrands with the whole batch, so a function that
    accidentally reduced across the batch axis would produce a plausible but
    wrong plot rather than an error.
    """
    dim = 3
    function = GENZ_FAMILY[name](dim)
    rng = np.random.default_rng(0)
    points = rng.random((16, dim))

    batched = np.asarray(function(points))
    assert batched.shape == (16,), f"{name} returned shape {batched.shape}, expected (16,)"

    one_at_a_time = np.array([float(np.asarray(function(p[None, :]))[0]) for p in points])
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-12, atol=0.0)


def test_as_array_rejects_a_wrong_length_parameter():
    """A mismatched parameter vector must raise rather than silently broadcast."""
    with pytest.raises(ValueError, match="Expected 1 or 3 parameters"):
        _as_array([1.0, 2.0], 3)


def test_relative_error_is_computed_against_the_closed_form():
    function = GENZ_FAMILY["gaussian"](2)
    assert function.relative_error(function.exact) == 0.0
    assert function.relative_error(function.exact * 1.5) == pytest.approx(0.5)

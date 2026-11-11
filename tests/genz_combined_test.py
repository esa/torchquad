"""Validation of the combined Genz integrand.

The combined integrand sums three normalized Genz functions, so its exact value
is the number of components by construction. That makes the closed form trivially
right and the *implementation* the thing that can be wrong: a component dropped,
double-counted, or normalized by the wrong constant would still produce a
plausible-looking number.

These tests therefore check the implementation against the mathematics, not the
formula against itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarking"))

from genz_functions import (  # noqa: E402
    BENCHMARK_INTEGRANDS,
    GENZ_FAMILY,
    _COMBINED_COMPONENTS,
    combined,
)

try:
    from scipy.integrate import cubature
except ImportError:
    cubature = None


@pytest.mark.parametrize("dim", [1, 2, 3, 6, 10])
def test_exact_is_the_component_count(dim):
    """Each component is normalized to unit integral, so the total is their number."""
    function = combined(dim)
    assert function.exact == pytest.approx(float(len(_COMBINED_COMPONENTS)))


@pytest.mark.parametrize("dim", [2, 3])
def test_evaluation_equals_the_sum_of_normalized_components(dim):
    """The implementation must be exactly the sum it claims to be.

    Rebuilding the components independently and summing them by hand catches a
    dropped term or a wrong normalizing constant, neither of which would show up
    in the closed form.
    """
    multiplier = 1.0
    function = combined(dim, a=multiplier)

    rng = np.random.default_rng(0)
    points = rng.random((32, dim))

    expected = np.zeros(32)
    for builder, scale in _COMBINED_COMPONENTS:
        part = builder(dim, a=(scale * multiplier) / dim, u=0.5)
        expected += np.asarray(part(points)) / part.exact

    np.testing.assert_allclose(np.asarray(function(points)), expected, rtol=1e-13, atol=0.0)


@pytest.mark.skipif(cubature is None, reason="requires scipy>=1.15 for cubature")
@pytest.mark.parametrize("dim", [2, 3])
def test_matches_an_independent_numerical_reference(dim):
    """Integrating the combined function numerically must give the component count.

    Run at a reduced difficulty so adaptive cubature can actually converge; the
    point is to confirm the integrand is the function whose integral we claim,
    not to benchmark anything.
    """
    function = combined(dim, a=0.1)
    result = cubature(function, np.zeros(dim), np.ones(dim), rtol=1e-10)
    assert result.status == "converged", result.status

    value = float(np.asarray(result.estimate).reshape(()))
    assert value == pytest.approx(function.exact, rel=1e-8), (
        f"combined at dim={dim} integrates to {value!r}, not {function.exact!r}"
    )


def test_difficulty_multiplier_makes_it_harder():
    """A larger multiplier must actually change the integrand, not be ignored."""
    points = np.random.default_rng(1).random((16, 3))
    easy = np.asarray(combined(3, a=0.1)(points))
    hard = np.asarray(combined(3, a=2.0)(points))

    assert not np.allclose(easy, hard), "the difficulty multiplier had no effect"


def test_every_component_contributes_comparably():
    """No component may be negligible, or a method could ignore it for free.

    This is why the components are normalized: at dim 10 the corner peak
    integrates to 9.9e-12 against the C0 term's 1.0e-02, nine orders apart.
    """
    dim = 10
    multiplier = 1.0
    points = np.random.default_rng(2).random((256, dim))

    contributions = []
    for builder, scale in _COMBINED_COMPONENTS:
        part = builder(dim, a=(scale * multiplier) / dim, u=0.5)
        contributions.append(float(np.mean(np.abs(np.asarray(part(points)) / part.exact))))

    assert min(contributions) > 0.0
    spread = max(contributions) / min(contributions)
    assert spread < 1e3, f"components differ by {spread:.1e}x in scale: {contributions}"


def test_combined_is_requestable_but_not_a_family_member():
    """It must be reachable by name, and must not claim the family's contract.

    Family builders accept a per-dimension sequence for ``a``; the combined
    integrand's ``a`` is a single difficulty multiplier, so putting it in
    GENZ_FAMILY would make the family-wide tests exercise a signature it does
    not implement.
    """
    assert BENCHMARK_INTEGRANDS["combined"] is combined
    assert "combined" not in GENZ_FAMILY

    with pytest.raises(TypeError):
        combined(3, a=[0.7, 1.3, 2.1])

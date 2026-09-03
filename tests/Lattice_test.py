"""Tests for the rank-1 lattice quasi-Monte Carlo sampler.

Lattice points plugged into ``MonteCarlo`` via the ``rng`` slot must (a)
integrate smooth functions accurately, (b) beat plain pseudo-random Monte
Carlo at the same sample count, and (c) reproduce bit-for-bit for a fixed
seed.

The construction is a few lines of array arithmetic, dispatched across
backends via ``autoray``, so it runs natively on all four supported backends
without any per-backend branching or additional dependency. The
accuracy/beats-mc/determinism tests below therefore run on every backend.

``Lattice.uniform`` casts its ``size`` components with plain ``int(...)``,
relying on the fact that the only real caller
(``MonteCarlo.calculate_sample_points``) always passes an already-validated
``N`` (checked by ``BaseIntegrator._check_inputs``) and a ``dim`` derived from
``integration_domain.shape[0]``, which is always a clean Python int. There is
therefore no dedicated test here for malformed ``size`` components (e.g. bools
or non-whole floats).

The remaining tests target properties that are specific to the rank-1 lattice
construction: the closed-form point formula, the one-shot random shift (drawn
once and reused for the sampler's lifetime), the optional Baker's transform
(``tent=True``), and input validation on N and dim. These run on the torch
backend only -- the per-backend tests above already cover cross-backend
correctness at the integration level.
"""

import numpy as np
import pytest
import torch
from autoray import numpy as anp

from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.qmc import (
    Lattice,
    load_lattice_vector,
    _MIN_POINTS,
    _MAX_POINTS,
    _max_dimension,
)
from torchquad.utils.set_up_backend import set_up_backend
from helper_functions import setup_test_for_backend, compute_integration_test_errors

# A smooth, non-separable-into-a-polynomial integrand:
# prod_i cos(pi/2 * x_i) over [0, 1]^dim integrates to (2/pi)^dim. QMC should
# shine here.
_DIM = 3
_N = (
    2**13
)  # 8192 points: a power of 2, inside the [_MIN_POINTS, _MAX_POINTS] range Lattice requires
_DOMAIN = [[0.0, 1.0]] * _DIM
_EXPECTED = (2.0 / np.pi) ** _DIM


def _to_float(result):
    """Convert a scalar backend tensor (possibly on GPU) to a Python float."""
    if hasattr(result, "cpu"):
        result = result.cpu()
    return float(np.asarray(result))


def _integrand(x):
    return anp.prod(anp.cos(x * (np.pi / 2.0)), axis=1)


def _nonperiodic_integrand(x):
    """A smooth, genuinely non-periodic integrand: prod_i x_i, integrating to
    (1/2)^dim over [0, 1]^dim. f(0) = 0 but f(1) = 1 per axis, so its periodic
    extension has a jump discontinuity -- exactly the case Baker's transform
    is meant to fix.
    """

    return anp.prod(x, axis=1)


# ---------------------------------------------------------------------------
# Accuracy, beating plain MC, determinism, and gradient flow. Runs per
# backend.
# ---------------------------------------------------------------------------


def _lattice_collection_test(backend, dtype_name=None):
    """Lattice-based MC must integrate the shared analytic test-function
    collection accurately, at each dimension it covers.

    Bounds were derived empirically across all four backends (numpy, torch,
    tensorflow, jax) with seed=0.
    """
    mc = MonteCarlo()
    cases = [(1, 2**12, 0.5), (3, 2**12, 0.07), (10, 2**11, 6e-4)]
    for integration_dim, N, bound in cases:
        errors, funcs = compute_integration_test_errors(
            mc.integrate,
            {"N": N, "dim": integration_dim, "rng": Lattice(backend=backend, seed=0)},
            integration_dim=integration_dim,
            use_complex=True,
            backend=backend,
        )
        for error, test_function in zip(errors, funcs):
            assert test_function.get_order() > 0 or error == 0.0
            assert error < bound, (
                f"Lattice dim={integration_dim} error {error} exceeds {bound} "
                f"for {type(test_function).__name__}"
            )


test_lattice_collection_numpy = setup_test_for_backend(_lattice_collection_test, "numpy", "float64")
test_lattice_collection_torch = setup_test_for_backend(_lattice_collection_test, "torch", "float64")
test_lattice_collection_tensorflow = setup_test_for_backend(
    _lattice_collection_test, "tensorflow", "float64"
)
test_lattice_collection_jax = setup_test_for_backend(_lattice_collection_test, "jax", "float64")


def _lattice_accuracy_test(backend, dtype_name=None):
    """Lattice MC must integrate a smooth function close to the analytic value."""
    mc = MonteCarlo()
    result = _to_float(
        mc.integrate(
            _integrand,
            dim=_DIM,
            N=_N,
            integration_domain=_DOMAIN,
            rng=Lattice(backend=backend, seed=0),
        )
    )
    assert abs(result - _EXPECTED) < 1e-3, (
        f"Lattice result {result} too far from analytic {_EXPECTED}"
    )


def _lattice_beats_mc_test(backend, dtype_name=None):
    """At equal N, Lattice must be more accurate than pseudo-random Monte Carlo.

    Both use a fixed seed, so this comparison is deterministic (no flakiness), and
    the QMC error is orders of magnitude smaller for a smooth integrand.
    """
    mc = MonteCarlo()
    lattice_error = abs(
        _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Lattice(backend=backend, seed=0),
            )
        )
        - _EXPECTED
    )
    mc_error = abs(
        _to_float(mc.integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0))
        - _EXPECTED
    )
    assert lattice_error < mc_error, (
        f"Lattice error {lattice_error} not below Monte Carlo error {mc_error}"
    )


def _lattice_determinism_test(backend, dtype_name=None):
    """A fixed seed must reproduce the same result bit-for-bit, across fresh
    ``Lattice`` instances (the shift is drawn once per instance, from ``seed``)."""
    mc = MonteCarlo()

    def run():
        return _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Lattice(backend=backend, seed=42),
            )
        )

    assert run() == run(), "Lattice integration is not reproducible for a fixed seed"


test_lattice_accuracy_numpy = setup_test_for_backend(_lattice_accuracy_test, "numpy", "float64")
test_lattice_accuracy_torch = setup_test_for_backend(_lattice_accuracy_test, "torch", "float64")
test_lattice_accuracy_tensorflow = setup_test_for_backend(
    _lattice_accuracy_test, "tensorflow", "float64"
)
test_lattice_accuracy_jax = setup_test_for_backend(_lattice_accuracy_test, "jax", "float64")

test_lattice_beats_mc_numpy = setup_test_for_backend(_lattice_beats_mc_test, "numpy", "float64")
test_lattice_beats_mc_torch = setup_test_for_backend(_lattice_beats_mc_test, "torch", "float64")
test_lattice_beats_mc_tensorflow = setup_test_for_backend(
    _lattice_beats_mc_test, "tensorflow", "float64"
)
test_lattice_beats_mc_jax = setup_test_for_backend(_lattice_beats_mc_test, "jax", "float64")

test_lattice_determinism_numpy = setup_test_for_backend(
    _lattice_determinism_test, "numpy", "float64"
)
test_lattice_determinism_torch = setup_test_for_backend(
    _lattice_determinism_test, "torch", "float64"
)
test_lattice_determinism_tensorflow = setup_test_for_backend(
    _lattice_determinism_test, "tensorflow", "float64"
)
test_lattice_determinism_jax = setup_test_for_backend(_lattice_determinism_test, "jax", "float64")


def test_lattice_preserves_gradient():
    """Lattice points are constants, so autodiff through the integral must survive."""
    set_up_backend("torch", "float64")
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    # integral over [0, 1] of parameter * x is parameter / 2, so d/dparameter = 1/2.
    def parametric(x):
        return parameter * x[:, 0]

    mc = MonteCarlo()
    result = mc.integrate(
        parametric,
        dim=1,
        N=_N,
        integration_domain=[[0.0, 1.0]],
        rng=Lattice(backend="torch", seed=0),
    )
    result.backward()
    assert abs(_to_float(parameter.grad) - 0.5) < 1e-3, (
        "Gradient did not flow through the Lattice-sampled Monte Carlo integral"
    )


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------


def test_lattice_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        Lattice(backend="bogus")


# ---------------------------------------------------------------------------
# Lattice-specific tests: the closed-form construction, the one-shot shift,
# the optional Baker's transform, and input validation on N and dim.
# Torch-only -- the per-backend tests above already cover cross-backend
# correctness.
# ---------------------------------------------------------------------------


def test_lattice_matches_closed_form_without_shift():
    """Unshifted points must equal frac(i * z / N) exactly, for the actual
    generating vector on disk. This pins down the construction itself,
    independently of the downstream integration-accuracy tests above.
    """
    set_up_backend("torch", "float64")
    dim = 4
    n = 1024  # smallest supported N, keeps the manual computation cheap

    z = load_lattice_vector(d=dim).astype(np.int64)
    i = np.arange(n, dtype=np.int64)
    expected = ((i[:, None] * z[None, :]) % n) / n

    points = Lattice(backend="torch", shift=False).uniform([n, dim], torch.float64).cpu().numpy()

    np.testing.assert_allclose(
        points,
        expected,
        atol=1e-12,
        err_msg="Unshifted Lattice points do not match the closed-form frac(i*z/N) formula",
    )


def test_lattice_same_seed_reproduces_shift_across_calls():
    """The shift is redrawn on every call to ``uniform``, not cached -- but with
    a fixed seed, each redraw must reproduce the same values, so successive
    calls on the same instance return identical points."""
    set_up_backend("torch", "float64")
    lattice = Lattice(backend="torch", seed=0, shift=True)

    first = lattice.uniform([1024, 2], torch.float64).clone()
    second = lattice.uniform([1024, 2], torch.float64)

    assert torch.equal(first, second), (
        "Calling uniform() twice with the same seed must return identical points, "
        "since the seeded shift draw is reproducible"
    )


def test_lattice_unseeded_shift_differs_across_calls():
    """Without a seed, each call to ``uniform`` draws its own independent
    shift, so successive calls on the same instance must return different
    points."""
    set_up_backend("torch", "float64")
    lattice = Lattice(backend="torch", shift=True)  # no seed

    first = lattice.uniform([1024, 2], torch.float64).clone()
    second = lattice.uniform([1024, 2], torch.float64)

    assert not torch.equal(first, second), (
        "Calling uniform() twice without a seed should not return identical "
        "points, since each call draws its own independent shift"
    )


def test_lattice_shift_changes_points():
    """With shift=True, the returned points must differ from the unshifted
    lattice -- otherwise the shift would be silently having no effect."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    unshifted = Lattice(backend="torch", shift=False).uniform([n, dim], torch.float64)
    shifted = Lattice(backend="torch", seed=0, shift=True).uniform([n, dim], torch.float64)

    assert not torch.equal(unshifted, shifted), (
        "Shifted points must not be identical to the unshifted lattice"
    )


def test_lattice_different_seeds_give_different_shifts():
    """Two instances with different seeds must (almost surely) produce different
    point sets, since each draws its own independent random shift."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    points_a = Lattice(backend="torch", seed=1, shift=True).uniform([n, dim], torch.float64)
    points_b = Lattice(backend="torch", seed=2, shift=True).uniform([n, dim], torch.float64)

    assert not torch.equal(points_a, points_b), (
        "Different seeds should not produce the same shifted lattice"
    )


def test_lattice_seed_ignored_when_shift_false():
    """When shift=False, the seed argument must have no effect: unshifted
    lattice points depend only on N and dim, never on the seed."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    points_a = Lattice(backend="torch", seed=1, shift=False).uniform([n, dim], torch.float64)
    points_b = Lattice(backend="torch", seed=99, shift=False).uniform([n, dim], torch.float64)

    assert torch.equal(points_a, points_b), "Unshifted lattice points must not depend on the seed"


def test_lattice_points_lie_in_unit_cube():
    """Every coordinate must lie in [0, 1), both with and without a shift
    (tent=False in both cases; see test_lattice_tent_points_lie_in_closed_unit_cube
    for the tent=True case, whose range is the closed [0, 1])."""
    set_up_backend("torch", "float64")
    n, dim = 2048, 5

    for shift in (False, True):
        points = Lattice(backend="torch", seed=0, shift=shift).uniform([n, dim], torch.float64)
        assert torch.all(points >= 0) and torch.all(points < 1), (
            f"Lattice points (shift={shift}) fall outside [0, 1)"
        )


def test_lattice_rejects_non_power_of_two_point_count():
    """The simple lattice construction is only valid when N is a power of 2;
    a non-power-of-two N (even if within [_MIN_POINTS, _MAX_POINTS]) must
    raise, since the construction's validity guarantee does not hold
    otherwise."""
    set_up_backend("torch", "float64")
    with pytest.raises(ValueError, match="power of 2"):
        Lattice(backend="torch", shift=False).uniform([1030, 2], torch.float64)


@pytest.mark.parametrize("bad_n", [_MIN_POINTS - 1, _MAX_POINTS + 1, 500, 2_000_000])
def test_lattice_rejects_out_of_range_point_counts(bad_n):
    """N below _MIN_POINTS or above _MAX_POINTS must raise, near the boundary
    or far outside it."""
    with pytest.raises(ValueError):
        Lattice(backend="torch").uniform([bad_n, 2], torch.float64)


def test_lattice_rejects_out_of_range_dimensions():
    """dim below 1 or above the generating vector file's max must raise, near
    the boundary or far outside it.

    Not parametrized: the upper bound is derived from the actual data file
    via ``_max_dimension()``, which must run inside a test function rather
    than at module collection time, to avoid loading the file on import.
    """
    max_dim = _max_dimension()
    for bad_dim in [0, -1, max_dim + 1, max_dim + 10000]:
        with pytest.raises(ValueError):
            Lattice(backend="torch").uniform([1024, bad_dim], torch.float64)


def test_lattice_tent_points_lie_in_closed_unit_cube():
    """With tent=True, points must lie in the CLOSED [0, 1], not the
    half-open [0, 1) used everywhere else in this sampler: the tent map's
    peak is exactly 1, reached whenever a point lands exactly on 0.5 (see
    test_lattice_tent_reaches_exact_upper_bound_at_midpoint for the
    structural case where this is guaranteed, not just possible)."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    points = Lattice(backend="torch", seed=0, tent=True).uniform([n, dim], torch.float64)
    assert torch.all(points >= 0) and torch.all(points <= 1), (
        "Tent-transformed Lattice points fall outside the closed interval [0, 1]"
    )


def test_lattice_tent_reaches_exact_upper_bound_at_midpoint():
    """The tent map's peak is exactly 1, reached whenever a lattice point
    lands exactly on 0.5 -- which happens systematically (not by rare
    chance) at index N/2 whenever the generating vector's first component
    is 1, since N is always a power of 2. Regression test for the exact
    boundary case flagged in review: N=1024, i=512, dim=1, shift=False.
    """
    set_up_backend("torch", "float64")
    n = 1024
    points = Lattice(backend="torch", shift=False, tent=True).uniform([n, 1], torch.float64)
    assert points[512, 0].item() == 1.0, (
        "Expected the tent-transformed midpoint (i=N/2) to be exactly 1.0"
    )
    assert torch.all(points >= 0) and torch.all(points <= 1), (
        "Tented points must lie in the closed interval [0, 1]"
    )


def test_lattice_tent_changes_points():
    """With tent=True, the returned points must differ from the
    (shifted-but-not-tent-transformed) lattice -- otherwise the transform
    would be silently having no effect."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    plain = Lattice(backend="torch", seed=0, tent=False).uniform([n, dim], torch.float64)
    tented = Lattice(backend="torch", seed=0, tent=True).uniform([n, dim], torch.float64)

    assert not torch.equal(plain, tented), (
        "Tent-transformed points must not be identical to the plain lattice"
    )


def test_lattice_tent_improves_nonperiodic_convergence():
    """Baker's transform periodizes the integrand, which should improve
    convergence for a smooth but non-periodic integrand, averaged over
    several independent shifts to avoid relying on a single seed's draw.

    Measured with the real generating vector, averaged over 15 seeds:
    tent=True was ~502x more accurate than tent=False on prod_i(x_i) at
    N=4096, dim=3. This threshold (20x) is set well below that measurement
    to leave a comfortable margin -- a single seed's ratio was found to vary
    quite a bit (as low as ~28x for seed=0 alone), so the margin needs to
    absorb real run-to-run variance, not just be a token safety factor.
    """
    n, dim, trials = 4096, 3, 15
    mc = MonteCarlo()
    exact = 0.5**dim

    errors_plain, errors_tent = [], []
    for seed in range(trials):
        error_plain = abs(
            _to_float(
                mc.integrate(
                    _nonperiodic_integrand,
                    dim=dim,
                    N=n,
                    integration_domain=[[0.0, 1.0]] * dim,
                    rng=Lattice(backend="torch", seed=seed, tent=False),
                )
            )
            - exact
        )
        error_tent = abs(
            _to_float(
                mc.integrate(
                    _nonperiodic_integrand,
                    dim=dim,
                    N=n,
                    integration_domain=[[0.0, 1.0]] * dim,
                    rng=Lattice(backend="torch", seed=seed, tent=True),
                )
            )
            - exact
        )
        errors_plain.append(error_plain)
        errors_tent.append(error_tent)

    mean_plain = float(np.mean(errors_plain))
    mean_tent = float(np.mean(errors_tent))
    assert mean_tent > 0, "degenerate test: tent-transformed error measured as exactly zero"
    assert mean_plain / mean_tent > 20, (
        f"tent=True only {mean_plain / mean_tent:.1f}x better than tent=False "
        f"on average over {trials} seeds, expected >> 20x (Baker's transform)"
    )


if __name__ == "__main__":
    for _backend in ["numpy", "torch", "tensorflow", "jax"]:
        _lattice_accuracy_test(_backend)
        _lattice_beats_mc_test(_backend)
        _lattice_determinism_test(_backend)
    test_lattice_preserves_gradient()
    test_lattice_rejects_invalid_backend()
    test_lattice_matches_closed_form_without_shift()
    test_lattice_same_seed_reproduces_shift_across_calls()
    test_lattice_unseeded_shift_differs_across_calls()
    test_lattice_shift_changes_points()
    test_lattice_different_seeds_give_different_shifts()
    test_lattice_seed_ignored_when_shift_false()
    test_lattice_points_lie_in_unit_cube()
    test_lattice_rejects_non_power_of_two_point_count()
    test_lattice_rejects_out_of_range_dimensions()
    test_lattice_tent_points_lie_in_closed_unit_cube()
    test_lattice_tent_reaches_exact_upper_bound_at_midpoint()
    test_lattice_tent_changes_points()
    print("All Lattice tests passed!")

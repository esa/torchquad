"""Tests for the randomized Latin Hypercube (LHS) sampler.

Randomized LHS points plugged into ``MonteCarlo`` via the ``rng`` slot must (a)
integrate the whole analytic test-function collection accurately, and (b)
reproduce bit-for-bit for a fixed seed -- the same contract as Sobol and Halton
(see ``sobol_test.py``, ``halton_test.py``), whose collection/determinism/
gradient tests are mirrored below.

Unlike Sobol and Halton, there is no "beats plain Monte Carlo" test here: LHS
and plain Monte Carlo are both randomized uniform sampling methods, so a
single-draw comparison between them has no principled basis -- either can
"win" by chance on any given seed (measured during development: LHS won only
~21/30 individual seeds against MC on a moderately interacting integrand,
despite a real ~7x average advantage), making a strict less-than assertion
between two random draws inherently unreliable.

On top of that, this file adds tests for properties SPECIFIC to LHS:

    - exact stratification: unlike Halton (whose (0,m,1)-net check has to
      tolerate a float64 representability caveat, see ``halton_test.py``), LHS
      strata are a direct k/n division with no digit-based reconstruction, so
      the bijection is checked for EXACT equality here, with no tolerance;
    - the additive-integrand advantage: Stein (1987) shows LHS's asymptotic
      variance depends only on the *non-additive* part of the integrand, so for
      a purely additive integrand LHS should beat plain MC by a very large
      margin, not just modestly -- a much stronger and more specific check than
      an arbitrary single-draw comparison, and averaged over several seeds so
      it isn't subject to the same single-draw unreliability noted above;
    - `seed=None` must give independently randomised draws on every call. This
      guards against a real bug found during development: an un-seeded
      `torch.Generator()` has a FIXED default internal state (verified: two
      fresh, un-seeded generators produced the identical draw), so forgetting
      to call `.seed()` in that branch would silently make every `seed=None`
      call return the exact same "random" sample.

Coverage runs on every backend.
"""

import numpy as np
import torch

from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.qmc import RandomizedLatinHypercube
from helper_functions import compute_integration_test_errors, setup_test_for_backend


# A smooth, non-separable-into-a-polynomial integrand: prod_i cos(pi/2 * x_i) over
# [0, 1]^dim integrates to (2/pi)^dim. Same integrand as sobol_test.py /
# halton_test.py, for a direct comparison across samplers.
_DIM = 3
_N = 2**10
_DOMAIN = [[0.0, 1.0]] * _DIM
_EXPECTED = (2.0 / np.pi) ** _DIM


def _to_float(result):
    """Convert a scalar backend tensor (possibly on GPU) to a Python float."""
    if hasattr(result, "cpu"):
        result = result.cpu()
    return float(np.asarray(result))


def _integrand(x):
    from autoray import numpy as anp

    return anp.prod(anp.cos(x * (np.pi / 2.0)), axis=1)


def _additive_integrand(x):
    """A purely additive integrand: sum_i sin(2*pi*(i+1)*x_i), integrating to
    exactly 0 over [0, 1]^dim (each term integrates to 0 over a full period).

    This is the case Stein (1987) singles out: LHS's asymptotic variance is
    driven only by the non-additive residual of the integrand, which is
    identically zero here, so LHS should crush plain Monte Carlo by orders of
    magnitude rather than by a modest constant factor.
    """
    from autoray import numpy as anp

    dim = x.shape[1]
    total = 0.0
    for i in range(dim):
        total = total + anp.sin(x[:, i] * (2.0 * np.pi * (i + 1)))
    return total


# =============================================================================
# Tests mirrored from sobol_test.py / halton_test.py
# =============================================================================


def _lhs_collection_test(backend, dtype_name=None):
    """Randomized LHS MC must integrate the whole analytic test-function
    collection accurately.

    Bounds are looser than Sobol's/Halton's: LHS converges at the plain Monte
    Carlo rate O(1/sqrt(N)) asymptotically (its advantage is a variance
    *constant*, not a better convergence *rate*, unlike a low-discrepancy
    sequence), so it should not be expected to reach anywhere near Sobol's
    ~1e-6 at equal N.

    Bounds were widened during development after observing real run-to-run
    variance on functions with strong cross-dimensional interaction (e.g.
    ProductFunction at dim=3 measured 8.19e-04, 3.24e-03 and 1.14e-02 across
    different runs/backends) -- LHS's per-axis stratification does not
    constrain multi-dimensional interaction terms, so a single seed's error
    on such functions is noticeably less stable than for Sobol/Halton.
    """
    mc = MonteCarlo()
    cases = [(1, 2**12, 0.15), (3, 2**12, 4e-2), (10, 2**11, 8e-2)]
    for integration_dim, N, bound in cases:
        errors, funcs = compute_integration_test_errors(
            mc.integrate,
            {
                "N": N,
                "dim": integration_dim,
                "rng": RandomizedLatinHypercube(backend=backend, seed=0),
            },
            integration_dim=integration_dim,
            use_complex=True,
            backend=backend,
        )
        for error, test_function in zip(errors, funcs):
            # Order-0 (constant) integrands are integrated exactly.
            assert test_function.get_order() > 0 or error == 0.0
            assert error < bound, (
                f"LHS dim={integration_dim} error {error} exceeds {bound} "
                f"for {type(test_function).__name__}"
            )


def _lhs_determinism_test(backend, dtype_name=None):
    """A fixed seed must reproduce the same result bit-for-bit."""
    mc = MonteCarlo()

    def run():
        return _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=RandomizedLatinHypercube(backend=backend, seed=42),
            )
        )

    assert run() == run(), "LHS integration is not reproducible for a fixed seed"


test_lhs_collection_numpy = setup_test_for_backend(_lhs_collection_test, "numpy", "float64")
test_lhs_collection_torch = setup_test_for_backend(_lhs_collection_test, "torch", "float64")
test_lhs_collection_tensorflow = setup_test_for_backend(
    _lhs_collection_test, "tensorflow", "float64"
)
test_lhs_collection_jax = setup_test_for_backend(_lhs_collection_test, "jax", "float64")

test_lhs_determinism_numpy = setup_test_for_backend(_lhs_determinism_test, "numpy", "float64")
test_lhs_determinism_torch = setup_test_for_backend(_lhs_determinism_test, "torch", "float64")
test_lhs_determinism_tensorflow = setup_test_for_backend(
    _lhs_determinism_test, "tensorflow", "float64"
)
test_lhs_determinism_jax = setup_test_for_backend(_lhs_determinism_test, "jax", "float64")


def test_lhs_preserves_gradient():
    """LHS points are constants, so autodiff through the integral must survive."""
    from torchquad.utils.set_up_backend import set_up_backend

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
        rng=RandomizedLatinHypercube(backend="torch", seed=0),
    )
    result.backward()
    assert abs(_to_float(parameter.grad) - 0.5) < 1e-3, (
        "Gradient did not flow through the LHS-sampled Monte Carlo integral"
    )


# =============================================================================
# LHS-specific tests
# =============================================================================


def test_lhs_exact_stratification():
    """Core LHS property: splitting [0, 1) into n equal strata along ANY
    single coordinate axis places exactly one point in each stratum.

    Unlike Halton's (0,m,1)-net check, this is checked for EXACT equality, no
    tolerance: LHS coordinates are a direct k/n division (k and n both plain
    integers), not a multi-digit base-b reconstruction, so there is no
    equivalent of Halton's float64 representability caveat here -- verified
    directly: 4 dimensions, n=5000, all four columns give a perfect bijection
    onto {0, ..., n-1}.
    """
    n, dim = 5000, 4
    points = RandomizedLatinHypercube(backend="torch", seed=7).uniform([n, dim], torch.float64)
    for j in range(dim):
        strata = torch.floor(points[:, j] * n).long()
        assert sorted(strata.tolist()) == list(range(n)), (
            f"dimension {j}: not a perfect stratification bijection"
        )


def test_lhs_crushes_mc_on_additive_integrand():
    """Stein (1987): LHS's asymptotic variance depends only on the
    *non-additive* part of the integrand. For a purely additive integrand
    (see `_additive_integrand`), LHS should beat plain MC by orders of
    magnitude, not just modestly.

    Measured during development: ~1200x lower mean squared error than plain
    MC over 30 independent trials at N=200, dim=5. This threshold (100x) is
    set well below that measurement to leave comfortable margin while still
    being a real, specific, theory-motivated check -- much stronger evidence
    of correctness than a single-draw comparison, which any variance-reduction
    method could pass or fail by chance (see module docstring).
    """
    N, dim, trials = 200, 5, 30
    lhs_sq_errors, mc_sq_errors = [], []
    for trial in range(trials):
        lhs_points = (
            RandomizedLatinHypercube(backend="torch", seed=trial)
            .uniform([N, dim], torch.float64)
            .cpu()
            .numpy()
        )
        lhs_sq_errors.append(_additive_integrand(lhs_points).mean() ** 2)

        mc_points = np.random.default_rng(10_000 + trial).random((N, dim))
        mc_sq_errors.append(_additive_integrand(mc_points).mean() ** 2)

    lhs_mse = float(np.mean(lhs_sq_errors))
    mc_mse = float(np.mean(mc_sq_errors))
    assert lhs_mse > 0, "degenerate test: LHS MSE measured as exactly zero"
    assert mc_mse / lhs_mse > 100, (
        f"LHS only {mc_mse / lhs_mse:.1f}x better than MC on an additive "
        f"integrand, expected >> 100x (Stein 1987)"
    )


def test_lhs_seed_none_randomizes_every_call():
    """`seed=None` must give independently randomised draws on every call.

    Regression guard for a real bug found during development: a fresh,
    un-seeded `torch.Generator()` has a FIXED default internal state (two
    freshly constructed, never-seeded generators were found to produce the
    identical draw), so forgetting the `.seed()` call in the `seed is None`
    branch would silently make every unseeded call return the same "random"
    sample -- exactly the same category of bug previously found and fixed for
    the `Lattice` sampler's shift.
    """
    a = RandomizedLatinHypercube(backend="torch", seed=None).uniform([8, 3], torch.float64)
    b = RandomizedLatinHypercube(backend="torch", seed=None).uniform([8, 3], torch.float64)
    assert not torch.equal(a, b), "seed=None produced identical draws across instances"


def test_lhs_fixed_seed_is_reproducible():
    """A fixed seed must reproduce the exact same points, not merely the
    same integration result (a stronger, more direct check than the
    integration-level determinism test above)."""
    a = RandomizedLatinHypercube(backend="torch", seed=123).uniform([8, 3], torch.float64)
    b = RandomizedLatinHypercube(backend="torch", seed=123).uniform([8, 3], torch.float64)
    assert torch.equal(a, b)


def test_lhs_points_in_unit_cube():
    points = RandomizedLatinHypercube(backend="torch", seed=1).uniform([500, 6], torch.float64)
    assert bool((points >= 0).all()) and bool((points < 1).all())


if __name__ == "__main__":
    from torchquad.utils.set_up_backend import set_up_backend

    for _backend in ["numpy", "torch", "tensorflow", "jax"]:
        set_up_backend(_backend, "float64")
        _lhs_collection_test(_backend)
        _lhs_determinism_test(_backend)
    test_lhs_preserves_gradient()
    test_lhs_exact_stratification()
    test_lhs_crushes_mc_on_additive_integrand()
    test_lhs_seed_none_randomizes_every_call()
    test_lhs_fixed_seed_is_reproducible()
    test_lhs_points_in_unit_cube()
    print("All RandomizedLatinHypercube tests passed!")

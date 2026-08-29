"""Tests for VEGAS error reporting via ``return_error=True``.

VEGAS already computes an error estimate, chi-squared and goodness-of-fit
internally; ``return_error=True`` surfaces them as a :class:`VEGASResult` instead
of discarding them. VEGAS supports only the numpy and torch backends.
"""

import numpy as np
from autoray import numpy as anp

from torchquad.integration.vegas import VEGAS
from torchquad.integration.vegas_result import VEGASResult
from helper_functions import get_test_functions, setup_test_for_backend

# prod_i sin(x_i) over [0, pi]^2 integrates to 2^2 = 4.
_DIM = 2
_N = 40000
_DOMAIN = [[0.0, np.pi], [0.0, np.pi]]
_EXPECTED = 4.0


def _to_float(result):
    """Convert a scalar backend tensor (possibly on GPU) to a Python float."""
    if hasattr(result, "cpu"):
        result = result.cpu()
    return float(np.asarray(result))


def _integrand(x):
    return anp.prod(anp.sin(x), axis=1)


def _vegas_result_fields_test(backend, dtype_name=None):
    """return_error=True must yield a well-formed, self-consistent VEGASResult."""
    result = VEGAS().integrate(
        _integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0, return_error=True
    )
    assert isinstance(result, VEGASResult)

    integral = _to_float(result.integral)
    sdev = _to_float(result.sdev)
    chi2 = _to_float(result.chi2)

    # The integral is accurate and the reported error is a sane, positive number.
    assert abs(integral - _EXPECTED) < 0.05, f"VEGAS integral {integral} far from {_EXPECTED}"
    assert 0.0 < sdev < 0.05, f"Unreasonable VEGAS sdev {sdev}"
    assert chi2 >= 0.0

    # The error estimate must actually bracket the truth (generous factor to stay
    # robust to GPU-reduction non-determinism, not so loose as to be meaningless).
    assert abs(integral - _EXPECTED) < 8.0 * sdev, (
        f"Error estimate {sdev} does not bracket the true error {abs(integral - _EXPECTED)}"
    )

    # Degrees of freedom and goodness-of-fit are well-formed, and Q matches an
    # independent evaluation of the chi-squared survival function (guards against
    # a swapped-argument or wrong-formula regression, not just a plausible range).
    from scipy.special import gammaincc

    assert result.dof >= 1
    assert result.Q is not None and 0.0 <= result.Q <= 1.0
    assert abs(result.Q - float(gammaincc(result.dof / 2.0, chi2 / 2.0))) < 1e-12
    assert result.nr_of_fevals > 0
    # __repr__ must not raise.
    assert "VEGASResult" in repr(result)


def _vegas_backward_compat_test(backend, dtype_name=None):
    """The default (return_error=False) must still return the bare integral."""
    result = VEGAS().integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0)
    assert not isinstance(result, VEGASResult)
    assert abs(_to_float(result) - _EXPECTED) < 0.05


def _vegas_result_matches_bare_test(backend, dtype_name=None):
    """The bundled integral must equal the bare integral for the same run.

    numpy only: its RNG is instance-local and its reductions are deterministic,
    so two identically-seeded runs agree bit-for-bit. (torch here runs on the GPU,
    whose reductions are not bit-reproducible.)
    """
    bare = VEGAS().integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=3)
    bundled = VEGAS().integrate(
        _integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=3, return_error=True
    )
    assert _to_float(bundled.integral) == _to_float(bare)


def _vegas_determinism_test(backend, dtype_name=None):
    """A fixed seed reproduces the integral and its error (numpy only, see above)."""
    first = VEGAS().integrate(
        _integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=11, return_error=True
    )
    second = VEGAS().integrate(
        _integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=11, return_error=True
    )
    assert _to_float(first.integral) == _to_float(second.integral)
    assert _to_float(first.sdev) == _to_float(second.sdev)


def _vegas_error_calibration_test(backend, dtype_name=None):
    """VEGAS's reported error must bracket the true error across the whole collection.

    For every real, scalar-integrand analytic test function (VEGAS supports only
    real, non-vectorized integrands, so complex and multi-integrand functions are
    skipped) the estimated ``sdev`` has to be a meaningful error bar: VEGAS
    converges to the closed-form value, and the actual error stays within a few
    ``sdev`` of it. Validates the estimate against ground truth rather than merely
    range-checking it.
    """
    for integration_dim, N in [(1, 40000), (3, 60000)]:
        for test_function in get_test_functions(
            integration_dim, backend, use_multi_dim_integrand=False
        ):
            if test_function.is_complex or not test_function.is_integrand_1d:
                continue
            result = test_function.evaluate(
                VEGAS().integrate,
                {"dim": integration_dim, "N": N, "seed": 0, "return_error": True},
            )
            exact = float(np.asarray(test_function.expected_result))
            integral = _to_float(result.integral)
            sdev = _to_float(result.sdev)
            true_error = abs(integral - exact)

            label = f"{type(test_function).__name__} dim={integration_dim}"
            assert sdev > 0.0, f"{label}: non-positive sdev"
            assert result.dof >= 1
            assert result.Q is not None and 0.0 <= result.Q <= 1.0
            # VEGAS actually converged on this function (worst observed ~0.8%).
            assert true_error < 0.05 * max(abs(exact), 1.0), (
                f"{label}: VEGAS did not converge, error {true_error}"
            )
            # ...and its self-reported sdev brackets the true error. The worst
            # true_error/sdev observed across the collection is ~3.4; 8x leaves
            # headroom for the stochastic (and GPU-nondeterministic) spread.
            assert true_error < 8.0 * sdev, (
                f"{label}: true error {true_error} exceeds 8*sdev = {8.0 * sdev}"
            )


test_vegas_result_fields_numpy = setup_test_for_backend(
    _vegas_result_fields_test, "numpy", "float64"
)
test_vegas_result_fields_torch = setup_test_for_backend(
    _vegas_result_fields_test, "torch", "float64"
)

test_vegas_backward_compat_numpy = setup_test_for_backend(
    _vegas_backward_compat_test, "numpy", "float64"
)
test_vegas_backward_compat_torch = setup_test_for_backend(
    _vegas_backward_compat_test, "torch", "float64"
)

test_vegas_error_calibration_numpy = setup_test_for_backend(
    _vegas_error_calibration_test, "numpy", "float64"
)
test_vegas_error_calibration_torch = setup_test_for_backend(
    _vegas_error_calibration_test, "torch", "float64"
)

# numpy-only: bit-for-bit equality/reproducibility (see docstrings).
test_vegas_result_matches_bare_numpy = setup_test_for_backend(
    _vegas_result_matches_bare_test, "numpy", "float64"
)
test_vegas_determinism_numpy = setup_test_for_backend(_vegas_determinism_test, "numpy", "float64")


if __name__ == "__main__":
    for _backend in ["numpy", "torch"]:
        _vegas_result_fields_test(_backend)
        _vegas_backward_compat_test(_backend)
        _vegas_error_calibration_test(_backend)
    _vegas_result_matches_bare_test("numpy")
    _vegas_determinism_test("numpy")
    print("All VEGAS result tests passed!")

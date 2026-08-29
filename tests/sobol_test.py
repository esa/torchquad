"""Tests for the Sobol quasi-Monte Carlo sampler.

Sobol points plugged into ``MonteCarlo`` via the ``rng`` slot must (a) integrate
the whole analytic test-function collection accurately, (b) beat plain
pseudo-random Monte Carlo at the same sample count, and (c) reproduce bit-for-bit
for a fixed seed. Coverage runs on every backend.
"""

import numpy as np

from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.qmc import Sobol
from helper_functions import compute_integration_test_errors, setup_test_for_backend

# A smooth, non-separable-into-a-polynomial integrand: prod_i cos(pi/2 * x_i) over
# [0, 1]^dim integrates to (2/pi)^dim. QMC should shine here.
_DIM = 3
_N = 2**13
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


def _sobol_collection_test(backend, dtype_name=None):
    """Sobol MC must integrate the whole analytic test-function collection accurately.

    Runs every function in ``integration_test_functions`` (real and complex,
    including the multi-dimensional integrands) in 1-D, 3-D and 10-D and checks the
    error against the closed-form value. The per-dimension bounds are far tighter
    than the plain Monte Carlo ones (see ``monte_carlo_test.py``): with these
    sample counts Sobol reaches ~1e-6 on the SciPy backends. torch's native
    SobolEngine is weaker in 10-D, so that bound is looser accordingly.
    """
    mc = MonteCarlo()

    # (integration_dim, N, per-function error bound). N is a power of two so the
    # Sobol balance property holds.
    cases = [(1, 2**16, 1e-4), (3, 2**16, 1e-4), (10, 2**13, 1e-2)]
    for integration_dim, N, bound in cases:
        errors, funcs = compute_integration_test_errors(
            mc.integrate,
            {"N": N, "dim": integration_dim, "rng": Sobol(backend=backend, seed=0)},
            integration_dim=integration_dim,
            use_complex=True,
            backend=backend,
        )
        for error, test_function in zip(errors, funcs):
            # Order-0 (constant) integrands are integrated exactly.
            assert test_function.get_order() > 0 or error == 0.0
            assert error < bound, (
                f"Sobol dim={integration_dim} error {error} exceeds {bound} "
                f"for {type(test_function).__name__}"
            )


def _sobol_beats_mc_test(backend, dtype_name=None):
    """At equal N, Sobol must be more accurate than pseudo-random Monte Carlo.

    Both use a fixed seed, so this comparison is deterministic (no flakiness), and
    the QMC error is orders of magnitude smaller for a smooth integrand.
    """
    mc = MonteCarlo()
    sobol_error = abs(
        _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Sobol(backend=backend, seed=0),
            )
        )
        - _EXPECTED
    )
    mc_error = abs(
        _to_float(mc.integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0))
        - _EXPECTED
    )
    assert sobol_error < mc_error, (
        f"Sobol error {sobol_error} not below Monte Carlo error {mc_error}"
    )


def _sobol_determinism_test(backend, dtype_name=None):
    """A fixed seed must reproduce the same result bit-for-bit."""
    mc = MonteCarlo()

    def run():
        return _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Sobol(backend=backend, seed=42),
            )
        )

    assert run() == run(), "Sobol integration is not reproducible for a fixed seed"


test_sobol_collection_numpy = setup_test_for_backend(_sobol_collection_test, "numpy", "float64")
test_sobol_collection_torch = setup_test_for_backend(_sobol_collection_test, "torch", "float64")
test_sobol_collection_tensorflow = setup_test_for_backend(
    _sobol_collection_test, "tensorflow", "float64"
)
test_sobol_collection_jax = setup_test_for_backend(_sobol_collection_test, "jax", "float64")

test_sobol_beats_mc_numpy = setup_test_for_backend(_sobol_beats_mc_test, "numpy", "float64")
test_sobol_beats_mc_torch = setup_test_for_backend(_sobol_beats_mc_test, "torch", "float64")
test_sobol_beats_mc_tensorflow = setup_test_for_backend(
    _sobol_beats_mc_test, "tensorflow", "float64"
)
test_sobol_beats_mc_jax = setup_test_for_backend(_sobol_beats_mc_test, "jax", "float64")

test_sobol_determinism_numpy = setup_test_for_backend(_sobol_determinism_test, "numpy", "float64")
test_sobol_determinism_torch = setup_test_for_backend(_sobol_determinism_test, "torch", "float64")
test_sobol_determinism_tensorflow = setup_test_for_backend(
    _sobol_determinism_test, "tensorflow", "float64"
)
test_sobol_determinism_jax = setup_test_for_backend(_sobol_determinism_test, "jax", "float64")


def test_sobol_preserves_gradient():
    """Sobol points are constants, so autodiff through the integral must survive."""
    import torch

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
        rng=Sobol(backend="torch", seed=0),
    )
    result.backward()
    assert abs(_to_float(parameter.grad) - 0.5) < 1e-3, (
        "Gradient did not flow through the Sobol-sampled Monte Carlo integral"
    )


if __name__ == "__main__":
    for _backend in ["numpy", "torch", "tensorflow", "jax"]:
        _sobol_collection_test(_backend)
        _sobol_beats_mc_test(_backend)
        _sobol_determinism_test(_backend)
    test_sobol_preserves_gradient()
    print("All Sobol tests passed!")

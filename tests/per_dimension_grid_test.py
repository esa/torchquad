"""Per-dimension point counts for the grid-based integrators.

Checks that ``N=(n_0, n_1, ...)`` puts each count on its own axis, on every
backend and in the compiled path, against closed-form integrals.
"""

import math
import warnings

import autoray as ar
import pytest
from autoray import numpy as anp

from torchquad import Boole, GaussLegendre, Simpson, Trapezoid
from helper_functions import compute_integration_test_errors, setup_test_for_backend

INTEGRATORS = [Trapezoid, Simpson, Boole, GaussLegendre]
# Valid for every rule: odd for Simpson and of the form 4n + 1 for Boole.
COLLECTION_COUNTS = (9, 13, 17)
# Twice the largest error on the 3-D analytic collection with COLLECTION_COUNTS.
# The measured values are 4.6e-1, 3.2e-3, 1.4e-4 and 4.3e-6, identical on all
# four backends and always on the exponential test function.
COLLECTION_BOUNDS = {
    Trapezoid: 9.3e-1,
    Simpson: 6.4e-3,
    Boole: 2.9e-4,
    GaussLegendre: 8.7e-6,
}


def _domain(bounds, backend, dtype_name):
    return anp.array(bounds, like=backend, dtype=ar.to_backend_dtype(dtype_name, like=backend))


def _oscillating_integrand(points):
    """Oscillates quickly along the first axis and is linear along the second."""
    return anp.sin(20.0 * points[:, 0]) * (1.0 + 0.1 * points[:, 1])


OSCILLATING_EXACT = (1.0 - math.cos(20.0)) / 20.0 * 1.05


def _run_collection_tests(backend, _precision):
    """Unequal counts must integrate the whole analytic collection, real and complex."""
    for integrator_class in INTEGRATORS:
        errors, functions = compute_integration_test_errors(
            integrator_class().integrate,
            {"N": COLLECTION_COUNTS, "dim": 3},
            integration_dim=3,
            use_complex=True,
            backend=backend,
        )
        for error, test_function in zip(errors, functions):
            assert error.max() < COLLECTION_BOUNDS[integrator_class], (
                f"{integrator_class.__name__} error {error} on {type(test_function).__name__}"
            )


def _run_axis_order_tests(backend, dtype_name):
    """Each count must land on its own axis.

    The integrand needs many points along the first axis and two along the
    second. Swapping the counts leaves an error of about 0.1, orders of
    magnitude above the bound, so a rule that mixed up the axes could not pass.
    """
    domain = _domain([[0.0, 1.0], [0.0, 1.0]], backend, dtype_name)
    cases = [(Trapezoid, (801, 5), 2e-4), (Simpson, (201, 5), 1e-7), (Boole, (201, 5), 1e-8)]
    for integrator_class, counts, bound in cases:
        integrator = integrator_class()
        result = integrator.integrate(
            _oscillating_integrand, dim=2, N=counts, integration_domain=domain
        )
        assert integrator._nr_of_fevals == counts[0] * counts[1]
        assert abs(float(ar.to_numpy(result)) - OSCILLATING_EXACT) < bound
        swapped = integrator_class().integrate(
            _oscillating_integrand, dim=2, N=counts[::-1], integration_domain=domain
        )
        assert abs(float(ar.to_numpy(swapped)) - OSCILLATING_EXACT) > 0.05

    # Gauss-Legendre: two nodes are exact along the linear axis, 20 resolve the sine.
    result = GaussLegendre().integrate(
        _oscillating_integrand, dim=2, N=(20, 2), integration_domain=domain
    )
    assert abs(float(ar.to_numpy(result)) - OSCILLATING_EXACT) < 1e-12
    swapped = GaussLegendre().integrate(
        _oscillating_integrand, dim=2, N=(2, 20), integration_domain=domain
    )
    assert abs(float(ar.to_numpy(swapped)) - OSCILLATING_EXACT) > 1e-3


def _run_equal_counts_tests(backend, dtype_name):
    """An equal tuple, a list and a one-shot iterator all match the scalar form exactly."""
    domain = _domain([[0.0, 1.0], [0.0, 2.0]], backend, dtype_name)
    for integrator_class in INTEGRATORS:
        scalar = integrator_class().integrate(
            _oscillating_integrand, dim=2, N=9 * 9, integration_domain=domain
        )
        for counts in [(9, 9), [9, 9], iter((9, 9))]:
            result = integrator_class().integrate(
                _oscillating_integrand, dim=2, N=counts, integration_domain=domain
            )
            assert float(ar.to_numpy(result)) == float(ar.to_numpy(scalar))


def _run_adjusted_count_tests(backend, dtype_name):
    """A count the rule cannot use is lowered with a warning, on its own axis only."""
    domain = _domain([[0.0, 1.0], [0.0, 2.0]], backend, dtype_name)
    cases = [(Simpson, (4, 7), (3, 7)), (Simpson, (2, 7), (3, 7)), (Boole, (8, 9), (5, 9))]
    for integrator_class, requested, expected in cases:
        integrator = integrator_class()
        with pytest.warns(UserWarning, match="N per dim"):
            result = integrator.integrate(
                lambda points: points[:, 0] + points[:, 1],
                dim=2,
                N=requested,
                integration_domain=domain,
            )
        assert integrator._nr_of_fevals == expected[0] * expected[1]
        assert abs(float(ar.to_numpy(result)) - 3.0) < 1e-12
    # Counts that already fit must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Boole().integrate(lambda points: points[:, 0], dim=2, N=(5, 9), integration_domain=domain)


def _run_compiled_tests(backend, dtype_name):
    """The compiled path agrees with the eager one for every rule."""
    domain = _domain([[0.0, 1.0], [0.0, 1.0]], backend, dtype_name)
    cases = [(Trapezoid, (41, 5)), (Simpson, (41, 5)), (Boole, (41, 5)), (GaussLegendre, (12, 2))]
    for integrator_class, counts in cases:
        eager = integrator_class().integrate(
            _oscillating_integrand, dim=2, N=counts, integration_domain=domain
        )
        compiled = integrator_class().get_jit_compiled_integrate(
            dim=2, N=counts, integration_domain=domain
        )
        result = compiled(_oscillating_integrand, domain)
        assert abs(float(ar.to_numpy(result)) - float(ar.to_numpy(eager))) < 1e-13


def _run_validation_tests(backend, dtype_name):
    """Malformed per-dimension counts raise before any grid is built."""
    domain = _domain([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], backend, dtype_name)
    for integrator_class in INTEGRATORS:
        for bad_counts, message in [
            ((5, 5), "must contain 3 values"),
            ((5, 0, 5), "positive integers"),
            ((5, 2.5, 5), "have to be integers"),
        ]:
            with pytest.raises(ValueError, match=message):
                integrator_class().integrate(
                    lambda points: points[:, 0], dim=3, N=bad_counts, integration_domain=domain
                )


for _backend in ["numpy", "torch", "jax", "tensorflow"]:
    for _name, _function in [
        ("collection", _run_collection_tests),
        ("axis_order", _run_axis_order_tests),
        ("equal_counts", _run_equal_counts_tests),
        ("adjusted_counts", _run_adjusted_count_tests),
        ("validation", _run_validation_tests),
    ]:
        globals()[f"test_{_name}_{_backend}"] = setup_test_for_backend(
            _function, _backend, "float64"
        )
    if _backend != "numpy":
        globals()[f"test_compiled_{_backend}"] = setup_test_for_backend(
            _run_compiled_tests, _backend, "float64"
        )

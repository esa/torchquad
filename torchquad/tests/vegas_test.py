import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype, astype
import timeit
import cProfile
import pstats
from unittest.mock import patch

from integration.vegas import VEGAS
from integration.rng import RNG

from helper_functions import (
    compute_integration_test_errors,
    setup_test_for_backend,
)


def _run_example_integrations(backend, dtype_name):
    """Test the integrate method in VEGAS for the given backend and example test functions using compute_integration_test_errors"""
    print(f"Testing VEGAS+ with example functions with {backend}, {dtype_name}")
    vegas = VEGAS()

    # 1D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 1, "seed": 0},
        dim=1,
        use_complex=False,
        backend=backend,
    )
    print("1D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors[:3]:
        assert error < 5e-3

    for error in errors:
        assert error < 9.0

    for error in errors[6:]:
        assert error < 6e-3

    # 3D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 3, "seed": 0},
        dim=3,
        use_complex=False,
        backend=backend,
    )
    print("3D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 0.61

    # 10D Tests
    N = 10000
    errors, _ = compute_integration_test_errors(
        vegas.integrate,
        {"N": N, "dim": 10, "seed": 0},
        dim=10,
        use_complex=False,
        backend=backend,
    )
    print("10D VEGAS Test: Passed N =", N, "\n", "Errors: ", errors)
    for error in errors:
        assert error < 12.5


def _run_vegas_accuracy_checks(backend, dtype_name):
    """Test VEGAS+ with special peak integrands where it should be significantly more accurate than MonteCarlo"""
    print(f"Testing VEGAS+ accuracy with {backend}, {dtype_name}")
    dtype = to_backend_dtype(dtype_name, like=backend)
    integrator = VEGAS()

    print("Integrating a function with a single peak")
    integration_domain = anp.array(
        [[1.0, 5.0], [-4.0, 4.0], [2.0, 6.0]], dtype=dtype, like=backend
    )
    dim = integration_domain.shape[0]

    def integrand_hypercube_peak(x):
        """An integrand which is close to zero everywhere except in a hypercube of volume 1."""
        # A product corresponds to logical And
        in_cube = anp.prod((x >= 3.0) * (x < 4.0), axis=1)
        # Add 0.01 since VEGAS+ does not yet support integrands which evaluate
        # to zero for all passed points
        return astype(in_cube, dtype_name) + 0.001

    reference_integral = (
        anp.prod(integration_domain[:, 1] - integration_domain[:, 0]) * 0.001 + 1.0
    )

    # Use multiple seeds to reduce luck
    for seed in [0, 1, 2, 3, 41317]:
        integral = integrator.integrate(
            integrand_hypercube_peak,
            dim,
            N=30000,
            integration_domain=integration_domain,
            seed=seed,
        )
        assert anp.abs(integral - reference_integral) < 0.03

    print("Integrating a function with peaks on the diagonal")
    peak_distance = 100.0
    integration_domain = anp.array(
        [[1.0, 1.0 + peak_distance], [-4.0, -4.0 + peak_distance]],
        dtype=dtype,
        like=backend,
    )
    dim = 2

    def integrand_diagonal_peaks(x):
        """An integrand which is close to zero everywhere except two corners of the integration domain."""
        a = anp.exp(anp.sum(integration_domain[:, 0] - x, axis=1))
        b = anp.exp(anp.sum(x - integration_domain[:, 1], axis=1))
        return a + b

    # If the integration domain is [r_1, r_1 + c]x[r_2, r_2 + c]
    # for some numbers r_1, r_2,
    # the integral of integrand_diagonal_peaks is the integral of
    # exp(-x_1) exp(-x_2) + exp(x_1 - c) exp(x_2 - c) over x in [0, c]^2.
    # indefinite integral:
    # F(x) = exp(-x_1) exp(-x_2) + exp(x_1 - c) exp(x_2 - c)
    # definite integral:
    # F((c,c)) - F((c,0)) - F((0,c)) + F((0,0)) = 2 - 4 exp(-c) + 2 exp(-2c)
    reference_integral = (
        2.0
        - 4.0 * anp.exp(-peak_distance, like="numpy")
        + 2.0 * anp.exp(-2.0 * peak_distance, like="numpy")
    )

    # Use multiple seeds to reduce luck
    for seed in [0, 1, 2, 3, 41317]:
        integral = integrator.integrate(
            integrand_diagonal_peaks,
            dim,
            N=30000,
            integration_domain=integration_domain,
            seed=seed,
        )
        assert anp.abs(integral - reference_integral) < 0.03


class ModifiedRNG(RNG):
    """A modified Random Number Generator which replaces some of the random numbers with 0.0 and 1.0"""

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        rng_uniform = self.uniform
        self.uniform = lambda *args, **kargs: self.modify_numbers(
            rng_uniform(*args, **kargs)
        )

    def modify_numbers(self, numbers):
        """Change the randomly generated numbers"""
        zeros = anp.zeros(numbers.shape, dtype=numbers.dtype, like=numbers)
        ones = anp.ones(numbers.shape, dtype=numbers.dtype, like=numbers)
        # Replace half of the random values randomly with 0.0 or 1.0
        return anp.where(
            numbers < 0.5, numbers * 2.0, anp.where(numbers < 0.75, zeros, ones)
        )


def _run_vegas_special_case_checks(backend, dtype_name):
    """Test VEGAS+ in special cases, for example an integrand which is zero everywhere"""
    print(f"Testing VEGAS+ special cases with {backend}, {dtype_name}")
    integrator = VEGAS()

    print("Testing VEGAS with an integrand which is zero everywhere")
    integral = integrator.integrate(
        lambda x: x[:, 0] * 0.0,
        2,
        N=10000,
        integration_domain=[[0.0, 3.0]] * 2,
        seed=0,
        backend=backend,
    )
    assert anp.abs(integral) == 0.0

    print("Testing VEGAS with a constant integrand")
    integral = integrator.integrate(
        lambda x: x[:, 0] * 0.0 + 10.0,
        2,
        N=10000,
        integration_domain=[[0.0, 3.0]] * 2,
        seed=0,
        backend=backend,
    )
    assert anp.abs(integral - 90.0) < 1e-13

    print("Testing VEGAS with random numbers which are 0.0 and 1.0")
    # This test may be helpful to detect rounding and indexing errors which
    # would happen with a low probability with the usual RNG
    with patch("integration.vegas.RNG", ModifiedRNG):
        integral = integrator.integrate(
            lambda x: anp.sum(x, axis=1),
            2,
            N=10000,
            integration_domain=[[0.0, 1.0]] * 2,
            seed=0,
            backend=backend,
        )
    assert isinstance(integrator.rng, ModifiedRNG)
    assert anp.abs(integral - 1.0) < 0.1


def _run_vegas_tests(backend, dtype_name):
    """Test if VEGAS+ works with example functions and is accurate as expected"""
    _run_vegas_accuracy_checks(backend, dtype_name)
    _run_vegas_special_case_checks(backend, dtype_name)
    _run_example_integrations(backend, dtype_name)


test_integrate_numpy = setup_test_for_backend(_run_vegas_tests, "numpy", "float64")
test_integrate_torch = setup_test_for_backend(_run_vegas_tests, "torch", "float64")


if __name__ == "__main__":
    # used to run this test individually
    test_integrate_numpy()

    profile_torch = False
    if profile_torch:
        profiler = cProfile.Profile()
        profiler.enable()
        start = timeit.default_timer()
        test_integrate_torch()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats()
        stop = timeit.default_timer()
        print("Test ran for ", stop - start, " seconds.")
    else:
        test_integrate_torch()

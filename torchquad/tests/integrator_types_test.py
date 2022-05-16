#!/usr/bin/env python3
"""
Additional integration tests to check if dtypes, shapes and similar
backend-specific properties
"""
import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, to_backend_dtype
from itertools import product

from integration.trapezoid import Trapezoid
from integration.simpson import Simpson
from integration.boole import Boole
from integration.monte_carlo import MonteCarlo
from integration.vegas import VEGAS
from utils.set_precision import set_precision
from helper_functions import setup_test_for_backend


def _run_simple_integrations(backend):
    """
    Integrate a simple 2D constant function to check the following:
    * The integrators do not crash with the numerical backend
    * The evaluation points have the correct backend, dtype and shape
    * The integration_domain argument dtype takes precedence over a globally
      configured dtype
    * The globally configured dtype or the backend's default dtype is used if
      the integration_domain argument is a list
    * MonteCarlo and the Newton Cotes composite integrators integrate a
      constant function (almost) exactly.
    """
    integrators_all = [Trapezoid(), Simpson(), Boole(), MonteCarlo(), VEGAS()]
    Ns_all = [13**2, 13**2, 13**2, 20, 1000]

    expected_dtype_name = None

    # Test only integrand output dtypes which are the same as the input dtype
    def fn_const(x):
        assert infer_backend(x) == backend
        assert get_dtype_name(x) == expected_dtype_name
        assert len(x.shape) == 2 and x.shape[1] == 2
        return 0.0 * x[:, 0] - 2.0

    for dtype_global, dtype_arg, (integrator, N) in product(
        ["float32", "float64"],
        [None, "float32", "float64"],
        zip(integrators_all, Ns_all),
    ):
        # JAX ignores the dtype argument when an array is created and always
        # uses the global precision.
        if (backend, dtype_global, dtype_arg) in [
            ("jax", "float32", "float64"),
            ("jax", "float64", "float32"),
        ]:
            continue
        integrator_name = type(integrator).__name__
        # VEGAS supports only numpy and torch
        if integrator_name == "VEGAS" and backend in ["jax", "tensorflow"]:
            continue

        # Set the global precision
        set_precision(dtype_global, backend=backend)

        integration_domain = [[0.0, 1.0], [-2.0, 0.0]]
        if dtype_arg is not None:
            # Set the integration_domain dtype which should have higher priority
            # than the global dtype
            integration_domain = anp.array(
                integration_domain,
                dtype=to_backend_dtype(dtype_arg, like=backend),
                like=backend,
            )
            assert infer_backend(integration_domain) == backend
            assert get_dtype_name(integration_domain) == dtype_arg
            expected_dtype_name = dtype_arg
        else:
            expected_dtype_name = dtype_global

        print(
            f"[2mTesting {integrator_name} with {backend}, argument dtype"
            f" {dtype_arg}, global/default dtype {dtype_global}[m"
        )
        if integrator_name in ["MonteCarlo", "VEGAS"]:
            extra_kwargs = {"seed": 0}
        else:
            extra_kwargs = {}
        result = integrator.integrate(
            fn=fn_const,
            dim=2,
            N=N,
            integration_domain=integration_domain,
            backend=backend,
            **extra_kwargs,
        )
        assert infer_backend(result) == backend
        assert get_dtype_name(result) == expected_dtype_name
        # VEGAS seems to be bad at integrating constant functions currently
        max_error = 0.03 if integrator_name == "VEGAS" else 1e-5
        assert anp.abs(result - (-4.0)) < max_error


test_integrate_numpy = setup_test_for_backend(_run_simple_integrations, "numpy", None)
test_integrate_torch = setup_test_for_backend(_run_simple_integrations, "torch", None)
test_integrate_jax = setup_test_for_backend(_run_simple_integrations, "jax", None)
test_integrate_tensorflow = setup_test_for_backend(
    _run_simple_integrations, "tensorflow", None
)


if __name__ == "__main__":
    try:
        test_integrate_numpy()
        test_integrate_torch()
        test_integrate_jax()
        test_integrate_tensorflow()
    except KeyboardInterrupt:
        pass

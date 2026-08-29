"""Deterministic-rule accuracy against analytic ground truth, on every backend.

Correctness across all four backends is torchquad's headline guarantee, so these
run each Newton-Cotes / Gaussian rule over smooth 1D densities and a 3D separable
integrand and assert rule-appropriate relative error in float64.
"""

import pytest

from torchquad import Trapezoid, Simpson, Boole, GaussLegendre

import _integrands as ig
from conftest import rel_error

# (rule class, N for a 1D integral, relative tolerance). N is generous because
# this suite is release-only and may be slower than CI. Tolerances reflect each
# rule's convergence order, not an arbitrary loose bound.
RULES_1D = [
    (Trapezoid, 100001, 1e-4),
    (Simpson, 100001, 1e-8),
    (Boole, 100001, 1e-9),
    (GaussLegendre, 201, 1e-9),
]


@pytest.mark.parametrize("integrand", ig.SMOOTH_1D, ids=[c[0] for c in ig.SMOOTH_1D])
@pytest.mark.parametrize("rule_cls,n_points,tol", RULES_1D, ids=[r[0].__name__ for r in RULES_1D])
def test_smooth_1d_accuracy(backend_f64, rule_cls, n_points, tol, integrand):
    """Every rule integrates smooth 1D densities to its expected precision."""
    name, fn, domain, expected = integrand
    integrator = rule_cls()
    result = integrator.integrate(
        fn, dim=1, N=n_points, integration_domain=domain, backend=backend_f64
    )
    error = rel_error(result, expected)
    assert error < tol, (
        f"{rule_cls.__name__} on {name} ({backend_f64}): rel err {error:.2e} >= {tol:.0e}"
    )


# (rule class, N for a 3D integral, relative tolerance). N**(1/3) points per axis.
RULES_3D = [
    (Simpson, 101**3, 1e-6),  # 101 is odd -> valid Simpson panel count per axis
    (GaussLegendre, 20**3, 1e-9),
]


@pytest.mark.parametrize("rule_cls,n_points,tol", RULES_3D, ids=[r[0].__name__ for r in RULES_3D])
def test_separable_3d_accuracy(backend_f64, rule_cls, n_points, tol):
    """Rules integrate sum_i sin(x_i) over [0,1]^3 (multi-dim grid correctness)."""
    integrator = rule_cls()
    domain = [[0.0, 1.0]] * 3
    result = integrator.integrate(
        ig.sum_sin, dim=3, N=n_points, integration_domain=domain, backend=backend_f64
    )
    expected = ig.sum_sin_expected(3)
    error = rel_error(result, expected)
    assert error < tol, f"{rule_cls.__name__} 3D ({backend_f64}): rel err {error:.2e} >= {tol:.0e}"

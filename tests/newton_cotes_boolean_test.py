"""Boolean quadrature and fixed-format compiled integration regressions."""

from functools import partial
from fractions import Fraction

import autoray as ar
from autoray import numpy as anp
import numpy as np
import pytest

from torchquad import Boole, Simpson, Trapezoid, set_up_backend


RULES = [
    (Trapezoid, Fraction(3, 8)),
    (Simpson, Fraction(5, 12)),
    (Boole, Fraction(13, 30)),
]


def _boolean_integrand(points, pattern, matrix):
    if pattern == "true":
        values = points[:, 0] == points[:, 0]
    elif pattern == "false":
        values = points[:, 0] != points[:, 0]
    else:
        values = points[:, 0] < 0.25
    if matrix:
        return anp.stack([values, ~values], axis=1)[:, :, None]
    return values


def _expected_result(pattern, matrix, dim, mixed_fraction):
    # Five nodes on [0, 0.5] give the mixed mask [True, True, False, False, False].
    # The fractions are its exact quadrature weights on a unit-width interval.
    fraction = {"true": 1, "false": 0, "mixed": mixed_fraction}[pattern]
    volume = 0.5**dim
    expected = float(fraction) * volume
    if matrix:
        return np.array([[expected], [volume - expected]])
    return np.array(expected)


@pytest.mark.parametrize("backend", ["numpy", "torch", "tensorflow", "jax"])
@pytest.mark.parametrize("integrator_class,mixed_fraction", RULES)
@pytest.mark.parametrize("pattern", ["true", "false", "mixed"])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dim", [1, 2])
def test_boolean_integrand(backend, integrator_class, mixed_fraction, pattern, matrix, dim):
    pytest.importorskip(backend)
    set_up_backend(backend, "float64", torch_enable_cuda=False)
    domain = anp.array(
        [[0.0, 0.5]] * dim, like=backend, dtype=ar.to_backend_dtype("float64", like=backend)
    )
    result = integrator_class().integrate(
        _boolean_integrand,
        dim=dim,
        N=5**dim,
        integration_domain=domain,
        args=(pattern, matrix),
    )
    expected = _expected_result(pattern, matrix, dim, mixed_fraction)
    assert result.shape == expected.shape
    np.testing.assert_allclose(ar.to_numpy(result), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("backend", ["torch", "tensorflow", "jax"])
@pytest.mark.parametrize("integrator_class,mixed_fraction", RULES)
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dim", [1, 2])
def test_boolean_jit_fixed_format(backend, integrator_class, mixed_fraction, matrix, dim):
    pytest.importorskip(backend)
    set_up_backend(backend, "float64", torch_enable_cuda=False)
    domain = anp.array(
        [[0.0, 0.5]] * dim, like=backend, dtype=ar.to_backend_dtype("float64", like=backend)
    )
    compiled = integrator_class().get_jit_compiled_integrate(
        dim=dim, N=5**dim, integration_domain=domain
    )
    # Every call to this compiled function has the same Boolean dtype and shape.
    for pattern in ["true", "false", "mixed"]:
        result = compiled(partial(_boolean_integrand, pattern=pattern, matrix=matrix), domain)
        expected = _expected_result(pattern, matrix, dim, mixed_fraction)
        assert result.shape == expected.shape
        np.testing.assert_allclose(ar.to_numpy(result), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("backend", ["torch", "tensorflow", "jax"])
@pytest.mark.parametrize("integrator_class", [Trapezoid, Simpson, Boole])
@pytest.mark.parametrize("compiled", [False, True])
def test_boolean_domain_gradient(backend, integrator_class, compiled):
    module = pytest.importorskip(backend)
    set_up_backend(backend, "float64", torch_enable_cuda=False)
    domain = anp.array(
        [[0.0, 0.5]], like=backend, dtype=ar.to_backend_dtype("float64", like=backend)
    )
    if backend == "torch":
        domain.requires_grad_(True)
    integrand = partial(_boolean_integrand, pattern="true", matrix=True)
    integrator = integrator_class()
    if compiled:
        integrate = partial(
            integrator.get_jit_compiled_integrate(dim=1, N=5, integration_domain=domain),
            integrand,
        )
    else:

        def integrate(bounds):
            return integrator.integrate(integrand, dim=1, N=5, integration_domain=bounds)

    if backend == "torch":
        result = integrate(domain)
        gradient = module.autograd.grad(result.sum(), domain)[0]
    elif backend == "tensorflow":
        with module.GradientTape() as tape:
            tape.watch(domain)
            result = integrate(domain)
            total = anp.sum(result)
        gradient = tape.gradient(total, domain)
    else:
        result = integrate(domain)
        gradient = module.grad(lambda bounds: anp.sum(integrate(bounds)))(domain)
    np.testing.assert_allclose(ar.to_numpy(result), [[0.5], [0.0]], rtol=1e-14, atol=0)
    np.testing.assert_allclose(ar.to_numpy(gradient), [[-1.0, 1.0]], rtol=1e-14, atol=0)


def test_trapezoid_subnormal_float():
    smallest = np.nextafter(0.0, 1.0)
    result = Trapezoid().integrate(
        lambda points: np.full(points.shape[0], smallest),
        dim=1,
        N=2,
        integration_domain=np.array([[0.0, 1.0]]),
    )
    assert result == smallest

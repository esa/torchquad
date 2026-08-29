"""Differentiability through the integrator — the feature that sets torchquad apart.

Covers the two gradient patterns seen in real usage: gradient of the integral
w.r.t. the integration-domain bounds (tutorial) and w.r.t. a learned integrand
parameter (the vebglm variational-Bayes pattern). Ground truth is the Leibniz
rule / differentiation under the integral sign.
"""

import math

import pytest

from torchquad import Simpson, GaussLegendre, set_up_backend


def test_torch_gradient_wrt_domain():
    """d/db integral = f(b), d/da integral = -f(a) (Leibniz), via torch autograd."""
    import torch

    set_up_backend("torch", data_type="float64")
    a, b = 0.5, 2.0
    domain = torch.tensor([[a, b]], dtype=torch.float64, requires_grad=True)
    simpson = Simpson()
    # Simpson is exact for x^2, so the quadrature gradient equals the exact Leibniz value.
    result = simpson.integrate(lambda x: x[:, 0] ** 2, dim=1, N=10001, integration_domain=domain)
    result.backward()
    grad = domain.grad
    assert abs(grad[0, 0].item() - (-(a**2))) < 1e-5, f"d/da wrong: {grad[0, 0].item()}"
    assert abs(grad[0, 1].item() - (b**2)) < 1e-5, f"d/db wrong: {grad[0, 1].item()}"


def test_torch_gradient_wrt_parameter():
    """Backprop into an integrand parameter: d/dtheta integral exp(-theta x^2)."""
    import torch

    set_up_backend("torch", data_type="float64")
    theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    simpson = Simpson()
    result = simpson.integrate(
        lambda x: torch.exp(-theta * x[:, 0] ** 2),
        dim=1,
        N=10001,
        integration_domain=[[-6.0, 6.0]],
        backend="torch",
    )
    result.backward()
    # integral = sqrt(pi/theta) on (-inf,inf); d/dtheta = -0.5*sqrt(pi)*theta^(-3/2).
    expected_grad = -0.5 * math.sqrt(math.pi) * theta.item() ** -1.5
    assert abs(theta.grad.item() - expected_grad) < 1e-3, (
        f"param grad {theta.grad.item()} != {expected_grad}"
    )


def test_jax_gradient_wrt_domain():
    """jax.grad of the integral w.r.t. an upper bound equals f(b)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    set_up_backend("jax", data_type="float64")
    gl = GaussLegendre()

    def integral_to_b(b):
        domain = jnp.array([[0.0, b]])
        return gl.integrate(
            lambda x: x[:, 0] ** 2, dim=1, N=101, integration_domain=domain, backend="jax"
        )

    b = 2.0
    grad = float(jax.grad(integral_to_b)(b))
    assert abs(grad - b**2) < 1e-5, f"jax d/db wrong: {grad} != {b**2}"

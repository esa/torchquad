from fractions import Fraction
import numpy as np
import pytest
from torchquad import Trapezoid, Simpson, Boole

RULES = [
    (Trapezoid, 2, Fraction(1, 2)),
    (Simpson, 3, Fraction(1, 6)),
    (Boole, 5, Fraction(7, 90)),
]
CASES = [
    ("numpy", "float64", "int64", 53, "eager"),
    ("numpy", "float64", "int64", 53, "manual"),
    ("torch", "float64", "int64", 53, "eager"),
    ("torch", "float64", "int64", 53, "manual"),
    ("torch", "float64", "int64", 53, "jit"),
    ("torch", "float32", "int32", 24, "eager"),
    ("torch", "float32", "int32", 24, "manual"),
    ("torch", "float32", "int32", 24, "jit"),
]


@pytest.mark.parametrize("rule,n,expected", RULES)
@pytest.mark.parametrize("backend,domain_dtype,value_dtype,exponent,entry", CASES)
def test_preserve_exact_integer_cancellation(
    rule, n, expected, backend, domain_dtype, value_dtype, exponent, entry
):
    values = [2**exponent + 1, *([0] * (n - 2)), -(2**exponent)]
    if backend == "torch":
        import torch

        domain = torch.tensor([[0.0, 1.0]], dtype=getattr(torch, domain_dtype))
        samples = torch.tensor(values, dtype=getattr(torch, value_dtype))
    else:
        domain = np.array([[0.0, 1.0]], dtype=domain_dtype)
        samples = np.array(values, dtype=value_dtype)

    def integrand(points):
        assert points.shape[0] == n
        return samples

    integrator = rule()
    if entry == "eager":
        result = integrator.integrate(integrand, dim=1, N=n, integration_domain=domain)
    elif entry == "manual":
        points, widths, count = integrator.calculate_grid(n, domain)
        evaluated, _ = integrator.evaluate_integrand(integrand, points)
        result = integrator.calculate_result(
            function_values=evaluated, dim=1, n_per_dim=count, hs=widths, integration_domain=domain
        )
    else:
        compiled = integrator.get_jit_compiled_integrate(dim=1, N=n, integration_domain=domain)
        result = compiled(integrand, domain)
    assert result.item() == pytest.approx(float(expected), rel=1e-6, abs=1e-12)
    observed = samples.tolist()
    assert observed == values

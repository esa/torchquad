"""Complex-valued, oscillatory integrands — a real pattern in physics usage.

Fourier transforms / characteristic functions (e.g. the bottomonia energy-loss
kernel) integrate a complex integrand over a wide domain. These verify the real
and imaginary parts independently against closed-form transforms.
"""

import math

from torchquad import Simpson

import _integrands as ig
from conftest import to_numpy


def test_cauchy_characteristic_function(backend_f64):
    """int exp(-|t|) exp(-i f t) dt = 2/(1+f^2) (real); imaginary part vanishes."""
    f = 2.0
    simpson = Simpson()
    result = simpson.integrate(
        ig.cauchy_char_integrand(f),
        dim=1,
        N=200001,
        integration_domain=[[-50.0, 50.0]],
        backend=backend_f64,
    )
    value = complex(to_numpy(result).reshape(()))
    expected_real = 2.0 / (1.0 + f**2)
    assert abs(value.real - expected_real) < 1e-3, f"real {value.real} != {expected_real}"
    assert abs(value.imag) < 1e-6, f"imag part should vanish, got {value.imag}"


def test_gaussian_characteristic_function(backend_f64):
    """int exp(-t^2) exp(-i f t) dt = sqrt(pi) exp(-f^2/4) (real); imag vanishes."""
    f = 2.0
    simpson = Simpson()
    result = simpson.integrate(
        ig.gaussian_char_integrand(f),
        dim=1,
        N=100001,
        integration_domain=[[-8.0, 8.0]],
        backend=backend_f64,
    )
    value = complex(to_numpy(result).reshape(()))
    expected_real = math.sqrt(math.pi) * math.exp(-(f**2) / 4.0)
    assert abs(value.real - expected_real) < 1e-4, f"real {value.real} != {expected_real}"
    assert abs(value.imag) < 1e-6, f"imag part should vanish, got {value.imag}"

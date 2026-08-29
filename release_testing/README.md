# Release testing

Heavier, slower end-to-end checks that mirror how torchquad is **actually used in
the wild**. This suite is deliberately separate from `tests/`:

| | `tests/` (per-PR CI) | `release_testing/` (release only) |
|---|---|---|
| When | every push / PR | before a release + on GitHub release |
| Backends | the pinned CI environment (`environment_all_backends.yml`) | the **latest released** torch / JAX / TensorFlow, so a release cannot silently break against a new framework version |
| Speed | fast | slow is fine (large `N`, high dimensions) |
| Focus | unit-level correctness of each integrator | realistic user scenarios, cross-backend, autodiff, JIT |

The scenarios are drawn from a survey of real torchquad usage (particle/nuclear
physics, cosmology, Bayesian statistics, score-based ML, quantitative finance) —
see the mix of smooth 1D densities, high-dimensional sums, complex-valued Fourier
transforms, and gradient-through-the-integrand patterns below.

## What it covers

| File | Scenario | Ground truth |
|---|---|---|
| `test_core_accuracy.py` | Trapezoid / Simpson / Boole / GaussLegendre on smooth 1D densities and a 3D separable integrand, every backend, float64 | closed-form integrals |
| `test_stochastic.py` | canonical 2D MonteCarlo (float32), 10D MC/VEGAS, seeded determinism | analytic value + bit-for-bit reproducibility |
| `test_autodiff.py` | gradient w.r.t. domain bounds and w.r.t. a learned integrand parameter | Leibniz rule / differentiation under the integral |
| `test_complex.py` | complex-valued oscillatory Fourier kernels | characteristic-function transforms |
| `test_jit.py` | `get_jit_compiled_integrate` (torch/JAX/TF) vs the eager path | eager result at the same seed |

Analytic integrands and their exact integrals live in `_integrands.py`; shared
fixtures/helpers in `conftest.py`.

## Running it

Unlike `tests/`, this suite runs against an **installed** torchquad, so run it
from the repository root (not from inside the folder):

```bash
# In an environment with the newest backends you want to validate against:
pip install torch "jax[cpu]" tensorflow      # latest releases
pip install -e .
pytest release_testing/ -v -ra --durations=15
```

In CI this is the **Release testing** workflow
(`.github/workflows/release_testing.yml`), triggered manually
(`workflow_dispatch`) or automatically when a GitHub release is created. It runs a
Python 3.10 / 3.11 / 3.12 matrix against the latest released backends. Running it
is a step in the release checklist (`.github/ISSUE_TEMPLATE/release.md`).

## Adding a scenario

1. Add the integrand and its exact integral to `_integrands.py` (keep it
   backend-agnostic via `from autoray import numpy as anp`).
2. Add a test that asserts a **rule-appropriate** tolerance — tie the bound to the
   method's convergence order, not an arbitrary loose number. Document why if you
   loosen it.
3. Prefer covering all four backends via the `backend_f64` fixture unless the
   scenario is backend-specific (e.g. torch autograd, VEGAS on torch/numpy).

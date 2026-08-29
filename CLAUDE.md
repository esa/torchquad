# CLAUDE.md

Guidance for Claude Code when working with the torchquad repository.

## Project Overview

torchquad is a multi-backend numerical integration library for GPUs and autodiff. The same `integrate()` API runs on PyTorch, JAX, TensorFlow, or NumPy via [autoray](https://github.com/jcmgray/autoray), and all methods are fully differentiable on the framework backends. Used in published research across cosmology, biology, soft robotics, and quantum computing — **correctness is the headline feature, not performance alone**.

Maintained by ESA's Advanced Concepts Team. Originally published in JOSS (2021).

## Branching Model

- **`main`** — releases, hotfixes, doc fixes only
- **`develop`** — features and refactors target this branch
- Never push directly to `main`; open a PR into `develop`

## Environment & Commands

```bash
# Activate conda environment (REQUIRED — installs all four backends)
conda activate torchquad

# Install editable
pip install -e .
```

The runtime dependencies are `loguru`, `scipy`, `numpy` and `autoray` (see `[project.dependencies]`); `environment.yml` is the base *development* environment and also carries test and docs tooling. `environment_all_backends.yml` adds PyTorch, JAX, and TensorFlow for full multi-backend test coverage. CI uses `environment_all_backends.yml`.

```bash
# Run all tests (must be from tests/ — sys.path.append-based imports)
cd tests/ && pytest

# Single test file / single test
cd tests/ && pytest boole_test.py
cd tests/ && pytest monte_carlo_test.py::test_monte_carlo

# Full CI command (coverage + skip-as-error)
cd tests/ && pytest -ra --error-for-skips --junitxml=pytest.xml \
    --cov-report=term-missing:skip-covered --cov=../torchquad .

# Lint + formatting + docstrings (must pass before committing)
ruff check .
ruff format --check .
pydoclint torchquad/

# Apply formatting
ruff format .

# Severe-error-only check (matches CI gate)
ruff check --select=E9,F63,F7,F82 .

# Dead-code check (100% tier blocks CI)
vulture torchquad .vulture_whitelist.py --min-confidence 100
```

**Shell note:** Do not prepend `cd` to the project root — the working directory is already set. Only `cd` for subdirectories (e.g. `tests/`).

## Coding Instructions

1. Follow PEP 8; format with `ruff format` (line length 100); lint clean under `ruff check` and `pydoclint torchquad/` (config in `pyproject.toml`).
2. **CORRECTNESS FIRST.** This library is cited in numerical-result papers. A silent precision regression is worse than a crash.
   - When changing a numerical kernel, run the full test sweep across all four backends, not just one.
   - Compare against analytic ground truth in `tests/integration_test_functions.py`. If a tolerance loosens, the PR description must explain why.
   - Preserve numerical determinism for fixed seeds — VEGAS, MonteCarlo, and any RNG-using path must reproduce bit-for-bit when given the same `RNG`.
3. **BACKEND AGNOSTIC.** Code under `torchquad/integration/` and `torchquad/utils/` must use `from autoray import numpy as anp` and `infer_backend(...)`, never raw `torch.*` / `jnp.*` / `tf.*` calls. Backend-specific paths belong only in `utils/enable_cuda.py`, `utils/set_precision.py`, and `utils/set_up_backend.py`, plus the RNG-style samplers `integration/rng.py` and `integration/qmc.py`, which necessarily branch per backend to reach each framework's native (pseudo/quasi-)random generators.
4. **VECTORIZE OR FAIL.** Integrand evaluation must be batched along the first dimension. No Python-level loops over sample points. If you can't vectorize a step on the GPU, that's a design red flag — discuss before implementing.
5. **FAIL HARD, NO FALLBACKS.**
   - Direct `dict["key"]` access, not `.get()` with defaults.
   - No silent precision downgrades; raise on backend/dtype mismatches.
   - No bare `except: pass`. Use `logger.warning()` for recoverable issues, raise for broken invariants.
6. **LOGGING.** Use `from loguru import logger`. `__init__.py` calls `logger.disable("torchquad")` so the library is silent by default and never pollutes a host app (issue #184). `set_log_level()` is the single sanctioned config lever: it `logger.enable("torchquad")`s and manages one tracked, `torchquad`-filtered stderr sink (replacing only its own handler id, never the host's). Do not add other `logger.add(...)`/`logger.remove()` calls in library code. `__init__.py` calls `set_log_level()` at import only when `TORCHQUAD_LOG_LEVEL` is set in the environment; there is no build-time logging switch.
7. **DOCSTRINGS.** Google style (`Args`, `Returns`, `Raises`). Public functions and classes must have one. Comments explain WHY, not what — a comment that paraphrases the next line is noise.
8. **NO ABBREVIATIONS** in new code (`integration_domain` not `int_dom`, `function_values` not `fvals`).
9. **TESTS REQUIRED FOR ALL CHANGES.** Bug fixes need a regression test. New features need both a correctness test (analytic integral) and a backend-coverage test. A new or changed integrator, sampler, or error estimate must be checked against the **whole** analytic test-function collection (real + complex, all dimensionalities) via `helper_functions.py::compute_integration_test_errors`, not just one or two ad-hoc integrands. Any reported error (including an *estimated* error like VEGAS's `sdev`) must be validated against the closed-form ground truth in `tests/integration_test_functions.py`.
10. **DEPRECATIONS.** When changing a public signature, keep the old path working for one release with a `warnings.warn(..., DeprecationWarning)`, and grep `pyproject.toml`'s `filterwarnings` to make sure the warning will actually be visible.
11. **COMMIT MESSAGES.** Conventional Commits in the form `<type>(<scope>): <description>`, e.g. `fix(vegas): reseed RNG per iteration`. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`. Scope is optional but preferred (integrator or module name). Never mention Claude, Claude Code, Anthropic, or AI assistance.
12. **SIZE BUDGETS.** Keep functions under ~100 lines and files under ~1000 lines; split before crossing. Prefer small functions and dataclasses over loose dicts passed between methods.
13. **DEAD CODE.** Delete unused code rather than whitelisting or commenting it out — git history is the archive. If a symbol must stay for API reasons, it needs a docstring saying why.

## Pull Requests

- Title: specific action under 70 chars. Body: 1–3 bullets on what and why (no code walk-throughs), `Closes #N` links, and a brief test plan.
- Reviews follow `REVIEW.md` (correctness-first, backend-agnostic, vectorize-or-fail, fail-hard). Skim it before opening a PR.

**CRITICAL:** Run the full test sweep after any change to `integration/` or `utils/`. ALL tests must pass on all four backends.
**CRITICAL:** Run `ruff format --check .`, `ruff check .`, `pydoclint torchquad/`, and `vulture torchquad .vulture_whitelist.py --min-confidence 100` before committing — CI gates on all of them.

## Architecture

Top-level layout:

```
torchquad/
  torchquad/                  # Library
    __init__.py               # Public API surface
    integration/              # Integrators + supporting machinery
    utils/                    # Backend / device / precision / logging setup
  tests/                      # pytest suite (run from inside tests/)
  benchmarking/               # Cross-backend benchmark harness
  docs/                       # Sphinx (Read the Docs) — RTD theme
  paper/                      # JOSS 2021 paper sources
```

### Public API (`torchquad/__init__.py`)

| Symbol            | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `Trapezoid`, `Simpson`, `Boole` | Newton–Cotes deterministic rules                            |
| `GaussLegendre`, `Gaussian`     | Gaussian quadrature                                          |
| `MonteCarlo`                    | Plain MC                                                     |
| `VEGAS`                         | Adaptive importance sampling                                 |
| `IntegrationGrid`               | Tensor-product grid generator (used by Newton–Cotes)         |
| `BaseIntegrator`, `GridIntegrator` | Subclassing entry points for new methods                  |
| `RNG`                           | Backend-agnostic RNG wrapper                                 |
| `set_up_backend`, `enable_cuda`, `set_precision`, `set_log_level` | Setup helpers |

### Integrator hierarchy

```
BaseIntegrator                 # integration/base_integrator.py
├── GridIntegrator             # integration/grid_integrator.py
│   ├── NewtonCotes            # integration/newton_cotes.py
│   │   ├── Trapezoid          # integration/trapezoid.py
│   │   ├── Simpson            # integration/simpson.py
│   │   └── Boole              # integration/boole.py
│   └── GaussLegendre          # integration/gaussian.py (and Gaussian base)
├── MonteCarlo                 # integration/monte_carlo.py
└── VEGAS                      # integration/vegas.py (+ vegas_map.py, vegas_stratification.py)
```

`BaseIntegrator.evaluate_integrand` is the single point where the integrand is invoked — it enforces backend consistency, broadcasting weights, and shape checks. Anything that calls `fn(points, *args)` directly is a bug.

### Backend abstraction

- `torchquad/utils/set_up_backend.py` is the canonical entry point. It dispatches to `enable_cuda` (PyTorch only) and `set_precision` per backend.
- All array math goes through `from autoray import numpy as anp`. Never import the backend module directly inside `integration/`.
- Backend is inferred from input tensors via `autoray.infer_backend(x)` — passing in a torch tensor selects torch math even if `set_up_backend` was called with another name.

### Logging

- Library uses `loguru.logger` directly. `__init__.py` calls `logger.disable("torchquad")` at import, so the library is silent in every build.
- See issue #184: do not call `logger.add(...)` or `logger.remove(...)` from library code. Configuration is the application's job.
- `TORCHQUAD_LOG_LEVEL` is honored when it is set in the environment at import time; otherwise logging stays off until `set_log_level()` is called.

### Testing

Tests under `tests/` import torchquad via `sys.path.append("../torchquad")`, so they must be run from `tests/`. Each integrator has its own test file (`boole_test.py`, `simpson_test.py`, etc.). Cross-backend coverage is achieved by parameterizing the same test bodies via `helper_functions.py::setup_test_for_backend`.

Common analytic integrands live in `integration_test_functions.py`.

## Documentation

User-facing Sphinx docs live in `docs/source/`, deployed to Read the Docs from `main`. Tutorial / API reference are written by hand. Build env: `rtd_environment.yml`.

## CLAUDE.md Maintenance

Update this file when:
- The integrator hierarchy or backend abstraction changes
- A coding rule is added or relaxed
- The branching, lint, or test commands change

During sessions, use the `#` key to quickly capture notes for later incorporation here.

Keep it under ~200 lines of content.

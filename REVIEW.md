# REVIEW.md

Repo-specific review guidance for torchquad. These rules have the highest
priority — they reflect what actually goes wrong in this repo and what torchquad
promises its users. `CLAUDE.md` is binding and may be cited directly in reviews
("backend-specific calls belong only in `utils/`, per CLAUDE.md rule 3"): a change
that makes a `CLAUDE.md` statement outdated is itself a finding.

torchquad is cited in published research that reports numerical results, runs on
four backends (PyTorch, JAX, TensorFlow, NumPy) from one API, and is fully
differentiable on the framework backends. That shapes the priority order below:
**correctness outranks everything**, backend-agnosticism and vectorization are
non-negotiable structural rules, and only then the general fail-hard / dead-code /
simplicity concerns. Order is by how much damage the issue does here, not by how
often it is typed.

## 1. Correctness first — a silent precision regression is the worst outcome

This is the headline feature. A crash is recoverable; a wrong number in a cited
paper is not. Scrutinise anything touching a numerical kernel
(`integration/`, weights, grids, RNG, VEGAS map/stratification).

- **Compare against analytic ground truth.** New or changed integrators, samplers,
  and error estimates must be checked against the closed-form integrals in
  `tests/integration_test_functions.py`, not just "runs without error" — and
  against the **whole** collection via `helper_functions.py::compute_integration_test_errors`
  (real + complex, 1-D/3-D/10-D), not one or two ad-hoc integrands. A *reported
  error* (e.g. VEGAS's `sdev`/`chi2`) must itself be validated against the true
  error from these functions, not merely bound-checked. A bug fix needs a
  regression test that would fail on the old code.
- **A loosened tolerance is a finding until justified.** If an assertion's
  tolerance was widened to make a test pass, the PR must explain *why* the extra
  error is real hardware/reduction-order non-associativity and not a genuine
  regression (issue #178/#179 is the cautionary tale — tolerances bumped to hide a
  GPU precision gap). Prefer testing method correctness in fp64; document any
  per-(backend, dtype, device) tolerance with its rationale.
- **Determinism must survive.** VEGAS, MonteCarlo, and any RNG-using path must
  reproduce bit-for-bit given the same `RNG`. Flag any change that reorders RNG
  draws, reseeds differently, or makes results seed-order-dependent without a
  deterministic-snapshot test.
- **Differentiability is load-bearing.** torchquad's moat is autodiff through the
  integral. Flag a `.detach()`, `.item()`, `float(...)`, `int(...)`,
  in-place write, or NumPy round-trip on any tensor on the gradient path — it
  silently severs the graph. VEGAS map's fp64-cumsum branch is the known risk
  area. If the value flows to a gradient, a gradient test must cover it.
- **Cross-backend agreement is a property, not a hope.** The same
  (integrator, integrand, domain) should agree across backends to fp-ULP × a small
  constant. Dtype-promotion, `anp.meshgrid`-returns-a-list (#214), and
  integer-domain-produces-zero (#180) bugs hide here.

## 2. Backend-agnostic — no raw framework calls outside `utils/`

CLAUDE.md rule 3: code under `integration/` and `utils/` (except the three setup
modules) must go through `from autoray import numpy as anp` and
`infer_backend(...)`. This is the single most torchquad-specific structural rule.

- Flag every raw `torch.*`, `jnp.*` / `jax.*`, `tf.*`, or bare `np.*` in
  `integration/`. Backend-specific paths belong only in
  `utils/enable_cuda.py`, `utils/set_precision.py`, `utils/set_up_backend.py`.
- Backend must be **inferred from the input tensors** via `infer_backend(x)`, not
  assumed from a global or from `set_up_backend`'s last call. A torch tensor in
  must select torch math even if the default backend is something else.
- Inline `if infer_backend(x) == "torch": ...` special-cases scattered through a
  kernel are a smell — prefer a single named helper over duplicating the branch.
  A hard `ValueError` rejecting JAX/TF (as VEGAS does) should read from a single
  supported-backends source of truth, not be duplicated.
- New autoray `register`/`repeat`/`expand_dims` shims belong in a private compat
  module, not in `__init__.py`.

## 3. Vectorize or fail — no Python loops over sample points

CLAUDE.md rule 4. Integrand evaluation is batched along the first dimension.

- Flag any Python-level `for`/`while` loop over sample points, grid points, or
  strata that could be a batched tensor op. "It's clearer as a loop" does not win
  here — on a GPU it is a correctness-of-performance red flag worth discussing
  before merge.
- All integrand invocation goes through
  `BaseIntegrator.evaluate_integrand` — it enforces backend consistency, weight
  broadcasting, and shape checks. Any code calling `fn(points, *args)` directly is
  a bug; flag it.
- A genuinely un-vectorizable step is a design discussion, not a quiet loop.

## 4. Fail hard — no silent fallbacks

CLAUDE.md rule 5. For every defaulted or `None`-tolerant value ask: *should this
ever be absent? Should it raise instead?*

- No `dict.get(key, default)` or `getattr(obj, ..., default)` for config /
  required inputs — direct `dict["key"]` / attribute access; a missing value is a
  bug that should surface as `KeyError`/`AttributeError`.
- No silent precision downgrade and no silent backend/dtype coercion — raise on a
  dtype/backend mismatch rather than casting quietly (the integer-domain → 0 bug).
- No bare `except:` / `except Exception: pass`. Broken invariants raise;
  recoverable issues use `logger.warning()`. A swallowed exception on a numerical
  path is a top finding.

## 5. Logging hygiene — the library must not reconfigure loguru

CLAUDE.md rule 6, issue #184. The library pollutes downstream apps if it touches
global loguru state at import.

- Flag any `logger.add(...)`, `logger.remove()`, or loguru reconfiguration in
  library code (import-time or inside `set_log_level`). The correct lever is
  `logger.disable("torchquad")` / `enable`. Use `from loguru import logger`
  directly; do not construct new sinks.
- Logging level comes from `TORCHQUAD_LOG_LEVEL` via `set_log_level()` only.

## 6. Dead code — delete rather than whitelist

CLAUDE.md rule 13; vulture runs in CI. The recurring
probe is *"Can this ever happen? Is this branch reachable? Do we still need this?"*

- Flag functions never called, values assigned but never read, stubs that fulfil
  no base-class contract (the dead `RNG.uniform` stub is the archetype), and
  guards for states that cannot occur.
- Flag duplicated definitions — the same logic in two places is both a DRY and a
  correctness risk (e.g. `grid_func` duplicated between `integration_grid.py` and
  `grid_integrator.py`; the library-side `sys.path.append` in
  `integration/utils.py`). Require one definition others import.
- Delete rather than comment out or whitelist — git history is the archive. A
  symbol kept for API reasons needs a docstring saying why.

## 7. Deprecations & public API

CLAUDE.md rules 8, 10, 13.

- A changed public signature must keep the old path working for one release with a
  `warnings.warn(..., DeprecationWarning)`, **and** the warning must actually be
  visible — grep `pyproject.toml`'s `filterwarnings` so it isn't suppressed. A
  suppressed self-deprecation warning that never resolves is exactly the anti-pattern
  being removed in 0.6.
- A name in `__all__` must not start with an underscore — the two contradict each
  other. `_deployment_test` was the archetype (underscore *and* in `__all__`); the
  fix was to drop it from `__all__` and keep it private (it is an internal
  self-test, not user-facing), not to make it public. No abbreviations in new
  code: `integration_domain` not `int_dom`, `function_values` not `fvals`.

## 8. Simplicity, size budgets, dataclasses

CLAUDE.md rule 12.

- Guard clauses over deep nesting; keep bodies a couple of levels deep. Functions
  under ~100 lines, files under ~1000 — flag a PR that pushes past these instead of
  splitting.
- Prefer small functions and typed dataclasses over loose dicts threaded between
  methods (the VEGAS result object is the canonical example — a dataclass with
  value/sdev/chi²/dof, not a bare float or a dict). Put shared dataclasses in a
  dependency-free module to avoid circular imports (`deployment_test`'s circular
  import back into the initializing package is the warning here).

## 9. Docs, docstrings, tests

CLAUDE.md rules 7, 9.

- Google-style docstrings (`Args`, `Returns`, `Raises`) on public functions and
  classes. An integrator with an N constraint (Boole, Simpson's odd-N) or a
  backend restriction (VEGAS: no JAX/TF) must state it in a `Raises:` / `Note:`
  section — pydoclint is being added to gate this.
- Keep `README.md`, `docs/source/`, and CLAUDE.md in sync with code. Flag claims
  that overstate reality (stale benchmark numbers, "first GPU VEGAS", ROCm
  "investigating", deprecated `pytorch` conda channel). Comments explain **why**,
  not what — a comment paraphrasing the next line is noise, delete it.
- Every change needs a test: bug fixes get a regression test; new integrators get
  both an analytic-correctness test and a cross-backend coverage test
  (parametrized via `helper_functions.py`). Tests run from `tests/`.

## 10. Keep PRs focused

- Drop unrelated diffs (notebook/log/`egg-info` churn, formatting-only noise mixed
  into a logic change). Defer broader cleanups to their own issue/PR rather than
  widening the current one. Run `ruff check` + `ruff format` (line length 100)
  before review; CI gates on both, so a formatting-only round-trip in the diff is
  itself a finding.

# Changelog

All notable changes to torchquad are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The tutorial's vectorized-integrand example crashed with `RuntimeError: Expected
  all tensors to be on the same device` on any CUDA machine. It built its helper
  tensors with the legacy `torch.Tensor(...)` constructor, which — unlike
  `torch.tensor(...)` — ignores the `torch.set_default_device("cuda")` that
  `set_up_backend` performs, so the helpers stayed on the CPU while the sample
  points were on the GPU. A note now explains the distinction, since it bites in
  user code as readily as in the docs.
- The tutorial's import block used `matplotlib`, which is not a torchquad runtime
  dependency, so it failed after the installation the README documents. Called out
  as a prerequisite rather than added as a dependency.
- The README logo used a repository-relative path, so it did not render on the
  PyPI project page. It is now an absolute URL, matching the performance figures.

## [0.6.0] - 2026-08-23

The 0.6 line is a modernization and credibility release: modern tooling, honest
packaging, and closing long-open fixed issues.

**No changes to numerical results.** Every integrator returns what it returned
under 0.5.0 for the same function, method, `N` and seed, and no existing test
tolerance was loosened. The additions below are new entry points (`Sobol`,
`return_error`, `args`); the removals were code that could not run on a
supported Python or PyTorch. The one behavioural change is that `GaussLegendre`
now raises on impossible node counts instead of exhausting memory.

### Added
- `Sobol` quasi-Monte Carlo sampler, usable via
  `MonteCarlo.integrate(..., rng=Sobol(...))` for faster convergence on smooth
  integrands (#140).
- `VEGAS.integrate(..., return_error=True)` returns a `VEGASResult` bundling the
  integral with its error estimate (standard deviation, chi-squared, degrees of
  freedom and goodness-of-fit Q) instead of discarding them.
- `args` argument on every integrator's `integrate()` — extra parameters are
  forwarded to the integrand as `fn(points, *args)`, so parametric integrands no
  longer need a lambda wrapper (#187, #188).
- Optional-dependency extras: `dev`, `docs`, and CPU-convenience backend extras
  `torch`, `jax`, `tensorflow`, `all`.
- `release_testing/` suite — slower end-to-end checks run against the latest
  released backends (not the pinned CI env), grounded in real-world usage. It
  runs automatically on any push to a `release-*` / `release/**` branch and on
  any PR into `main`, so it gates a release before it ships. `workflow_dispatch`
  alone could not: GitHub registers a workflow only once it exists on the default
  branch or has already run, so a suite added on `develop` is undispatchable
  until it reaches `main`, leaving `release: created` — after shipping — as its
  first possible run.
- `uv` as the primary dev/CI toolchain, with `uv.lock` for reproducible envs.
- Experimental `pixi.toml` with per-backend environments.
- CI quality gates: a Python 3.10–3.12 backend matrix, isolated JAX/TensorFlow
  jobs, a wheel-build smoke test, a `sphinx-build -W` docs gate, and an enforced
  coverage floor.
- `pre-commit` + `pydoclint` docstring checks and a two-tier `vulture`
  dead-code job.
- `.github/RELEASE_NOTES_TEMPLATE.md`, a release-notes skeleton with a
  contributor-thanks section, referenced from the release checklist.
- New README performance figures: convergence (error vs N, all methods,
  float64), quasi-Monte Carlo vs Monte Carlo, error vs dimension at a fixed
  budget, CPU-vs-GPU runtime, and a like-for-like comparison against SciPy.
  The SciPy comparison covers Boole and VEGAS alongside Gauss-Legendre and
  Sobol, which matters: Boole beats Gauss-Legendre by six orders of magnitude
  at d=3 on this integrand, because a composite low-order rule handles a kink
  far better than a single global high-order one, and VEGAS goes from worst
  method at d=3 to best at d=10.
- `benchmarking/genz_functions.py` gains a `combined` integrand: a sum of
  normalised Genz functions that oscillates, peaks in a corner, and is not
  differentiable, so no single feature can flatter one method. Summing keeps the
  integral exact, since integration is linear.
- Dependabot for GitHub Actions.
- This `CHANGELOG.md`.

### Changed
- Switched linting/formatting from flake8 + black to `ruff`.
- Raised the Python floor to `>=3.10` and the PyTorch floor to `>=2.1`.
- Modernized all GitHub Actions to current majors; CI installs CPU wheels via
  `uv` instead of conda/micromamba.
- Install docs lead with pip/uv; dropped the deprecated `pytorch` conda channel;
  rebuilt `environment_all_backends.yml` on conda-forge.
- The README performance section is rebuilt around the new figures, all
  regenerated on one machine after the GPU timing fix. The claim of a broad
  efficiency win over SciPy is replaced by a measured comparison against SciPy's
  strongest configuration on a hard integrand, with both sides given the same
  50-million-evaluation budget. SciPy's algorithms are more efficient per
  evaluation at low dimension — at d=3 `nquad` reaches 4.4e-16 from 250k points
  where torchquad needs 38 million to reach 8.3e-12 — while torchquad's GPU
  throughput offsets that in wall-clock, and dimension decides it past d=6, where
  no SciPy configuration completes at d=10.
- `loguru` is disabled by default; `set_log_level` manages a single tracked sink
  instead of touching host-application handlers (#184).
- `TORCHQUAD_LOG_LEVEL` now works. Setting it to a non-empty value enables
  torchquad's logging at that level when the package is imported; leaving it
  unset or empty keeps the library silent, and an unrecognised level raises with
  a message naming the valid ones. Previously the variable was only read when
  the `TORCHQUAD_DISABLE_LOGGING` constant was `False`, which no release ever
  shipped, so it had no effect. Note that the level governs the handler
  torchquad adds; once records are enabled they also reach any other sink loguru
  has registered, including its unfiltered default. This is now documented in
  the README and in `set_log_level`.
- Rewrote the release checklist (`.github/ISSUE_TEMPLATE/release.md`): it now
  lists all six places the version lives, adds the missing tagging step, defers
  to CI instead of asking for manual re-runs, and covers conda-forge and
  Read the Docs. Review of the release PRs now comes *before* the Test PyPI
  upload rather than after it: a version number uploaded to an index can never
  be reused, so anything found in review after publishing costs a version.
- Raised the SciPy floor to `>=1.7.2` and declared `numpy` explicitly. The
  `Sobol` sampler uses `scipy.stats.qmc` on the non-torch backends, and that
  module only exists from SciPy 1.7.0 onward; 1.7.0 and 1.7.1 are themselves
  capped at Python <3.10, so 1.7.2 is the oldest release that is both new
  enough and installable on a supported Python. `numpy` is imported directly by
  `integration/gaussian.py` and previously arrived only as a SciPy transitive.
  Its floor is `>=1.21.3` on the same reasoning: 1.21.3 is the oldest release
  shipping cp310 wheels. The conda and pixi manifests, which had drifted to
  `scipy>=1.7.0`, now carry the same floors — pyproject remains the single
  source of truth. The whole stack is verified working at these floors on
  Python 3.10.
- License metadata moved to a PEP 639 SPDX expression, `GPL-3.0-only`, replacing
  the deprecated `license = { text = ... }` table and the `License ::` trove
  classifier. `-only` rather than `-or-later` because the original `setup.py`
  declared the `GPLv3` classifier, not `GPLv3+` — this records what torchquad has
  always shipped under and does not change the licence itself. Builds are now
  clean: the three `SetuptoolsDeprecationWarning`s are gone, and the published
  metadata is version 2.4 with a `License-Expression` field. Building from the
  sdist now needs `setuptools>=77.0.3`, and uploading needs `twine>=6.1.0`.

### Fixed
- A CI bug where `pytest | tee` masked a non-zero exit code, hiding failing
  suites; fixed with `set -o pipefail`, which surfaced (and fixed) an unseeded
  JIT Monte Carlo test.
- Documentation build warnings (`imgmath` → `mathjax`, removed an unsupported
  theme option, repaired a malformed tutorial code block).
- `GaussLegendre` (and any `Gaussian` subclass) now raises a clear `ValueError`
  when asked for more than 10 000 nodes per dimension, instead of reaching NumPy
  and failing with a bare `MemoryError: Unable to allocate 7450.6 GiB`. The nodes
  are eigenvalues of an `n x n` matrix, so the cost is quadratic in memory and
  cubic in time. The message states the node count, the limit, the memory implied,
  and that `N` is divided across the dimensions. Note this counts nodes *per
  dimension*: `dim=3, N=10**6` is 100 per axis and is unaffected.
- GPU timings in the `benchmarking/` harness. Nothing synchronized the device,
  so every clock stopped once the kernels were queued rather than once they had
  run: MonteCarlo at N=1e8 measured 1.35 ms where the true cost is 13.6 ms.
  The vectorized benchmark compared a loop that materialized every result inside
  the timed region against a batched call that materialized none, so its ratio
  was partly a synchronization artifact — at grid size 1, where the true speedup
  is about 1x, it reported 35x. Timed regions now materialize their result, both
  sides of that comparison are timed identically, and the vectorized benchmark
  discards a warm-up run like the others. Corrected, the vectorization speedup is
  close to linear in the number of integrands (0.81x at 1, 19.7x at 20, 186.8x at
  200), rather than the "exponential" growth the README claimed.
- A fabricated ground truth in the `benchmarking/` harness. When a reference
  value could not be computed it returned `1.0`, so every error on the plot was
  then measured against a made-up number — the same class of defect as the
  timing bug, but invisible in the output, since nothing looks unusual. It now
  raises and names the dimension that failed.

### Removed
- The library-side `sys.path.append` import hack.
- The legacy `set_default_tensor_type` branch in `set_precision`.
- The 3-year-old `(N,) → (N,1)` return-shape deprecation warning.
- The `TORCHQUAD_DISABLE_LOGGING` constant, a build-time switch that could only
  be flipped by editing `__init__.py` and was therefore always `True`. Use
  `TORCHQUAD_LOG_LEVEL` or `set_log_level()` instead.
- The `TORCHQUAD_RELEASE_BUILD` environment variable from the deploy workflows,
  which nothing read, and the `release: created` trigger on the Test PyPI
  workflow, which re-uploaded an already-published version and failed every time.
- Dead code (`RNG.uniform`, `Gaussian.name`).
- `matplotlib` and `tqdm` as runtime dependencies. Neither is imported by the
  shipped package: `matplotlib` is only used by the benchmarking harness (it
  moved to the `dev` extra) and `tqdm` was not used anywhere. Installing
  torchquad no longer pulls them in.
- `requirements.txt`, a vestigial second copy of the runtime dependencies that
  had already drifted from `pyproject.toml`. `pyproject.toml` is the single
  source of truth.

## [0.5.0] - 2025-08-03
### Changed
- Migrated packaging from `setup.py` to `pyproject.toml` (#223).
- Restructured the test suite to a root-level `tests/` directory (#218).
- Standardized formatting with Black at 100-character line length.
### Fixed
- Improved GPU device selection and `set_precision` backend/CUDA handling (#222).
### Added
- Parametric-integration and GPU-usage documentation, plus CI/CD docs (#219).

## [0.4.1] - 2024-11-25
### Fixed
- Compatibility with newer JAX (now requires `jax>=0.4.17`).
- TensorFlow global-precision handling and related bugfixes.
- Various CI and documentation fixes.

## [0.4.0] - 2023-06-14
### Added
- Vectorized multi-integrand evaluation in a single call.
- Gauss-Legendre integration and better support for custom integrators.
### Changed
- Test, docs, and workflow improvements; automatic coverage reporting.

## [0.3.0] - 2022-05-05
### Added
- NumPy, JAX, and TensorFlow support via autoray for most integrators.
- (JIT) compilation of integration (except VEGAS); custom `RNG` class.
- `TORCHQUAD_LOG_LEVEL` environment variable.
### Changed
- Large VEGAS performance improvements; Newton-Cotes refactor; stricter linting.

## [0.2.4] - 2021-08-31
### Added
- JOSS publication release (Zenodo archive).

## [0.2.3] - 2021-08-20
- Early public releases with the core Newton-Cotes, Monte Carlo, and VEGAS
  integrators on PyTorch.

[Unreleased]: https://github.com/esa/torchquad/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/esa/torchquad/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/esa/torchquad/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/esa/torchquad/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/esa/torchquad/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/esa/torchquad/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/esa/torchquad/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/esa/torchquad/releases/tag/v0.2.3

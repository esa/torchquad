# Changelog

All notable changes to torchquad are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The 0.6 line is a modernization and credibility release: modern tooling, honest
packaging, and closing long-open fixed issues.

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
  released backends (not the pinned CI env), grounded in real-world usage.
- `uv` as the primary dev/CI toolchain, with `uv.lock` for reproducible envs.
- Experimental `pixi.toml` with per-backend environments.
- CI quality gates: a Python 3.10–3.12 backend matrix, isolated JAX/TensorFlow
  jobs, a wheel-build smoke test, a `sphinx-build -W` docs gate, and an enforced
  coverage floor.
- `pre-commit` + `pydoclint` docstring checks and a two-tier `vulture`
  dead-code job.
- Dependabot for GitHub Actions.
- This `CHANGELOG.md`.

### Changed
- Switched linting/formatting from flake8 + black to `ruff`.
- Raised the Python floor to `>=3.10` and the PyTorch floor to `>=2.1`.
- Modernized all GitHub Actions to current majors; CI installs CPU wheels via
  `uv` instead of conda/micromamba.
- Install docs lead with pip/uv; dropped the deprecated `pytorch` conda channel;
  rebuilt `environment_all_backends.yml` on conda-forge.
- `loguru` is disabled by default; `set_log_level` manages a single tracked sink
  instead of touching host-application handlers (#184).
- Raised the SciPy floor to `>=1.7.2` and declared `numpy` explicitly. The
  `Sobol` sampler uses `scipy.stats.qmc` on the non-torch backends, and that
  module only exists from SciPy 1.7.0 onward; 1.7.0 and 1.7.1 are themselves
  capped at Python <3.10, so 1.7.2 is the oldest release that is both new
  enough and installable on a supported Python. `numpy` is imported directly by
  `integration/gaussian.py` and previously arrived only as a SciPy transitive.

### Fixed
- A CI bug where `pytest | tee` masked a non-zero exit code, hiding failing
  suites; fixed with `set -o pipefail`, which surfaced (and fixed) an unseeded
  JIT Monte Carlo test.
- Documentation build warnings (`imgmath` → `mathjax`, removed an unsupported
  theme option, repaired a malformed tutorial code block).
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
  The plots in the README predate this fix and still need regenerating.
- A fabricated ground truth in the `benchmarking/` harness. When a reference
  value could not be computed it returned `1.0`, so every error on the plot was
  then measured against a made-up number — the same class of defect as the
  timing bug, but invisible in the output, since nothing looks unusual. It now
  raises and names the dimension that failed.

### Removed
- The library-side `sys.path.append` import hack.
- The legacy `set_default_tensor_type` branch in `set_precision`.
- The 3-year-old `(N,) → (N,1)` return-shape deprecation warning.
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

[Unreleased]: https://github.com/esa/torchquad/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/esa/torchquad/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/esa/torchquad/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/esa/torchquad/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/esa/torchquad/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/esa/torchquad/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/esa/torchquad/releases/tag/v0.2.3

# Contributing to torchquad

Thanks for your interest in improving torchquad! The full guide lives in the
[contributing documentation](https://torchquad.readthedocs.io/en/main/contributing.html);
this file is the quick reference GitHub shows when you open a pull request.

## Branching model

torchquad uses a two-branch model:

- **`develop`** — the integration branch. **All feature and fix PRs target
  `develop`.**
- **`main`** — release branch. It only ever receives releases (via a
  `develop` → `main` merge) and the occasional documentation hotfix for the
  currently released version.

So, unless you are fixing docs for the *current release*, branch off `develop`
and open your PR against `develop`. Releases are cut by merging `develop` into
`main`, tagging, and publishing; `main` therefore stays a subset of `develop`.

## Development setup

Fork and clone the repo, then set up an editable install with all backends and
the dev tooling. Either uv or conda works:

```bash
# uv (fast, CPU wheels)
uv pip install -e ".[dev,all]"

# or conda (installs every backend, CPU)
conda env create -f environment_all_backends.yml
conda activate torchquad
pip install -e ".[dev]"
```

See the [installation docs](https://torchquad.readthedocs.io/en/main/install.html)
for GPU builds and per-backend options.

## Before you push

CI enforces formatting, linting, docstrings, dead-code, and the full test suite.
Run these locally first (see `CLAUDE.md` / the CI docs for the full list):

```bash
ruff format .
ruff check .
pydoclint torchquad/
cd tests/ && pytest
```

Or install the hooks once with `pre-commit install` so they run automatically.

## Review

PRs are reviewed against `REVIEW.md` (correctness-first, backend-agnostic,
vectorize-or-fail, fail-hard). Skimming it before you open a PR helps your
change land faster.

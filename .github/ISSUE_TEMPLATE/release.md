---
name: New Release
about: Creating a new release version for torchquad. Only for Maintainers.
title: "Release "
labels: "release"
assignees: "gomezzz"
---

# Release X.Y.Z

Draft the release notes from [`.github/RELEASE_NOTES_TEMPLATE.md`](https://github.com/esa/torchquad/blob/develop/.github/RELEASE_NOTES_TEMPLATE.md)
as you go — the "thanks" section is much easier to fill in while the changes are
still fresh than after the tag is pushed.

## What Needs to Be Done (chronologically)

### 1. Prepare

- [ ] Verify `main` holds nothing `develop` lacks:
      `git fetch origin && git rev-list --count origin/develop..origin/main` must print `0`.
      If it does not, PR `main` → `develop` first and get it reviewed and merged. Let GitHub
      create a branch for any conflict fixes; never commit to `main` directly.
- [ ] Confirm every PR in scope is merged into `develop` and the milestone is empty.
- [ ] Cut the release branch off `develop`:
      `git switch develop && git pull && git switch -c release/X.Y.Z`

### 2. Version bump (all six locations)

- [ ] `pyproject.toml` — `version = "X.Y.Z"`
- [ ] `pyproject.toml` — `Development Status` classifier
      (`3 - Alpha` → `4 - Beta` for 0.6.x; `5 - Production/Stable` only at 1.0)
- [ ] `torchquad/__init__.py` — `__version__ = "X.Y.Z"`
- [ ] `docs/source/conf.py` — `release = "X.Y.Z"`
- [ ] `uv.lock` — regenerate with `uv lock`, do not hand-edit. The root package entry pins
      the project version, and a stale lock makes `uv sync --locked` fail. Commit the result.
- [ ] `CHANGELOG.md` — rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh
      empty `## [Unreleased]`, and update the link block at the bottom of the file:
      `[Unreleased]: https://github.com/esa/torchquad/compare/vX.Y.Z...HEAD` and
      `[X.Y.Z]: https://github.com/esa/torchquad/compare/v<previous>...vX.Y.Z`
- [ ] Sanity check that nothing was missed:
      `git grep -n "<previous version>"` should return only unrelated dependency floors.

### 3. Dependency + metadata consistency

- [ ] `[project.dependencies]` in `pyproject.toml` is the single source of truth. Confirm
      `environment.yml`, `environment_all_backends.yml`, `rtd_environment.yml` and
      `pixi.toml` do not contradict it.
- [ ] Every dependency floor is high enough for the features actually being shipped, not
      just high enough to import the package.
- [ ] `requires-python` matches the tested matrix (`>=3.10, <4` against CI 3.10–3.12) and
      the `Programming Language :: Python ::` classifiers list exactly those versions.
- [ ] Build locally and check the output for packaging warnings: `uv build`
      (or `python -m build`). One known set of `SetuptoolsDeprecationWarning`s is
      outstanding — the pre-PEP-639 `license = { text = ... }` table and the
      `License ::` classifier. Fixing it means committing to an SPDX identifier
      (`GPL-3.0-only` vs `GPL-3.0-or-later`), which is bundled with the deferred
      licensing decision. Anything *else* that warns is new and should be fixed.

### 4. Verify (CI is the gate — do not re-run green jobs by hand)

- [ ] `Running tests` workflow green on the release branch: `lint`,
      `test (all backends, py3.10|3.11|3.12)`, `test (jax isolated)`,
      `test (tensorflow isolated)`, `wheel-smoke`, `docs-build`.
- [ ] `dead_code` workflow green (the `vulture` job, 100%-confidence tier).
- [ ] Trigger [Release testing](https://github.com/esa/torchquad/actions/workflows/release_testing.yml)
      on the release branch — this is the run against the *latest released* torch/JAX/
      TensorFlow rather than the pinned CI versions. See
      [`release_testing/README.md`](https://github.com/esa/torchquad/blob/develop/release_testing/README.md).
- [ ] **GPU check — not covered by any CI.** In a CUDA runtime (e.g.
      [Colab](https://colab.research.google.com/drive/1lFpdtY5zV7VpW88aazedA3n4khedHDQP?usp=sharing)):
      ```bash
      pip install -e ".[dev,all]"
      cd tests/ && pytest -ra
      cd .. && pytest release_testing/ -v -ra
      ```
      Paste the summary into this issue.
- [ ] Documentation content review: every entry in the new changelog section has
      user-facing documentation (`docs/source/tutorial.rst`,
      `docs/source/integration_methods.rst` or API autodoc), and the Read the Docs build
      of the release branch renders correctly.
- [ ] Any changed numerical behaviour or loosened tolerance is called out explicitly in the
      changelog, not just in the diff.

### 5. Test PyPI

- [ ] Run [Upload Python Package to Test PyPI](https://github.com/esa/torchquad/actions/workflows/deploy_to_test_pypi.yml)
      and **select the release branch** in the "Run workflow" dropdown — it defaults to the
      repository's default branch.
- [ ] Install and smoke-test the TestPyPI artifact in a clean environment:
      ```bash
      uv venv --python 3.12 /tmp/tq && source /tmp/tq/bin/activate
      uv pip install --index-url https://test.pypi.org/simple/ \
                     --extra-index-url https://pypi.org/simple torchquad
      python -c "import torchquad; print(torchquad.__version__); torchquad._deployment_test()"
      ```
      (`pip install` with the same two index URLs works identically.)
- [ ] If this release adds or changes any non-Python package data, import something that
      reads that data here — `wheel-smoke` and `_deployment_test()` do not touch it, so a
      missing `[tool.setuptools.package-data]` entry is invisible until a user hits it.

### 6. Ship

- [ ] Finalize the release branch, then open PRs `release/X.Y.Z` → `main` **and**
      `release/X.Y.Z` → `develop`.
- [ ] Review both PRs against [`REVIEW.md`](https://github.com/esa/torchquad/blob/develop/REVIEW.md).
      Merge, but do not delete the branch yet.
- [ ] Tag the merge commit on `main`:
      `git tag -a vX.Y.Z -m "torchquad vX.Y.Z" && git push origin vX.Y.Z`.
      The `vX.Y.Z` form is load-bearing — the `CHANGELOG.md` compare links depend on it.
- [ ] Create the GitHub Release from tag `vX.Y.Z`, using the notes drafted from
      [`.github/RELEASE_NOTES_TEMPLATE.md`](https://github.com/esa/torchquad/blob/develop/.github/RELEASE_NOTES_TEMPLATE.md).
      Creating the release also auto-triggers `release_testing.yml`, which is expected.
- [ ] Run [Upload Python Package to PyPI](https://github.com/esa/torchquad/actions/workflows/deploy_to_pypi.yml)
      and **select `main`**.
- [ ] Verify the published artifact from a clean environment:
      ```bash
      uv run --with torchquad --no-project python -c \
        "import torchquad; print(torchquad.__version__); torchquad._deployment_test()"
      ```

### 7. conda-forge

- [ ] Wait for the regro-cf-autotick-bot PR on
      [`conda-forge/torchquad-feedstock`](https://github.com/conda-forge/torchquad-feedstock)
      (usually within a day of the PyPI upload), or bump `version` and `sha256` in
      `recipe/meta.yaml` by hand. See
      https://conda-forge.org/docs/maintainer/updating_pkgs.html
- [ ] Reconcile the feedstock's `host:`/`run:` requirements and its Python floor with
      `[project]` in `pyproject.toml`, then merge.
- [ ] Confirm `conda install torchquad -c conda-forge` resolves the new version.

### 8. Wrap up

- [ ] Confirm Read the Docs built the new tag and that the version selector shows it.
- [ ] Close the milestone and every issue this release fixes, linking the release notes.
- [ ] Thank the contributors, issue reporters and reviewers in the release notes (see the
      template) — this is the step that is easiest to skip and most worth doing.
- [ ] Delete the release branch.
- [ ] Update `claude_docs/roadmap.md` "Current status" for the next cycle.

---
name: New Release
about: Creating a new release version for torchquad. Only for Maintainers.
title: "Release "
labels: "release"
assignees: "gomezzz"
---

# Feature

## Changelog

_to be written during release process_

## What Needs to Be Done (chronologically)

- [ ] Create PR from `main` -> `develop` to incorporate hotfixes / documentation changes.
- [ ] In case of merge conflicts create PR to fix them which is then merged into main / fix on GitHub but make sure to let it create a new branch for the changes.
- [ ] Review the PR
- [ ] Create PR to merge from current develop into release branch
- [ ] Write Changelog in PR and request review
- [ ] Review the PR (if OK - merge, but DO NOT delete the branch)
- [ ] Minimize packages in requirements.txt and conda-forge submission. Update packages in pyproject.toml
- [ ] Check unit tests -> Check all tests pass on CPU and [GPU (e.g. on colab)](https://colab.research.google.com/drive/1lFpdtY5zV7VpW88aazedA3n4khedHDQP?usp=sharing#scrollTo=IbU2vypPQ-Ej) and that there are tests for all important features
- [ ] Check documentation -> Check presence of documentation for all features by locally building the docs on the release
- [ ] Change version number in pyproject.toml and docs (under conf.py)
- [ ] Trigger the Upload Python Package to testpypi GitHub Action (https://github.com/esa/torchquad/actions/workflows/deploy_to_test_pypi.yml) on the release branch (need to be logged in)
- [ ] Test the build on testpypi (with `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple torchquad`)
- [ ] Finalize release on the release branch
- [ ] Create PR: `release` → `main`, `release` -> `develop`
- [ ] PR Review
- [ ] Merge `release` back into `main`, and `develop`
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR) reviewed
- [ ] Upload to PyPI
- [ ] Update on conda following https://conda-forge.org/docs/maintainer/updating_pkgs.html

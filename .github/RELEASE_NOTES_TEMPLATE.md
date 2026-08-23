# Release notes template

Starting point for the body of a GitHub Release. Delete the sections that do not
apply — an empty heading is worse than no heading. The release checklist
(`.github/ISSUE_TEMPLATE/release.md`) points here.

Keep it readable by someone who has never seen the repository: say what changed
for *users*, not what changed in the diff.

---

## torchquad X.Y.Z

One or two sentences on the theme of the release. What can users do now that
they could not do before, or what should they trust more than they did before?

### Highlights

- **Feature name** — one line on what it does and why it matters, with a link
  to the tutorial or API docs section that covers it.
- **Second feature** — same.

Three to five entries at most. Everything else lives in the changelog.

### Numerical behaviour

Anything that changes results, tolerances, or reproducibility for existing code.
Say so plainly even when the change is an improvement — people cite this library
in papers and need to know whether their numbers move. Write "no changes to
numerical results" explicitly when that is the case; silence reads as an
oversight.

### Breaking changes / deprecations

What breaks, what to do instead, and which release the old path stops working in.
Omit the section entirely if there are none.

### Installation

```bash
pip install torchquad==X.Y.Z
uv add torchquad==X.Y.Z
conda install torchquad -c conda-forge
```

### Thanks

Thanks to everyone who contributed to this release.

### Full changelog

**Full changelog:** https://github.com/esa/torchquad/compare/v<previous>...vX.Y.Z

See [`CHANGELOG.md`](https://github.com/esa/torchquad/blob/main/CHANGELOG.md)
for the complete list.

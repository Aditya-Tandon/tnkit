# Building the docs

Not part of the published site (excluded in `conf.py`). The build is described in
`development.md`, section "Building the documentation". In short, from the repository root,
with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

`docs/api/public.rst`, `docs/api/blocks.rst`, `docs/api/symmetric.rst`, `docs/api/opsum.rst`,
`docs/api/thermal.rst` and `docs/api/generated/` are written by the build and are not
committed.

Read the Docs builds from `.readthedocs.yaml`. It builds pull requests only once "Build pull
requests" is enabled in the project's settings, and it can only build a public repository.

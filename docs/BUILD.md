# Building the docs

Not part of the published site (excluded in `conf.py`). From the repository root, with
Python 3.11:

```bash
pip install -e . -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`docs/requirements.txt` pins the build dependencies; the `docs` extra in `pyproject.toml`
gives the same packages with lower bounds only, for local use. `docs/api/public.rst`,
`docs/api/blocks.rst`, `docs/api/opsum.rst` and `docs/api/generated/` are written by the
build and are not committed.

Read the Docs builds from `.readthedocs.yaml`. It builds pull requests only once "Build pull
requests" is enabled in the project's settings, and it can only build a public repository.

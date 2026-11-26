# Development

This page is for working on tnkit itself: setting up the environment, running the tests and
building this documentation. To use tnkit, {doc}`getting-started/install` is enough.

## Environment

The development environment is managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins
every package, for Linux x86_64 and macOS on Apple silicon, and `.python-version` selects
Python 3.11. From a clone:

```bash
git clone https://github.com/Aditya-Tandon/tnkit.git
cd tnkit
uv sync --extra torch     # tnkit (editable), the test tools and torch
```

`uv sync` alone leaves torch out. The torch extras resolve differently under uv than under
pip:

| | `--extra torch` | `--extra cuda` |
|---|---|---|
| Linux | the CPU-only torch wheel | the CUDA 13.2 build from <https://download.pytorch.org/whl/cu132> |
| macOS | PyPI's wheel | PyPI's wheel |

`--extra cuda` needs an NVIDIA driver recent enough for CUDA 13, and the two extras cannot be
combined. Under pip, both extras install PyPI's wheel, which on Linux is a CUDA build (see
{doc}`getting-started/install`).

After changing the version or the dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`; CI fails if the lock is out of date.

## Tests

```bash
uv run pytest                 # everything
uv run pytest -m "not slow"   # without the tests that take minutes
```

Tests marked `cuda` need a CUDA device and are skipped without one. Without uv,
`pip install -e ".[test]"` and then `pytest` run the same tests.

## Building the documentation

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The `docs` dependency group in `pyproject.toml` pins the build dependencies and `uv.lock`
pins the rest; Read the Docs installs the same, and fails on any warning, as `-W` does here.
The API pages under `docs/api/` (except `index.rst`) are written by the build from the
package and are not committed. Code examples in the documentation are not run by the build.

## Contraction conventions

The source code and its comments use the vocabulary of the TAPP tensor-contraction interface
([arXiv:2601.07827](https://arxiv.org/abs/2601.07827)) for contractions:

- a tensor's indices are its **modes** (the user guide calls them legs), and the size of a mode
  is its **extent**;
- in a contraction, a mode that appears in both inputs and is summed over is
  **contracted**; one that appears in one input and in the output is **free**;
- a mode that appears in both inputs *and* the output is a **batch** (Hadamard) mode.

For example, extending the left environment $L$ by one site contracts the bond and physical
modes of $L$, $A$, $W$ and $\bar A$, and leaves three free modes:

$$L'_{b'\,w'\,b} = \sum_{a'\,w\,a\,s\,s'} L_{a'\,w\,a}\, \mps{\bar A}{s}{a'}{b'}\,
\mpo{W}{s}{s'}{w}{w'}\, \mps{A}{s'}{a}{b} .$$

Environments are stored in the order (bra, MPO, ket). tnkit evaluates this as three pairwise
contractions, each dispatched through the backend's `einsum`; it is an MPS library built on
a contraction layer (numpy, PyTorch, cuBLAS and cuSOLVER), and does not implement one.

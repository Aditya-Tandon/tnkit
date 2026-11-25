# Install

tnkit needs Python 3.11 or later. The core depends only on numpy and scipy.

```bash
pip install tnkit
```

For development, install from a clone instead:

```bash
git clone https://github.com/Aditya-Tandon/tnkit.git
cd tnkit
pip install -e .
```

Add PyTorch for the `torch-cpu`, `torch-cuda` and `torch-mps` backends:

::::{tab-set}

:::{tab-item} CPU or Apple silicon
```bash
pip install -e ".[torch]"
```
:::

:::{tab-item} NVIDIA GPU
On Linux, PyPI's torch wheel is a CUDA build, so the default index is enough:
```bash
pip install -e ".[cuda]"
```
On Windows, or to choose a CUDA version or a CPU-only build, install torch first with the
selector at <https://pytorch.org>, then `pip install -e .`.
:::

::::

`import tnkit` does not import PyTorch, and nothing needs it unless you choose a torch
backend.

To run the test suite, install the `test` extra and run `pytest tests/`.

## Developing with uv

The repository's development environment is managed with [uv](https://docs.astral.sh/uv/):
`uv.lock` pins every package, for Linux x86_64 and macOS on Apple silicon, and
`.python-version` selects Python 3.11. From a clone:

```bash
uv sync --extra torch     # tnkit (editable), the test tools and torch
uv run pytest
```

`uv sync` alone leaves torch out. On Linux, `--extra torch` installs the CPU-only torch wheel.
For an NVIDIA GPU, `--extra cuda` installs instead the CUDA 13.2 build from
<https://download.pytorch.org/whl/cu132>, which needs an NVIDIA driver recent enough for
CUDA 13; the two extras cannot be combined. On macOS both install PyPI's wheel. `uv sync --group docs` adds the
documentation build (see `docs/BUILD.md`).

After changing the version or the dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`; CI fails if the lock is out of date.

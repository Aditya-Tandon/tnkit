# Install

tnkit needs Python 3.11 or later. The core depends only on numpy and scipy.

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

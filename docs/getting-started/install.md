# Install

tnkit needs Python 3.11 or later. The core depends only on numpy and scipy, and runs on the
CPU with the numpy backend:

```bash
pip install tnkit
```

The `torch-cpu`, `torch-cuda` and `torch-mps` backends also need PyTorch. Two extras install
it from PyPI:

::::{tab-set}

:::{tab-item} CPU or Apple silicon
```bash
pip install "tnkit[torch]"
```
:::

:::{tab-item} NVIDIA GPU (Linux)
```bash
pip install "tnkit[cuda]"
```
:::

::::

On Linux the two extras install the same package: PyPI's torch wheel for Linux is a CUDA
build, which bundles the CUDA libraries and is a large download. `tnkit[torch]` on Linux
therefore also runs on an NVIDIA GPU. For a CPU-only build of torch, for a specific CUDA
version, or on Windows, install torch first with the selector at <https://pytorch.org>, then
run `pip install tnkit`.

`import tnkit` does not import PyTorch; a torch backend loads it the first time it is used.

To install from a clone of the repository, run `pip install .` in it, with an extra if you need
one (`pip install ".[torch]"`). To work on tnkit itself, see {doc}`../development`.

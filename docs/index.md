# tnkit

tnkit is a matrix-product-state (MPS) library for one-dimensional quantum many-body
problems. It finds ground states with DMRG and evolves states in time with TDVP and TEBD.
The same code runs on numpy, on PyTorch on the CPU, and on NVIDIA GPUs; you choose with one
argument.

```python
from tnkit import dmrg, tfim_mpo

E, mps, info = dmrg(tfim_mpo(64, g=1.0), chi=64)                        # CPU
E, mps, info = dmrg(tfim_mpo(64, g=1.0), chi=64, backend="torch-cuda")  # GPU
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install, run a first DMRG and a first TDVP, pick a backend.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

MPS and MPO conventions, solver options, symmetries, GPU notes.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function and class, generated from the code.
:::

:::{grid-item-card} Related software
:link: related-software
:link-type: doc

Where tnkit sits next to TeNPy, ITensor, YASTN and others.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
user-guide/index
api/index
benchmarks
related-software
changelog
```

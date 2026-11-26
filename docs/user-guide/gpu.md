# GPUs

Pass `backend="torch-cuda"` to run on an NVIDIA GPU. By default the arithmetic is double
precision (complex128, or float64 with `-real`), and the factorisations use cuSOLVER. This
page shows how to check and time a GPU run, and describes the switches that make small GPU
operations cheaper.

## Checking and timing a run

```python
import time
import torch
from tnkit import dmrg, get_backend, tfim_mpo

print(torch.cuda.is_available(), torch.cuda.get_device_name(0))   # True, and the GPU's name
be = get_backend("torch-cuda")
print(be.device, be.dtype)                   # the CUDA device, torch.complex128

mpo = tfim_mpo(64, g=1.0)
dmrg(mpo, chi=32, backend="torch-cuda", n_sweeps=2)      # warm-up: CUDA start-up, first kernels
t0 = time.perf_counter()
E, mps, info = dmrg(mpo, chi=128, backend="torch-cuda")
print(f"{time.perf_counter() - t0:.1f} s", info["backend"], mps[0].device, mps[0].dtype)
print(info["sweep_s"])                       # seconds per sweep
```

The first call on a GPU pays for starting CUDA and loading its kernels, so time a second call.
`E` is a Python float, so the GPU has finished when `dmrg` returns, and the wall time needs
no extra synchronisation. `info["backend"]` names the backend that ran, and the tensors of
`mps` stay on the device. To compare with the CPU, run the same call with
`backend="numpy"`; at small `chi` the CPU is often faster.

## Where the time goes

On a GPU, small MPS operations cost more in launch overhead than in arithmetic. tnkit keeps
the whole Lanczos loop of DMRG on the device, so no vector crosses to the host per
matrix-vector product. For one-site TDVP it also removes most launches with CUDA graphs.

## CUDA graphs (one-site TDVP)

`tdvp1_evolve(..., graph="auto")`, the default, records each site update once as a CUDA graph
and replays it on later steps, one submission instead of hundreds of kernel launches. The
replay gives the same numbers as the ordinary path.

- `"auto"` captures only on a complex torch backend on the current CUDA device. A real dtype, a
  block backend and any other device run the ordinary path.
- `"on"` requires capture and raises if it is not possible.
- `"off"` never captures.

`"torch-cuda"` names the CUDA device that is current when the backend is built
(`cuda:<current>`). To run on GPU `k`, call `torch.cuda.set_device(k)` before the backend is
built (before the call, when you pass the name as a string), or pass `"torch-cuda:k"`
explicitly.

A graph records one set of tensor shapes, so sites whose bond dimension is still growing run
on the ordinary path. So does any step that would need a deeper Krylov space than
`graph_m_max` (default 12) allows; that step then gives exactly the ordinary path's result.
`graph_cache_cap` (default 1024) bounds how many graphs are kept. If you call `tdvp1_evolve`
many times in one run, build one cache and pass it as `prefix_graph=`, so capture is paid
once. `"auto"` also runs the ordinary path with `adaptive_dt`, a fixed `fast_expm` depth or a
`krylov_m` other than 30; `"on"` raises in those cases.

This section is for dense tensors. Block tensors refuse `graph="on"`; their Krylov
matrix-vector products have a separate switch, `TNKIT_BLOCK_GRAPH`, described under
[Block TDVP](#block-tdvp-host-work-and-cuda-graphs) below.

## SVD driver

On CUDA tnkit uses cuSOLVER's `gesvd`, which on MPS truncation blocks is faster and more
accurate than PyTorch's default. `TNKIT_CUDA_SVD_DRIVER` (single SVDs) and
`TNKIT_CUDA_SVD_DRIVER_BATCHED` (batched TDVP) select another driver; `default` restores
PyTorch's choice.

## Caches

Contractions of block and symmetric tensors are planned once and the plans are cached; the
limits of those caches, and how to inspect them, are in {doc}`performance`.

## Block TDVP: host work and CUDA graphs

At the bond dimensions where kernel launches and host work dominate a block TDVP step on a
GPU, two switches reduce them. Both are provisional, and each gives results identical to the
ordinary path, bit for bit.

**Block Lanczos with less host work (provisional), on by default.** For block tensors, the
Krylov exponential runs its vector algebra on flat buffers when the vectors share a layout,
and reads the Lanczos coefficients from the device once per iteration.
`TNKIT_BLOCK_LANCZOS_FAST=0` restores the plain loop, which reads every coefficient from the
device on its own, and `tnkit.block_lanczos.set_enabled(False)` does the same at run time. Values other
than empty, `0` and `1` raise at import.

**CUDA graphs for block matrix-vector products (provisional), off by default.** Set
`TNKIT_BLOCK_GRAPH=1` or call `tnkit.block_graph.set_enabled(True)`. In a block TDVP solve on
CUDA, the first matrix-vector product per block layout within each Krylov solve runs as
usual, the second is recorded as a graph, and later ones replay it. The graphs are dropped when the solve ends, so
nothing carries over between solves. The valid values of the variable are empty, `0` and `1`;
anything else raises at import. If a recording fails, that layout stays on the ordinary path
for the rest of the solve. This is the block-tensor counterpart of the dense `graph=` option
above, which block tensors do not accept.

```python
from tnkit import block_graph

block_graph.set_enabled(True)
# ... run a block TDVP step on CUDA ...
print(block_graph.info())        # solves, captures, replays, failed, ...
block_graph.set_enabled(False)   # also closes every live graph
```

- The gather indices a graph reads (see {doc}`performance`) count toward the cache limit,
  and the cache cannot free them while the graph lives. The CUDA memory pool that graphs share is not counted.
- `block_graph.clear()` closes every live graph. `block_graph.info()` returns the counters.
  See {py:func}`~tnkit.block_graph.clear` and {py:func}`~tnkit.block_graph.info`.

All environment variables are listed in {doc}`environment`.

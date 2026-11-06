# GPU notes

Pass `backend="torch-cuda"` to run on an NVIDIA GPU. By default everything is double
precision (complex128, or float64 with `-real`), and the factorisations use cuSOLVER.

## Where the time goes

On a GPU, small MPS operations cost more in launch overhead than in arithmetic. tnkit keeps
the whole Lanczos loop of DMRG on the device, so no vector crosses to the host per
matrix-vector product. For one-site TDVP it also removes most launches with CUDA graphs.

## CUDA graphs (one-site TDVP)

`tdvp1_evolve(..., graph="auto")`, the default, records each site update once as a CUDA graph
and replays it on later steps, one submission instead of hundreds of kernel launches. The
replay gives the same numbers as the ordinary path.

- `"auto"` captures only on a complex torch backend on the default CUDA device (the
  `torch-cuda` spec, `torch-cuda:0`, or such a `Backend` object), with a visible device. A real
  dtype, a block backend and any other device run the ordinary path. After
  `torch.cuda.set_device(k)` with `k != 0`, pass the unindexed `torch-cuda` or `graph="off"`.
- `"on"` requires capture and raises if it is not possible.
- `"off"` never captures.

A graph records one set of tensor shapes, so sites whose bond dimension is still growing run
on the ordinary path. So does any step that would need a deeper Krylov space than
`graph_m_max` (default 12) allows; that step then gives exactly the ordinary path's result.
`graph_cache_cap` (default 1024) bounds how many graphs are kept. If you call `tdvp1_evolve`
many times in one run, build one cache and pass it as `prefix_graph=`, so capture is paid
once. `"auto"` also runs the ordinary path with `adaptive_dt`, a fixed `fast_expm` depth or a
`krylov_m` other than 30; `"on"` raises in those cases.

This section is for dense tensors. Block tensors refuse `graph="on"`; their Krylov matvecs
have a separate switch, `TNKIT_BLOCK_GRAPH`, described under "Block TDVP" below.

## SVD driver

On CUDA tnkit uses cuSOLVER's `gesvd`, which on MPS truncation blocks is faster and more
accurate than PyTorch's default. `TNKIT_CUDA_SVD_DRIVER` (single SVDs) and
`TNKIT_CUDA_SVD_DRIVER_BATCHED` (batched TDVP) select another driver; `default` restores
PyTorch's choice.

## Memory and plan caches

Block-sparse contractions are planned once per pair of tensor layouts and the plans are
cached in two levels, each bounded in bytes:

- Plans and their gather indices, in total over the host and all devices: 4 GiB by default,
  lowered to 30% of the device's memory, if that is less, the first time an index reaches a
  CUDA device. `TNKIT_BLOCK_CACHE_MB` or `set_cache_limit` replaces this default.
- Symbolic plans (which blocks meet, independent of sector sizes): 512 MiB by default, or
  `TNKIT_BLOCK_LABEL_CACHE_MB`.

Both variables are read at import. `tnkit.blocks.set_cache_limit(max_bytes, label_max_bytes)`
changes the bounds and evicts at once; `cache_info()` reports hits, misses, evictions and
bytes held for both caches, and where the first limit came from (`limit_source`: `env`, `set`,
`default-host` or `default-device`); `clear_cache()` empties them.

A gather index is expanded the first time it is used on a device. `TNKIT_BLOCK_EXPAND` sets
where that first expansion happens, and is read at each expansion:

- `auto` (default): on the host, then copied, for the CPU and for indices of at most 2^17
  elements; on the device itself for larger ones.
- `host`: on the host, then copied.
- `device`: on the device.

An index that already has a host copy, or was already expanded on another device, is copied
in every mode. Any other value raises `ValueError` the first time an index is used on a torch
device; numpy runs do not read it. Uploads to a CUDA device of at most 1 MiB go through pinned
host memory without waiting for the device. That pinned pool is outside the cache limit above.
Larger uploads are synchronous.

`cache_info()` also reports the contractions run and the GEMMs with a nonzero inner dimension
(`contractions`, `gemms`) and the build time in wall and CPU seconds (`build_s`,
`build_cpu_s`).

These defaults are provisional and may change before 1.0.

On the numpy backend, small dense contractions are planned too (`TNKIT_EINSUM_PLAN=0`
disables this). Each planned contraction is bit-identical to `numpy.einsum`.

## Block TDVP: host work and CUDA graphs

At the bond dimensions where kernel launches and host work dominate a block TDVP step on a
GPU, two switches reduce them. Both are provisional. Each gives results identical to the plain
path, bit for bit; this is tested for both, and for graph replay on CUDA.

**Host Lanczos path (provisional), on by default.** For block tensors, the Krylov exponential runs its
vector algebra on flat buffers when the vectors share a layout, and reads the Lanczos
coefficients once per iteration. `TNKIT_BLOCK_LANCZOS_FAST=0` restores the eager loop, and
`tnkit.block_lanczos.set_enabled(False)` does the same at run time. Values other than empty,
`0` and `1` raise at import.

**CUDA graphs for block Krylov matvecs (provisional), off by default.** Set
`TNKIT_BLOCK_GRAPH=1` or call `tnkit.block_graph.set_enabled(True)`. In a block TDVP solve on
CUDA, the first matvec per layout within each Krylov solve runs as usual, the second is
recorded as a graph, and later ones replay it. The graphs are dropped when the solve ends, so
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

- The gather indices a graph reads count toward the cache limit, and the cache cannot free
  them while the graph lives. The CUDA memory pool that graphs share is not counted.
- `block_graph.clear()` closes every live graph. `block_graph.info()` returns the counters.
  See {py:func}`~tnkit.block_graph.clear` and {py:func}`~tnkit.block_graph.info`.

All environment variables are listed in {doc}`environment`.

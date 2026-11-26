# Performance tuning and caches

Block and symmetric tensors spend much of their time working out which blocks meet in a
contraction and where their entries go. tnkit does that work once per pair of tensor layouts
and caches the result, a *contraction plan*. The caches are bounded, and this page describes
their limits and how to inspect them. The defaults suit most runs; change them when a long
run grows its memory, or to measure what the caches save.

## Block-tensor plans

Plans for block tensors are cached at two levels, each bounded in bytes:

- Plans and their *gather indices* (the integer arrays that move each block's entries
  between the layout of a tensor and that of a matrix product), in total over the host and all
  devices: 4 GiB by default, lowered to 30% of the device's memory, if that is less, the first
  time an index reaches a CUDA device. `TNKIT_BLOCK_CACHE_MB` or `set_cache_limit` replaces
  this default.
- Symbolic plans (which blocks meet, independent of sector sizes): 512 MiB by default, or
  `TNKIT_BLOCK_LABEL_CACHE_MB`.

Both variables are read at import. In `tnkit.blocks`:

```python
from tnkit.blocks import cache_info, clear_cache, set_cache_limit

set_cache_limit(2 * 1024**3, 256 * 1024**2)   # plans and indices 2 GiB, symbolic plans 256 MiB
info = cache_info()
clear_cache()
```

`set_cache_limit(max_bytes, label_max_bytes)` changes the bounds and evicts at once.
`cache_info()` reports hits, misses, evictions and bytes held for both caches, and where the
first limit came from (`limit_source`: `env`, `set`, `default-host` or `default-device`). It
also reports the contractions run and the matrix products with a nonzero inner dimension
(`contractions`, `gemms`), and the time spent building plans in wall and CPU seconds
(`build_s`, `build_cpu_s`). `clear_cache()` empties both caches.

A gather index is expanded to its full form the first time it is used on a device.
`TNKIT_BLOCK_EXPAND` sets where that first expansion happens, and is read at each expansion:

- `auto` (default): on the host, then copied, for the CPU and for indices of at most 2^17
  elements; on the device itself for larger ones.
- `host`: on the host, then copied.
- `device`: on the device.

An index that already has a host copy, or was already expanded on another device, is copied
in every mode. Any other value raises `ValueError` the first time an index is used on a torch
device; numpy runs do not read it. Uploads to a CUDA device of at most 1 MiB go through pinned
host memory without waiting for the device. That pinned pool is outside the cache limit above.
Larger uploads are synchronous.

These defaults are provisional and may change before 1.0.

## Symmetric-tensor caches

Symmetric tensors cache the maps between block layouts, the recoupling matrices and the
fusion-tree structures they are built from. Each cache has its own limit, set by a
`TNKIT_SYMMETRIC_*` variable read at import; {doc}`environment` lists them with their
defaults.

## Dense contractions on numpy

On the numpy backend, small dense contractions are planned too (`TNKIT_EINSUM_PLAN=0`
disables this). Each planned contraction is bit-identical to `numpy.einsum`.

## Check level

`TNKIT_CHECK=off` turns off the shape checks that every solver runs on its inputs, for timing
runs; see {doc}`environment`.

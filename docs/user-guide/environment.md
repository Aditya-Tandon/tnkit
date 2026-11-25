# Environment variables

All are optional. Unless noted, a variable is read when it is used, so it can be changed
while a program runs.

| Variable | Values (default first) | Effect |
|---|---|---|
| `TNKIT_CHECK` | `entry`, `debug`, `off` | Shape checks on MPS, MPO and environment inputs, which raise `tnkit.invariants.InvariantError` on a mismatch. `entry` checks once per solver call; `debug` also checks after every sweep; `off` never checks (for timing runs). |
| `TNKIT_EINSUM_PLAN` | `1`, `0` | Plans small dense contractions on the numpy backend. `0` turns plans off. Read at import. |
| `TNKIT_CUDA_SVD_DRIVER` | `gesvd`, `default`, or another cuSOLVER driver name | The CUDA driver for single SVDs. `default` uses PyTorch's choice. |
| `TNKIT_CUDA_SVD_DRIVER_BATCHED` | `gesvd`, `default`, or another driver name | The same for batched SVDs (batched TDVP). |
| `TNKIT_EXPM_ADAPTIVE` | `1`, `0` | `0` makes the Krylov time-step exponential use its full dimension instead of stopping adaptively. For comparisons only. |
| `TNKIT_BLOCK_CACHE_MB` | unset: 4096, or 30% of the CUDA device's memory if less | Limit, in MiB, of the block-sparse plan and index cache. Unset, the device fraction is taken the first time an index reaches a CUDA device. Read at import. |
| `TNKIT_BLOCK_LABEL_CACHE_MB` | `512` | Limit, in MiB, of the symbolic-plan cache. Read at import. |
| `TNKIT_BLOCK_LANCZOS_FAST` | `1`, `0` | Block TDVP's Krylov exponential with less host work per iteration, bitwise equal to the eager loop (provisional). `0` restores the eager loop. Values other than empty, `0` and `1` raise at import. Read at import. |
| `TNKIT_BLOCK_EXPAND` | `auto`, `host`, `device` | Where block gather indices are expanded. `auto`: on the host, then copied, for the CPU and for small indices; on the device for large ones. `host`: on the host, then copied. `device`: on the device. These apply to an index's first expansion; an index with a host copy, or already expanded on another device, is copied in every mode. Any other value raises `ValueError` on first use on a torch device. Numpy runs do not read it. |
| `TNKIT_BLOCK_GRAPH` | unset or `0`, `1` | `1` replays block TDVP Krylov matvecs from CUDA graphs on CUDA (provisional, off by default). Any other value raises. Read at import. |
| `TNKIT_SYMMETRIC_INDEX_CACHE_MB` | `1024` | Limit, in MiB, of the maps between layouts of symmetric tensors on the host (numpy and torch on the CPU). Read at import. |
| `TNKIT_SYMMETRIC_DEVICE_INDEX_CACHE_MB` | unset: 2048, or 10% of the CUDA device's memory if less | The same for maps on other devices. Unset, the device fraction is taken the first time a map reaches a CUDA device. Read at import. |
| `TNKIT_SYMMETRIC_PERMUTE_CACHE_MB` | `1024` | Limit, in MiB, of the symmetric permute structures (tree pairs and recoupling matrices by sectors). Read at import. |
| `TNKIT_SYMMETRIC_RECOUPLING_CACHE_MB` | `256` | Limit, in MiB, of the symmetric recoupling matrices. Read at import. |
| `TNKIT_SYMMETRIC_STRUCTURE_CACHE_MB` | `512` | Limit, in MiB, of the symmetric fusion-tree structures of one side. Read at import. |
| `TNKIT_SYMMETRIC_RECENT_LAYOUTS_MB`, `TNKIT_SYMMETRIC_RECENT_LAYOUTS` | `256`, `32768` | Limits, in MiB and in entries, of the symmetric layouts kept alive after use. Read at import. |
| `TNKIT_SYMMETRIC_DENSE_TREE_CACHE_MB`, `TNKIT_SYMMETRIC_DENSE_PAIR_CACHE_MB` | `256`, `256` | Limits, in MiB, of the dense fusion trees and tree pairs, used only to convert symmetric tensors to and from dense arrays. Read at import. |

See {doc}`gpu` for the caches, CUDA graphs and the SVD drivers.

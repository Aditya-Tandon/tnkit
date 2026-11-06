# Choosing a backend

Every solver takes a `backend` argument. Nothing else in your code changes.

| `backend=` | Runs on | Precision | Needs |
|---|---|---|---|
| `"numpy"` | CPU | complex128 | nothing extra |
| `"torch-cpu"` | CPU | complex128 | `.[torch]` |
| `"torch-cuda"` | NVIDIA GPU | complex128 | `.[cuda]` and a CUDA device |
| `"torch-mps"` | Apple GPU | single precision | `.[torch]` on Apple silicon |

Add `-real` (for example `"numpy-real"`, `"torch-cuda-real"`) to use float64 instead of
complex128. Real arithmetic is valid for ground states of real Hamiltonians and needs half
the memory. Time evolution needs a complex backend; a real state is converted when evolution
starts.

`dmrg`'s default is `backend="auto"`: `"numpy-real"` when the MPO is real and `"numpy"`
otherwise. The time-evolution functions default to `"numpy"`.

::::{tab-set}

:::{tab-item} CPU
```python
E, mps, info = dmrg(mpo, chi=64, backend="numpy")
```
:::

:::{tab-item} GPU
```python
E, mps, info = dmrg(mpo, chi=64, backend="torch-cuda")
S = mps.entropies()      # measured on the GPU; only the result comes back
mps_host = mps.to_numpy()
```
:::

::::

The state stays on the device it was computed on. The observables in `tnkit` accept states on
any device, compute there and return numpy arrays. Call `mps.to_numpy()` when you need the
tensors themselves on the host.

**Which to use.** At small bond dimension a GPU gains little, because every GPU operation has
a fixed launch cost that the arithmetic does not cover. The GPU pays off as `chi` grows; time
your own problem on both before committing to one. `torch-mps` is single precision
(complex64), so it cannot reach the accuracy of the double-precision backends.

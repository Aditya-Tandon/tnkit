# Backends and GPUs

A backend decides where the arithmetic runs and in which number type. Every solver takes a
`backend` argument, and nothing else in your code changes.

| `backend=` | Runs on | dtype | dtype with `-real` | Needs |
|---|---|---|---|---|
| `"numpy"` | CPU | complex128 | float64 | nothing extra |
| `"torch-cpu"` | CPU | complex128 | float64 | `tnkit[torch]` |
| `"torch-cuda"` | NVIDIA GPU | complex128 | float64 | `tnkit[cuda]` and a CUDA device |
| `"torch-mps"` | Apple GPU | complex64 | float32 | `tnkit[torch]` on Apple silicon |

A `-real` suffix (for example `"numpy-real"`, `"torch-cuda-real"`) selects real arithmetic.
Real arithmetic is valid for ground states of real Hamiltonians and needs half the memory.
Time evolution needs a complex backend; a real state is converted to complex when evolution
starts. `torch-mps` is single precision in both forms, so it cannot reach the accuracy of the
double-precision backends.

`dmrg`'s default is `backend="auto"`: `"numpy-real"` when the MPO is real and `"numpy"`
otherwise. The time-evolution functions default to `"numpy"`.

::::{tab-set}

:::{tab-item} CPU
```python
from tnkit import dmrg, tfim_mpo

mpo = tfim_mpo(32, g=1.0)
E, mps, info = dmrg(mpo, chi=64, backend="numpy")
```
:::

:::{tab-item} GPU
```python
from tnkit import dmrg, tfim_mpo

mpo = tfim_mpo(32, g=1.0)
E, mps, info = dmrg(mpo, chi=64, backend="torch-cuda")
S = mps.entropies()      # computed on the GPU; only the result comes back
mps_host = mps.to_numpy()
```
:::

::::

The state stays on the device it was computed on. The observables in `tnkit` accept states on
any device, compute there and return numpy arrays. Call `mps.to_numpy()` when you need the
tensors themselves on the host.

**Which to use.** At small bond dimension a GPU gains little, because every GPU operation has
a fixed launch cost that the arithmetic does not cover. The GPU pays off as `chi` grows; time
your own problem on both before committing to one. {doc}`../user-guide/gpu` shows how to check
the device and dtype of a run and how to time it.

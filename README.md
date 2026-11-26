# tnkit

tnkit is a Python library for matrix-product states (MPS) and matrix-product operators (MPO),
the standard tensor-network description of one-dimensional quantum many-body systems. It
finds ground and excited states with DMRG (the density-matrix renormalisation group), evolves
states in time with TDVP (the time-dependent variational principle) and TEBD, and can conserve
abelian and non-abelian symmetries such as U(1) and SU(2). It also has thermal states by
purification, and iDMRG and VUMPS for infinite chains (the transverse-field Ising model, or a
translation-invariant model given as one bulk MPO tensor).

The same code runs on numpy and on PyTorch, on the CPU or on a GPU; one argument chooses the
backend. On NVIDIA GPUs the arithmetic is double precision by default. tnkit is alpha software.

Documentation: <https://tnkit.readthedocs.io>

## Install

tnkit needs Python 3.11 or later. The core depends only on numpy and scipy.

```bash
pip install tnkit
```

For the PyTorch backends, install an extra:

```bash
pip install "tnkit[torch]"     # CPU or Apple silicon
pip install "tnkit[cuda]"      # NVIDIA GPU on Linux
```

On Linux both extras install the same package, PyPI's torch wheel, which is a CUDA build and
bundles the CUDA libraries (a large download); `tnkit[torch]` on Linux therefore also runs on
an NVIDIA GPU. For a CPU-only build or a specific CUDA version, or on Windows, install torch
first with the selector at <https://pytorch.org>, then `pip install tnkit`. To install from
source, clone the repository and run `pip install .` in it.

`import tnkit` does not import PyTorch; a torch backend loads it on first use.

## Ground state with DMRG

The transverse-field Ising model on an open chain of `N` sites,
`H = -sum_i X_i X_{i+1} - g sum_i Z_i`, where `X` and `Z` are the Pauli matrices (`SX`, `SZ`),
at its critical point `g = 1`:

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, structure_factor, SX

N = 10
mpo = tfim_mpo(N, g=1.0)
E, mps, info = dmrg(mpo, chi=32, n_sweeps=10)

print(E - tfim_ground_energy(N, g=1.0))   # 0.0: exact for this small chain, up to rounding
S = mps.entropies()                       # entanglement entropy at each of the N - 1 bonds
m2, xx_sum = structure_factor(mps, SX)    # M^2, and N^2 M^2 = sum_ij <X_i X_j>
```

`chi` is the bond dimension, the largest number of states the MPS keeps on any bond; larger
is more accurate and slower. `mps` is a list of rank-3 tensors with legs in the order
`(left, physical, right)`, and `info` holds the energy per sweep, the discarded weight and
timings. `structure_factor` returns the squared order parameter
`M^2 = (1/N^2) sum_ij <X_i X_j>` and the sum `sum_ij <X_i X_j> = N^2 M^2`.

A run stops early once the energy changes between sweeps by less than
`max(tol, rtol * |E|)`. Both `tol` and `rtol` default to `1e-10`; pass `tol=0, rtol=0` to run
every sweep.

## Time evolution with TDVP

Prepare the ground state at `g = 1.5`, then evolve it under the Hamiltonian at `g = 1` (a
quench) and record the mean magnetisation `<Z>`, the average of `<Z_i>` over the sites:

```python
from tnkit import dmrg, tdvp2_evolve, tfim_mpo, local_expectations, SZ

N = 16
_, psi0, _ = dmrg(tfim_mpo(N, g=1.5), chi=16, n_sweeps=6)

psi, records = tdvp2_evolve(
    psi0, tfim_mpo(N, g=1.0), chi=32, dt=0.05, n_steps=40,
    measure=lambda m: local_expectations(m, SZ).mean(), meas_every=10,
)
for t, mz in records:
    print(f"t = {t:.2f}   <Z> = {mz:.6f}")
```

States evolve as `|psi(t)> = exp(-iHt) |psi(0)>` with hbar = 1, so `dt` is in units of the
inverse energy of the MPO. `tdvp2_evolve` (two-site) lets the bond dimension grow up to `chi`.
`tdvp1_evolve` (one-site) keeps it fixed and conserves the energy of a time-independent H, up
to the Krylov tolerance (the accuracy of the matrix exponential computed at each step).

## Backends and GPUs

Every solver takes `backend=`:

| `backend=` | Runs on | dtype | dtype with `-real` |
|---|---|---|---|
| `"numpy"` | CPU | complex128 | float64 |
| `"torch-cpu"` | CPU | complex128 | float64 |
| `"torch-cuda"` | NVIDIA GPU | complex128 | float64 |
| `"torch-mps"` | Apple GPU | complex64 | float32 |

Add `-real` (for example `"numpy-real"`) for real arithmetic. `-real` is for ground states
only: time evolution needs a complex backend, and a real state is converted when it starts.
`dmrg` defaults to `backend="auto"`, which picks `numpy-real` for a real MPO, so the DMRG
example above runs in real double precision. Moving a run to the GPU changes one argument:

```python
from tnkit import dmrg, tfim_mpo

E, mps, info = dmrg(tfim_mpo(32, g=1.0), chi=32, backend="torch-cuda")
S = mps.entropies()       # the SVDs run on the GPU; only the singular values come back
```

The state stays on the device: contractions, factorisations and the Krylov vectors never leave
it. The small tridiagonal eigenproblem and the small matrix exponential inside the Krylov
solvers are solved on the host, and TDVP's `measure` receives host copies of the state unless
`keep_device=True`. The observables accept states on any device and return numpy arrays;
`mps.to_numpy()` copies the tensors to the host.

## Symmetries

If a Hamiltonian conserves a quantity, such as particle number, total `S^z`, parity or total
spin, most blocks of every tensor are zero by symmetry, and tnkit can store and multiply only
the others. Nothing is symmetric unless you ask. To use a symmetry, give the charge of every
basis state of every site, convert the MPO, and pass the total charge of the state you want.
The Ising model conserves parity, a `Zn(2)` charge (0 for `Z = +1`, 1 for `Z = -1`):

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, get_backend, Zn, to_block_mpo, to_dense_mps

N = 10
site_charges = [[0, 1]] * N            # the parity charge of each basis state of each site
Wb, _ = to_block_mpo(tfim_mpo(N, g=1.0), site_charges, get_backend("numpy-real"), sym=Zn(2))
E, mps, info = dmrg(Wb, 16, charge=0)  # ground state with total parity charge 0

print(E - tfim_ground_energy(N, g=1.0))
dense = to_dense_mps(mps, site_charges)
```

Block tensors take the abelian symmetries `U1`, `Zn` and their products. Non-abelian
symmetries such as SU(2) and U(1) x SU(2) run on symmetric tensors, in
`tnkit.symmetric_backend`, with `dmrg` (also with `sites=1`), `dmrg_excited`, `tdvp2_evolve`
and `tdvp1_evolve`.
Both are provisional and may change before version 1.0. The symmetries guide in the
documentation says which to choose, and which solvers take them.

## Tests

From a clone, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra torch     # Python 3.11, the test tools and torch, from uv.lock
uv run pytest
```

Under uv, `--extra torch` installs the CPU-only torch wheel on Linux, and `--extra cuda` a
CUDA build. Or with pip: `pip install -e ".[test]"`, then `pytest`.

## Benchmarks

Timings against other tensor-network libraries, and the conditions behind them, are in
[BENCHMARKS.md](BENCHMARKS.md).

## Related software

Several libraries overlap with tnkit. [TeNPy](https://arxiv.org/abs/1805.00055) (Python) and
[ITensor](https://arxiv.org/abs/2007.14822) (Julia) implement MPS algorithms with abelian
symmetries; [YASTN](https://arxiv.org/abs/2405.12196) provides abelian-symmetric tensors with
numpy and PyTorch backends; [Quantum TEA](https://arxiv.org/abs/2409.03818) simulates
many-body systems and quantum circuits on CPUs and GPUs; and
[block2](https://arxiv.org/abs/2310.03920) is a DMRG code for quantum chemistry with abelian and
SU(2) symmetries. [TAPP](https://arxiv.org/abs/2601.07827) (Tensor Algebra Processing
Primitives) is a proposed standard interface for tensor contraction; tnkit's source code uses
its vocabulary for contractions (modes, extents, and free, contracted and batch indices).

## Questions and contributions

Questions, bug reports and suggestions go to
[GitHub Issues](https://github.com/Aditya-Tandon/tnkit/issues); pull requests are welcome.
You can also email Aditya Tandon at adityatandon29@gmail.com.

## Licence and citation

tnkit is released under the MIT licence; see [LICENSE](LICENSE). If you use it in published
work, please cite it as described in [CITATION.cff](CITATION.cff).

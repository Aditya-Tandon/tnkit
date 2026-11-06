# tnkit

tnkit is a Python library for matrix-product states (MPS) and matrix-product operators (MPO),
the standard tensor-network description of one-dimensional quantum many-body systems. It
finds ground states with DMRG (the density-matrix renormalization group) and evolves states
in time with TDVP (the time-dependent variational principle). It also has TEBD, infinite-chain
iDMRG and VUMPS, and excited states.

The same code runs on numpy and on PyTorch, on the CPU or on a GPU; one argument chooses the
backend. On NVIDIA GPUs the arithmetic is double precision by default. tnkit is alpha software.

Documentation: <https://tnkit.readthedocs.io> (placeholder address: the site is published when
the repository becomes public).

## Install

tnkit needs Python 3.11 or later. The core depends only on numpy and scipy.

```bash
git clone https://github.com/Aditya-Tandon/tnkit.git
cd tnkit
pip install .
```

For the PyTorch backends, install an extra:

```bash
pip install ".[torch]"     # CPU, or Apple silicon
pip install ".[cuda]"      # NVIDIA GPU on Linux: PyPI's torch wheel is a CUDA build
```

On Linux both extras install the same PyPI wheel, which bundles the CUDA libraries (a large
download). On Windows, or to choose a CUDA version or a CPU-only build, install torch first with
the selector at <https://pytorch.org>, then `pip install .`.

`import tnkit` does not import PyTorch; a torch backend loads it on first use.

## Ground state with DMRG

The transverse-field Ising model on an open chain of `N` sites,
`H = -sum_i X_i X_{i+1} - g sum_i Z_i`, at its critical point `g = 1`:

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, structure_factor, SX

N = 10
mpo = tfim_mpo(N, g=1.0)
E, mps, info = dmrg(mpo, chi=32, n_sweeps=10)

print(E - tfim_ground_energy(N, g=1.0))   # about 1e-14: exact for this small chain
S = mps.entropies()                       # entanglement entropy at every bond
m2, Sq = structure_factor(mps, SX)        # squared order parameter
```

`chi` is the bond dimension, the largest matrix size the MPS may use; larger is more accurate
and slower. `mps` is a list of rank-3 tensors with an index order of `(left, physical, right)`.
`info` holds the energy per sweep, the discarded weight and timings.

A run stops early once the energy changes between sweeps by less than
`max(tol, rtol * |E|)`. Both `tol` and `rtol` default to `1e-10`; pass `tol=0, rtol=0` to run
every sweep.

## Time evolution with TDVP

Prepare the ground state at `g = 1.5`, then evolve it under the Hamiltonian at `g = 1` (a
quench) and record the mean of `Z`:

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

`tdvp2_evolve` (two-site) lets the bond dimension grow up to `chi`. `tdvp1_evolve` (one-site)
keeps it fixed and conserves energy up to the Krylov tolerance.

## Backends and GPUs

Every solver takes `backend=`:

| `backend=` | Runs on | Precision |
|---|---|---|
| `"numpy"` | CPU | complex128 |
| `"torch-cpu"` | CPU | complex128 |
| `"torch-cuda"` | NVIDIA GPU | complex128 |
| `"torch-mps"` | Apple GPU | complex64 |

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

## Symmetries and block tensors

If a Hamiltonian conserves a charge, such as particle number, total spin `S^z` or parity,
every tensor splits into blocks of definite charge and most blocks are zero. tnkit can store
and multiply only the nonzero blocks. It supports abelian symmetries; non-abelian symmetries
such as SU(2) are not supported yet:

```python
from tnkit import U1, Zn, ProductSymmetry

U1()                             # integer charges that add, e.g. particle number
Zn(2)                            # charges 0..n-1 that add modulo n, e.g. parity
ProductSymmetry((U1(), Zn(2)))   # both; charges are tuples such as (1, 0)
```

Nothing is symmetric unless you ask. To use a symmetry, give the charge of every physical
state of every site, convert the MPO, and pass the total charge you want. The Ising model
conserves parity, a `Zn(2)` charge:

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, get_backend, Zn, to_block_mpo, to_dense_mps

N = 10
site_charges = [[0, 1]] * N            # the charge of each physical state of each site
Wb, _ = to_block_mpo(tfim_mpo(N, g=1.0), site_charges, get_backend("numpy-real"), sym=Zn(2))
E, mps, info = dmrg(Wb, 16, charge=0)  # ground state with total parity 0

print(E - tfim_ground_energy(N, g=1.0))
dense = to_dense_mps(mps, site_charges)
```

- `sym=` is required and keyword-only on `to_block_mpo`, `to_block_mps` and
  `project_to_block_mps`; there is no default symmetry. `to_block_mpo` and `mpo_charges` accept
  `U1`, `Zn` and `ProductSymmetry`. `to_block_mps` and `project_to_block_mps` convert a dense
  MPS and accept `U1()` only; anything else raises `NotImplementedError`.
- A block MPS must be converted with `to_dense_mps` before `entropies()`, `to_numpy()` or the
  observables. `dmrg_excited` takes dense MPOs only.
- With a block MPO, DMRG uses subspace expansion in its opening sweeps by default, so that
  charge sectors missing from the random start can fill. `expand=0` turns it off.
- `tdvp1_evolve` and `tdvp2_evolve` take block states through `BlockBackend`.
- The block-tensor and symmetry API is provisional and may change before version 1.0.

## Tests

```bash
pip install -e ".[test]"
pytest
```

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
Primitives) is a proposed standard interface for tensor contraction; tnkit's documentation
uses its vocabulary for contractions (modes, extents, and free, contracted and batch indices).

## Questions and contributions

Questions, bug reports and suggestions go to
[GitHub Issues](https://github.com/Aditya-Tandon/tnkit/issues); pull requests are welcome.
You can also email Aditya Tandon at adityatandon29@gmail.com.

## Licence and citation

tnkit is released under the MIT licence; see [LICENSE](LICENSE). If you use it in published
work, please cite it as described in [CITATION.cff](CITATION.cff).

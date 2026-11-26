# Abelian symmetries

Block tensors conserve abelian symmetries, those whose charges simply add: U(1) for a
conserved integer and $\mathbb{Z}_n$ for an integer conserved modulo $n$. This page shows
how to run DMRG and TDVP with them.

```python
from tnkit import U1, Zn, ProductSymmetry

U1()                             # integer charges that add, e.g. particle number or 2 S^z
Zn(2)                            # charges 0 .. n-1 that add modulo n, e.g. parity
ProductSymmetry((U1(), Zn(2)))   # both at once; charges are tuples such as (1, 0)
```

## Symmetric DMRG

Three steps turn a dense DMRG into a symmetric one: give the charge of every basis state of
every site, convert the MPO with {py:func}`~tnkit.to_block_mpo`, and pass the total charge
of the state sought to {py:func}`~tnkit.dmrg`.

The transverse-field Ising model conserves the parity $P = \prod_i \sigma^z_i$, a
$\mathbb{Z}_2$ symmetry. State 0 of a site has $\sigma^z = +1$ and parity charge 0; state 1
has $\sigma^z = -1$ and charge 1. The total charge, the number of down spins modulo 2, is 0
for $P = +1$ and 1 for $P = -1$:

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, get_backend, Zn, to_block_mpo, to_dense_mps

N = 10
W = tfim_mpo(N, g=1.0)
site_charges = [[0, 1]] * N                  # parity charge of (sigma^z = +1, sigma^z = -1)
Wb, _ = to_block_mpo(W, site_charges, get_backend("numpy-real"), sym=Zn(2))
E, mps, info = dmrg(Wb, 16, charge=0)        # the lowest state of total charge 0

print(E - tfim_ground_energy(N, g=1.0))      # about 5e-15
print(info["blocks"], info["expand"])        # True 4
dense = to_dense_mps(mps, site_charges)      # an ordinary list of dense tensors
```

The third argument of `to_block_mpo` is the dense backend that sets the dtype and device of
the stored blocks. A real Hamiltonian can be stored in real blocks, `get_backend("numpy-real")`,
which halves the memory, and `dmrg` then runs in real arithmetic.

## Symmetric TDVP

{py:func}`~tnkit.tdvp1_evolve` and {py:func}`~tnkit.tdvp2_evolve` evolve a block MPS under a
block MPO when `backend` is a {py:class}`~tnkit.BlockBackend` around a complex dense backend.
Real-time evolution needs complex numbers, and the TDVP functions, unlike `dmrg`, do not wrap
a dense backend in a `BlockBackend` themselves. `measure` receives the block tensors. Continuing
the parity example with a quench to $g = 1.5$:

```python
from tnkit import tdvp2_evolve, BlockBackend, local_expectations, SZ

be = BlockBackend(get_backend("numpy"))      # complex128 blocks
Wq, _ = to_block_mpo(tfim_mpo(N, g=1.5), site_charges, be, sym=Zn(2))
psi, records = tdvp2_evolve(
    mps, Wq, 16, 0.05, 20, backend=be, meas_every=10,
    measure=lambda m: local_expectations(to_dense_mps(m, site_charges), SZ).mean(),
)
```

## U(1): the XXZ chain

The spin-1/2 XXZ chain conserves the total $S^z$; the charges are $2S^z$, so that
they are integers:

```python
from tnkit import U1
from tnkit.opsum import OpSum, compile_mpo, spin_half

N, Delta = 12, 0.5
terms = OpSum()
for i in range(N - 1):
    terms += (0.5, "Sp", i, "Sm", i + 1)
    terms += (0.5, "Sm", i, "Sp", i + 1)
    terms += (0.25 * Delta, "Z", i, "Z", i + 1)
W, _ = compile_mpo(terms, [spin_half()] * N, dtype="auto")
site_charges = [[1, -1]] * N                 # 2 S^z of (up, down)
Wb, _ = to_block_mpo(W, site_charges, get_backend("numpy-real"), sym=U1())
E0, _, _ = dmrg(Wb, 32, charge=0)            # total S^z = 0: -4.3381483309
E1, _, _ = dmrg(Wb, 32, charge=2)            # total S^z = 1: -4.1411841527
```

Write hopping terms with $S^\pm$. Written as $XX + YY$, the MPO has channels that carry two
charges, and `to_block_mpo` raises.

## Rules

- `sym=` is keyword-only and required on {py:func}`~tnkit.to_block_mpo`,
  {py:func}`~tnkit.block_backend.mpo_charges`, {py:func}`~tnkit.to_block_mps` and
  {py:func}`~tnkit.project_to_block_mps`; there is no default symmetry. The last two convert a
  dense MPS and take `U1()` only.
- With a block MPO, `dmrg` needs exactly one of `charge=` and a block `init_mps`. The random
  start is {py:func}`~tnkit.random_block_mps`, whose `charge` is likewise required.
- The returned `mps` holds block tensors. {py:func}`~tnkit.mpo_expectation`,
  {py:func}`~tnkit.mpo_variance` and `mps_inner` (in `tnkit.dmrg_excited`) accept them.
  `mps.entropies()`, `mps.to_numpy()`, `local_expectations` and the other observables of
  `tnkit.observables` refuse them; {py:func}`~tnkit.to_dense_mps` converts them first.
- Truncation keeps the `chi` largest singular values over all sectors together, as a dense
  truncation would.
- Block runs of `dmrg` use subspace expansion in their opening sweeps by default (see
  {doc}`../dmrg`), so that sectors missing from the random start can fill; `expand=0` turns it
  off.

## Limits

- `trunc="svd"` only, in DMRG and in TDVP, and no `eig="lanczos-mixed"`.
- Block TDVP takes no `fast_expm` and no CUDA graphs.
- `dmrg_block` and `dmrg_tower(pair_solve=True)` from `tnkit.dmrg_excited` take dense MPOs
  only. {py:func}`~tnkit.dmrg_excited` and `dmrg_tower` take block MPOs with `charge=`, as
  `dmrg` does, and do not expand the subspace.
- With or without expansion, a block run at small `chi` can stop at a different fixed point
  than dense DMRG.

These mistakes raise an error rather than run:

| Mistake | Error |
|---|---|
| no `sym=` | `TypeError`: missing keyword-only argument `'sym'` |
| `sym="U1"` | `TypeError: ... sym= must be a Symmetry such as U1(), got 'U1'` |
| a non-abelian symmetry such as SU(2) in `to_block_mpo` | `NotImplementedError: ... needs unique fusion, which block tensors require` |
| an MPO term that changes the charge by two amounts at once, such as $XX + YY$ | `ValueError: MPO site ..., right channel ...: charges [...]` |
| block MPO without `charge=` | `ValueError: a block MPO needs exactly one of charge= ... and a block init_mps ...` |
| `charge=` with a dense MPO | `ValueError: charge= needs a block MPO: convert the MPO with to_block_mpo(W, site_charges, be, sym=...) first` |
| `trunc="gram"` on a block MPO | `ValueError: a block MPO needs trunc='svd' and an eig other than 'lanczos-mixed'` |
| a string `backend=` in block TDVP | `ValueError: block tensors need a block MPS, a block MPO and backend=BlockBackend(dense) ...` |

# Symmetries and block-sparse tensors

```{warning}
Provisional. The symmetry and block-tensor API may change before 1.0.
```

A Hamiltonian that conserves a charge (particle number, total $S^z$, parity) never mixes
states of different charge. If every tensor index is split into sectors of definite charge,
most blocks of every tensor are zero by symmetry and need not be stored or multiplied. A
block-sparse tensor keeps only the allowed blocks.

Nothing is symmetric unless you ask for it: `dmrg(W, chi)` on an ordinary MPO stays dense.

## Symmetries

```python
from tnkit import U1, Zn, ProductSymmetry

U1()                          # integer charges that add, e.g. particle number or 2 S^z
Zn(2)                         # charges 0..n-1 that add modulo n, e.g. parity
ProductSymmetry((U1(), Zn(2)))   # both at once; charges are tuples such as (1, 0)
```

All three are abelian: two charges combine into exactly one charge. SU(2) is not yet
supported.

## Symmetric DMRG

Convert the MPO with {py:func}`~tnkit.to_block_mpo`, giving the charge of every physical state
of every site, then pass the total charge you want to {py:func}`~tnkit.dmrg`. The
transverse-field Ising model conserves the parity $\prod_i \sigma^z_i$, a $\mathbb{Z}_2$
charge: the two states of each site carry charges 0 and 1.

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, get_backend, Zn, to_block_mpo, to_dense_mps

N = 10
W = tfim_mpo(N, g=1.0)
site_charges = [[0, 1]] * N                  # the charge of each physical state
Wb, _ = to_block_mpo(W, site_charges, get_backend("numpy-real"), sym=Zn(2))
E, mps, info = dmrg(Wb, 16, charge=0)        # charge: the total charge sought

print(E - tfim_ground_energy(N, g=1.0))      # about 1e-14
print(info["blocks"], info["expand"])        # True 4
dense = to_dense_mps(mps, site_charges)      # an ordinary list of dense tensors
```

- `sym=` is keyword-only and required on {py:func}`~tnkit.to_block_mpo`,
  {py:func}`~tnkit.block_backend.mpo_charges`, {py:func}`~tnkit.to_block_mps` and
  {py:func}`~tnkit.project_to_block_mps`. There is no default symmetry. The last two convert
  a dense MPS and take `U1()` only.
- With a block MPO, `dmrg` needs exactly one of `charge=` and a block `init_mps`. The random
  start is {py:func}`~tnkit.random_block_mps`, whose `charge` is likewise required.
- The returned `mps` holds block tensors. `mps.entropies()` and `mps.to_numpy()` refuse them;
  {py:func}`~tnkit.to_dense_mps` converts them.
- Truncation keeps the `chi` largest singular values over all sectors together, as a dense
  truncation would.
- Block runs use subspace expansion in their opening sweeps by default (see
  {doc}`dmrg`), so that sectors missing from the random start can fill.

## Symmetric TDVP

{py:func}`~tnkit.tdvp1_evolve` and {py:func}`~tnkit.tdvp2_evolve` take a block MPS and a block
MPO when `backend` is a {py:class}`~tnkit.BlockBackend` around a complex dense backend.
`measure` receives the block tensors:

```python
from tnkit import tdvp2_evolve, BlockBackend, local_expectations, SZ

be = BlockBackend(get_backend("numpy"))
Wb1, _ = to_block_mpo(tfim_mpo(N, g=1.5), site_charges, be, sym=Zn(2))
psi, records = tdvp2_evolve(
    mps, Wb1, 16, 0.05, 20, backend=be, meas_every=10,
    measure=lambda m: local_expectations(to_dense_mps(m, site_charges), SZ).mean(),
)
```

## Limits

- Abelian symmetries only.
- `trunc="svd"` only, in DMRG and in TDVP, and no `eig="lanczos-mixed"`.
- Block TDVP takes no `fast_expm` and no CUDA graphs.
- {py:func}`~tnkit.dmrg_excited`, and `dmrg_block` and `dmrg_tower` from
  `tnkit.dmrg_excited`, take dense MPOs only.
- Block runs expand the subspace in their opening sweeps. With or without that, a block run
  at small `chi` can stop at a different fixed point than dense DMRG.

These mistakes raise an error rather than run:

| Mistake | Error |
|---|---|
| no `sym=` | `TypeError`: missing keyword-only argument `'sym'` |
| `sym="U1"` | `TypeError: ... sym= must be a Symmetry such as U1(), got 'U1'` |
| a non-abelian symmetry such as SU(2) | `NotImplementedError: ... needs unique fusion, which block tensors require` |
| block MPO without `charge=` | `ValueError: a block MPO needs exactly one of charge= ... and a block init_mps ...` |
| `charge=` with a dense MPO | `ValueError: charge= needs a block MPO: convert the MPO with to_block_mpo(W, site_charges, be, sym=...) first` |
| `trunc="gram"` on a block MPO | `ValueError: a block MPO needs trunc='svd' and an eig other than 'lanczos-mixed'` |

## Block-sparse tensors

The tensors themselves live in `tnkit.blocks`, and `tnkit.block_backend.BlockBackend` lets
the solvers' kernels run on them.

```python
import numpy as np
from tnkit.blocks import Leg, from_dense, tensordot, truncated_svd

leg = Leg(charges=(0, 1), dims=(2, 3))      # two sectors: charge 0 (extent 2), charge 1 (extent 3)
x = np.random.default_rng(0).standard_normal((5, 5))

A, dropped = from_dense(x, (leg, leg.flip()), charge=0)
print(A)                                    # BlockTensor(shape=(5, 5), charge=0, blocks=2, stored=13)

B = tensordot(A, A, ((1,), (0,)))           # contract mode 1 of A with mode 0 of A
U, S, Vh, discarded = truncated_svd(B, chi=3)
```

- A {py:class}`~tnkit.blocks.Leg` lists a mode's charge sectors, their extents, a sign and a
  symmetry (`U1()` if not given). A block is allowed when the signed charges of its sectors
  combine to the tensor's total charge.
- A contracted pair of modes must have the same sectors and opposite signs (`leg.flip()`).
- {py:func}`~tnkit.blocks.truncated_svd` truncates across all sectors together, as a dense
  truncated SVD of the same matrix would.
- Contraction plans are cached. `tnkit.blocks.cache_info()` reports the caches and
  `set_cache_limit()` bounds them; see {doc}`gpu`.

See {doc}`../api/blocks` for the full reference.

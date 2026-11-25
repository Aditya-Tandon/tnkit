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

All three are abelian: two charges combine into exactly one charge. Block tensors take only
these. Non-abelian symmetries such as SU(2) run on symmetric tensors; see
[Symmetric tensors](#symmetric-tensors).

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

- Abelian symmetries only (for SU(2), see [Symmetric tensors](#symmetric-tensors)).
- `trunc="svd"` only, in DMRG and in TDVP, and no `eig="lanczos-mixed"`.
- Block TDVP takes no `fast_expm` and no CUDA graphs.
- `dmrg_block` and `dmrg_tower(pair_solve=True)` from `tnkit.dmrg_excited` take dense MPOs
  only. {py:func}`~tnkit.dmrg_excited` and `dmrg_tower` take block MPOs with `charge=`, as
  `dmrg` does.
- Block runs of `dmrg` expand the subspace in their opening sweeps; `dmrg_excited` and
  `dmrg_tower` do not (`info["expand"]` is 0). With or without expansion, a block run at small
  `chi` can stop at a different fixed point than dense DMRG.

These mistakes raise an error rather than run:

| Mistake | Error |
|---|---|
| no `sym=` | `TypeError`: missing keyword-only argument `'sym'` |
| `sym="U1"` | `TypeError: ... sym= must be a Symmetry such as U1(), got 'U1'` |
| a non-abelian symmetry such as SU(2) in `to_block_mpo` | `NotImplementedError: ... needs unique fusion, which block tensors require` |
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

## Symmetric tensors

```{warning}
Provisional, and not exported from `tnkit`: import from the modules named below. The API, and
the file layout of saved states, may change before 1.0.
```

Block tensors need abelian symmetries. Symmetric tensors (`tnkit.symmetric_tensor`) take any
symmetry that `tnkit.symmetry` describes, non-abelian ones included. A tensor that commutes with
the group is stored as one reduced matrix per coupled sector (the total irrep of its legs), over
the fusion trees of its legs; the Clebsch-Gordan coefficients are never stored. Under SU(2) a
bond keeps whole multiplets: one reduced value of spin $j$ stands for $2j + 1$ dense states.

Symmetries:

- `U1()`, `Zn(n)` and `SU2()` from `tnkit.symmetry`, and any `ProductSymmetry` of them, such as
  `ProductSymmetry((U1(), SU2()))`. SU(2) sectors are labelled by twice the spin: 0, 1 (spin
  1/2), 2 (spin 1), ...
- Other groups through a subclass of `Symmetry` that gives the fusion rules and the
  Clebsch-Gordan, F and R coefficients, including fusions in which an outcome appears more than
  once (`fusion = "generic"`).

### Spaces and tensors

A leg is a {py:class}`~tnkit.symmetric_tensor.Space`: its sectors, the number of multiplets of
each, and whether the leg is dual. A tensor maps its domain legs to its codomain legs.

```python
import numpy as np
from tnkit.symmetry import SU2
from tnkit.symmetric_tensor import Space, random, from_dense

V = Space(SU2(), (0, 1, 2), (2, 1, 1))   # spin 0 twice, spin 1/2 once, spin 1 once
print(V.dim)                             # 7 dense states
t = random((V, V.dual()), (), rng=np.random.default_rng(0))
print(t)          # SymmetricTensor(shape=(7, 7), codomain legs=2, sectors=1, stored=6)

s, dropped = from_dense(t.to_dense(), (V, V.dual()))   # project a dense array back
```

`permute`, `contract`, `fuse` and `split` in `tnkit.symmetric_tensor`, and `svd`,
`truncated_svd` and `qr` in `tnkit.symmetric_linalg`, act on the reduced blocks. A truncated SVD
keeps or drops whole multiplets.

### DMRG, excited states and TDVP

{py:func}`~tnkit.symmetric_backend.symmetric_mpo` converts a dense MPO, given a
{py:class}`~tnkit.symmetric_backend.SiteSpace` for each site. It raises if the MPO does not
commute with the symmetry. `dmrg`, `dmrg_excited` and `tdvp2_evolve` then run as they do on
block MPOs. For the spin-1/2 Heisenberg chain under SU(2):

```python
from tnkit import dmrg, dmrg_excited, get_backend, tdvp2_evolve
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.symmetric_backend import SymmetricBackend, spin_half_site, symmetric_mpo, to_dense_mps

N = 10
os = OpSum()
for i in range(N - 1):                       # S_i . S_(i+1)
    os += (0.5, "Sp", i, "Sm", i + 1)
    os += (0.5, "Sm", i, "Sp", i + 1)
    os += (0.25, "Z", i, "Z", i + 1)
W, _ = compile_mpo(os, [spin_half()] * N, dtype="auto")
sites = [spin_half_site()] * N               # one spin 1/2 of SU(2) per site
Ws, _ = symmetric_mpo(W, sites)

E0, gs, _ = dmrg(Ws, 32, charge=0)           # lowest singlet: -4.2580
Et, trip, _ = dmrg(Ws, 32, charge=2)         # lowest triplet (spin 1): -3.9307
E1, ex, _ = dmrg_excited(Ws, 32, prev_states=[gs], charge=0)   # next singlet

be = SymmetricBackend(get_backend("numpy"))  # complex, for real time
psi, _ = tdvp2_evolve(list(gs), Ws, 32, 0.05, 10, backend=be)
dense = to_dense_mps(trip, sites)            # dense tensors; the last bond holds the triplet
```

- `charge` is the total sector: twice the total spin under SU(2), a tuple such as `(0, 2)`
  under U(1) x SU(2). As with block MPOs, exactly one of `charge=` and a symmetric `init_mps`
  is needed.
- `chi` (and TDVP's `chi_max`) counts multiplets, so the dense bond is larger: a bond of `chi`
  multiplets holds the sum of their dimensions in states. `dmrg_excited` reports both
  (`info["max_bond"]`, `info["max_bond_dense"]`).
- A state in a sector of nonzero spin keeps its whole multiplet on its last bond. `mps_inner`,
  `mpo_expectation` and `mpo_variance` in `tnkit.dmrg_excited` return the values of one state
  of it.
- `save_mps` and `load_mps` in `tnkit.symmetric_backend` write a symmetric MPS to one `.npz`
  and read it back onto any dense backend.
- `models_lgt2f.site_spaces_2f` gives the sites of the two-flavour Schwinger model under
  U(1) x SU(2).
- The limits of block tensors apply: `trunc="svd"` only, no `eig="lanczos-mixed"`, no
  `fast_expm` and no CUDA graphs in TDVP, and `dmrg_block` and `dmrg_tower(pair_solve=True)`
  take dense MPOs only.

### GPUs

Pass `backend="torch-cuda"` to `dmrg` or `dmrg_excited`, or
`SymmetricBackend(get_backend("torch-cuda"))` to `tdvp2_evolve`. The reduced blocks, the
contractions, the factorisations and the maps between block layouts stay on the device; the
small recoupling matrices are computed on the host and cached. The cache limits are set by the
`TNKIT_SYMMETRIC_*` variables in {doc}`environment`.

See {doc}`../api/symmetric` for the full reference.

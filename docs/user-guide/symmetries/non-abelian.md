# Non-abelian symmetries

```{warning}
Provisional, and not exported from `tnkit`: import from the modules named below. The API, and
the file layout of saved states, may change before 1.0.
```

The Heisenberg chain conserves not only the total $S^z$ but the total spin: it commutes with
the whole SU(2) group of spin rotations. Conserving SU(2) rather than U(1) finds states of
definite total spin directly, and it shrinks the tensors much further, because the states of
each multiplet are no longer stored separately.

## Reduced tensors

Block tensors cannot hold SU(2), because two spins combine into several total spins, not one.
By the Wigner–Eckart theorem, a tensor that commutes with SU(2) splits into a reduced part and
Clebsch–Gordan coefficients, which the group fixes. tnkit stores only the reduced part: one
matrix $t_c$ for each total spin $c$ of the tensor's legs (its coupled sector). Each row and
column of $t_c$ stands for a whole multiplet, of $2j+1$ states for spin $j$, so a bond of
`chi` multiplets holds more than `chi` dense states.

For a tensor with two legs, an operator $T$ that commutes with every rotation, the theorem
reads

$$\bra{c\, m\, \alpha} T \ket{c'\, m'\, \beta} = \delta_{cc'}\, \delta_{mm'}\, (t_c)_{\alpha\beta} ,$$

where $m$ is the $S^z$ of a state within its multiplet, and $\alpha$, $\beta$ number the
multiplets of spin $c$ on each leg. All $2c + 1$ values of $m$ share the one matrix $t_c$. For
tensors with more legs, Clebsch–Gordan coefficients take the place of $\delta_{mm'}$.

These *symmetric tensors* live in `tnkit.symmetric_tensor`, and
`tnkit.symmetric_backend` runs the DMRG and TDVP drivers on them. They take any symmetry
that `tnkit.symmetry` describes. Sectors of SU(2) are labelled by twice the spin, so that
labels are integers: 0 (spin 0), 1 (spin 1/2), 2 (spin 1), and so on.

## DMRG and excited states

{py:func}`~tnkit.symmetric_backend.symmetric_mpo` converts a dense MPO to symmetric tensors,
given one {py:class}`~tnkit.symmetric_backend.SiteSpace` per site, which says how the basis
states of the site form multiplets. It raises if the MPO does not commute with the symmetry.
`dmrg` (two-site, or one-site with `sites=1`) and `dmrg_excited` then run as they do on block
MPOs, with `charge=` naming the sector of the state sought. For the spin-1/2 Heisenberg chain, $H = \sum_i \mathbf{S}_i \cdot
\mathbf{S}_{i+1}$:

```python
from tnkit import dmrg, dmrg_excited, get_backend, tdvp2_evolve
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.symmetric_backend import SymmetricBackend, spin_half_site, symmetric_mpo
from tnkit.symmetric_backend import to_dense_mps as symmetric_to_dense

N = 10
terms = OpSum()
for i in range(N - 1):                       # S_i . S_(i+1)
    terms += (0.5, "Sp", i, "Sm", i + 1)
    terms += (0.5, "Sm", i, "Sp", i + 1)
    terms += (0.25, "Z", i, "Z", i + 1)
W, _ = compile_mpo(terms, [spin_half()] * N, dtype="auto")
sites = [spin_half_site()] * N               # one spin 1/2 of SU(2) per site
Ws, _ = symmetric_mpo(W, sites)

E0, gs, _ = dmrg(Ws, 32, charge=0)           # lowest singlet: -4.2580352073
Et, trip, _ = dmrg(Ws, 32, charge=2)         # lowest triplet (total spin 1): -3.9306735895
E1, ex, info = dmrg_excited(Ws, 32, prev_states=[gs], charge=0)   # next singlet: -3.3961982690
```

In `spin_half()`, `"Z"` is the Pauli matrix $\sigma^z$, and `"Sp"`, `"Sm"` are $S^+$ and
$S^-$, so the terms above are

$$\mathbf{S}_i \cdot \mathbf{S}_{i+1} = \tfrac12 \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right)
+ \tfrac14\, \sigma^z_i \sigma^z_{i+1} .$$

The energy of charge 0 equals that of DMRG without symmetry on `W`. The default penalty of
`dmrg_excited`, 1, suffices here because the gap between the two singlets, about 0.86, is
smaller; see {doc}`../excited-states`.

- The argument `charge` names the sector of the state sought. Under SU(2) this label is twice
  the total spin, so `charge=2` asks for spin 1; under U(1) $\times$ SU(2) it is a tuple such
  as `(0, 2)`. As with block MPOs, exactly one of `charge=` and a symmetric `init_mps` is
  needed.
- `chi` (and TDVP's `chi_max`) counts multiplets, so the dense bond is larger: a bond of `chi`
  multiplets holds the sum of their dimensions in states. `dmrg_excited` reports both
  (`info["max_bond"]`, `info["max_bond_dense"]`; 10 and 32 above).
- `prev_states` of `dmrg_excited` must be in the same sector as the state sought.

`tnkit.symmetric_backend` has its own `to_dense_mps`, which converts a symmetric MPS to dense
tensors. It shadows {py:func}`tnkit.to_dense_mps`, which converts block tensors, and the two
take a different second argument: a list of `SiteSpace` here, a list of site charges there.
The examples import it under the distinct name `symmetric_to_dense`.

## Reading a state of nonzero spin

A state of total spin $j > 0$ comes back as its whole multiplet. After `symmetric_to_dense`,
the last bond has dimension $2j + 1$, with one index per value of the total $S^z$, from the
highest, $S^z = j$, down. The observables then average over the multiplet: the triplet above
gives $\sum_i \expval{\sigma^z_i} = 0$, and its entanglement entropies are not those of any
one of its three states. To measure one state, keep one index of the last bond, then
normalise: each state of the multiplet holds $1/(2j + 1)$ of the norm, $1/3$ here.

```python
import numpy as np
from tnkit import MPS, local_expectations, mps_overlap, SZ

d = symmetric_to_dense(trip, sites)
print(d[-1].shape)                           # (6, 2, 3): the last bond holds the 3 states
d[-1] = d[-1][..., :1]                       # keep S^z = +1
print(round(mps_overlap(d, d).real, 6))      # 0.333333: a third of the norm
d[-1] = d[-1] / np.sqrt(mps_overlap(d, d).real)
one = MPS(d)
print(round(mps_overlap(one, one).real, 6))  # 1.0
print(round(local_expectations(one, SZ).sum(), 6))   # 2.0, which is 2 S^z
```

Keeping index 0, 1 or 2 gives the state with total $S^z = +1$, 0 or $-1$. `local_expectations`
normalises by itself, but overlaps and the tensors do not, so normalise before using them. `mps_inner`,
`mpo_expectation` and `mpo_variance` in `tnkit.dmrg_excited` accept symmetric states directly
and return the values of one state of the multiplet.

## Time evolution

{py:func}`~tnkit.tdvp2_evolve` and {py:func}`~tnkit.tdvp1_evolve` evolve a symmetric MPS when
`backend` is a
{py:class}`~tnkit.symmetric_backend.SymmetricBackend` around a complex dense backend;
`measure` receives the symmetric tensors. Evolving an eigenstate under its own Hamiltonian
only changes its phase, so this example quenches: it prepares the ground state of a
dimerised chain, with couplings alternating between 1 and 0.2, and evolves it under the
uniform chain `Ws` from above:

```python
from tnkit import MPS

dimer = OpSum()
for i in range(N - 1):
    J = 1.0 if i % 2 == 0 else 0.2
    dimer += (0.5 * J, "Sp", i, "Sm", i + 1)
    dimer += (0.5 * J, "Sm", i, "Sp", i + 1)
    dimer += (0.25 * J, "Z", i, "Z", i + 1)
Wd, _ = symmetric_mpo(compile_mpo(dimer, [spin_half()] * N, dtype="auto")[0], sites)
_, psi0, _ = dmrg(Wd, 32, charge=0)

be = SymmetricBackend(get_backend("numpy"))  # complex, for real time
middle = lambda m: MPS(symmetric_to_dense(m, sites)).entropies()[N // 2 - 1]
psi, records = tdvp2_evolve(psi0, Ws, 32, 0.05, 20, backend=be,
                            measure=middle, meas_every=10)
```

`records` holds the entanglement entropy of the middle bond at $t = 0$, $0.5$ and $1$.

{py:func}`~tnkit.tdvp1_evolve` takes the same arguments without `chi`, and keeps the bond
sectors and multiplet counts of the initial state:

```python
from tnkit import tdvp1_evolve

psi1, records1 = tdvp1_evolve(psi, Ws, 0.05, 20, backend=be, measure=middle, meas_every=10)
```

## Site spaces

A `SiteSpace(space, basis)` describes one physical leg. `space` is a
{py:class}`~tnkit.symmetric_tensor.Space`: the sectors the leg holds, and the number of
multiplets of each. `basis` is the unitary matrix that takes a state written in the model's
basis to the leg's own basis, in which each multiplet lists its states from the highest
$S^z$ down: $x_{\text{leg}} = \texttt{basis}\, x_{\text{model}}$. `spin_half_site()` is
`SiteSpace(Space(SU2(), (1,), (1,)), np.eye(2))`, one spin 1/2 in the basis (up, down) that
`spin_half()` uses.

A spin 1 is one multiplet of label 2. In the basis $S^z = +1, 0, -1$ its basis matrix is the
identity:

```python
import numpy as np
from tnkit import dmrg
from tnkit.opsum import OpSum, compile_mpo, qudit
from tnkit.symmetry import SU2
from tnkit.symmetric_tensor import Space
from tnkit.symmetric_backend import SiteSpace, symmetric_mpo

Sz = np.diag([1.0, 0.0, -1.0])
Sp = np.sqrt(2) * np.diag([1.0, 1.0], k=1)
site = qudit(3, {"Sz": Sz, "Sp": Sp, "Sm": Sp.T})
N = 8
terms = OpSum()
for i in range(N - 1):
    terms += (0.5, "Sp", i, "Sm", i + 1)
    terms += (0.5, "Sm", i, "Sp", i + 1)
    terms += (1.0, "Sz", i, "Sz", i + 1)
W, _ = compile_mpo(terms, [site] * N, dtype="auto")
spin1 = SiteSpace(Space(SU2(), (2,), (1,)), np.eye(3))
Ws, _ = symmetric_mpo(W, [spin1] * N)
E0, gs, _ = dmrg(Ws, 32, charge=0)           # -10.1246372224, as exact diagonalisation gives
```

`tnkit.models_lgt2f.site_spaces_2f` gives the sites of the two-flavour Schwinger model under
U(1) $\times$ SU(2).

## Other symmetries

`U1()`, `Zn(n)` and `SU2()` from `tnkit.symmetry`, and any `ProductSymmetry` of them, such as
`ProductSymmetry((U1(), SU2()))`, work on symmetric tensors. Another group needs a subclass of
`tnkit.symmetry.Symmetry`. It sets the attribute `fusion` and implements:

- for every symmetry: `unit()` (the trivial sector), `dual(a)`, `label(a)` (the canonical form
  of a sector label, raising for a label that is not a sector) and, optionally, `sort_key(a)`
  (the order of sectors along a leg);
- for `fusion = "unique"` (abelian): `fuse(a, b)`, the one sector in $a \otimes b$;
- for `fusion = "simple"` (several outcomes, each once, as for SU(2)): `fusions(a, b)`, the
  sectors in $a \otimes b$ with multiplicity 1; `dim(a)`; the Clebsch–Gordan coefficients
  `cg(a, b, c)`; the recoupling coefficients `F(a, b, c, d, e, f)` (6j symbols for SU(2)); the
  exchange coefficients `R(a, b, c)`; and `frobenius_schur(a)`, the sign of a self-dual
  sector ($-1$ for half-integer spins);
- for `fusion = "generic"` (an outcome may appear more than once): the same methods, with a
  multiplicity index on each fusion vertex.

The module docstring of `tnkit.symmetry` states the conventions these coefficients follow.

## Saving states

`save_mps(file, mps)` and `load_mps(file, sym, be=None)` in `tnkit.symmetric_backend` write a
symmetric MPS to one `.npz` file and read it back onto any dense backend.

## Limits

- `trunc="svd"` only, and no `eig="lanczos-mixed"`.
- No `fast_expm` and no CUDA graphs in TDVP.
- `dmrg_block` and `dmrg_tower(pair_solve=True)` take dense MPOs only.
- Thermal states (`tnkit.thermal`) take dense or block tensors only.

## GPUs

Pass `backend="torch-cuda"` to `dmrg` or `dmrg_excited`, or
`SymmetricBackend(get_backend("torch-cuda"))` to `tdvp2_evolve`. The reduced blocks, the
contractions, the factorisations and the maps between block layouts stay on the device; the
small recoupling matrices are computed on the host and cached. The cache limits are set by the
`TNKIT_SYMMETRIC_*` variables in {doc}`../environment`.

The low-level tensors are described in {doc}`tensors`, and the full reference is
{doc}`../../api/symmetric`.

# MPS and MPO conventions

Every solver in tnkit takes its state and Hamiltonian as plain lists of arrays, so a model or
state built by hand must follow the same leg order as tnkit's own. This page states those
orders. A *leg* is one index of a tensor; its *dimension* is the number of values the index
takes.

## States

An MPS is a list of rank-3 tensors, one per site. tnkit stores site $i$ as

$$\mps{A}{s}{a}{b} \quad\text{with legs in the order}\quad (a, s, b),$$

the left bond, the physical leg and the right bond, so that

$$\ket{\psi} = \sum_{s_1 \dots s_N} A^{s_1}_{[1]} A^{s_2}_{[2]} \cdots A^{s_N}_{[N]}
\ket{s_1 s_2 \dots s_N},$$

where $A^{s}_{[i]}$ is the tensor of site $i$ at physical index $s$, a matrix over $(a, b)$.
Neighbouring bonds match, `mps[i].shape[2] == mps[i + 1].shape[0]`, and the two outer bonds
have dimension 1. The formulas number the sites from 1 to $N$; code numbers them from 0, as
Python numbers the list, so site $i$ of a formula is `mps[i - 1]`. Site numbers passed to
tnkit functions, such as those in an `OpSum`, count from 0.

The solvers return an {py:class}`~tnkit.MPS`, a subclass of `list`, so code that treats a
state as a list of arrays keeps working. It adds:

- `mps.entropies()`: the entanglement entropy at every bond, from one sweep;
- `mps.to_numpy()`: a copy with every tensor on the host as a numpy array;
- `mps.center`: the site of the orthogonality centre when it is known, else `None`.

Changing the list (assigning a tensor, appending) resets `center` to `None`.

## Operators

An MPO tensor has four legs in a different order,

$$\mpo{W}{s}{s'}{a}{b} \quad\text{with legs in the order}\quad (a, b, s, s'),$$

the left bond, the right bond, and two physical legs. The tensor acts on a site as

$$W_{ab} = \sum_{s s'} W^{s s'}_{ab} \ket{s}\!\bra{s'} ,$$

so $s$ is the row (output) index and $s'$ the column (input) index: for a one-site operator
$O$ placed in an MPO, `W[a, b] = O`, the matrix with entries $\bra{s} O \ket{s'}$. The operator
of the whole chain is the matrix product of the $W_{ab}$ over the bonds, and the first tensor's
left bond and the last tensor's right bond have dimension 1.

An MPO tensor is therefore *not* an MPS tensor with one more leg appended; mixing the two
orders is the most common mistake when building models by hand. Any list of arrays in the
$(a, b, s, s')$ order with outer bonds of dimension 1 is a valid MPO.

tnkit ships MPOs for the transverse-field Ising model ({py:func}`~tnkit.tfim_mpo`,
{py:func}`~tnkit.tfim_ring_mpo`) and the lattice Schwinger model
({py:func}`~tnkit.schwinger_mpo`). For other models, see {doc}`opsum`.

## Boundary environments

The solvers contract the tensors to the left and to the right of the sites they update into
*environments*, rank-3 tensors with legs in the order (bra bond, MPO bond, ket bond). On an
open chain they start from the dimension-1 outer bonds and you never see them. The TDVP
functions and `tnkit.window` also take fixed boundary environments, `L0` and `RN` in
{py:class}`~tnkit.TDVPOptions`, which describe a finite window embedded in an infinite chain;
`dmrg` does not take them. {doc}`../development` gives the contraction that builds an
environment.

# Symmetries

```{warning}
Provisional. The symmetry, block-tensor and symmetric-tensor API may change before 1.0.
```

A Hamiltonian that conserves a quantity, such as particle number, total $S^z$, parity or
total spin, never mixes states in which that quantity differs. Every state then has a
definite value of it, its *sector* (for an abelian symmetry, also called its *charge*), and
the Hamiltonian is block diagonal in the sectors. The same holds for each tensor of an MPS
and an MPO once every leg is split into sectors: most blocks of every tensor are zero by
symmetry. A symmetric DMRG or TDVP stores and multiplies only the allowed blocks, which saves
memory and time, and it can target one sector directly.

Nothing is symmetric unless you ask for it: `dmrg(W, chi)` on an ordinary MPO stays dense.

## Abelian and non-abelian symmetries

For U(1) (a conserved integer, such as particle number or $2S^z$) and $\mathbb{Z}_n$ (an
integer conserved modulo $n$, such as parity), two sectors combine into exactly one sector:
charges add. Such symmetries are *abelian*. A tensor leg is then a list of sectors, each with
a number of states, and a block of a tensor is allowed when the charges of its legs add up to
the tensor's total charge $q$:

$$T_{s_1 s_2 \dots} \neq 0 \quad\text{only if}\quad \sum_n \epsilon_n\, q(s_n) = q ,$$

where $q(s_n)$ is the charge of the sector that index $s_n$ of leg $n$ lies in, and
$\epsilon_n = \pm 1$ is the sign of that leg: a leg that is contracted with another carries the opposite sign of
its partner. For $\mathbb{Z}_n$ the sum is taken modulo $n$.

SU(2), the symmetry of the total spin, is *non-abelian*: two spins combine into several total
spins (spin 1/2 and spin 1/2 into spin 0 and spin 1), not one. Its sectors are the total
spins $j$, and each is a multiplet of $2j + 1$ states.

## Block tensors and symmetric tensors

tnkit has two kinds of symmetric tensor:

- **Block tensors** (`tnkit.blocks`) store one dense block for each allowed combination of
  sectors of their legs. They take abelian symmetries only: `U1`, `Zn` and products of them.
- **Symmetric tensors** (`tnkit.symmetric_tensor`) store, by the Wigner–Eckart theorem, only
  the part of a tensor that the symmetry does not fix. They take any symmetry that
  `tnkit.symmetry` describes, non-abelian ones such as SU(2) and U(1) $\times$ SU(2) included.

We recommend block tensors for abelian symmetries and symmetric tensors for non-abelian
ones. Block tensors are exported from `tnkit` and run in more of the library, thermal states
among it. Symmetric tensors are needed as
soon as the symmetry group is not abelian.

| | dense | block tensors | symmetric tensors |
|---|---|---|---|
| `dmrg` (two-site and `sites=1`), `dmrg_excited` | yes | yes | yes |
| `tdvp2_evolve`, `tdvp1_evolve` | yes | yes | yes |
| thermal states (`tnkit.thermal`) | yes | yes | no |
| `dmrg_block`, `dmrg_tower(pair_solve=True)` | yes | no | no |
| `idmrg`, `vumps`, `tebd_evolve`, `evolve_batch` | yes | no | no |
| GPU backends | yes | yes | yes |

## Pages

- {doc}`abelian`: DMRG and TDVP with U(1), $\mathbb{Z}_n$ and their products.
- {doc}`non-abelian`: DMRG and TDVP with SU(2), and how to read the states it returns.
- {doc}`tensors`: the tensors themselves, for writing your own algorithms.

```{toctree}
:hidden:

abelian
non-abelian
tensors
```

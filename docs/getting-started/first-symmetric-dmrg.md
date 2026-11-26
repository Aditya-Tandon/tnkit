# A first symmetric DMRG

When a Hamiltonian conserves a quantity, such as the total magnetisation $S^z$, its
eigenstates can be found one sector of that quantity at a time. DMRG that conserves the
quantity is cheaper, because most blocks of every tensor are zero and are never stored, and it
targets a sector directly: the lowest state with total $S^z = 1$ is then an ordinary ground
state. This page finds the lowest states of two sectors of the spin-1/2 XXZ chain,

$$H = \sum_i \left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1} \right)
= \sum_i \left[ \tfrac12 \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right)
+ \tfrac{\Delta}{4}\, \sigma^z_i \sigma^z_{i+1} \right],$$

which conserves the total $S^z = \tfrac12 \sum_i \sigma^z_i$. Here $S^\pm = S^x \pm i S^y$
raise and lower one spin, and $\sigma^z = 2 S^z$ is the Pauli matrix.

```python
from tnkit import dmrg, get_backend, U1, to_block_mpo
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

The example has three steps:

1. `OpSum` and `compile_mpo` build the MPO from a list of terms; `spin_half()` names the
   operators of one site, with `"Z"` the Pauli matrix and `"Sp"`, `"Sm"` the operators
   $S^\pm$. See {doc}`../user-guide/opsum`.
2. {py:func}`~tnkit.to_block_mpo` converts the MPO to block tensors. It needs the charge of
   every basis state of every site, here $2S^z$ so that charges are integers: $+1$ for up and
   $-1$ for down. `U1()` says that charges add.
3. `dmrg(Wb, chi, charge=q)` finds the lowest state whose charges sum to `q`. Charge 0 is
   total $S^z = 0$, and charge 2 is total $S^z = 1$.

The energy of charge 0 equals that of DMRG without symmetry on `W`, since the ground state
of this chain has $S^z = 0$.

Write hopping terms with $S^\pm$. Written as $XX + YY$, the MPO has channels that carry two
charges, and `to_block_mpo` raises.

The Heisenberg chain ($\Delta = 1$) also conserves the total spin, an SU(2) symmetry that is
not abelian. It needs symmetric tensors instead of block tensors; see
{doc}`../user-guide/symmetries/non-abelian`. For which symmetry type to choose, see
{doc}`../user-guide/symmetries/index`.

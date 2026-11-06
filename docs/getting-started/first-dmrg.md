# A first DMRG

DMRG finds the ground state of a Hamiltonian given as a matrix-product operator (MPO). Here
the Hamiltonian is the transverse-field Ising model (TFIM) on an open chain of $N$ sites,

$$H = -\sum_i \sigma^x_i \sigma^x_{i+1} - g \sum_i \sigma^z_i ,$$

at its critical point $g = 1$.

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, structure_factor, SX

N = 10
mpo = tfim_mpo(N, g=1.0)
E, mps, info = dmrg(mpo, chi=32, n_sweeps=10)

print(E - tfim_ground_energy(N, g=1.0))   # about 1e-14: exact for this small chain
S = mps.entropies()                       # entanglement entropy at every bond
m2, Sq = structure_factor(mps, SX)        # squared order parameter
```

`chi` is the bond dimension: the largest matrix size the MPS may use. Larger `chi` is more
accurate and more expensive. `dmrg` returns:

- `E`, the energy;
- `mps`, the state, as an {py:class}`~tnkit.MPS` (a list of tensors, one per site);
- `info`, a dict with the per-sweep energies, the number of sweeps, the discarded weight and
  timings.

The run stops before `n_sweeps` once the energy changes between sweeps by less than
$\max(\texttt{tol}, \texttt{rtol}\,|E|)$; both default to `1e-10`. Here that took 2 sweeps.
`tol=0, rtol=0` runs every sweep.

`tfim_ground_energy` diagonalises the full Hamiltonian, so it only works up to about 12 sites.
It is there to check results.

## Grouping settings

Every keyword of `dmrg` is also a field of {py:class}`~tnkit.DMRGOptions`. An options object
is convenient for scans:

```python
import dataclasses
from tnkit import DMRGOptions

base = DMRGOptions(n_sweeps=10)
for chi in (4, 8, 16):
    E, mps, info = dmrg(mpo, chi, base)
```

The two styles run the same code; `dataclasses.replace(base, ...)` changes one field. See
{doc}`../user-guide/dmrg` for the options that matter.

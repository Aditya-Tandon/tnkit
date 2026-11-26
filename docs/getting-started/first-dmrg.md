# A first DMRG

DMRG (the density-matrix renormalisation group) finds the ground state of a Hamiltonian
written as a matrix-product operator (MPO). This page finds the ground state of the
transverse-field Ising model (TFIM) on an open chain of $N$ sites,

$$H = -\sum_{i} \sigma^x_i \sigma^x_{i+1} - g \sum_i \sigma^z_i ,$$

at its critical point $g = 1$. Here $\sigma^x$ and $\sigma^z$ are the Pauli matrices (`SX` and
`SZ` in tnkit), and the first sum runs over the $N - 1$ pairs of neighbouring sites.

```python
from tnkit import dmrg, tfim_mpo, tfim_ground_energy, structure_factor, SX

N = 10
mpo = tfim_mpo(N, g=1.0)
E, mps, info = dmrg(mpo, chi=32, n_sweeps=10)

print(E - tfim_ground_energy(N, g=1.0))   # 0.0: exact for this small chain, up to rounding
S = mps.entropies()                       # entanglement entropy at each of the N - 1 bonds
m2, xx_sum = structure_factor(mps, SX)    # M^2, and N^2 M^2 = sum_ij <X_i X_j>
```

`chi` is the bond dimension: the largest number of states the MPS keeps on any bond between
two sites. A larger `chi` is more accurate and more expensive. `dmrg` returns:

- `E`, the energy;
- `mps`, the state, as an {py:class}`~tnkit.MPS` (a list of tensors, one per site);
- `info`, a dict with the energy of every sweep, the number of sweeps, the discarded weight
  and timings (see {doc}`../user-guide/dmrg`).

`mps.entropies()` returns the entanglement entropy $S = -\sum_r \lambda_r^2 \ln \lambda_r^2$
at each bond, where the $\lambda_r$ are the Schmidt values of the cut there.
{py:func}`~tnkit.structure_factor` returns two numbers: the squared order parameter
$M^2 = N^{-2} \sum_{i,i'} \expval{\sigma^x_i \sigma^x_{i'}}$, summed over all pairs of
sites, and the sum itself, $\sum_{i,i'} \expval{\sigma^x_i \sigma^x_{i'}} = N^2 M^2$.

A sweep optimises every site once from left to right and once back. The run stops before
`n_sweeps` once the energy changes between sweeps by less than
$\max(\texttt{tol}, \texttt{rtol}\,|E|)$; both default to `1e-10`. Here that took 2 sweeps.
`tol=0, rtol=0` runs every sweep.

`tfim_ground_energy` diagonalises the full Hamiltonian, so it only works up to about 12 sites.
It is there to check results.

## Grouping settings

Every keyword of `dmrg` is also a field of {py:class}`~tnkit.DMRGOptions`. An options object
is convenient for scans, and `dataclasses.replace` makes a copy with one field changed:

```python
import dataclasses
from tnkit import DMRGOptions

base = DMRGOptions(n_sweeps=10)
for chi in (4, 8, 16):
    E, mps, info = dmrg(mpo, chi, base)

short = dataclasses.replace(base, n_sweeps=3)   # the same settings, at most 3 sweeps
E, mps, info = dmrg(mpo, 16, short)
```

The two styles run the same code. See {doc}`../user-guide/dmrg` for the options that matter.

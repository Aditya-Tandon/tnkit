# A first TDVP

TDVP (the time-dependent variational principle) evolves an MPS in time. This example
prepares the ground state at $g = 1.5$, then evolves it under the Hamiltonian at $g = 1$ (a
*quench*) and records the mean $\expval{\sigma^z}$.

```python
from tnkit import dmrg, tdvp2_evolve, tfim_mpo, local_expectations, SZ

N = 16
_, psi0, _ = dmrg(tfim_mpo(N, g=1.5), chi=16, n_sweeps=6)

psi, records = tdvp2_evolve(
    psi0, tfim_mpo(N, g=1.0), chi=32, dt=0.05, n_steps=40,
    measure=lambda m: local_expectations(m, SZ).mean(), meas_every=10,
)
for t, mz in records:
    print(f"t = {t:.2f}   <sz> = {mz:.6f}")
```

`records` is a list of `(time, value)` pairs, one every `meas_every` steps, where `value` is
whatever `measure` returns.

There are two integrators:

- {py:func}`~tnkit.tdvp2_evolve` (two-site) lets the bond dimension grow up to `chi`, and
  truncates when it would exceed it.
- {py:func}`~tnkit.tdvp1_evolve` (one-site) keeps the bond dimension fixed. It never
  truncates and conserves energy, but it cannot add entanglement.

A common pattern is to start with the two-site integrator and switch to the one-site one once
the bond dimension has reached its cap:

```python
from tnkit import tdvp1_evolve

psi, more = tdvp1_evolve(psi, tfim_mpo(N, g=1.0), dt=0.05, n_steps=20,
                         measure=lambda m: local_expectations(m, SZ).mean(), meas_every=10)
```

The times in `more` start again at 0; add the time already evolved (here 2.0) to place them
after `records`.

See {doc}`../user-guide/tdvp` for the options.

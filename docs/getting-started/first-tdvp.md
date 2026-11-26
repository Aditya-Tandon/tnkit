# A first TDVP

TDVP (the time-dependent variational principle) evolves an MPS in real time. This example
prepares the ground state of the transverse-field Ising model at $g = 1.5$, then evolves it
under the Hamiltonian at $g = 1$ (a *quench*: a sudden change of the Hamiltonian) and records
the mean magnetisation $N^{-1} \sum_i \expval{\sigma^z_i}$, printed as `<Z>`.

```python
from tnkit import dmrg, tdvp2_evolve, tfim_mpo, local_expectations, SZ

N = 16
_, psi0, _ = dmrg(tfim_mpo(N, g=1.5), chi=16, n_sweeps=6)

psi, records = tdvp2_evolve(
    psi0, tfim_mpo(N, g=1.0), chi=32, dt=0.05, n_steps=40,
    measure=lambda m: local_expectations(m, SZ).mean(), meas_every=10,
)
for t, mz in records:
    print(f"t = {t:.2f}   <Z> = {mz:.6f}")
```

```text
t = 0.00   <Z> = 0.886833
t = 0.50   <Z> = 0.714471
t = 1.00   <Z> = 0.669142
t = 1.50   <Z> = 0.650382
t = 2.00   <Z> = 0.652075
```

States evolve as $\ket{\psi(t)} = e^{-iHt} \ket{\psi(0)}$ with $\hbar = 1$, so the time step
`dt` is in units of the inverse of the energy unit of the MPO. `records` is a list of
`(time, value)` pairs, one at the start and one every `meas_every` steps, where `value` is
whatever `measure` returns.

There are two integrators:

- {py:func}`~tnkit.tdvp2_evolve` (two-site) lets the bond dimension grow up to `chi`, and
  truncates when it would exceed it.
- {py:func}`~tnkit.tdvp1_evolve` (one-site) keeps every bond dimension fixed. It never
  truncates and conserves the energy of a time-independent $H$, up to the Krylov tolerance
  (the accuracy of the matrix exponential computed at each step), but it cannot add
  entanglement.

A common pattern is to start with the two-site integrator and switch to the one-site one once
the bond dimension has reached `chi`:

```python
from tnkit import tdvp1_evolve

psi, more = tdvp1_evolve(psi, tfim_mpo(N, g=1.0), dt=0.05, n_steps=20,
                         measure=lambda m: local_expectations(m, SZ).mean(), meas_every=10)
```

Every call starts its clock at 0, so the times in `more` are measured from the end of the
first run; add the time already evolved (here $40 \times 0.05 = 2.0$) to place them after
`records`. The first record of `more`, at time 0, is the state the first run ended with, so it
repeats the last record of `records`.

See {doc}`../user-guide/tdvp` for the options.

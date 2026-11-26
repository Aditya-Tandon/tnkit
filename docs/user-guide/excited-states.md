# Excited states

{py:func}`~tnkit.dmrg_excited` finds the states above the ground state, one at a time: each
run is a DMRG that is kept away from the states already found. Use it for gaps and low-lying
spectra of finite chains.

```python
from tnkit import dmrg, dmrg_excited, tfim_mpo

W = tfim_mpo(10, g=1.5)
E0, gs, _ = dmrg(W, 32)
E1, ex1, info = dmrg_excited(W, 32, prev_states=[gs], penalty=10.0)
E2, ex2, _ = dmrg_excited(W, 32, prev_states=[gs, ex1], penalty=10.0)
print(f"{E0:.8f} {E1:.8f} {E2:.8f}")   # -16.53525495 -15.36801737 -14.94368177
```

The three energies are the lowest three eigenvalues of this chain. `dmrg_excited` takes the
MPO, the bond dimension and the states to avoid, `prev_states`, and returns
`(E, mps, info)` as `dmrg` does. `E` is the energy $\bra{\psi} H \ket{\psi}$ of the state
found, so it can be compared across states.

## Deflation

`deflate="penalty"` (the default) minimises the energy of

$$H + w \sum_k \ket{\psi_k}\!\bra{\psi_k} ,$$

where the $\ket{\psi_k}$ are the states in `prev_states` and $w$ is `penalty` (default 1).
This lifts each previous state by $w$. The penalty must therefore be larger than the gap
between the state sought and the previous states; otherwise the run finds a previous state
again. In the example, the gap $E_1 - E_0$ is about 1.17, so the default penalty of 1 returns
the ground state a second time. `info["penalty_overlaps"]` lists $|\braket{\psi_k}{\psi}|$ for
every previous state, and an overlap near 1 shows this failure.

`deflate="project"` instead removes the previous states from every two-site update, so it
needs no estimate of the gap; `penalty` then only matters for the updates where the
projection leaves too little room, which fall back to the penalty method
(`info["project_fallbacks"]` counts them).

To check that a state is an eigenstate, compute its energy variance with
{py:func}`~tnkit.mpo_variance`; it returns the variance and the round-off level below which
the variance is not resolved.

## Other settings and output

The settings of `dmrg_excited` are the fields of {py:class}`~tnkit.DMRGExcitedOptions`, and
most have the meaning they have in `dmrg`: `n_sweeps` (default 12), `tol`, `eig_tol`, `seed`,
`init_mps`, `backend` and `charge`. Two are specific to it:

`time_budget_s`
: a wall-clock limit in seconds; no sweep starts that would end after it, and a run stopped
  this way has `info["cut"]`.

`start_noise`
: a random vector of this relative norm added to each local start vector of the first sweep,
  for levels the start has no weight on.

`info` reports, besides the energies and timings, `penalty_overlaps`, `converged` (whether
the last sweep changed the penalised energy by less than `tol`) and the largest bond
(`max_bond`).

## Symmetric tensors

With a block or symmetric MPO, `dmrg_excited` runs in one sector, as `dmrg` does: pass
`charge=`, and give `prev_states` from the same sector. A state in another sector is never
found by DMRG in this one, so it needs no deflation; the lowest state of a sector is an
ordinary `dmrg` run with that `charge`. Subspace expansion is not used. See
{doc}`symmetries/index`.

## Several levels

The module `tnkit.dmrg_excited` has two drivers for more than a few levels (see their
docstrings):

- `dmrg_tower(W, chi, nlevels)` finds the lowest `nlevels` states with `dmrg_excited` in turn,
  and raises the penalty to three times the spread of the levels found so far, which avoids
  the failure described above;
- `dmrg_block(W, chi, nstates)` finds `nstates` states together, in one shared basis. Inside a
  group of nearly degenerate levels this is better posed than one state at a time, but the
  shared basis needs a larger `chi` for the same accuracy.

`dmrg_block` and `dmrg_tower(..., pair_solve=True)` take dense MPOs only; `dmrg_tower` without
`pair_solve` also takes block and symmetric MPOs, with `charge=`.

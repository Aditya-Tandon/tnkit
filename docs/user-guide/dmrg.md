# DMRG

{py:func}`~tnkit.dmrg` finds the ground state of a finite open chain. It sweeps over the
chain, optimising two neighbouring sites at a time by default, until the energy stops
changing. It takes the MPO, the bond dimension `chi`, and either keyword arguments or a
{py:class}`~tnkit.DMRGOptions`:

```python
from tnkit import dmrg, tfim_mpo

E, mps, info = dmrg(tfim_mpo(32, g=1.0), chi=64, n_sweeps=10)
```

The defaults are a sound start; this page lists the options worth knowing and what `info`
reports.

## Stopping

`n_sweeps`
: the most sweeps to run (one sweep is left to right and back). Default 8.

`tol`, `rtol`
: the run stops once the energy changes between sweeps by less than
  $\max(\texttt{tol}, \texttt{rtol}\,|E|)$, the looser of the two. Both default to `1e-10`.
  A `tol` below `rtol * |E|` therefore has no effect; for a tighter stop set `rtol=0` too.
  `tol=0, rtol=0` runs all `n_sweeps`. The relative stop is there because at a truncated
  `chi` the sweep energy keeps moving by about the truncation error, so a small absolute
  `tol` alone may never be met.

`eig_tol`, `eig_stop`
: the tolerance of the local eigensolver and what it is tested against. The default,
  `eig_stop="residual"`, stops on the residual $\lVert Hx - ex \rVert$, so the state itself
  is converged. `eig_stop="energy"` stops when the local energy stops changing; because the
  energy converges quadratically faster than the state, this leaves the state only about
  $\sqrt{\texttt{eig\_tol}}$ converged, and nothing in the output shows it. Keep the default
  unless you are reproducing an old result.

## Sweeps and subspace expansion

`sites`
: `2` (default) optimises two sites at a time. `1` optimises one site, which is cheaper per
  step but cannot change a bond's size or charge sectors on its own; use it with `expand`.

`expand`
: the number of opening sweeps that use subspace expansion. Expansion adds to each bond the
  states the Hamiltonian couples to (Hubig et al., PRB 91, 155115 (2015)), so the run can leave
  a poor starting basis and fill empty charge sectors. The default, `None`, means `0` (off) on
  a dense MPO and `max(0, min(4, n_sweeps - 2))` on a block or symmetric MPO; `info["expand"]`
  reports the value used. It must be less than `n_sweeps`, and a run with expansion does at least `expand + 2`
  sweeps. Expanding sweeps truncate with an exact SVD whatever `trunc` says.

`expand_alpha`, `expand_decay`
: the weight of the expansion term in sweep `k` is `expand_alpha * expand_decay**k`
  (defaults `3e-3` and `0.3`), and `0` once expansion ends, so the final sweeps are unbiased.

Expansion can move a run to another fixed point, and not always a lower one: at small `chi` it
can end higher than a run without it.

## Local eigensolver

Each step of a sweep finds the lowest eigenvector of the effective Hamiltonian of two sites
with a Krylov method: it builds the space spanned by $v, Hv, H^2v, \dots$ from a start vector
$v$ (the Krylov space) and solves the small eigenproblem there.

`eig`
: `"lanczos-restart"` (default, thick-restart Lanczos), `"lanczos"`, `"lanczos-mixed"`
  (Lanczos with single-precision matrix-vector products, refined in double precision),
  `"lobpcg"`, or `"scipy-host"` (scipy's `eigsh` on the host; a baseline for comparison, not
  for production).

`max_iter`
: the largest dimension of the Krylov space. The default, `None`, grows with `chi`, because a
  fixed cap starves the local solve at large bond dimension.

## Truncation and device

`trunc`
: how each two-site block is split: `"svd"` (default), `"randomized"`, `"randomized-mixed"`,
  `"cutn"` or `"gram"`. `"density"` is deprecated and warns; use `"svd"`.

  `"gram"` is opt-in, for speed on GPUs: it replaces the SVD by a matrix product and an
  eigendecomposition of the smaller Gram matrix $\Theta^\dagger \Theta$ of the two-site block
  $\Theta$. `gram` is faster on GPUs, but
  its rounding differs from `svd`, so its results are not reproducible against `svd` runs.
  It does not resolve singular values below about $10^{-8}$ of the largest, so do not read
  small entanglement spectra from it. Use it only in double precision, for energies, local
  observables and entropies; single precision and block tensors refuse it.

`backend`
: see {doc}`../getting-started/backends`.

`init_mps`, `seed`
: a starting state, or the seed of the random one.

`charge`
: block and symmetric MPOs only: the sector of the state sought, its total charge. See
  {doc}`symmetries/index`.

## What `info` contains

`sweeps`
: the number of sweeps run.

`energies`, `sweep_s`
: the energy and the wall time of every sweep.

`backend`, `blocks`
: the backend used, and whether the tensors were block or symmetric tensors.

`eig`
: the local eigensolver used.

`discarded_weight`
: the summed discarded weight of all truncations.

`matvec_calls`, `phase_time`
: the number of local matrix-vector products, and a dict of the seconds spent in the
  eigensolver, the truncations and the environment updates (`eig`, `trunc`, `env`).

`mixed_counts`
: with `eig="lanczos-mixed"`, the number of matrix-vector products done in single
  (`lo`) and in double (`hi`) precision; both are 0 for the other eigensolvers.

`sites`, `expand`, `expand_alphas`
: the sweep type, the number of expanding sweeps, and the expansion weight of every sweep
  (`0` where none).

## Beyond the ground state

- {doc}`excited-states` finds the states above the ground state.
- {doc}`infinite-chains` works directly in the thermodynamic limit.
- {py:func}`~tnkit.mpo_variance` returns the energy variance $\expval{H^2} - \expval{H}^2$,
  with the round-off floor below which it is not resolved. It measures how close a state is
  to an eigenstate: some eigenvalue of $H$ lies within the square root of the variance of
  $\expval{H}$.

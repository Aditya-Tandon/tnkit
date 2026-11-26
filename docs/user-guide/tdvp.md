# Time evolution

tnkit evolves an MPS in real time with TDVP (the time-dependent variational principle) for
any MPO, and with TEBD (time-evolving block decimation) for nearest-neighbour Hamiltonians.
The two TDVP functions are

```text
tdvp2_evolve(mps, mpo, chi, dt, n_steps, ...)   # two-site: bonds grow up to chi
tdvp1_evolve(mps, mpo, dt, n_steps, ...)        # one-site: bonds stay as they are
```

where `...` stands for keyword arguments or a {py:class}`~tnkit.TDVPOptions`. Both return
`(state, records)`: the evolved {py:class}`~tnkit.MPS`, and the list of `(time, value)` pairs
that `measure` produced. {doc}`../getting-started/first-tdvp` has a complete example.

## Time convention

States evolve as

$$\ket{\psi(t)} = e^{-iHt} \ket{\psi(0)} \qquad (\hbar = 1),$$

so `dt` is in units of the inverse of the energy unit of the MPO: for an MPO with couplings
of order 1, `dt = 0.05` is a twentieth of the basic time scale. Each call evolves for
`n_steps * dt` and starts its record times at 0.

## Choosing an integrator

{py:func}`~tnkit.tdvp2_evolve` (two-site)
: lets bonds grow up to `chi` (or `chi_max`) and truncates beyond. Use it while entanglement is
  growing.

{py:func}`~tnkit.tdvp1_evolve` (one-site)
: keeps every bond dimension fixed. It never truncates and conserves the energy of a
  time-independent $H$, up to the Krylov tolerance, but it cannot add entanglement. Use it once
  the bonds have reached their cap.

{py:func}`~tnkit.tebd_evolve`
: second-order Trotter evolution for nearest-neighbour Hamiltonians; see [TEBD](#tebd).

{py:func}`~tnkit.evolve_batch`
: TDVP of several states of the same geometry at once, as one batched program on one device;
  see [Batched TDVP](#batched-tdvp).

## What a step does

One TDVP step sweeps once from left to right and once back, evolving each site (or each pair
of sites, for the two-site integrator) for half the step on each pass. Each update applies
the exponential of a local effective Hamiltonian $H_{\text{eff}}$ to the site tensor $v$. It
does so with a Krylov method: from the vectors $v, H_{\text{eff}}v, H_{\text{eff}}^2v, \dots$
it builds an orthonormal basis of a few vectors (the Krylov space), and computes
$e^{-iH_{\text{eff}}\,dt/2}\,v$ within it. The number of vectors is the *Krylov depth*. By
default each update adds vectors until its error estimate is below the tolerance (the Krylov
tolerance), up to `krylov_m` (default 30).

## Common options

`measure`, `meas_every`
: `measure` is called every `meas_every` steps, and at the start, on a host copy of the
  state; its return values are collected in `records` as `(time, value)` pairs.
  `keep_device=True` hands it the tensors on the device instead.

`backend`
: as for DMRG; the default is `"numpy"`. Time evolution needs a complex backend.

`fast_expm`
: fix the Krylov depth of every update instead of choosing it adaptively. On a GPU this
  removes the waits for the host that the adaptive choice needs. An integer fixes the depth;
  `"auto"` runs the first sweep adaptively, fixes the depth 2 above the largest it needed, and
  measures it again every 25 steps; a dict does the same and stores the depth in it, under
  `"m"`, for the next call (`"recal"` sets the interval, 0 never measures again). It cannot be
  combined with `adaptive_dt`.

`adaptive_dt`
: `None` (fixed steps) or a dict that configures an adaptive step-size controller.

`L0`, `RN`
: fixed boundary environments, for a finite window embedded in an infinite chain (see
  {doc}`mps-mpo`). The default, `None`, is an open chain.

## Time-dependent Hamiltonians

Both integrators accept a function `t -> mpo` in place of the MPO. It is evaluated at the
midpoint of each step, which keeps the integrator second order, at the cost of rebuilding the
environments once per step. Use it for ramps of a coupling.

## Two-site only

`chi_max`
: hard cap on any bond (default `chi`). Set it when the input state already has bonds larger
  than `chi`.

`svd_tol`
: relative discarded-weight budget per bond. `0` (default) always keeps `chi_max` states; a
  positive value lets the bond dimension adapt.

`trunc`
: as for DMRG; `"svd"` is recommended.

## One-site only

`graph`
: `"auto"` (default), `"on"` or `"off"`: whether to replay each site update from a CUDA graph.
  See {doc}`gpu`.

## Block and symmetric tensors

Both integrators also evolve a block MPS under a block MPO, conserving the charges, when
`backend` is a {py:class}`~tnkit.BlockBackend` around a complex dense backend, for example
`BlockBackend(get_backend("numpy"))`, and a symmetric MPS under a symmetric MPO with a
{py:class}`~tnkit.symmetric_backend.SymmetricBackend`. Block runs truncate with `trunc="svd"` only, and take no
`fast_expm` and no CUDA graphs; `measure` receives the block tensors. See
{doc}`symmetries/abelian`; for SU(2) and other non-abelian symmetries, see
{doc}`symmetries/non-abelian`. One-site TDVP on block and symmetric tensors keeps the bond
sectors of the initial state, as it keeps the bond dimensions.

## TEBD

TEBD splits $e^{-iH\,dt}$ into gates on pairs of neighbouring sites, applied in a symmetric
(second-order) order, so its error per unit time is of order $dt^2$:

```python
from tnkit import dmrg, tebd_evolve, tfim_mpo, local_expectations, SZ

N = 12
_, psi0, _ = dmrg(tfim_mpo(N, g=1.5), chi=16)
psi, records = tebd_evolve(psi0, N, 1.0, 32, 0.05, 20, meas_every=10,
                           measure=lambda m: local_expectations(m, SZ).mean())
```

The arguments are `tebd_evolve(mps, N, g, chi, dt, n_steps, ...)`: the transverse-field Ising
model at field `g`, or, with `g=None` and `bond_h=`, any list of the $N - 1$ two-site terms
whose sum is $H$ (each of shape `(d*d, d*d)` or `(d, d, d, d)`). Split any one-site terms
between the bonds yourself. `imaginary=True` evolves in imaginary time.

## Batched TDVP

{py:func}`~tnkit.evolve_batch` evolves several states with the same number of sites and the
same physical dimension in one program, which keeps a GPU busy when each state alone is too
small to. A scan over quench couplings is the typical use:

```python
from tnkit import dmrg, evolve_batch, tfim_mpo, local_expectations, SZ

N = 12
_, psi0, _ = dmrg(tfim_mpo(N, g=1.5), chi=16)
hams = [tfim_mpo(N, g=g) for g in (0.5, 1.0, 2.0)]      # one quench per state
states, records = evolve_batch(
    [psi0] * 3, hams, 0.05, 20, meas_every=10,
    measure=lambda ms: [local_expectations(m, SZ).mean() for m in ms],
)
```

`measure` receives the list of states. The MPOs may differ in their couplings but not in
their bond structure. The default integrator is one-site TDVP (`evolver="1tdvp"`); `"2tdvp"`
truncates at `chi_max`. The settings are the fields of {py:class}`~tnkit.BatchTDVPOptions`.
A batch whose states cannot run as one program raises
{py:class}`~tnkit.BatchGeometryError`.

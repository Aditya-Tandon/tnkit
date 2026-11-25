# TDVP options

Both integrators take the state, the MPO, the time step `dt` and the number of steps (the
two-site one also takes the bond dimension `chi`, before `dt`), plus keyword arguments or a
{py:class}`~tnkit.TDVPOptions`. Each returns `(state, records)`; the record times start at 0
on every call.

## Choosing an integrator

{py:func}`~tnkit.tdvp2_evolve` (two-site)
: Lets bonds grow up to `chi` (or `chi_max`) and truncates beyond. Use it while entanglement is
  growing.

{py:func}`~tnkit.tdvp1_evolve` (one-site)
: Keeps every bond dimension fixed. No truncation, exact energy conservation, but it cannot add
  entanglement. Use it once the bonds have reached their cap.

{py:func}`~tnkit.tebd_evolve`
: Second-order Trotter evolution for nearest-neighbour Hamiltonians: the transverse-field
  Ising model at field `g`, or any list of two-site terms passed as `bond_h=` (with
  `g=None`). `imaginary=True` evolves in imaginary time.

{py:func}`~tnkit.evolve_batch`
: TDVP of many states with the same geometry at once, as one batched program on one device.

## Common options

`measure`, `meas_every`
: by default `measure` is called on a host copy of the state every `meas_every` steps;
  its return values are collected in `records` as `(time, value)` pairs.

`backend`
: as for DMRG; the default is `"numpy"`. Time evolution needs a complex backend.

`fast_expm`
: fix the Krylov depth of the time-step exponential instead of choosing it adaptively. It
  removes host synchronisation on the GPU. `"auto"` measures the depth on the first sweep.

`adaptive_dt`
: `None` (fixed steps) or a dict that configures an adaptive step-size controller.

Time-dependent Hamiltonians
: pass a function `t -> mpo` instead of an MPO. It is evaluated at the midpoint of each step.

## Two-site only

`chi_max`
: hard cap on any bond (default `chi`). Set it when the input state already has bonds larger
  than `chi`.

`svd_tol`
: relative discarded-weight budget per bond. `0` (default) always keeps `chi_max`; a positive
  value lets the rank adapt.

`trunc`
: as for DMRG; `"svd"` is recommended.

## One-site only

`graph`
: `"auto"` (default), `"on"` or `"off"`: whether to replay each site update from a CUDA graph.
  See {doc}`gpu`.

## Block tensors

Both integrators also evolve a block MPS under a block MPO, conserving the charges, when
`backend` is a {py:class}`~tnkit.BlockBackend` around a complex dense backend, for example
`BlockBackend(get_backend("numpy"))`. Block runs truncate with `trunc="svd"` only, and take no
`fast_expm` and no CUDA graphs; `measure` receives the block tensors. See {doc}`symmetries`.

## Thermal states

{py:mod}`tnkit.thermal` stores a Gibbs state as a pure state of the chain and one ancilla per
site, fused into sites of dimension `d*d`. {py:func}`~tnkit.thermal.thermal_state` cools the
infinite-temperature state in imaginary time with two-site TDVP and returns the state and
`log Z`; `charges=` restricts it to one charge sector. Expectation values use
{py:func}`~tnkit.thermal.purify` and {py:func}`~tnkit.thermal.purify_mpo`, and real-time
evolution uses either integrator on `purify_mpo(H, twin=True)`, which evolves the ancillas
backward so that the thermal state itself does not change:

```python
from tnkit import mpo_expectation, tdvp2_evolve
from tnkit.models_lgt import schwinger_charges, schwinger_mpo
from tnkit.thermal import purify_mpo, thermal_state

H = schwinger_mpo(24, x=4.0, mu=0.2)
rho, log_z = thermal_state(H, beta=0.5, chi=256, dbeta=0.01,
                           charges=schwinger_charges(24), svd_tol=1e-10)
E = mpo_expectation(rho, purify_mpo(H)).real
S = log_z + 0.5 * E                                   # entropy
rho_t, records = tdvp2_evolve(rho, purify_mpo(H, twin=True), 512, 0.025, 40)
```

With `blocks=True` the same state is held in block tensors under
`tnkit.thermal.PURIFIED`, U(1) x U(1): the physical and the ancilla charge are
conserved separately. The block MPOs come from `purify_mpo(H, twin=True, charges=q)`, the
integrators run with `backend=BlockBackend(get_backend("numpy"))`, a packet is applied with
{py:func}`~tnkit.block_backend.apply_block_mpo`, and observables take
`to_dense_mps(rho, purified_charges(q))`:

```python
from tnkit import SZ, BlockBackend, get_backend, local_expectations, to_dense_mps
from tnkit.thermal import purified_charges, purify

q = schwinger_charges(24)
rho, log_z = thermal_state(H, beta=0.5, chi=256, dbeta=0.01, charges=q, blocks=True,
                           svd_tol=1e-10)
rho_t, _ = tdvp2_evolve(rho, purify_mpo(H, twin=True, charges=q), 512, 0.025, 40,
                        backend=BlockBackend(get_backend("numpy")))
z = local_expectations(to_dense_mps(rho_t, purified_charges(q)), purify(SZ))
```

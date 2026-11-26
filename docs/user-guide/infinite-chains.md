# Infinite chains

A finite chain has edges, and its energy per site approaches the bulk value only as $1/N$.
For a translation-invariant chain, {py:func}`~tnkit.idmrg` and {py:func}`~tnkit.vumps` work
in the thermodynamic limit directly: they optimise the tensors of one repeating unit of an
infinite MPS and return the energy per site without edge effects.

```python
from tnkit import idmrg, vumps

res = idmrg(1.5, chi=16)                 # TFIM at g = 1.5: a dict
gs = vumps(1.5, chi=16)                  # the same model: a UniformMPS
print(f"{res['e0']:.10f} {gs.e0:.10f}")  # both -1.6719262215
```

The model is the transverse-field Ising model, $H = -\sum_i \sigma^x_i \sigma^x_{i+1} - g
\sum_i \sigma^z_i$, on the infinite chain. Its exact energy per site,
$e_0 = -\pi^{-1} \int_0^\pi \sqrt{1 + g^2 + 2g\cos k}\, dk$, is $-1.6719262215$ at $g = 1.5$.
Both methods converge slowly near the critical point $g = 1$, where the correlation length
diverges. There `idmrg` is also not variational: it shows the qualitative growth of the
entanglement with `chi`, but not clean values of the central charge $c$ or of the exponent
$\kappa$ in $\xi \propto \chi^\kappa$. Use `vumps` for those.

## Which models

Both functions solve the transverse-field Ising model when given its field `g`. Other models
are passed as a *bulk MPO tensor*: the one rank-4 tensor $W$ of shape `(D, D, d, d)`, in the
leg order of {doc}`mps-mpo`, that repeats on every site. Channel 0 of $W$ must start a term
and channel `D - 1` end one, as in the tensor that `tfim_bulk_mpo(g)` from `tnkit.idmrg`
returns.

{py:func}`~tnkit.idmrg`
: takes either `g` or a bulk tensor, `idmrg(W, chi)`, of any physical dimension `d`.

{py:func}`~tnkit.vumps`
: takes `g`, or a bulk tensor as `mpo=W` together with the operator that measures its energy.
  VUMPS reads the energy per site from a local expectation value, not from the MPO, so a
  custom `mpo` needs either `bond_h`, the two-site energy density of shape `(d, d, d, d)` with
  legs (first site out, second site out, first site in, second site in), or `triple_h`, a
  three-site one of shape `(d,) * 6` in the same order; without either, `vumps` raises
  `EnergyReadoutError`. The operator must assign each site's one-site terms to one bond only,
  for example the bond on its right. `vumps` handles spin-1/2 sites only ($d = 2$).

The mixed-field Ising chain, which adds $-h \sum_i \sigma^x_i$, has helpers for both:

```python
from tnkit.idmrg import mixed_field_bulk_mpo, mixed_field_bond_h

g, h = 1.5, 0.1
W = mixed_field_bulk_mpo(g, h)
res = idmrg(W, chi=16)
gs = vumps(g, chi=16, mpo=W, bond_h=mixed_field_bond_h(g, h))
print(f"{res['e0']:.10f} {gs.e0:.10f}")  # both -1.6948274546
```

## Results

`idmrg` returns a dict with `e0` (energy per site), `S` (entanglement entropy of a bond),
`svals` (its Schmidt values), `xi` (a correlation length from the transfer matrix of the left
tensor), `iters`, `e_hist` (the energy per site at each iteration), and the two tensors `A`
and `B` of its two-site unit cell, on the host.

`vumps` returns a {py:class}`~tnkit.UniformMPS`, a dict whose keys are also attributes:
`e0`, `S`, `svals`, `xi` and `xi3` (correlation lengths from the second and third largest
eigenvalues of the transfer matrix), `eps` (the norm of the energy gradient at the last
iteration, its convergence measure), `iters`, and the tensors of the uniform state in its
mixed canonical form, `AL`, `AR`, `AC` and `C`, on the run's backend. Its settings are the
fields of {py:class}`~tnkit.VUMPSOptions`, such as `n_iter`, `tol` and `backend`.

The settings of `idmrg` are those of {py:class}`~tnkit.IDMRGOptions`: `n_iter`, `tol`, `seed`
and `backend`.

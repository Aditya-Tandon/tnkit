# Thermal states

An MPS describes a pure state, but a system in equilibrium at temperature $T = 1/\beta$ is in
the mixed Gibbs state $\rho = e^{-\beta H}/Z$, with $Z = \Tr e^{-\beta H}$. {py:mod}`tnkit.thermal`
writes $\rho$ as a pure state of a larger chain, a *purification*, so that every MPS solver and
observable applies to it. Use it for thermodynamics and for real-time evolution at nonzero
temperature.

```python
from tnkit import mpo_expectation
from tnkit.models_lgt import schwinger_charges, schwinger_mpo
from tnkit.thermal import purify_mpo, thermal_state

N, beta = 8, 0.5
H = schwinger_mpo(N, x=4.0, mu=0.2)
q = schwinger_charges(N)                         # charge of each basis state of each site
rho, log_z = thermal_state(H, beta=beta, chi=64, dbeta=0.05, charges=q)
E = mpo_expectation(rho, purify_mpo(H)).real     # Tr(rho H)
s_thermal = log_z + beta * E                     # thermal entropy
print(f"log Z = {log_z:.6f}, E = {E:.6f}, entropy = {s_thermal:.6f}")
# log Z = 8.445290, E = -15.061355, entropy = 0.914612
```

The model is the lattice Schwinger model (quantum electrodynamics in one spatial dimension)
in the form of Bañuls et al., JHEP 11 (2013) 158, Eq. (2.6), with its electric field
eliminated by Gauss's law. Its two parameters are $x = 1/(e\, a_0)^2$ and
$\mu = 2 m_{\mathrm f}/(e^2 a_0)$, for the gauge coupling $e$, the fermion mass $m_{\mathrm f}$
and the lattice spacing $a_0$; the module docstring of
`tnkit.models_lgt` gives the operator in full. With `charges=`, the state is restricted to
total charge 0, the physical sector of the open chain.
The energy and $\ln Z$ above agree with exact diagonalisation of that sector to within
$10^{-6}$.

## Purification

Each physical site of dimension $d$ is paired with an *ancilla*, a copy of it, and the pair is
stored as one site of dimension $d^2$, with fused index $s d + a$ for physical state $s$ and
ancilla state $a$. The infinite-temperature state is the product of the pairs
$\sum_s \ket{s}\ket{s}$, normalised. Evolving it in imaginary time under $H \otimes 1$ gives

$$\ket{\psi(\beta)} = \left(e^{-\beta H/2} \otimes 1\right) \ket{\psi(0)},
\qquad \bra{\psi(\beta)} O \otimes 1 \ket{\psi(\beta)} \propto \Tr\left(e^{-\beta H} O\right),$$

so expectation values in $\ket{\psi(\beta)}$ are thermal averages. The norm that each step
removes is accumulated into $\ln Z$.

{py:func}`~tnkit.thermal.thermal_state` performs this cooling with two-site TDVP, in
`round(beta / dbeta)` steps of `dbeta`, and returns the normalised state and $\ln Z$; `chi`
caps its bonds. `charges=` (an array of shape `(N, d)`, the integer charge of every basis
state of every site) restricts the state to one total charge, `total` (default 0).
{py:func}`~tnkit.thermal.cool` continues the cooling of a given state.

Observables of the physical chain become operators on the fused sites:
{py:func}`~tnkit.thermal.purify` turns a one-site operator $O$ into $O \otimes 1$, for
`local_expectations`, and {py:func}`~tnkit.thermal.purify_mpo` does the same for an MPO. The
thermal entropy then follows from $\ln Z$ and the energy as
$-\Tr(\rho \ln \rho) = \ln Z + \beta \expval{H}$, which is `s_thermal` in the example above.

## Real-time evolution

Evolving the purified state under $H \otimes 1$ alone would let its bond dimension grow
quickly. `purify_mpo(H, twin=True)` instead gives $H \otimes 1 - 1 \otimes H^{\mathsf T}$,
which evolves the ancillas backward in time; this leaves the thermal state of $H$ unchanged
and slows the growth of entanglement (Karrasch, Bardarson and Moore, PRL 108, 227206 (2012)).
Either TDVP integrator runs on it:

```python
from tnkit import tdvp2_evolve

rho_t, records = tdvp2_evolve(rho, purify_mpo(H, twin=True), 64, 0.025, 20)
```

Since $\rho$ is the thermal state of the same $H$, this run leaves every expectation value
unchanged; it shows the call. To study dynamics, evolve under a different Hamiltonian, or apply
an operator to `rho` first.

## Block tensors

With `blocks=True`, the purified state is held in block tensors under
`tnkit.thermal.PURIFIED`, the symmetry U(1) $\times$ U(1): the physical and the ancilla
charge are conserved separately. The block MPOs come from `purify_mpo(H, twin=True,
charges=q)`, the integrators run with `backend=BlockBackend(get_backend("numpy"))`, and
observables need the dense state, `to_dense_mps(rho, purified_charges(q))`:

```python
from tnkit import SZ, BlockBackend, get_backend, local_expectations, to_dense_mps
from tnkit.thermal import purified_charges, purify

rho, log_z = thermal_state(H, beta=beta, chi=64, dbeta=0.05, charges=q, blocks=True)
rho_t, _ = tdvp2_evolve(rho, purify_mpo(H, twin=True, charges=q), 64, 0.025, 20,
                        backend=BlockBackend(get_backend("numpy")))
z = local_expectations(to_dense_mps(rho_t, purified_charges(q)), purify(SZ))
```

To apply an operator given as a block MPO to a block state, for example to create an
excitation before evolving, use {py:func}`~tnkit.block_backend.apply_block_mpo`, which also
compresses the result to a given bond dimension.

The full reference is {doc}`../api/thermal`.

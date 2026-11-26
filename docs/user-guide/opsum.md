# Building Hamiltonians

Writing an MPO by hand is error-prone beyond nearest-neighbour terms. `tnkit.opsum` builds
one from the Hamiltonian written as a sum of products of named one-site operators: you list
the terms, and {py:func}`~tnkit.opsum.compile_mpo` returns the MPO. Use it for any model that
tnkit does not ship.

```python
from tnkit import dmrg
from tnkit.opsum import OpSum, compile_mpo, spin_half

N, Delta = 12, 0.5
terms = OpSum()
for i in range(N - 1):                       # sites are numbered 0 .. N-1
    terms += (0.5, "Sp", i, "Sm", i + 1)     # (coefficient, operator, site, operator, site)
    terms += (0.5, "Sm", i, "Sp", i + 1)
    terms += (0.25 * Delta, "Z", i, "Z", i + 1)
W, report = compile_mpo(terms, [spin_half()] * N, dtype="auto")

E, mps, info = dmrg(W, 32)
print(f"{E:.10f}")                           # -4.3381483309
```

This is the spin-1/2 XXZ chain,

$$H = \sum_i \left[ \tfrac12 \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right)
+ \tfrac{\Delta}{4}\, \sigma^z_i \sigma^z_{i+1} \right],$$

with $S^\pm = S^x \pm i S^y$ and $\sigma^z$ the Pauli matrix.

## Terms

`terms += (c, "A", i, "B", j, ...)` adds the product of the number `c` with operator `"A"` on
site `i`, operator `"B"` on site `j`, and so on. Each name must be an operator of the site
type, and site numbers count from 0. Operators multiply in the order written, and a term
may hold any number of factors on any sites, so long-range and multi-site terms need nothing
special. A constant is a term on the identity, `(c, "Id", 0)`.

Two methods add many terms at once without listing them:

- `terms.add_quadratic_form(J, "A", "B")` adds `J[i, j]` times `"A"` on site `i` and `"B"` on
  site `j`, for every pair `i < j`, given an `N` by `N` strictly upper-triangular array `J`;
- `terms.add_sum_of_squares(A, b, "O")` adds, for every row `n` of the array `A`, the square of
  `sum_k A[n, k] O_k + b[n]`, expanded exactly, as the electric-field energy of the Schwinger
  model needs.

Both take bosonic operators only.

## Site types

A site type names the operators of one site; `compile_mpo` takes one per site, so a chain
may mix site types.

`spin_half()`
: the Pauli matrices `"X"`, `"Y"`, `"Z"`, with `"Z"` $= \operatorname{diag}(1, -1)$ in the basis
  (up, down), and `"Sp"` $= S^+$, `"Sm"` $= S^-$. Note that `"Z"` is $\sigma^z = 2S^z$, not
  $S^z$.

`spinless_fermion()`
: the annihilator `"C"`, the creator `"Cdag"`, the number operator `"N"` and the parity
  `"F"`, equal to 1 minus twice `"N"`, in the basis $(\ket{0}, \ket{1})$. `"C"` and `"Cdag"` are fermionic:
  `compile_mpo` inserts the Jordan–Wigner strings, so a hopping term is written as
  `(-1.0, "Cdag", i, "C", i + 1)` whatever the order of its sites.

`qudit(d, ops)`
: a site of dimension `d` with the operators in the dict `ops`, each a `d` by `d` array;
  `"Id"` is added. For example, a spin 1 in the basis $S^z = +1, 0, -1$:

  ```python
  import numpy as np
  from tnkit.opsum import qudit

  Sz = np.diag([1.0, 0.0, -1.0])
  Sp = np.sqrt(2) * np.diag([1.0, 1.0], k=1)
  spin1 = qudit(3, {"Sz": Sz, "Sp": Sp, "Sm": Sp.T})
  ```

## Compiling

`compile_mpo(terms, sites, dtype=..., max_width=None)` returns `(mpo, report)`. The MPO
follows the conventions of {doc}`mps-mpo`. The compiler lets terms that share operators share
bond channels, and compresses a bond by SVD where that lowers its dimension; the MPO stays
exact up to the relative tolerance `rank_tol` (default `1e-13`).

`dtype`
: `"complex128"` (the default, as for every model tnkit ships) or `"auto"`, which gives float64
  tensors when every coefficient and operator used is real. `dmrg` picks a real backend for a
  real MPO, which halves the memory.

`max_width`
: raise {py:class}`~tnkit.opsum.MPOWidthError` at the first bond wider than this.

`report`
: a dict with, for each cut between sites (from the left end to the right end), the bond
  dimension (`width`), the channel counts before and after the SVD pass (`structural`,
  `numerical`) and the largest singular value dropped (`dropped`); and the `dtype` chosen.

For small chains, {py:func}`~tnkit.opsum.to_dense` contracts an MPO into a dense matrix, a
check of a model against exact diagonalisation:

```python
import numpy as np
from tnkit.opsum import to_dense

small = OpSum()
for i in range(5):                           # the same chain on 6 sites
    small += (0.5, "Sp", i, "Sm", i + 1)
    small += (0.5, "Sm", i, "Sp", i + 1)
    small += (0.25 * Delta, "Z", i, "Z", i + 1)
W6, _ = compile_mpo(small, [spin_half()] * 6, dtype="auto")
print(np.linalg.eigvalsh(to_dense(W6))[0], dmrg(W6, 16)[0])   # both -2.0978876383
```

The full reference is {doc}`../api/opsum`.

# Low-level tensors

The solvers create and convert block and symmetric tensors themselves. This page is for
writing your own algorithms on them: it shows how the tensors are built, contracted and
factorised.

## Block tensors

The block tensors live in `tnkit.blocks`, and `tnkit.block_backend.BlockBackend` lets the
solvers' kernels run on them.

```python
import numpy as np
from tnkit.blocks import Leg, from_dense, tensordot, truncated_svd

leg = Leg(charges=(0, 1), dims=(2, 3))      # two sectors: charge 0 (2 states), charge 1 (3 states)
x = np.random.default_rng(0).standard_normal((5, 5))

A, dropped = from_dense(x, (leg, leg.flip()), charge=0)
print(A)                                    # BlockTensor(shape=(5, 5), charge=0, blocks=2, stored=13)

B = tensordot(A, A, ((1,), (0,)))           # contract leg 1 of A with leg 0 of A
U, sv, Vh, discarded = truncated_svd(B, chi=3)
```

- A {py:class}`~tnkit.blocks.Leg` lists a leg's charge sectors, the number of states in each,
  a sign and a symmetry (`U1()` if not given). A block is allowed when the signed charges of
  its sectors combine to the tensor's total charge.
- `from_dense` keeps the allowed blocks of a dense array and returns, as `dropped`, the
  squared norm of the part outside them: 0 for an array that already conserves the charge.
  Here `x` is random, so its two allowed blocks, of $2 \times 2$ and $3 \times 3$ entries, keep
  13 of its 25 numbers.
- A contracted pair of legs must have the same sectors and opposite signs (`leg.flip()`).
- {py:func}`~tnkit.blocks.truncated_svd` keeps the `chi` largest singular values across all
  sectors together, as a dense truncated SVD of the same matrix would. It returns them as
  `sv`, in sector order rather than sorted, and returns as `discarded` the sum of the squares
  of those it dropped.
- Contraction plans are cached. `tnkit.blocks.cache_info()` reports the caches and
  `set_cache_limit()` bounds them; see {doc}`../performance`.

See {doc}`../../api/blocks` for the full reference.

## Symmetric tensors

A symmetric tensor is a linear map from the tensor product of its *domain* legs (its inputs)
to that of its *codomain* legs (its outputs). An MPS or MPO tensor has all its legs in the
codomain and an empty domain.

A leg is a {py:class}`~tnkit.symmetric_tensor.Space`: its sectors, the number of multiplets of
each, and whether the leg is dual (pointing the other way, as a leg contracted with this one
must). The tensor stores one reduced matrix per coupled sector, as
{doc}`non-abelian` explains.

```python
import numpy as np
from tnkit.symmetry import SU2
from tnkit.symmetric_tensor import Space, random, from_dense

V = Space(SU2(), (0, 1, 2), (2, 1, 1))   # spin 0 twice, spin 1/2 once, spin 1 once
print(V.dim)                             # 7 dense states
t = random((V, V.dual()), (), rng=np.random.default_rng(0))
print(t)          # SymmetricTensor(shape=(7, 7), codomain legs=2, sectors=1, stored=6)

t2, dropped = from_dense(t.to_dense(), (V, V.dual()))   # project a dense array back
```

`random(codomain, domain)` draws a random invariant tensor; here `()` is an empty domain:
like every MPS tensor, all legs are in the codomain. The invariant part of $V \otimes V^*$ has
one $2 \times 2$ matrix for the two spin-0 copies and one $1 \times 1$ entry each for spin 1/2
and spin 1, so 6 numbers stand for the $7 \times 7 = 49$ entries of the dense array. As for
block tensors, `from_dense` returns the squared norm of the part of the array that is not
invariant; it is 0 (to rounding) here.

`permute`, `contract`, `fuse` and `split` in `tnkit.symmetric_tensor`, and `svd`,
`truncated_svd` and `qr` in `tnkit.symmetric_linalg`, act on the reduced blocks. A truncated SVD
keeps or drops whole multiplets.

See {doc}`../../api/symmetric` for the full reference.

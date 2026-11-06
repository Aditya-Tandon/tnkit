# MPS and MPO basics

## Index order

An MPS is a list of rank-3 tensors, one per site. tnkit stores site $i$ as

$$\mps{A}{s}{a}{b} \quad\text{in the order}\quad (a, s, b),$$

the left bond, the physical index and the right bond. Neighbouring bonds match,
`mps[i].shape[2] == mps[i + 1].shape[0]`, and the two outer bonds have extent 1.

An MPO tensor has four indices in a different order,

$$\mpo{W}{s}{s'}{a}{b} \quad\text{in the order}\quad (a, b, s, s'),$$

left bond, right bond, outgoing physical index, incoming physical index. An MPO tensor is
therefore *not* an MPS tensor with one more index appended; mixing the two orders is the
most common mistake when building models by hand.

The solvers return an {py:class}`~tnkit.MPS`, a subclass of `list`, so code that treats a
state as a list of arrays keeps working. It adds:

- `mps.entropies()`: the entanglement entropy at every bond, from one sweep;
- `mps.to_numpy()`: a copy with every tensor on the host as a numpy array;
- `mps.center`: the site of the orthogonality centre when it is known, else `None`.

Changing the list (assigning a tensor, appending) resets `center` to `None`.

## Contraction vocabulary

The docs use the terms of the TAPP tensor-contraction interface
([arXiv:2601.07827](https://arxiv.org/abs/2601.07827)):

- a tensor's indices are its **modes**, and the size of a mode is its **extent**;
- in a contraction, a mode that appears in both inputs and is summed over is
  **contracted**; one that appears in one input and in the output is **free**;
- a mode that appears in both inputs *and* the output is a **batch** (Hadamard) mode.

For example, extending the left environment $L$ by one site contracts the bond and
physical modes of $L$, $A$, $W$ and $\bar A$, and leaves three free modes:

$$L'_{b'\,w'\,b} = \sum_{a'\,w\,a\,s\,s'} L_{a'\,w\,a}\, \mps{\bar A}{s}{a'}{b'}\,
\mpo{W}{s}{s'}{w}{w'}\, \mps{A}{s'}{a}{b} .$$

Environments are stored in the order (bra, MPO, ket), and you pass them in that order as
`L0` and `RN`.

tnkit evaluates this as three pairwise contractions, each dispatched through the backend's
`einsum`. tnkit is an MPS library built on top of a contraction layer
(numpy, PyTorch, cuBLAS/cuSOLVER); it does not implement one.

## Building Hamiltonians

`tnkit` ships ready-made MPOs for the transverse-field Ising model
({py:func}`~tnkit.tfim_mpo`, {py:func}`~tnkit.tfim_ring_mpo`) and the lattice Schwinger model
({py:func}`~tnkit.schwinger_mpo`). For other models, `tnkit.opsum` compiles a sum of
operator strings into an MPO; see {doc}`../api/opsum`.

Any list of arrays in the $(a, b, s, s')$ order is a valid MPO, as long as the first
tensor's left bond and the last tensor's right bond have extent 1.

# Related software

tnkit is an MPS library: DMRG, TDVP and TEBD for one-dimensional systems, with the same code
on CPUs and GPUs. It sits above the tensor-contraction layer and calls numpy or PyTorch for
the contractions themselves. These projects overlap with it or sit beneath it.

**Tensor-network libraries**

- **TeNPy** — Python library for MPS algorithms (DMRG, TEBD, TDVP, iDMRG) with abelian
  symmetries. J. Hauschild and F. Pollmann, *SciPost Phys. Lect. Notes* 5 (2018),
  [arXiv:1805.00055](https://arxiv.org/abs/1805.00055).
- **ITensor** — Julia library for tensor networks with index-based contraction and
  abelian symmetries. M. Fishman, S. R. White and E. M. Stoudenmire, *SciPost Phys. Codebases*
  4 (2022), [arXiv:2007.14822](https://arxiv.org/abs/2007.14822).
- **YASTN** — Python library for abelian-symmetric tensors and tensor networks, with numpy and
  PyTorch backends. M. M. Rams et al., *SciPost Phys. Codebases* 52 (2025),
  [arXiv:2405.12196](https://arxiv.org/abs/2405.12196).
- **Quantum TEA** — tensor-network library for quantum many-body and quantum-circuit
  simulation, with CPU and GPU support. D. Jaschke et al.,
  [arXiv:2409.03818](https://arxiv.org/abs/2409.03818).
- **block2** — DMRG code focused on quantum chemistry, with abelian and SU(2) symmetries.
  H. Zhai et al., *J. Chem. Phys.* 159, 234801 (2023),
  [arXiv:2310.03920](https://arxiv.org/abs/2310.03920).

**Contraction interfaces**

- **TAPP** (Tensor Algebra Processing Primitives) — a proposed standard C interface for tensor
  contraction, which separates applications from contraction backends as BLAS did for
  matrices. These docs use its vocabulary (modes, extents, free, contracted and batch
  indices). J. Brandejs et al., [arXiv:2601.07827](https://arxiv.org/abs/2601.07827).

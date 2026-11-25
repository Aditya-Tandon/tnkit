# Fused two-site matvec: GPU timing and DMRG check

Scripts and results behind the fused, transpose-free two-site matvec in dense DMRG (#2).
Standalone: numpy, torch and tnkit only, no pytest.

- `gpu_matvec_prof.py OUT.json` — times `apply_two_site` against the fused matvec on one GPU, for
  a Heisenberg 6-row cylinder MPO (width 20) and an XXZ chain MPO (width 5), at χ 256, 512 and
  1024, and records the relative difference of the outputs.
- `fused_dmrg_check.py LABEL OUT.json` — dense two-site DMRG with whichever tnkit is importable.
  Run it once per version with the same script; `fused_dmrg_compare.py A.json B.json OUT.json`
  compares the per-sweep energies and sweep times.

`gpu-20261106/` holds the outputs from one NVIDIA H100 NVL MIG 1g.12gb slice
(torch 2.5.1+cu124, fp64), with the run logs and an md5 list:

- `matvec/`: fused matvec 1.46–2.34× faster; outputs agree to ≤ 1.3e-14 relative.
- `fused_dmrg/`: tnkit before #2 (ca14103) against after (b933cd4). Final energies agree to
  ≤ 1.0e-14 relative and every sweep agrees to ≤ 1.2e-14; a sweep is 1.29–1.36× faster on the
  cylinder and 1.03–1.04× on the chain.

## Symmetric tensors: SU(2) layout, recoupling and contraction times

- `symmetric_su2_timing.py OUT.json` — times layout building, `permute` (first call, rebuilt
  buffer indices with cached recoupling matrices, warm), one two-site effective-Hamiltonian
  application and `fuse` on SU(2) bonds of 13–40 spins, numpy backend, and records the cache sizes.
- `symmetric_batched_apply.py DEVICE [OUT.json]` — on `numpy` or a torch device: warm `permute`
  and effective-Hamiltonian times, the batched recoupling against one product per group, peak
  memory and index bytes, on the 13-spin SU(2) bond and a 42-sector U(1) x SU(2) bond.
- `symmetric_product_bonds.py DEVICE [OUT.json]` — U(1) x SU(2) bonds of 32–40 sectors built
  from two-colour fermion sites, of dense dimension 500, 1000 and 2000: cold layout times at
  ranks 3 to 5, recoupling-matrix reuse and batched products per permute, and stored entries,
  first and warm two-site effective-Hamiltonian times, a permute and peak memory against SU(2)
  alone, U(1) block tensors and dense arrays.

## Purified thermal states: block tensors against dense

- `purified_blocks.py [--backend numpy|torch-cuda] [--N ...] [--chi ...] [--out OUT.json]` —
  a purified Schwinger chain (x 4, m/g 0.05, T/g 0.5) cooled in U(1) x U(1) blocks, a packet
  applied, then real time with backward twins until the bond dimension reaches each cap. At each
  cap the same state takes one step on blocks and one on dense tensors: time of the first and of
  a steady step, peak memory, the contraction cache, and the sectors on every bond.
  `--cache-mb` limits the block contraction cache.

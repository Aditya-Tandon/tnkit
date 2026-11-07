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

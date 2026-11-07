# Changelog

Notable changes to `tnkit`. Versions follow [semantic versioning](https://semver.org), written
in Python's form for pre-releases (`0.2.0rc1`).

## [Unreleased]

### Changed

- Dense two-site DMRG applies the effective Hamiltonian with a fused, transpose-free matvec
  (one GEMM per environment and one batched GEMM for the MPO pair). Results agree with the
  previous matvec to round-off but are not bit-identical. Measured per sweep: 1.3x faster on
  CPU (cylinder, 2 threads); on an H100 MIG 1g slice, 1.29-1.36x on a 6x8 cylinder and
  1.03-1.04x on a 64-site chain. The matvec alone is 1.46-2.34x faster on the GPU.

## [0.2.0rc2] — 2026-11-05

No numerical change: the library code is 0.2.0rc1's apart from comments, docstrings and the
wording of three error messages (two in `vumps`, one in `itdvp_step`).

### Changed

- The `test` extra no longer installs matplotlib: no test in this repository imports it.
- `bench/suite`: the `tf_collision` workload needs a saved state that is not distributed; it is
  now recorded as unsupported when `TNBENCH_DATA` is unset, instead of failing.
- README: a "Questions and contributions" section; CITATION.cff gains a contact email.

### Tests

- `test_trim.py` no longer depends on code outside the package.
- `test_counters` compares block against dense contraction to 1e-13 rather than bitwise (a
  one-ulp difference on another BLAS failed it).
- Every test that needs a GPU carries the `cuda` marker, so `-m cuda` selects all of them.
- A lighter suite: the critical-VUMPS test runs at bond dimension 16 (was 24) with a tighter
  energy bar (1e-6, was 1e-4).

## [0.2.0rc1] — 2026-10-30

Block-sparse tensors with abelian symmetries (U(1), Z_n and their products), with DMRG and
TDVP on them (provisional); the `OpSum` MPO compiler; DMRG subspace expansion; the lattice
Schwinger models; input checks; a documentation site. Python 3.11 is now the minimum. Results
change only where an entry says so, chiefly through the two new `dmrg` defaults, the CUDA
batched-SVD driver (round-off) and the `evolve_growing_window` fix.

0.1.1, 0.1.2 and 0.1.3 were tagged without sections of their own; their changes are included
below. 0.1.1 is 0.1.0 plus two fixes: a README paragraph that described the module layout
backwards, and a shim rebind guard that listed three of the six shim modules (the set is now
computed from `__shim_source__`). 0.1.3 also corrected `pyproject.toml`, which still said 0.1.1
after 0.1.2 was tagged.

### Breaking

- **`dmrg_excited`'s third positional is now `options`** (a `DMRGExcitedOptions`), as in `dmrg`;
  it was `prev_states`. Passing a list there raises `TypeError`. No caller in the package or its
  tests passes it positionally. `idmrg`'s third positional is likewise `options` (it was `n_iter`),
  and its first parameter is renamed `g` -> `mpo_or_g`, so `idmrg(g=...)` no longer works.
- `TNKIT_CHECK` input checks (see Added) turn inputs with wrongly sized legs into
  `InvariantError`; they used to run and give wrong numbers.
- **Python 3.11 or later is required** (was 3.9). Minimum dependency versions are now
  numpy 1.24, scipy 1.12 and, for the `test` extra, matplotlib 3.6: the lowest that run on 3.11.
- **Missed from 0.1.0's list:** `options` replaced `dmrg`'s third positional argument
  (`n_sweeps`), `vumps`' third (`n_iter`), `tdvp1_evolve`'s fifth and `tdvp2_evolve`'s sixth
  (both `backend`).

### Changed

- **The block plan cache holds 4 GiB by default, or 30% of the CUDA device's memory if less**
  (was 1 GiB). The device fraction is taken from `torch.cuda.get_device_properties` the first
  time an index reaches a CUDA device, so importing tnkit does not touch CUDA. Host-only runs
  (numpy, torch-cpu) get 4 GiB. `TNKIT_BLOCK_CACHE_MB` and `set_cache_limit` still set the limit
  exactly; `cache_info()["limit_source"]` says which applied (`env`, `set`, `default-host`,
  `default-device`). Only eviction changes, so results do not. In a 10-step block TDVP run on a
  12 GB MIG slice (U(1) x U(1), working set 3.5 GB), the default first evicted in step 10 and
  cost 0.6% in step time against no limit; 1 GiB evicted throughout.
- **`graph="auto"` decides from the resolved backend's device and dtype**, not the literal
  spec `"torch-cuda"`: it captures for any complex torch backend on `cuda` or `cuda:0`, so
  `"torch-cuda:0"` and CUDA `Backend` objects capture by default and `graph="on"` accepts them.
  Real dtypes, `BlockBackend` and other CUDA devices (`"torch-cuda:1"`) stay eager. CPU results
  are unchanged.
- **Block TDVP now uses the host Lanczos path by default** (`tnkit.block_lanczos`, see Added).
  Results are bitwise identical; set `TNKIT_BLOCK_LANCZOS_FAST=0` to restore the eager loop.
- **`dmrg` stops on a relative energy change by default** (`rtol=1e-10`, was 0). The stop is
  |E - E_prev| < max(tol, rtol |E|), the looser of the two, so results are no longer
  bit-identical to earlier versions: a run whose energy change falls below 1e-10 |E| before it
  falls below `tol` stops earlier, and an explicit `tol` below 1e-10 |E| no longer tightens the
  stop. `tol=0` no longer runs every sweep; pass `rtol=0` as well for that, or for a tight stop.
  Measured over the 225 `dmrg` calls in the tests, examples, benchmark and a set of fixed
  configurations that leave `rtol` at its default: 33 stop earlier, by 1-17 sweeps, and the
  energy changes by at most 1.24e-8 (relative 7.7e-11; benchmark Ising, N 128, chi 16, 20 -> 3
  sweeps, ending lower). That count was measured with block expansion off. A block XXZ example
  (N 64, chi 64, U(1)) goes from 7 sweeps to 3 (-1.55e-9) with `rtol`
  alone, and to 6 (-9.5e-12) with both new defaults.
- **Block MPOs expand the subspace by default.** `DMRGOptions.expand` defaults to None: 0 (off)
  on a dense MPO, and min(4, n_sweeps - 2), at least 0, on a block MPO. `info["expand"]` is the
  value used. The stop is read only from the second sweep after expansion, so block runs now take
  at least 6 sweeps at the default `n_sweeps=8` (2-3 before, on easy problems, with no energy
  change beyond 2e-13). Expansion can end at a higher fixed point at small `chi`: on 12-qubit
  chains, chi 8-32, 10 seeds each, it ended worse by more than 1e-8 in 2 of 100 runs (two-flavour
  chain, chi 12: +2.0e-2; Heisenberg, chi 12: +6.1e-8) and better in 6 of 100 (two-flavour,
  chi 8 and 16: up to 6.2e-3 lower). Pass `expand=0` for the old behaviour.

- `mpo_charges` (so `to_block_mpo`) no longer raises on MPO channels that no path of nonzero
  entries joins to both ends, such as the unused channels of a compiled `OpSum` MPO. They get
  the trivial sector, and their entries are ignored when charges are assigned; where those
  entries do not fit the trivial sector, `to_block_mpo` drops them. Either way they add nothing
  to the operator.
- Block tensors refuse to become numpy arrays: `np.asarray` of a `BlockTensor`, and the
  observables (`MPS.to_numpy`, `MPS.entropies`, ...) on a block MPS, raise `TypeError` naming
  `to_dense_mps`. `np.asarray` used to return a one-element object array.
- The block plan cache keeps a running byte total (`cache_info()["bytes"]`) instead of
  recounting every entry on each insertion. An index's bytes (host runs or positions, plus each
  device copy) are added when its first holder enters or it reaches a new device, and removed
  when its last holder is evicted, so an index shared by a plan and a gather counts once. Each
  entry's host metadata (a plan's output layout, a gather's item offsets) counts with it.
- **Block-cache counters mean different things** after the two-level plans (see Added), so
  `cache_info` numbers from before and after are not comparable:
  - `misses` counts gather entries and plan entries separately, so a contraction miss is now
    up to three misses (`plan_misses` counts contractions alone);
  - `build_s` is the wall time of builds not nested in another build, plus index expansions,
    which run outside builds. It now includes symbolic plans and index expansion (`expand_s`),
    each counted once;
  - `bytes` counts runs (until an index is expanded, when they are dropped), expanded
    positions and the entries' host metadata, which were not counted before. On small chains
    the metadata dominates, so the same limit holds fewer entries: a 3-step N = 5 block TDVP
    run (numpy, `expand=3`) holds 1.43 MB, of which 251 kB are indices; before, it counted
    286 kB.
- **Block layouts are interned:** there is one `Struct` per layout content (labels, sector sizes,
  offsets, strides), so a plan is found again however its layouts were built, and fewer plans
  are built. Vector algebra on two tensors of equal layout no longer canonicalises them first.
  An output layout held by several cached plans counts once in `cache_info()["bytes"]`, with the
  copy of its offsets and strides that interns it. Pickled and copied layouts are the interned
  ones, so an unpickled block MPS finds its cached plans. Block tensors pickled by this version
  cannot be loaded by older tnkit.
- `TNKIT_BLOCK_EXPAND` takes `auto` (the new default), `host` or `device`. `auto` expands an
  index on the host for the CPU, and for a GPU when it has at most 2^17 elements, then copies it
  through pinned memory without waiting for the device; larger ones expand on the device, as
  every index did before (now `device`). Any other value raises `ValueError` when an index is
  first used on a torch device; numpy runs do not read it. Only uploads of
  at most 1 MiB are pinned; the pinned buffers come from torch's pinned-memory pool, which the
  cache limit does not bound.
- `to_block_mps` contracts its overlap pairwise (cost D^3, not D^4). Its tensors are unchanged;
  its `fidelity` diagnostic can differ from before in the last bits.
- On CUDA, `TorchBackend.svd_batched` (the stacked SVD of batched 2TDVP) now uses cuSOLVER `gesvd`
  instead of `gesvdj`. On real MPS truncation blocks `gesvd` was never slower (1.1-3x per call,
  1.04x on a batched 2TDVP step), has tighter singular values and needs no fallback; `gesvdj`
  wins only on random matrices. **GPU results that go through `evolve_batch` change at
  round-off level** (kept ranks identical, energies to 6e-14 in a side-by-side run); CPU is unchanged.
  `TNKIT_CUDA_SVD_DRIVER_BATCHED=gesvdj|gesvda|default` restores another driver, and the
  non-convergence fallback to `gesvd` is kept for those.
- The prefix-graph capture moved from `tnkit/ift/prefix_graph.py` to `tnkit/prefix_graph_impl.py`,
  since `tdvp1_evolve(graph="auto")` uses it. Pure relocation: `tnkit.prefix_graph` and
  `tnkit.ift.prefix_graph` are shims over it and give the same objects.
- `tnkit.__version__` reads the checkout's `pyproject.toml` first and installed metadata
  second. A leftover `tnkit.egg-info/` used to win, so a script run from the repository root
  could report the version of other code than it ran.
- `tnkit.observables._profile` returns a contiguous array rather than a stride-2 view into
  a complex buffer. Nothing observable moves today; `structure_factor` sums this array into
  a published number and numpy is free to branch its pairwise summation on stride.
- `tnkit._vumps._check_energy_readout` gives its own message, not a torch `TypeError`, when
  the `mpo` is a device tensor that cannot become a numpy array.
- CUDA tests carry `@pytest.mark.cuda`, so `pytest -m cuda` selects them. The marker was
  declared and applied nowhere, so `-m cuda` selected no tests and passed.
- `DMRGOptions`' docstring no longer advertises hashing. None of the options classes is
  frozen, so `hash(DMRGOptions())` raises.

### Deprecated

- `trunc='density'` (`dmrg`, `dmrg_excited`, `tdvp2_evolve`, `TDVPEvolver`) warns once at the call.
  Its right factor is not an isometry for small singular values, which breaks the canonical
  form on right-to-left steps. Use `trunc='svd'`.

### Removed

- `tnkit.api.SHADOWED_BY_MODULE`, empty since the module reorganisation and kept alive only by a test that it
  stayed empty. The invariant it encoded is now stated directly:
  `set(api.CURATED) <= set(tnkit.__all__)`.

### Fixed

- `BlockBackend.array` returns a contiguous buffer when the dense backend gives a strided view
  (a complex block tensor on a real backend), which `BlockTensor` refused.
- `evolve_growing_window` re-canonicalises on its first step and after every left growth. 1TDVP
  needs the orthogonality centre at site 0, which neither a built window nor an AL-prepended one
  has; results move at about 1e-6 in `<sigma^z>` (against an exact free-fermion reference:
  5.6e-7 before, 5.2e-10 after, chi = 16, t = 1.2). `evolve_growing_window` also takes
  `grow_every` and `symmetric`, defaulting to the old behaviour, and `edge_deviations` returns
  the two edges separately.
- The `bond_h` static check in `tests/test_vumps_energy_readout.py` walks every source
  tree present (here `tnkit/`, `tests/`, `bench/`) recursively and resolves `vumps(**name)`
  against the dict assigned to `name`. It used to scan three directories, flat, and missed
  keyword-built calls.
- A CUDA-graph capture that fails on the device (a host sync, an invalid capture) no longer leaves
  torch's default CUDA generator marked as capturing. Until a later capture succeeded, every
  replay of a graph on that device, and every random op on it, raised "Offset increment outside
  graph capture encountered unexpectedly" (PyTorch 2.5.1). The failed-capture cleanup now ends
  with a one-kernel capture, never replayed, that clears the mark; seed and offset are unchanged.

### Added

- Documentation site in `docs/` (Sphinx, PyData theme, MyST), built by Read the Docs from
  `.readthedocs.yaml`; the API reference is generated from the package. `pip install -e ".[docs]"`
  installs the build dependencies.
- DMRG subspace expansion and one-site sweeps: `DMRGOptions.expand`, `expand_alpha`,
  `expand_decay` (expansion in the opening sweeps, after Hubig et al., PRB 91, 155115 (2015),
  with the expansion term projected off the span of the centre), `sites` (1 or 2) and `rtol` (see
  Changed); `info` gains `sites`, `expand`, `expand_alphas` (alpha per sweep) and `sweep_s`
  (wall time per sweep). Dense and block MPOs. Expanding sweeps cut with an exact SVD whatever
  `trunc` says, `'gram'` included.
- `trunc='gram'` for `dmrg`, `dmrg_excited` and the two-site TDVP integrators: GEMM + `eigh` of
  the smaller Gram matrix in place of an SVD, with each new canonical site an exact isometry.
  Opt-in and an exception: it failed its speed/accuracy acceptance test and is kept for GPU
  speed. It is not reproducible against `'svd'`: in chaotic two-site TDVP, observables drift from
  an `'svd'` run by about 10–50× the spread between `'svd'` reruns (under a tenth of the
  truncation budget's own effect). Safe in float64 with `svd_tol >= 1e-12` for energies, local
  observables and entropies; not for runs that must reproduce `'svd'`, for `svd_tol < 1e-12`
  (warns), or for reading `S` below ~1e-8 of its largest value. Single precision, block tensors
  and `degenerate_tol` are refused; there is no floor guard. `'svd'` stays the default and its
  output is bitwise unchanged.
- DMRG on block tensors (**provisional**): `dmrg` takes a block MPO (`to_block_mpo`) and runs
  the same two-site sweep in one charge sector. A block MPO needs exactly one of the new
  `DMRGOptions.charge` (the total charge sought; it sets the random start, and has no default)
  and a block `init_mps`; `charge=` with a dense MPO is refused. The returned `MPS` holds block
  tensors; `to_dense_mps` converts it. Limits:
  - abelian symmetries only (`U1`, `Zn`, products; no SU(2));
  - `trunc='svd'` only, and no `eig='lanczos-mixed'`;
  - at small `chi` it can stop at a different fixed point than dense DMRG, with or without
    subspace expansion (on by default for block MPOs; see Changed);
  - `dmrg_excited`, `dmrg_block` and `dmrg_tower` refuse block MPOs.

  A new `info["blocks"]` key, on dense runs too, records which path ran; dense DMRG is
  otherwise byte-identical to before.
- `block_backend.random_block_mps`: a random block MPS of total charge `charge` (required) on
  the physical legs of a block MPO, at most `chi` states per bond, with sectors that chain from
  end to end.
- `sym=`, keyword-only and required (no default symmetry), on `block_backend.mpo_charges`,
  `to_block_mpo`, `to_block_mps` and `project_to_block_mps` (the last two take `U1()` only):
  MPOs in `Zn` and product sectors. For a symmetry other than U(1), `mpo_charges` returns lists
  of labels; for U(1) it still returns int arrays. `to_dense_mps` works for any symmetry.
- `tnkit` exports `Symmetry`, `U1`, `Zn`, `ProductSymmetry`, the block converters
  (`to_block_mpo`, `to_block_mps`, `project_to_block_mps`, `to_dense_mps`), `random_block_mps`
  and `BlockBackend` (**provisional**; `BlockBackend` because block TDVP takes one). Nothing is
  symmetric unless a symmetry is passed: `dmrg(W, chi)` on a dense MPO stays dense.
- `tnkit.symmetry` (provisional): the `Symmetry` interface (trivial sector, dual, fusion, sector
  order; `F`, `R` and `cg`, trivial for abelian groups) with `U1`, `Zn` and `ProductSymmetry`.
  Block legs take one: `blocks.Leg(charges, dims, sign, sym)` (default `U1()`), with labels
  checked against it, so block tensors carry Z_n and product charges (tuples) as well
  as integers. `Leg.from_charges(q, sign, sym)`. Only unique (abelian) fusion is supported.
- `tnkit.su2` and `tnkit.symmetry.SU2` (**provisional, not exported**): exact SU(2)
  Clebsch-Gordan, 6j, F and R coefficients, and an `SU2` symmetry with `fusion = "simple"`.
  `ProductSymmetry` now accepts multiplicity-free factors such as `SU2`, so U(1) x SU(2) works
  at the interface level (`fusions`, `dim`, `F`, `R`, `cg`). Block tensors and the block
  converters still refuse any symmetry whose fusion is not unique.
- Block contraction plans in two levels: a symbolic plan per set of sector labels, reused when
  only sector sizes change, and numeric offsets and gather indices from it. Gather indices are
  stored as runs and expanded once per device, where `TNKIT_BLOCK_EXPAND` says (see Changed);
  each operand's gather is cached apart, so it is reused when only the other
  operand changes. Symbolic plans live in their own cache (`TNKIT_BLOCK_LABEL_CACHE_MB`,
  default 512; `set_cache_limit(max_bytes, label_max_bytes)`). New `cache_info` keys:
  `plan_misses`, `expand_s`, and `label_hits`, `label_misses`, `label_evictions`,
  `label_build_s`, `label_entries`, `label_bytes`, `label_max_bytes`. Results are byte-identical
  to before.
- `cache_info` keys `build_cpu_s` (the thread CPU time of `build_s`), `contractions` and
  `gemms` (block contractions and the GEMMs with a nonzero inner dimension they issued).
- `tnkit.blocks`, `block_backend`, `block_graph` and `block_lanczos` declare `__all__`, so
  `from tnkit.blocks import *` and the API reference list only their public names.
- `tnkit.block_graph` (**provisional**, off by default): with `TNKIT_BLOCK_GRAPH=1` (values
  other than `0` and `1` raise) or `block_graph.set_enabled(True)`, block TDVP on CUDA replays
  each Krylov solve's matvec from a CUDA graph captured on its second call per input layout.
  Results are bitwise those of the eager path. A graph keeps the gather indices it reads alive
  (`blocks.keep_operands`, `blocks.Kept`); while it lives they count in `cache_info()["bytes"]`
  and toward the limit, once each, and the cache cannot free them. Graphs are dropped, and their
  indices released, when the solve ends, on `block_graph.clear()`, `set_enabled(False)` and
  `blocks.clear_cache()`. `block_graph.info()` counts captures, replays and failed captures.
- `tnkit.block_lanczos` (**provisional**, on by default): `eigensolver.lanczos_expm` runs
  block-tensor solves with less host work per iteration: vector algebra on the flat buffers,
  alpha kept on the device and read once with beta. Results and `info` are bitwise those of the
  eager loop. It composes with `block_graph`. `TNKIT_BLOCK_LANCZOS_FAST=0` or
  `block_lanczos.set_enabled(False)` restores the eager loop. The variable is read once at
  import; a value other than `0` or `1` makes `import tnkit` raise. Dense vectors, block DMRG
  (`lanczos_ground`), `lanczos_expm_fixed` and the `prefix_graph` solvers are unchanged.
- Helpers for preparing block states and the two-flavour model:
  - `models_lgt2f.schwinger2f_opsum`/`schwinger2f_mpo(isospin_penalty=)` adds
    lambda I^2, which vanishes on isosinglets;
  - `block_backend.project_to_block_mps` projects a dense MPS whose bond bases mix charges
    onto one total charge and compresses it to `chi` (ties go to the lower charge), with
    `charge`, `weight`, `discarded` and `fidelity` diagnostics;
  - `to_block_mps(progress=)` and `project_to_block_mps(progress=)` report progress on long
    chains;
  - `TDVPEvolver.prepare()` canonicalises and builds the environments before the first `run`;
    the arithmetic is unchanged.
- `TDVPOptions.expand` (`TDVPEvolver` only, off by default; **provisional** on block tensors):
  global subspace expansion. Before every step each bond's right basis gains up to `expand`
  zero-weight states from H|psi>, so a step can reach directions and, on blocks, charge sectors
  a bond lacked. The returning sweep cuts each one-site bond back by SVD (applying
  `degenerate_tol`) to its size at the start of the step, or, outside `'1site'`, to the cap if
  a two-site cut had pinned it below. The expansion's own cuts (building H|psi>, choosing the
  new states) ignore `degenerate_tol`, so on a degenerate spectrum dense and block runs can keep
  different states.
  `tnkit.expansion.expand_right` does the expansion on a right-canonical MPS. New block helpers
  `blocks.concat` and `blocks.zeros`. With `expand=0` every path is byte-identical to before.
- TDVP on block tensors (provisional): `tdvp1_evolve`, `tdvp2_evolve` and `TDVPEvolver` (all
  three schedules) accept a block MPS and block MPO with `backend=BlockBackend(dense)`. Two-site
  steps truncate with `blocks.truncated_svd` (`trunc='svd'`, with `svd_tol` and `degenerate_tol`);
  one-site steps and `right_canonicalize` factorise with the new per-block `blocks.qr`. `measure`
  and the returned MPS get block tensors. Open boundaries only; `fast_expm`, step graphs
  (`graph="on"`, `cuda_graph`, `prefix_graph`) and a block MPS with a dense backend or dense MPO
  are refused (`tnkit.block_graph` replays Krylov matvecs instead). `degenerate_tol` on blocks follows the
  dense rule and is experimental. Dense paths are unchanged.
- `block_backend.to_block_mps` converts a dense MPS in one charge sector (bond bases rotated to
  definite charge, off-sector entries dropped and reported); `to_dense_mps` converts back.
  `BlockBackend.array` moves a block tensor to the dense backend's device and dtype.
  `truncated_svd(..., degenerate_tol=, info=)` keeps or drops whole groups of equal singular
  values across sectors by the dense rule (`_dmrg._cut`), and reports in `info["pinned"]` a cap
  cut that it moved down.
- `models_lgt2f` (provisional): the two-flavour Schwinger model, with `site_charges_2f`: the
  charge Q and isospin 2 I_z of each qubit packed into one integer, so U(1) blocks conserve both
  (`unpack_charge` reverses it).
- `tnkit.blocks`: block-sparse tensors with abelian charges, opt-in and off the default path.
  Provisional: the block-tensor API and the `BlockBackend` interface may change before 1.0.
  `BlockTensor` keeps every allowed block in one flat buffer; `tensordot` runs on plans cached
  per block structure, one gather per operand and one GEMM per contracted charge; permuting legs
  moves no data. Plans and gather indices share one LRU cache bounded by bytes
  (`set_cache_limit`, `TNKIT_BLOCK_CACHE_MB`; default under Changed) with hit, miss and eviction counts
  (`cache_info`). `truncated_svd` factorises sector by sector and truncates across sectors as a
  dense truncated SVD would, optionally padding sector sizes to a bucket so layouts repeat.
  Like the dense backend, it uses gesvd on CUDA and retries with gesvd when an SVD does not
  converge.
  `tnkit.legs.fuse`/`split` (also partial) dispatch to block tensors with a `BlockFuseRecord`,
  and refuse to split a dense array with one.
- `tnkit.block_backend`: `BlockBackend` wraps a dense backend so `_dmrg.apply_two_site`,
  `apply_one_site`, the environment updates and `lanczos_expm` run on block tensors unchanged;
  any other call with a block operand, also inside a list or tuple, raises, or with
  `strict=False` densifies and is counted in `fallbacks`. `mpo_charges` and `to_block_mpo`
  label the channels of a compiled MPO. Dense results are unchanged.
- `TDVPOptions.degenerate_tol`, **experimental** (two-site steps, `trunc='svd'`; default 0 leaves
  the cut and every bit unchanged). A value in (0, 1) keeps or drops whole a group of singular
  values equal within that relative tolerance of the value next to the cut: at the cap the cut
  moves down to the start of the group, below it the `svd_tol` cut moves up to its end. A group
  that starts at the largest value is split. `TDVPEvolver`'s '12site' schedule treats a bond cut
  below the cap this way as full. Values outside [0, 1), or above 0 with another `trunc`, raise
  when the options are resolved. Not shown to help: on a 12-qubit test it made the isospin leak
  1.56–1.65× larger where it acted, and lowered the fidelity. A DMRG version was rejected: at
  1e-6 and 1e-4 it never acted on the 80 target runs; it acted in 19 of 112 other runs and never
  lowered an energy; at 1e-1 it made one energy error about 90× larger.
- `MPS.center`: the site of the orthogonality centre, or None when unknown. `right_canonicalize`
  (on an `MPS`), `dmrg`, `dmrg_excited`, both TDVP integrators, `evolve_batch` (0) and `compress`
  (N-1) set it, and `build_window`/`grow_window` return an `MPS` centred at the inserted site;
  any change to the list's contents resets it. `dmrg_excited` now returns an `MPS`. No numerical
  change. An `MPS` pickled by this version does not load in an older tnkit.
- The field-theory helpers run on the backend adaptor: `ift.two_particle.SeparationBasis(..., be=)`
  and the window observables (`window_expectations`, `window_neighbor_correlations`,
  `grow_window`) accept torch tensors and keep them on their device. `PlainEinsumNumpyBackend`
  is numpy with plain `np.einsum` (no compiled plans, whose output layouts can change round-off);
  the numpy defaults use it and are bit-identical. `SeparationBasis` needs complex128.
- Front doors: `vumps` returns a `UniformMPS` (the same dict, keys also readable as
  attributes); `idmrg(mpo_or_g, chi, options=IDMRGOptions(...))` takes a bulk MPO tensor or the
  TFIM field, with `be=` kept as a deprecated alias of `backend=`; `dmrg_excited(mpo, chi,
  options=DMRGExcitedOptions(...))`; `tebd_evolve(..., bond_h=[...])` for any nearest-neighbour
  Hamiltonian (with `g=None`); `tfim_excitations(..., mpo=, bond_h=, triple_h=)`, which
  `mixed_field_excitations` now calls. TDVP step dicts carry `krylov_cap_hits`, the number of
  adaptive solves that stopped at the `krylov_m` cap (always 0 on fixed schedules). Existing
  call styles are bit-identical.
- `tnkit.invariants`: `check_mps`, `check_mpo` and `check_pair` check ranks, bond chaining,
  physical legs, and outer bonds against 1 or against `L0`/`RN`. `dmrg`, `dmrg_excited`,
  `tdvp1_evolve`, `tdvp2_evolve`, `evolve_batch`, `apply_mpo` and `compress` run them once per
  call (also `dmrg_block`); `TNKIT_CHECK=debug` also runs them after every sweep, where the
  solver's own lists must also have one dtype family and one device; `TNKIT_CHECK=off` skips them. **Inputs that break these rules now raise
  `InvariantError`** instead of running: einsum broadcasts a size-1 leg against any size, so an
  outer bond of the wrong size used to give wrong numbers silently, and `apply_mpo` silently
  dropped sites when the MPO was shorter. Valid inputs are bit-identical. A check costs about
  0.2 ms at N = 600, so drivers that call a solver once per step should time with
  `TNKIT_CHECK=off`.
- `tnkit.legs`: `fuse(x, groups, be)` fuses groups of legs into single legs and returns a
  `FuseRecord` (unfused shape and groups, and whether it copied); `split` is its
  checked inverse and can restore only the fused legs that survive a factorisation. New
  adaptor method `be.transpose(x, perm)` on both backends. The two-site truncations in
  `dmrg`, `tdvp2_evolve` and `evolve_batch` now reshape through them; bit-identical.
- `TDVPOptions.krylov_m` (default 30) caps the adaptive Krylov exponential in both TDVP
  integrators, replacing the `lanczos_expm` monkeypatch (`graph='auto'` resolves to eager when it
  is not 30, since the prefix graph replays solves of its own length). `step_callback` is now shared:
  `tdvp1_evolve` calls it every step with `{step, t, dt, krylov_max}`, and both integrators'
  step dicts carry `krylov_max`. Defaults are bit-identical.
- `tnkit.models_lgt`: the lattice Schwinger model (`schwinger_mpo`, `schwinger_opsum`,
  `schwinger_dense`, `charge_conjugation_mpo`), its profile helpers, `chiral_condensate`,
  `flux_correlator` and `energy_density`; `observables.two_site_entropy` and
  `observables.compress`. `schwinger_mpo`, `two_site_entropy`, `compress` and
  `mpo_expectation` are curated. `mpo_expectation` now defaults to the MPS's own backend.
- `tnkit.opsum`: `OpSum` and `compile_mpo`, which compile a sum of local operator products
  (spin-1/2, spinless fermions via Jordan-Wigner, any `qudit(d)`; dense long-range quadratic
  forms and sums of squares) into the MPO list the solvers take. Exact vertex-cover channels,
  SVD only where a cut's rank drops; reproduces `tfim_mpo`, `magnetization_mpo` and
  `longrange_mpo` bit for bit. `models.py` is unchanged.
- `mpo_variance` (curated) and `mpo_sq_expectation` in `tnkit.dmrg_excited`: the energy variance
  `<W^2> - <W>^2` of an MPS, from `<W^2>` contracted layer by layer without forming `W^2`
  (memory chi^2 D^2). Returns `(var, floor)`, `floor` being the round-off level of the difference.

### Known issues

- Dense TDVP2 on a charge-conserving problem with capped bonds can leak about 1e-7 of weight
  out of the charge sector: zero-weight singular vectors kept at a capped bond can mix sectors,
  and the one-site back-evolution fills them. Block TDVP conserves the charge exactly and is the
  accurate path there (20 sites, one step: block 2e-10 from exact evolution, dense 1.4e-7). A
  small `svd_tol` does not reliably remove the leak; use a block backend for symmetric problems.

## [0.1.0] — 2026-08-17

The first tagged release, and the whole of it: `tnkit` began this cycle with no licence, no
`pyproject.toml`, a CI job that ran no tests, and observables that threw on any state its own
GPU backend produced. Everything below turned that into something installable.

No numerical result changes ride along with it — with one deliberate, documented exception
under **Breaking**, which is a bug fix. A golden-number suite recomputes published
values from archived data through the current code and was green before and after every
change here; it is the check that says the refactor moved nothing, and it is a test result
rather than a claim. (That suite and its archived data stayed in the original research
repository; they are not part of this one.)

Measured at the tag: 169 passed / 5 skipped locally on three BLAS threads, and 172 passed /
2 skipped on an H100 NVL MIG 3g.47gb slice, where the three CUDA tests run instead of
skipping. The two remaining skips are the two golden numbers whose analysis chain is not yet
on `main`.

### Breaking

- **`vumps(mpo=<a custom MPO>)` now RAISES unless `bond_h=` or `triple_h=` is supplied.**
  Read this before upgrading; it is the one change here that can stop a working script.

  `vumps` reads the energy density `e0` off *local expectation values*, dispatching on the
  operator the caller gave it: `triple_h` → the three-site readout, `bond_h` → the
  caller's own nearest-neighbour operator, and **neither** → the TFIM formula
  `-<sx sx> - g<sz>`. A caller that passed a custom `mpo` without a bond operator fell
  through to the TFIM readout, which silently drops every term the custom MPO added. For a
  mixed-field chain that is exactly `-h<sx>` per site: at `g_x = 1`, `g_z = 0.10`, `chi =
  12` the reported `e0` was `-1.26819106` against a true `-1.35074499`. The *state* was
  correct throughout — it is optimised against the full MPO — so nothing looked wrong.

  The fix refuses rather than repairs. Reconstructing the missing term from the MPO would
  have fixed that instance and hidden the next one (a three-site term, a staggered field, a
  different automaton layout); the caller knows its own Hamiltonian and must say so. The
  new `tnkit.vumps.EnergyReadoutError` fires **before** the optimisation loop, so a
  mis-specified run costs a second rather than six hundred iterations, and its message
  names the fix. `mpo=tfim_bulk_mpo(g)` is still accepted — the check is against the
  operator, not against the presence of `mpo=` — and it catches for free an `mpo` built at
  a different `g` than the `g` argument.

  **Migration** is one keyword:

  ```python
  gs = vumps(g, chi, mpo=mixed_field_bulk_mpo(g, h),
             bond_h=mixed_field_bond_h(g, h))          # three-site: triple_h=...
  ```

  **To reproduce a pre-fix number**, `on_mpo_mismatch="ignore"` returns the old `e0`
  bit-for-bit, so archived results stay reproducible. It is for old results, not for new runs.

  Every caller is fixed (fourteen of them). `tnkit.window.build_window` forwards
  `bond_h`/`triple_h`, and a repo-wide AST sweep in `tests/test_vumps_energy_readout.py`
  fails the suite if a **new** call site reintroduces the combination — the guard alone
  would only fire at runtime, possibly hours into a run.

### Added

- **`tests/test_golden.py`** — published numbers recomputed from archived data through
  the current code and asserted to the tolerance of the original publication: the FSS exponents
  `beta` and `nu`, the exact free-fermion `M2(N)` instrument, `structure_factor` against it
  at `N = 48`, the `N = 64` entanglement profile and its Calabrese-Cardy central
  charge, the ACCC Aitken-in-`chi` ladder and its coupling continuation, DMRG energy against
  the Jordan-Wigner solution, and the repository's three `*_selftest.py` conventions. Two
  further numbers are skip-marked with their reproduction command. (Kept, with the archived
  data and the self-tests, in the original research repository; not part of this one.)
- **`observables.all_entropies(mps)`** — the whole `S(l)` profile from one sweep,
  `O(N chi^3)`. Promoted from an analysis script, where it measured 33 s
  against hours at `N = 600, chi = 192`. Asserted bit-identical to per-bond
  `entanglement_entropy`.
- **`observables.to_numpy(mps)`** — host copy of a device-resident state, for handing an MPS
  to numpy-only code.
- **`canon=` on `entanglement_entropy` and `local_expectations`** — share one right-canonical
  copy across repeated measurements of the same state. Asserted to be an exact no-op.
- **`pyproject.toml`** — `pip install tnkit`, with `[torch]` and `[cuda]` extras and
  `[test]` for the suite. Core dependencies are numpy and scipy only.
- **`.github/workflows/tests.yml`** — CI runs the suite on push and pull request, on three
  BLAS threads, without a GPU.
- **`BENCHMARKS.md`** — the host-loop eigensolver comparison: conditions table and its measured
  1.18x–7.15x span, with the withdrawn `chi^1.09` fit and why.
- **`CITATION.cff`**.

- **`tnkit.__version__` and `tnkit.__all__`** — the version is read from the installed
  distribution metadata, falling back to the `version` line of `pyproject.toml` when the
  package is used straight from a checkout. There is no
  second copy of the number to drift; `tests/test_api.py` asserts the two agree.
- **`tnkit.api`** — the curated surface in one flat namespace: the solvers, `MPS`,
  `DMRGOptions`, `TDVPOptions`, `VUMPSOptions`, the model builders, every observable, and
  `get_backend`. Every name in it is re-exported from `tnkit` itself.
- **`DMRGOptions`** — the nineteen keyword arguments of `dmrg` as one dataclass, with the same
  names and the same defaults. `dmrg(mpo, chi, options=None, **kwargs)` accepts either style,
  and a flat keyword overrides the corresponding field of an options object passed alongside
  it. `tests/test_api.py` asserts the two styles give bit-identical states and energies.
- **`TDVPOptions`** — the same treatment for `tdvp1_evolve` and `tdvp2_evolve`. One dataclass
  covers both; passing a field that belongs to the other integrator FLAT raises, while
  carrying it inside a shared bundle does not (2TDVP-then-1TDVP is the recommended protocol
  and needs one bundle for both).
- **`tnkit.mps.MPS`** — a `list` subclass, so an MPS is still a list of rank-3
  `(left, phys, right)` tensors and every existing script is unaffected. It gives the index
  convention a documented home (it lived in a module docstring) and carries `.to_numpy()` and
  `.entropies()`, which forward to `observables.to_numpy` and `observables.all_entropies`.
  `dmrg`, `tdvp1_evolve` and `tdvp2_evolve` now return one.
- **Type annotations** on the public surface: all of `observables.py`, plus the signatures and
  returns of the exported entry points.
- **`tests/test_api.py`** — version consistency, the curated surface, options equivalence,
  and the MPS type's list behaviour.
- **`VUMPSOptions`** — the sixteen keyword arguments of `vumps` as one dataclass, same
  names, same defaults, with `vumps(g, chi, options=None, **kwargs)` as the front door and
  a flat keyword overriding the matching field of an options object passed alongside it.
  `tests/test_api.py` asserts the two call styles agree on `e0`, `S`, `eps`, `iters` and on
  the bytes of `AL`, `AR`, `AC` and `C`. `chi` and `g` both remain acceptable as keywords,
  as older call sites do.
- **`tnkit.ift`** — the field-theory helpers as their own subpackage: `luscher` (finite-volume
  Lüscher analysis), `two_particle` (the relative-separation two-particle basis) and
  `prefix_graph` (the overshoot-and-truncate CUDA-graph capture). None is engine, and they
  are deliberately absent from
  `tnkit.__all__` and not imported by `import tnkit` — importable, not headline. The old
  paths `tnkit.luscher`, `tnkit.two_particle` and `tnkit.prefix_graph` keep working as
  shims and no caller changed.
- **`tnkit.vumps.EnergyReadoutError`** and `on_mpo_mismatch=` — see "Breaking" above.
- **`MANIFEST.in`** — non-library directories are pruned from the source distribution.
- **`tests/test_vumps_energy_readout.py`** — seven tests on the energy-readout guard,
  including the repo-wide AST sweep that stops it recurring at a new call site.

### Changed

- **Every observable is backend-aware.** `dmrg(backend="torch-cuda")` returns CUDA tensors,
  and until now every routine in `tnkit/observables.py` was numpy-only, so solve-then-measure
  raised `'Tensor' object has no attribute 'copy'` or `can't convert cuda:0 device type
  tensor to numpy`. Each routine now deduces its backend from the state, contracts on that
  device, and transfers only the result — once per routine, not once per site.
- **`structure_factor` canonicalises once**, not twice; the diagonal `<op_i^2>` pass shares
  the canonical form with the accumulated off-diagonal sum.
- **`eig_stop` is documented where it is used.** `dmrg` and `lanczos_ground` now state in
  their own docstrings that the energy stopping rule leaves the eigenvector only
  `sqrt(tol)` converged, and name the four comparisons it invalidated.
  It was written up in the README and `BENCHMARKS.md`; the functions themselves said nothing.
- **`from tnkit import dmrg` now gives the FUNCTION, and so do `vumps` and `idmrg`.** The
  implementations moved to `tnkit/_dmrg.py`, `tnkit/_vumps.py` and `tnkit/_idmrg.py`;
  `tnkit/dmrg.py` and its two siblings are compatibility shims that re-export the
  implementation's entire module namespace, so `from tnkit.dmrg import <anything>` is
  unchanged. `tnkit.api.SHADOWED_BY_MODULE` is now empty:
  every name on the curated surface is reachable from `import tnkit` alone.

  One consequence, for anyone who patches library internals from a driver:
  `import tnkit.dmrg as D` now binds the function, because CPython resolves
  `import a.b as c` by `getattr` on the package. **Mutating a registry still works from
  every import path** — `dmrg._EIGSOLVERS` and the new `vumps._HOOKS` are the same dict
  objects the implementation holds, never copies — **but assigning to a module global does
  not**, and never says so. `tnkit._vumps._HOOKS` therefore carries the two internals a
  legacy-replay driver replaces (`left_fixed_point`, `energy_channel`), mirroring
  `_EIGSOLVERS`. `tests/test_api.py` patches through each seam and asserts the patch was
  *called*, and refuses, by AST sweep, any new `<module alias>.<name> = ...`
  against a shadowed module. (That sweep covered driver scripts that are not part of this
  repository, and it stayed with them; `tests/test_api.py` here checks the shim set and that
  `tnkit.tdvp` and `tnkit.eigensolver` stay modules.)
- **`bench/` and `tests/` are excluded from the sdist and the wheel.** They stay in git, but a
  `pip install tnkit` gets the library and nothing else. `tests/test_packaging.py` builds
  a real sdist and a real wheel and asserts on their members rather than trusting the
  declaration.
- **The suite no longer errors without torch.** `tests/test_prefix_identity.py` and the two
  truncation tests in `tests/test_tnkit.py` skip their torch legs through
  `pytest.importorskip` and keep running their numpy legs, so a torch-free environment
  demonstrates the packaging claim instead of blowing up on an unguarded import: measured
  117 passed / 16 skipped / 0 errors with torch hidden from the interpreter.

### Known gaps

- The CUDA smoke tests in `tests/test_observables_backend.py` skip without a GPU. CI cannot
  run them; they run on a GPU before a tag.
- `tebd_evolve` and `idmrg` keep their flat signatures, and `vumps` returns a dict rather
  than an `MPS`. Only `dmrg` and the two TDVP integrators return the `MPS` type.
- `import tnkit.dmrg as D` binds the function, not the module — see "Changed". Patch
  through the registries, not by assignment.
- Annotations cover the public surface only. Most of the internals are still unannotated, and
  there is no type-checking step in CI.

# Sparse LU Reference TODO

- [x] Pin the Python linear-solver reference to SciPy 1.18.1 `splu` / SuperLU.
- [x] Add deterministic BVP-shaped sparse CSC fixtures with exact solutions.
- [x] Add Rust correctness comparison against faer's existing sparse-LU wrapper.
- [x] Add separate Rust timing scopes for symbolic analysis, numeric factorization,
  repeated numeric refactorization, solve, and matrix assembly.
- [x] Add shared fixture export and a compact Python `splu` timing runner.
- [x] Cross-language story checks SciPy version, CSC structural count, residual,
  and the known solution.
- [x] Keep the existing BVP_sci Python entry point as a tested compatibility
  shim over the shared runner instead of maintaining duplicate SuperLU code.
- [x] Run the debug Rust gate and cross-language fixture story: SciPy 1.18.1,
  dimension 50, CSC nnz 474, residual infinity norm `8.882e-16`.
- [x] Collect an initial release slice across small, medium, and large
  synthetic BVP-shaped matrices. Rust and SciPy Python fixture tests passed
  (`3/3`); the ignored same-fixture Rust/SciPy correctness story also passed.
  The report is archived under
  `test_reports/BVP_sci_release_manual/banded_refinement_ab_20261012/somelinalg_sparse_lu_reference/release/rust_microbenchmark.md`;
  Python `splu` rows were emitted for the same exported fixtures. Results and
  interpretation are recorded below.
- [x] Expose the kernel to BVP_sci as an explicit opt-in experimental sparse
  factorization policy and add a full-solve/continuation A/B story. The normal
  `StandardFaer` policy remains the default. Debug smoke (`3` repeats,
  `4` continuation rebinds) matched final states and per-series counters on
  both routes: `10` numeric factors, `6` symbolic analyses, `4` symbolic
  reuses. Report:
  `test_reports/BVP_sci_new/debug/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [ ] Add more representative matrices captured from actual BVP_sci Jacobians;
  one captured 1022x1022 Jacobian/RHS is now available, but one workload is not
  a matrix corpus.
- [ ] Explain fill-in / ordering behavior and compare SuperLU/faer options where
  exposed; do not infer a solver win from one synthetic pattern.
- [ ] Decide whether the reusable faer kernel merits production integration.
- [x] Record the BVP_sci release full-solve/continuation A/B (5 repeats, 8
  rebinds). Standard vs reusable experimental medians: solve series `1.648`
  vs `1.612 ms`; numeric-factorization scope `0.048` vs `0.029 ms`. Both
  returned identical mesh/state and the same counters (`16` numeric
  factorizations, `8` symbolic analyses, `8` symbolic reuses). The result
  supports the same-pattern reuse path on this workload, but the total solve
  gain is small and does not justify changing the default. Captured production
  Jacobians and memory retention are still open. Full report:
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [ ] Only after same-matrix evidence, evaluate a `SparseSciPy` backend name in
  BVP_sci and retire or retain its experimental Banded route.

## Explain the Captured BVP LU Gap

The captured 1022-by-1022 Jacobian has 124,684 stored CSC coordinates but
only 4,864 numerically nonzero values. Rust faer Sparse and SciPy/SuperLU
produce matching BVP work counts, yet on this matrix Rust numeric factorization
is materially slower. The synthetic BVP-shaped fixtures do not reproduce that
gap consistently, so tuning from synthetic cases alone is not justified.

- [x] Add an isolated microbenchmark for faer AUTO, forced simplicial, and
  forced supernodal LU, holding the COLAMD ordering settings constant.
- [x] Add a synthetic same-pattern control with approximately 92.8% of stored
  values numerically zero; assert that CSC structural coordinates are preserved.
  Report symbolic, cold factor, warm numeric factor, and solve timings with
  backward error and known-solution parity. Export the zero-valued fixtures so
  the existing SciPy `splu` runner can consume the same matrices.
- [x] Capture an exact production BVP_sci CSC matrix and RHS with full
  coordinates and values. This first capture has a high explicit-zero ratio;
  capture additional passes/workloads before generalizing.
- [x] Run SuperLU and each faer strategy on the same captured input. Compare
  available factor fill, backward error, and cold plus repeated numeric
  timings, with assembly outside factor timers. faer does not expose all of
  SuperLU's permutation/pivot diagnostics through the current public API.
- [ ] Capture additional Newton matrices with different values but the same
  pattern, plus at least one distinct workload/pattern.
- [ ] If mode/fill/zero controls do not explain the gap, add a local faer
  diagnostic build with opt-in counters for symbolic selection, factor storage
  growth, pivot work, supernodal updates, workspace allocation/reuse, and
  triangular solve stages. Keep the dependency/version pinned and verify that
  telemetry-disabled results match upstream faer.
- [ ] Only tune or promote a sparse implementation after the captured-matrix
  diagnostic reproduces the BVP gap and release A/B confirms the improvement.

### Isolated SuperLU Control (2026-10-12)

- [x] Add an opt-in, repository-only SuperLU 6.0.1 C bridge under
  `somelinalg`; pin the upstream source commit, build it separately, and load
  the bridge dynamically. This does not add a SuperLU dependency to the crate
  and is not connected to a production solver policy.
- [x] Reproduce SciPy's factor CSC count semantics instead of counting values
  in padded native supernode buffers. SciPy's `LU_to_csc` filters exact zeros,
  moves the upper triangle stored in the L supernodes into U, and emits L's
  unit diagonal. The corrected counts match SciPy exactly on all three shared
  synthetic fixtures and on the captured BVP Jacobian: captured
  `L.nnz=2530`, `U.nnz=4224`. The earlier 2210/2516 counts were not comparable
  and are superseded.
- [x] Re-run the release same-matrix diagnostic with 9 repeats on captured
  BVP Jacobian n=1022, stored CSC coordinates=124684, numerical nonzeros=4864.
  SuperLU native factor/solve medians were 2.363/0.045 ms; FFI total was
  2.587 ms; int32 conversion was 0.274 ms. SciPy 1.18.1 `splu` on the same
  fixture measured factor 2.856 ms, solve 0.074 ms, and CSC assembly 0.712 ms.
  The core-only SuperLU result looks faster, but after including Rust's CSC
  conversion, the route totals are close; the difference is too small and too
  build-sensitive to call a stable win. SciPy's wheel links OpenBLAS while
  this control uses SuperLU's bundled CBLAS, so this is diagnostic evidence,
  not a controlled backend victory.
- [ ] Before any backend decision, rebuild the control against the same BLAS
  implementation used by SciPy (or explicitly treat BLAS as a separate axis),
  repeat on an idle machine, and compare more captured matrices / BVP
  workloads. Keep all SuperLU experiments in `somelinalg`; do not expose them
  through BVP_sci until same-matrix and full-solve evidence justifies it.

Report: `test_reports/somelinalg_sparse_lu_reference/release/rust_microbenchmark.md`.
SciPy source contract: `_superluobject.c::LU_to_csc` in the pinned 1.18.1
reference; the bridge now mirrors the same zero filtering and L/U extraction.

### Debug Correctness and Comparison Contract (2026-10-12)

- [x] Fix captured-case reporting: when there is no known exact solution,
  `known_solution_max_diff` is now `n/a (captured RHS)`, never a fabricated
  zero. Backward/relative residual remains the numerical quality gate.
  The archived first-release table still contains the old misleading zero and
  is historical only; do not use that cell as parity evidence. Future release
  output from the fixed benchmark will print `n/a`.
- [x] Extend the SciPy runner to optionally emit its complete solution vector
  and SHA-256, and report its BLAS vendor/version, configured thread policy,
  and SuperLU ordering/pivot settings.
- [x] Add an ignored debug cross-language gate comparing Rust bridge and SciPy
  on the same CSC matrix and RHS: exact materialized `L.nnz/U.nnz`, complete
  solution-vector max difference, and residual. It passes both for the
  generated n=50 fixture and captured n=1022 BVP Jacobian. Captured result:
  `L/U=2530/4224` on both paths, solution max difference `4.177e-16`, relative
  residual `2.372e-20`.
- [x] Constrain SciPy OpenBLAS to one thread during the parity gate and record
  the distinct BLAS variants: SciPy OpenBLAS `0.3.31.dev` vs SuperLU's bundled
  sequential CBLAS. This controls thread count, not BLAS implementation; the
  two builds must stay separate in performance tables.
- [x] Allow the debug/release harness to take multiple semicolon-separated
  captured fixtures while retaining the singular environment variable alias.
  Current corpus has one real captured BVP Jacobian plus deterministic
  synthetic sizes and explicit-zero controls.
- [ ] Capture additional real BVP matrices across workloads and Newton
  iterations/patterns. Use the new fixture-list gate to establish correctness
  first; run release performance comparisons only after that debug matrix
  suite is populated.
- [ ] For a strict same-BLAS timing control, build a bridge against the exact
  SciPy OpenBLAS DLL/import ABI if it can be done without fragile symbol
  shims. Otherwise keep bundled CBLAS and SciPy OpenBLAS as explicitly named
  variants and avoid attributing small timing deltas to LU alone.

Debug evidence:
`test_reports/somelinalg_sparse_lu_reference/debug/superlu_bridge_matches_scipy_factorization_and_solution_on_shared_matrix.md`.
No release command was run for these correctness changes.

### First Strategy/Zero-Control Results (2026-10-12)

The targeted 5-repeat release control passed all backward-error and known-
solution checks. On these synthetic fixtures, AUTO and forced simplicial have
similar costs, while forced supernodal is much slower at the medium and large
sizes. At n=514 its cold factor was 24.3 ms versus 0.23 ms for AUTO; at n=1538
it was 32.4 ms versus 0.75 ms. Do not select forced supernodal for this BVP-like
pattern. AUTO's timings are consistent with the simplicial path, but the public
faer API does not expose which variant AUTO selected.

The same-pattern zero control retained every CSC coordinate and had 92.8%
zero stored values at n=514 and 1538. Faer numeric factor medians changed from
230.2 to 225.8 us and 746.4 to 739.2 us respectively: little effect at these
sizes. SuperLU factor medians changed from 279.3 to 244.9 us and 750.3 to
622.6 us; its reported `L+U` stored entries fell from 8,687 to 1,102 and from
25,989 to 3,307. This suggests explicit structural zeros influence SuperLU's
numeric fill/work differently than faer's path, but these fixtures are still
synthetic and do not establish the cause of the captured BVP gap. The n=50
control has 88.6% zero values because diagonal entries are retained.

Rust synthetic-control report:
`test_reports/BVP_sci_sparse_lu_diagnostics_20261012/reports/somelinalg_sparse_lu_reference/release/rust_microbenchmark.md`.
The six Python `splu` raw rows were captured from the same exported CSC
fixtures. The separately replayed captured production matrix and faer strategy
results are recorded in the following section.

### Captured BVP Matrix Strategy A/B (2026-10-12)

The existing BVP same-matrix diagnostic now exports the exact 1022-by-1022
Newton CSC matrix and RHS. Its report gives 124,684 stored coordinates,
4,864 numerical nonzeros, and SuperLU factor fill `L.nnz=2,530`, `U.nnz=4,224`.
Replaying that exact input through faer with five release samples produced:

| faer symbolic policy | cold symbolic + numeric (ms) | warm numeric median (ms) | solve median (ms) | backward error |
| --- | ---: | ---: | ---: | ---: |
| AUTO | 13.031 | 14.805 | 0.033 | 9.438e-17 |
| forced simplicial | 9.396 | 6.998 | 0.061 | 8.651e-17 |
| forced supernodal | 12.067 | 9.278 | 0.032 | 9.438e-17 |

The forced simplicial numeric refactor is about 2.1x faster than AUTO on this
matrix, while preserving backward error. The AUTO result is close to the slow
standard BVP Sparse factor timing; the result strongly suggests AUTO chooses a
less suitable strategy for this collocation structure. This is the first
specific high-value candidate for BVP-level A/B. It remains a single captured
Newton matrix and five samples, so keep it opt-in until the full BVP
continuation story confirms the gain on changing Jacobian values. The exact
capture and strategy report are under
`test_reports/BVP_sci_sparse_lu_diagnostics_20261012/`.

- [x] Add forced simplicial as a separate opt-in BVP_sci policy;
  `StandardFaer` and reusable AUTO remain unchanged.
- [x] Debug-smoke the expanded BVP A/B story. All three strategies returned
  identical final mesh/state with matching factorization and symbolic-reuse
  counters. The observed debug wall times are intentionally not interpreted as
  backend performance evidence.
- [x] Run the targeted release BVP full-solve/continuation A/B with standard
  faer, reusable AUTO, and reusable forced-simplicial (5 repeats, 8 parameter
  rebinds). Final mesh/state and counters matched (`16` numeric factors, `8`
  symbolic analyses, `8` reuses). Series medians were `1.639 / 1.607 / 1.602
  ms`; cumulative numeric-factor scopes were `0.041 / 0.031 / 0.029 ms`.
  Forced simplicial is about `2.3%` faster end-to-end than Standard on this
  single workload, so it remains opt-in. Report:
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [x] Repeat on a large nonlinear parameterized logistic BVP (512 nodes / 1024
  unknowns), with 9 paired repetitions and 8 continuation rebinds. Final
  solutions and counters matched. Standard / reusable AUTO / forced simplicial
  solve-series median/MAD was `15.224/0.046`, `14.888/0.030`, and
  `14.872/0.005 ms`; paired deltas versus Standard were `-2.57%` and `-2.44%`.
  Reusable numeric-factor median/MAD was `0.140/0.002 ms` for AUTO and
  `0.139/0.002 ms` for simplicial, versus `0.303/0.009 ms` Standard. Thus
  buffer reuse is the source of the factor-stage gain here; forced simplicial
  is indistinguishable from AUTO. The end-to-end gain is modest and
  workload-specific. Report:
  `test_reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [ ] Measure retained/peak memory and repeat on another representative BVP
  before deciding whether to promote a reusable strategy from opt-in status.
- [x] Add a BVP-level matched workload matrix for `stiff-decay` and
  `stiff-coupled`, comparing standard faer against reusable AUTO and forced
  simplicial at selectable mesh sizes. Debug smoke passed at 64/128 nodes;
  this validates correctness and telemetry only, not performance.
- [x] Run the matched release workload matrix at 64/256/1024 nodes with 9
  interleaved samples and compare its standard Sparse rows with the existing
  Banded-vs-Sparse scale matrix. All rows passed and matched correctness/counter
  gates. Fixed-mesh cases perform one numeric factorization and zero symbolic
  reuses, so they do not test amortized reuse. Reusable AUTO/simplicial have no
  consistent one-shot full-solve win; retain StandardFaer as default and keep
  experimental policies opt-in. Banded's structured path is competitive on
  `stiff-decay`; its `stiff-coupled` fallback is slower or near-tied with Sparse.
  Reports:
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/sparse_kernel_real_workload_matrix_ab.md`;
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/lambdify_layout_scale_matrix.md`.
- [ ] Repeat the release workload matrix independently, then add a prepared
  multi-workload continuation comparison that actually exercises repeated
  same-pattern numeric refactorization before considering a default change or
  retiring Banded.
- [x] Trace the large Banded diagnostic values on `stiff-coupled`: they are the
  structured candidate's absolute residual and solution infinity norm recorded
  before the `1e-8` residual guard. When rejected, fallback solves the original
  RHS but the candidate diagnostics intentionally remain in the snapshot; they
  are not measurements of the returned fallback solution. Telemetry docs and
  story table headings now state this, and the debug scale story asserts both
  candidate rejection and accurate final BVP state. Targeted debug run passed
  at 64/256/1024 nodes for both frontends (`6/6` rows). Do not treat these
  values as evidence of a fallback correctness failure.
- [x] Add a BVP-level policy A/B for the official 60-state/two-parameter
  problem, with the production ExprLegacy/Sparse route, one full solve and
  prepared parameter continuation. Debug smoke matched all policies on the
  17-node mesh and counters (7 numeric factors, 3 symbolic analyses, 4 reuses);
  maximum state drift was `2.925e-9`. Debug timing is not evidence.
- [ ] Run the official 60-state policy A/B in release with 9 interleaved
  repetitions and 4 continuation rebinds. This is the next full-solve test of
  reusable factorization on the workload behind the original large-system
  gap. Run through `scripts/bvp_sci_release_matrix.ps1` with
  `-IncludeSparseStrategyAb`; keep standard faer as the reference policy until
  the release table is reviewed.

## Comparison Contract

- Direct parity: same CSC matrix and RHS; compare solution and residual.
- Direct cold timing: Python `splu + solve` versus Rust `sp_lu + solve`, with
  assembly reported separately.
- Rust-only lifecycle timing: symbolic analysis, repeated numeric refactor, and
  repeated solve are reported separately.
- Warm Rust numeric refactor is not compared against `splu`, whose public API
  does not expose symbolic-factor reuse.

## Initial Release Evidence (2026-10-12)

The Rust microbenchmark used 9 samples per operation. Python `splu` used the
same exported CSC fixtures and 9 samples; fixture parsing was outside the
timers. Both report matrix assembly separately from factorization and solve.
The dimensions below are synthetic BVP-shaped systems, not captured production
collocation Jacobians.

| Dimension | CSC nnz | Rust faer cold factor + solve | SciPy `splu` factor + solve | Rust/SciPy | Assembly Rust / Python |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 474 | 12.2 us | 61.7 us | 0.20x | 2.6 / 48.5 us |
| 514 | 8,162 | 265.2 us | 314.1 us | 0.84x | 92.3 / 99.9 us |
| 1,538 | 24,546 | 802.9 us | 674.1 us | 1.19x | 213.9 / 156.6 us |

Rust faer cold factorization is faster on the 50- and 514-dimensional fixtures,
but slower on the largest fixture. This is evidence against a universal
"Rust wins" claim and is not yet a stable performance baseline. Assembly is
also a measurable part of the large case and is not included in the factor-plus-
solve ratio. Python CSC assembly is `COO -> CSC + sum_duplicates`; Rust assembly
builds CSC from the same triplets, so treat assembly as a broadly matched but
implementation-specific measurement.

The experimental reusable kernel does not improve one-shot cold work:

| Dimension | faer cold factor | reusable cold symbolic + numeric | reusable warm numeric refactor |
| ---: | ---: | ---: | ---: |
| 50 | 11.6 us | 11.4 us | 3.9 us |
| 514 | 260.0 us | 300.0 us | 78.6 us |
| 1,538 | 787.9 us | 837.9 us | 242.7 us |

The warm numeric column is useful only for repeated factorizations with an
unchanged CSC pattern; it is not comparable to a fresh `splu` call by itself.
At dimensions 514 and 1,538, reusable cold totals are about 15% and 6% slower
than the existing faer cold factorization, while later warm numeric refactors
are about 70% faster than a fresh faer factorization. This suggests possible
amortization for repeated BVP Newton matrices with stable structure, but does
not yet establish a production win.

One timing-attribution question remains: `reusable_cold_total` (300.0 / 837.9
us) is materially larger than the sum of separately measured symbolic setup
and warm numeric refactor (214.5 / 604.8 us). These are different lifecycle
states (fresh allocation vs warmed retained buffers), so the discrepancy is
plausible, but the next benchmark should expose first numeric-factor time
separately from warm refactor and record allocation/workspace setup scopes.

All residuals and known-solution errors remained near floating-point precision
(maximum observed relative residual `1.79e-15`; maximum solution error
`2.22e-15`). Correctness is established for these deterministic fixtures and
the small BVP restart gate only. The reusable kernel is now connected to
BVP_sci behind an opt-in policy, but is not the default; shared banded solvers
remain unchanged.

## Production-Shaped Full-Solve Continuation A/B (2026-10-12)

The official 60-state/two-parameter BVP was run through the production
ExprLegacy/Sparse route, comparing StandardFaer, reusable AUTO and reusable
forced-simplicial. Each of 9 interleaved samples included one initial solve and
4 prepared parameter rebinds. All strategies passed with matching work counts
(14 numeric factors, 3 symbolic analyses, 11 symbolic reuses) and identical
mesh; maximum final-state drift was `3.190e-9`.

| policy | full series median/MAD (ms) | paired delta | factor scope median/MAD (ms) |
|---|---:|---:|---:|
| StandardFaer | 233.131 / 72.356 | baseline | 183.378 / 73.991 |
| reusable AUTO | 175.626 / 42.057 | -11.31% | 127.908 / 40.731 |
| reusable forced-simplicial | 162.085 / 0.681 | -30.52% | 113.862 / 0.385 |

This is a promising production-shaped signal, especially for forced
simplicial, but the baseline spread is large and asymmetric. Keep both reusable
policies opt-in; run an independent idle-machine repeat and inspect paired
sample deltas before selecting a default. Full report:
`test_reports/BVP_sci_release_manual/20261012T174404Z/reports/BVP_sci_new/release/sparse_kernel_official_60_state_continuation_ab.md`.

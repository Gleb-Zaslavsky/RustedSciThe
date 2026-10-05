# BVP_Damp Story Tests: AOT Lifecycle and Chunking

AOT build/runtime lifecycle, toolchain selection, artifact policy, locking, retries, chunking, and generated-backend handoff. Cold, warm, and prebuilt measurements must not be compared as one metric.

## Chunking and ABI status (2026-09-21)

The existing generated AOT route already supports explicit-entry `whole`,
`chunk4` and `Auto` policies for Sparse and Banded callbacks. The release
evidence below records callback equivalence, actual chunk/job counts and the
toolchain-dependent break-even behavior. In particular, the Banded
`tcc/whole` versus `tcc/chunk4` rows in
`BVP_DAMP_STORY_PERFORMANCE.md` are valid evidence for the historical
explicit-entry ABI.

The new AtomView-native `BandedCompactValues` ABI is a separate, opt-in
full-slot contract. It currently uses one complete compact callback because
the output buffer contains `(kl + ku + 1) * cols` slots, including boundary
zeros. This does not remove or replace explicit-entry chunking. Compact
chunking will be enabled only after a disjoint column/slot ownership rule,
boundary-zero initialization and Rust/C/Zig parity tests are in place.

## Solver-level compact Banded gate (2026-09-21, debug)

`compact_banded_solver_parity_rebind_and_telemetry` is the first solver-facing
gate for the full-slot AtomView Banded ABI. It does not build an external
compiler artifact; instead it registers a typed in-process linked backend and
exercises the same callback adapter used by the BVP solver bundle.

```powershell
cargo test --lib --no-default-features symbolic::bvp::legacy::bvp_sparse_chunking_tests::compact_banded_solver_parity_rebind_and_telemetry -- --nocapture --test-threads=1
```

Result on 2026-09-21: `shape=8`, `storage=32`, two residual and two Jacobian
calls across two parameter bindings, with separate non-zero residual/Jacobian
telemetry. The matrix values changed after rebind and the second callback
published a distinct `BandedMatrixType` factor owner, so the old factor cannot
be reused accidentally. The report is also stored under
`test_reports/BVP_Damp_AOT/`; this debug result is a lifecycle/correctness
gate, not a release performance baseline.

## Cross-toolchain compact-manifest gate (2026-09-21, debug)

The compact Banded marker is validated by one shared function before Rust, C or
Zig registration opens a dynamic library. This keeps malformed bandwidth,
backend or storage-length metadata on the same diagnostic path and avoids
turning a manifest defect into an opaque loader error. The C and Zig tests use a
missing library on purpose; a compact storage mismatch must win over the file
system error.

```powershell
cargo test --lib --no-default-features symbolic::codegen::codegen_aot_runtime_link::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features symbolic::codegen::c_backend::codegen_c_aot_runtime_link::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::tests -- --nocapture --test-threads=1
```

Result: shared Rust validation, C pre-load rejection and Zig pre-load
rejection all passed in debug on 2026-09-21. This is a manifest/lifecycle gate,
not evidence that a real C or Zig compact artifact has been compiled and solved.
Materialized cross-toolchain numerical parity is covered by the dedicated gate
below; the large release matrix remains open.

The backend-comparison harness was also aligned with this contract: Sparse
uses the sparse registration function, while Banded uses the corresponding
Rust/C/Zig banded registration function. This prevents a Banded story from
silently bypassing layout validation.

## Materialized AtomView compact parity gate (2026-09-21, debug)

The previously open lifecycle gap is now covered by an ignored debug gate on a
real `small-damp1-24` BVP. The test prepares AtomView without an Expr round-trip,
materializes and loads Rust, C and Zig artifacts when the toolchain is present,
then invokes the same typed whole residual and full-slot compact Jacobian
callbacks that the solver consumes. The ExprLegacy/lambdify matrix is used only
as the numerical oracle; it is normalized into the AtomView slot map inside the
test, so different legacy sparsity pruning cannot masquerade as ABI drift.

```powershell
cargo test --lib --no-default-features bvp_atomview_native_compact_banded_materialized_cross_toolchain_parity -- --ignored --nocapture --test-threads=1
```

Debug result on 2026-09-21:

```text
Rust: residual_diff=0.000e0, compact_jacobian_diff=0.000e0, output_len=192
C:    residual_diff=0.000e0, compact_jacobian_diff=0.000e0, output_len=192
Zig:  residual_diff=3.469e-18, compact_jacobian_diff=0.000e0, output_len=192
```

The test is intentionally ignored because it starts external build processes.
It is now the required debug gate immediately before the expensive release
stories, not a release performance baseline. Missing C/Zig toolchains are
reported as skipped; an available toolchain that fails materialization,
linking or numerical parity fails the test.

The strict no-fallback boundary is also covered by
`bvp_native_compact_builder_rejects_incomplete_atom_route_without_fallback`.

## AtomView Sparse compatibility-bridge regression (2026-09-21, debug)

The first pre-release acceptance pass found four failures in the existing Rust
and TCC AOT smoke gates. The generated callbacks were valid, but the retained
solver bridge asked the Expr-compatible payload for Sparse coordinates. That
payload is intentionally empty on AtomView-native preparation, so the bridge
allocated a zero-length Jacobian buffer while the linked manifest correctly
expected `nnz` values.

The bridge now uses the prepared Atom-aware `sparse_structure()` and does not
reintroduce an Expr conversion or fallback. The following four gates passed
after the fix; the output and integer solver counters are captured by the
usual `test_reports/BVP_Damp_AOT/` records. These are debug correctness gates;
the dated release timing rows remain a separate baseline.

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::aot_rust_default_parallel_exact_examples_cover_parallel_modes_and_chunking -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::aot_tcc_smoke_exact_two_point_small_grid_solves -- --nocapture --test-threads=1
```

Result: all four passed on 2026-09-21. Combustion sequential/parallel
comparison reported `max_difference_seq_vs_par=0.000000e0`; the exact sequential
case reported `error=1.172771e-2`, the exact parallel case
`l2_error=2.870510e-4`, and the TCC two-point smoke reported
`error=3.045600e-3`.

## AtomView-native artifact-builder gate (2026-09-21, debug)

The BVP bridge now has an explicit opt-in builder for native compact Banded
artifacts. The historical builder is unchanged and keeps emitting the
explicit-entry compatibility ABI. The new builder derives the manifest from
the prepared codegen object itself, so the compact marker and full-slot count
cannot be lost through stale adapter metadata.

The solver-level compact gate also checks this builder path and reports
`shape=8`, `storage=32`, with the expected `BandedCompact { kl, ku }` manifest
layout. It remains a debug correctness gate; no compiler invocation is part
of this test.

```powershell
cargo test --lib --no-default-features symbolic::bvp::legacy::bvp_sparse_chunking_tests::compact_banded_solver_parity_rebind_and_telemetry -- --nocapture --test-threads=1
```

## Non-Ignored Acceptance and Tuning Checks


These are not the heavy release matrix, but they are useful quick gates before paying
for combustion-1000 release runs.


### `aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Checks production-style default Rust AOT acceptance on exact examples with sequential
execution. This is a smoke/parity gate, not a toolchain benchmark.

```powershell
cargo test --release aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps -- --nocapture --test-threads=1
```

Result:
[AOT solve] clairaut-220: solve took 163.7334ms
[AOT exact sequential] clairaut-220: n_steps=220, error=1.172771e-2
ok
```text
Date:
Conclusion:
```


### `aot_rust_default_parallel_exact_examples_cover_parallel_modes_and_chunking`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Checks default Rust AOT parallel modes and chunking on exact examples.

```powershell
cargo test --release aot_rust_default_parallel_exact_examples_cover_parallel_modes_and_chunking -- --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


### `aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Default Rust AOT acceptance coverage for combustion across sequential/parallel
execution and varied grids. This is a smaller preflight before the full 1000-step
release stories.

```powershell
cargo test --release aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids -- --nocapture --test-threads=1
```


### `aot_tcc_smoke_exact_two_point_small_grid_solves`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Compact non-Rust AOT smoke test. It runs a small exact two-point BVP through the
TCC sparse AtomView path. If TCC is not installed or the local artifact environment
cannot build/load TCC libraries, the test reports an environment skip; if TCC runs,
the numerical error is asserted.

```powershell
cargo test --release aot_tcc_smoke_exact_two_point_small_grid_solves -- --nocapture --test-threads=1
```

Result:
[AOT solve] tcc-smoke-two-point-40: solve took 212.9625ms
[AOT TCC smoke] two-point-40: error=3.045562e-3
```text
Date:
Conclusion:
```

Result:

```text
Date:
Conclusion:
```


### `combustion_sparse_aot_callback_chunking_parallelism_diagnostic`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

This is a diagnostic test, not an end-to-end story. It answers a narrower question:
does the sparse AtomView AOT runtime callback path itself benefit from whole-vs-chunked
execution once the artifact is already available? The test builds a whole sequential
callback and several explicit chunked variants, checks each chunked callback against
the whole callback numerically, then repeatedly evaluates residual and Jacobian
callbacks without running Newton, damping, mesh logic, or linear solves.

This diagnostic now runs sparse AtomView AOT for Rust, C `gcc`, C `tcc`, and Zig.
For each toolchain it builds a `whole-sequential` baseline and then compares explicit
chunked policies against that same toolchain's whole callback. Use
`combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix` when the
question is full solver behavior; use this diagnostic when the question is whether
linked sparse callback chunking really registered chunks and runtime jobs.

Use this test when CPU utilization or the end-to-end tuning table looks suspicious.
If this callback-only test shows speedup but the full BVP solve does not, then the
bottleneck is outside residual/Jacobian evaluation. If this test also shows no speedup,
then chunking is either too fine/coarse for the workload, dominated by scheduling
overhead, or not effectively parallel on the current machine.

  ```powershell
  cargo test --release combustion_sparse_aot_callback_chunking_parallelism_diagnostic -- --ignored --nocapture
  ```

  The guard version of this diagnostic intentionally uses a moderate grid (`n_steps=200`).
  Earlier `n_steps=1000` Rust-AOT diagnostic artifacts could fail during generated-crate
  compilation with a rustc stack overflow before reaching the runtime callback layer. That
  failure is useful information for large Rust AOT stress testing, but it is not the right
  substrate for a crisp "did runtime chunking actually bind?" regression gate.

The table also prints `workers`, `res_ch`, `jac_ch`, `res_jobs`, and `jac_jobs`.
These columns are deliberately low-level. If chunked rows show more than one chunk
and more than one job, the parallel runtime binding is active and the absence of
speedup should be interpreted as overhead/economics. If they show zero or one job,
the problem is configuration propagation, artifact registration, or callback rebinding.

Result:
CPU 4 Core historical run, kept for comparison with the 12 Core callback-binding
economics below. It proves the old callback-registration fix, but no longer
defines the current performance recommendation.
[BVP symbolic assembly diff] label=combustion-sparse-aot-callback-zig-par-16x16-jobs16-200, residual_max_diff=0.000000e0, jacobian_max_diff=0.000000e0
[BVP callback parallelism diagnostic] sparse AtomView AOT callbacks, n_steps=200, callback_iters=30, measurement_repeats=5
note: this isolates residual/Jacobian callback evaluation; bootstrap_ms is reported only to expose artifact overhead and is not part of callback throughput.
config                       | bootstrap_ms | workers |  res_ch |  jac_ch | res_jobs | jac_jobs | residual_ms        | jacobian_ms        | callback_total_ms  | speedup_vs_whole   |   residual_diff |   jacobian_diff
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rust/whole-sequential        |     5852.147 |       4 |      16 |      16 |       1 |       1 | 0.886 +/- 0.167    | 7.447 +/- 0.391    | 8.334 +/- 0.552    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
rust/par-4x4-jobs4           |     4395.443 |       4 |       4 |       4 |       4 |       4 | 1.097 +/- 0.444    | 8.683 +/- 0.659    | 9.780 +/- 1.063    | 0.861 +/- 0.086    |      0.000000e0 |      0.000000e0
rust/par-8x8-jobs8           |     4341.116 |       4 |       8 |       8 |       8 |       8 | 0.842 +/- 0.063    | 8.355 +/- 1.197    | 9.197 +/- 1.190    | 0.920 +/- 0.104    |      0.000000e0 |      0.000000e0
rust/par-16x16-jobs16        |     3491.071 |       4 |      16 |      16 |      16 |      16 | 1.209 +/- 0.353    | 8.857 +/- 0.742    | 10.065 +/- 0.911   | 0.835 +/- 0.080    |      0.000000e0 |      0.000000e0
gcc/whole-sequential         |     1924.336 |       4 |      16 |      16 |       1 |       1 | 0.996 +/- 0.086    | 7.909 +/- 0.862    | 8.905 +/- 0.941    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
gcc/par-4x4-jobs4            |     1842.794 |       4 |       4 |       4 |       4 |       4 | 0.749 +/- 0.004    | 7.063 +/- 0.159    | 7.811 +/- 0.158    | 1.140 +/- 0.023    |      0.000000e0 |      0.000000e0
gcc/par-8x8-jobs8            |     1627.682 |       4 |       8 |       8 |       8 |       8 | 0.790 +/- 0.105    | 6.979 +/- 0.352    | 7.769 +/- 0.456    | 1.150 +/- 0.062    |      0.000000e0 |      0.000000e0
gcc/par-16x16-jobs16         |     1613.404 |       4 |      16 |      16 |      16 |      16 | 0.926 +/- 0.130    | 6.955 +/- 0.198    | 7.881 +/- 0.269    | 1.131 +/- 0.038    |      0.000000e0 |      0.000000e0
tcc/whole-sequential         |      591.398 |       4 |      16 |      16 |       1 |       1 | 0.946 +/- 0.092    | 7.111 +/- 0.133    | 8.057 +/- 0.208    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
tcc/par-4x4-jobs4            |      520.016 |       4 |       4 |       4 |       4 |       4 | 0.975 +/- 0.040    | 7.660 +/- 0.887    | 8.635 +/- 0.870    | 0.942 +/- 0.087    |      0.000000e0 |      0.000000e0
tcc/par-8x8-jobs8            |      507.448 |       4 |       8 |       8 |       8 |       8 | 0.973 +/- 0.044    | 7.238 +/- 0.405    | 8.211 +/- 0.387    | 0.983 +/- 0.045    |      0.000000e0 |      0.000000e0
tcc/par-16x16-jobs16         |      528.367 |       4 |      16 |      16 |      16 |      16 | 1.011 +/- 0.019    | 7.267 +/- 0.300    | 8.279 +/- 0.309    | 0.975 +/- 0.036    |      0.000000e0 |      0.000000e0
zig/whole-sequential         |    31593.646 |       4 |      16 |      16 |       1 |       1 | 0.908 +/- 0.124    | 7.857 +/- 0.298    | 8.765 +/- 0.393    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
zig/par-4x4-jobs4            |    28172.316 |       4 |       4 |       4 |       4 |       4 | 0.989 +/- 0.070    | 8.198 +/- 0.652    | 9.187 +/- 0.623    | 0.958 +/- 0.062    |      0.000000e0 |      0.000000e0
zig/par-8x8-jobs8            |    29051.344 |       4 |       8 |       8 |       8 |       8 | 1.029 +/- 0.133    | 7.680 +/- 0.242    | 8.710 +/- 0.286    | 1.007 +/- 0.032    |      0.000000e0 |      0.000000e0
zig/par-16x16-jobs16         |    29874.693 |       4 |      16 |      16 |      16 |      16 | 1.004 +/- 0.032    | 8.512 +/- 0.797    | 9.516 +/- 0.780    | 0.927 +/- 0.074    |      0.000000e0 |      0.000000e0
test numerical::BVP_Damp::BVP_Damp_tests3::tests::combustion_sparse_aot_callback_chunking_parallelism_diagnostic ... ok
CPU 12 Core (current source of truth)

[BVP callback parallelism diagnostic] sparse AtomView AOT callbacks, n_steps=200, callback_iters=30, measurement_repeats=5
note: this isolates residual/Jacobian callback evaluation; bootstrap_ms is reported only to expose artifact overhead and is not part of callback throughput.
config                       | bootstrap_ms | workers |  res_ch |  jac_ch | res_jobs | jac_jobs | residual_ms        | jacobian_ms        | callback_total_ms  | speedup_vs_whole   |   residual_diff |   jacobian_diff
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rust/whole-sequential        |     2975.310 |      24 |      93 |      93 |       1 |       1 | 0.389 +/- 0.124    | 2.109 +/- 0.184    | 2.498 +/- 0.279    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
rust/par-4x4-jobs4           |     1737.750 |      24 |       4 |       4 |       4 |       4 | 0.270 +/- 0.022    | 1.854 +/- 0.161    | 2.124 +/- 0.180    | 1.184 +/- 0.093    |      0.000000e0 |      0.000000e0
rust/par-8x8-jobs8           |     1509.011 |      24 |       8 |       8 |       8 |       8 | 0.254 +/- 0.082    | 1.868 +/- 0.112    | 2.122 +/- 0.194    | 1.186 +/- 0.099    |      0.000000e0 |      0.000000e0
rust/par-16x16-jobs16        |     1309.818 |      24 |      16 |      16 |      16 |      16 | 0.241 +/- 0.060    | 1.824 +/- 0.194    | 2.065 +/- 0.253    | 1.225 +/- 0.127    |      0.000000e0 |      0.000000e0
gcc/whole-sequential         |      836.947 |      24 |      93 |      93 |       1 |       1 | 0.314 +/- 0.067    | 1.880 +/- 0.098    | 2.194 +/- 0.163    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
gcc/par-4x4-jobs4            |      687.352 |      24 |       4 |       4 |       4 |       4 | 0.223 +/- 0.031    | 1.809 +/- 0.082    | 2.032 +/- 0.113    | 1.083 +/- 0.056    |      0.000000e0 |      0.000000e0
gcc/par-8x8-jobs8            |      697.798 |      24 |       8 |       8 |       8 |       8 | 0.229 +/- 0.034    | 1.801 +/- 0.039    | 2.030 +/- 0.073    | 1.082 +/- 0.037    |      0.000000e0 |      0.000000e0
gcc/par-16x16-jobs16         |      662.370 |      24 |      16 |      16 |      16 |      16 | 0.229 +/- 0.034    | 1.834 +/- 0.259    | 2.063 +/- 0.292    | 1.081 +/- 0.126    |      0.000000e0 |      0.000000e0
tcc/whole-sequential         |      195.477 |      24 |      93 |      93 |       1 |       1 | 0.369 +/- 0.077    | 1.950 +/- 0.091    | 2.318 +/- 0.168    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
tcc/par-4x4-jobs4            |      180.693 |      24 |       4 |       4 |       4 |       4 | 0.268 +/- 0.041    | 1.947 +/- 0.247    | 2.215 +/- 0.288    | 1.062 +/- 0.116    |      0.000000e0 |      0.000000e0
tcc/par-8x8-jobs8            |      261.912 |      24 |       8 |       8 |       8 |       8 | 0.266 +/- 0.036    | 1.842 +/- 0.071    | 2.108 +/- 0.106    | 1.102 +/- 0.052    |      0.000000e0 |      0.000000e0
tcc/par-16x16-jobs16         |      178.218 |      24 |      16 |      16 |      16 |      16 | 0.318 +/- 0.098    | 1.838 +/- 0.078    | 2.155 +/- 0.175    | 1.082 +/- 0.078    |      0.000000e0 |      0.000000e0
zig/whole-sequential         |    12098.448 |      24 |      93 |      93 |       1 |       1 | 0.488 +/- 0.072    | 2.291 +/- 0.120    | 2.778 +/- 0.161    | 1.000 +/- 0.000    |      0.000000e0 |      0.000000e0
zig/par-4x4-jobs4            |    11984.866 |      24 |       4 |       4 |       4 |       4 | 0.352 +/- 0.031    | 2.118 +/- 0.154    | 2.470 +/- 0.185    | 1.131 +/- 0.076    |      0.000000e0 |      0.000000e0
zig/par-8x8-jobs8            |    11967.489 |      24 |       8 |       8 |       8 |       8 | 0.366 +/- 0.035    | 2.137 +/- 0.244    | 2.503 +/- 0.277    | 1.122 +/- 0.107    |      0.000000e0 |      0.000000e0
zig/par-16x16-jobs16         |    12023.846 |      24 |      16 |      16 |      16 |      16 | 0.365 +/- 0.034    | 2.079 +/- 0.065    | 2.444 +/- 0.098    | 1.138 +/- 0.043    |      0.000000e0 |      0.000000e0
test numerical::BVP_Damp::BVP_Damp_tests3::tests::combustion_sparse_aot_callback_chunking_parallelism_diagnostic ... ok
```text
Date: 2026-05-26
Status: passed; runtime parallel binding is proven, but medium-grid callback speedup is toolchain-dependent and usually small.
Important numbers:
  Every chunked row is numerically exact against its whole callback:
  `residual_diff = 0`, `jacobian_diff = 0`.
  Runtime binding is genuine: chunked rows report the requested `4`, `8`, or
  `16` linked residual/Jacobian chunks and the same number of executed jobs.
  On the current 12 Core run, every toolchain reports real multi-job callback
  execution and modest callback-total gains for at least one chunked layout:
  Rust improves from `2.498 ms` whole to `2.065 ms` at `16x16`, gcc from
  `2.194 ms` to about `2.03 ms`, tcc from `2.318 ms` to `2.108 ms`, and Zig
  from `2.778 ms` to `2.444 ms`.
  The older 4 Core rows are retained only as historical contrast; they showed
  weaker or negative economics for several toolchains.
Conclusion:
  This rerun closes the binary correctness question: chunk functions are
  exported, bound, and executed through a genuinely parallel runtime route.
  It also shows why `Auto` should be workload-aware rather than blindly
  sequential. On 12 Core, real parallel callback execution is visible and useful
  even for this moderate grid, but the gains are still small enough that full
  solver economics must be checked separately.
Follow-up:
  Use this test as a binding/correctness guard and callback-level scaling check,
  not as a complete solver-performance recommendation. Use the full solve
  stories for application-level economics.
```

Current verdict:

This diagnostic originally exposed a real runtime-registration gap. The solver-side
policy propagation was already correct: `AotExecutionPolicy::Parallel(...)` reached
the BVP handoff and `rebind_linked_runtime_callbacks(..., Some(config))`. The silent
fallback happened later because linked sparse AOT backends were registered only with
the whole exported ABI symbols, `rustedscithe_aot_eval_residual` and
`rustedscithe_aot_eval_jacobian_values`. The generated libraries could contain
internal chunk functions, and the whole wrapper could call them sequentially, but the
chunk functions were not exported and not registered as `LinkedResidualChunk` /
`LinkedSparseJacobianChunk`. As a result, `res_ch=0`, `jac_ch=0`, `res_jobs=0`, and
`jac_jobs=0` meant "whole callback fallback", not real runtime parallelism.

The codegen/runtime gap has now been fixed in the generator and linker layer. Generated
Rust, C, and Zig sparse AOT libraries emit `rustedscithe_aot_chunk_*` FFI symbols for
residual and Jacobian chunks, and sparse cdylib registration loads those symbols from
the manifest chunk metadata and attaches them through `with_chunked_evaluators(...)`.
The diagnostic test is now a real guard: chunked variants must register more than one
residual/Jacobian chunk and must produce more than one runtime job. If a future refactor
silently falls back to whole-callback execution again, this test panics instead of
printing a misleading performance table.

The guard intentionally uses `n_steps=200`. This keeps the test focused on runtime
chunk binding. Larger Rust-AOT diagnostic artifacts, especially `n_steps=1000`, can
fail during generated-crate compilation with a rustc stack overflow before reaching
the callback layer. That compiler-stress behavior belongs in a separate Rust-AOT
artifact scalability test, not in this runtime-parallelism guard.

Expected healthy signal: for `par-4x4-jobs4`, `par-8x8-jobs8`, and `par-16x16-jobs16`,
the `res_ch`, `jac_ch`, `res_jobs`, and `jac_jobs` columns should all be greater than
one. If they are zero, the artifact was likely built by an older generator or chunk
symbol loading failed.

Release interpretation from the current run:

The runtime binding is now healthy. The diagnostic reports real linked chunks and real
runtime jobs: `par-4x4-jobs4` has `res_ch=4`, `jac_ch=4`, `res_jobs=4`, `jac_jobs=4`;
`par-8x8-jobs8` and `par-16x16-jobs16` similarly show the expected chunk/job counts.
This closes the previous "parallel policy silently falls back to whole callback" failure
mode.

The performance signal is intentionally more conservative. At `n_steps=200`,
`whole-sequential` takes about `0.89 ms` for 30 residual calls and `7.57 ms` for 30
Jacobian calls. That is only about `0.03 ms` per residual evaluation and `0.25 ms` per
Jacobian evaluation. At that granularity, explicit chunk dispatch has little room to
win: each callback still pays for rayon scheduling and several FFI chunk calls. Older
measurements also paid for per-job temporary buffers, mutex-protected result
collection, and copy-back into the final output.

Follow-up fix:

The linked sparse AOT runtime path now writes chunk results directly into disjoint
slices of the caller-owned residual/Jacobian output buffers. The old
`Mutex<Vec<(offset, Vec<f64>)>>` aggregation path was removed from
`symbolic_functions_BVP.rs`, and the regression test
`linked_sparse_parallel_callbacks_write_directly_into_final_buffers` fails if a future
refactor starts passing temporary buffers to linked chunk callbacks again.

There is also a separate concurrency guard:
`linked_sparse_parallel_callbacks_actually_overlap_on_rayon_workers`. It runs the
linked sparse residual and Jacobian chunk dispatch inside a local four-thread rayon
pool and asserts that more than one chunk callback is active at the same time. This
test answers the binary question "are chunks actually executed concurrently?" It is
not a performance benchmark. A passing result means slow chunked rows should be
interpreted as overhead/economics, not as silent sequential execution.

Conclusion: sparse AOT callback chunking is now correct, actually bound at runtime,
and no longer carries the old temporary-buffer/copy-back overhead. The remaining
question is economic rather than correctness-related: for medium-grid combustion
callbacks, the work per FFI chunk may still be too small to beat a whole generated
callback. Re-run this diagnostic after the direct-write fix before treating older
speedup rows as authoritative.

Follow-up after the direct-write fix: linked AOT runtime `Auto` no longer uses the
old fixed `128/256` output thresholds to decide whether to fan out chunks. It now
delegates the decision to the same measured-overhead recommendation used by the
codegen executor layer. This does not hide explicit `chunk4` experiments: rows
using forced parallel execution still run as requested. It only prevents `Auto`
from quietly enabling a parallel path when the current machine and current chunk
granularity predict that scheduling/FFI overhead will dominate useful arithmetic.


## AOT Chunking and Parallel Execution Source of Truth


### `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Purpose: non-combustion Frozen coverage. The test solves a nonlinear polynomial
profile BVP with exact solution `y = 1 + x^2`, rewritten as a first-order system
`y' = z`, `z' = 2 + 0.1 * (y - (1 + x^2))^2`. This intentionally avoids the
combustion family while still exercising nonlinear symbolic residuals, Banded
AtomView assembly, Lambdify baseline, `tcc` AOT build, and strict
`RequirePrebuilt` reuse. The polynomial profile is deliberately used instead of
the earlier logarithmic Gaussian experiment because Frozen Newton has no damping
guard and the `ln(y)` route can step through negative intermediate states.

Command:

```powershell
cargo test --release frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture --test-threads=1
```

Result:
12 Core
[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: correctness/backend selection
source   | variant    | selected_backend | build_policy    | solve_diff
----------------------------------------------------------------------------------
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | build      | AotCompiled      | BuildIfMissing  | 0.000000e0
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 0.000000e0
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 0.000000e0

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: wall-clock and Newton stages; milliseconds
source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   | 2175.103 |    2000.000 |     0.000 |  0.000 |  1.000 |    14 |     14 |      1
AOT      | build      |   32.958 |      30.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1
AOT      | prebuilt   |    6.348 |       3.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1
AOT      | prebuilt   |    6.141 |       3.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: generated handoff and compiled callback stages; milliseconds
source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs
------------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |            5.692 |           0.571 |       NaN |          NaN |      NaN |      NaN
AOT      | build      |            3.372 |           0.457 |     0.407 |       12.019 |    1.000 |    1.000
AOT      | prebuilt   |            3.327 |           0.410 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |            3.360 |           0.360 |       NaN |          NaN |    1.000 |    1.000

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: repeated-run summary; milliseconds
source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------
AOT      | build      |    32.958 +/- 0.000     |       30.000 +/- 0.000     |      0.000 +/- 0.000     | 0.000000e0
AOT      | prebuilt   |     6.245 +/- 0.103     |        3.000 +/- 0.000     |      0.000 +/- 0.000     | 0.000000e0
Lambdify | AtomView   |  2175.103 +/- 0.000     |     2000.000 +/- 0.000     |      0.000 +/- 0.000     | 0.000000e0
ok
12 Core after refactor

BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: correctness/backend selection
source   | variant    | selected_backend | build_policy    | solve_diff
----------------------------------------------------------------------------------
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | build      | AotCompiled      | BuildIfMissing  | 0.000000e0
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 0.000000e0
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 0.000000e0

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: wall-clock and Newton stages; milliseconds
source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   | 2284.438 |    2000.000 |     0.000 |  0.000 |  1.000 |    14 |     14 |      1
AOT      | build      |   32.452 |      30.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1
AOT      | prebuilt   |    6.472 |       4.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1
AOT      | prebuilt   |    5.840 |       3.000 |     0.000 |  0.000 |  0.000 |    14 |     14 |      1

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: generated handoff and compiled callback stages; milliseconds
source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs
------------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |            5.794 |           0.713 |       NaN |          NaN |      NaN |      NaN
AOT      | build      |            3.091 |           0.352 |     0.349 |       10.905 |    1.000 |    1.000
AOT      | prebuilt   |            3.772 |           0.424 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |            3.294 |           0.385 |       NaN |          NaN |    1.000 |    1.000

[BVP Frozen story] nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: repeated-run summary; milliseconds
source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------
AOT      | build      |    32.452 +/- 0.000     |       30.000 +/- 0.000     |      0.000 +/- 0.000     | 0.000000e0
AOT      | prebuilt   |     6.156 +/- 0.316     |        3.500 +/- 0.500     |      0.000 +/- 0.000     | 0.000000e0
Lambdify | AtomView   |  2284.438 +/- 0.000     |     2000.000 +/- 0.000     |      0.000 +/- 0.000     | 0.000000e0
test numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story ... ok

Interpretation:

```text
12 Core release result: ok.

Correctness and lifecycle are locked for this non-combustion Frozen story:
Lambdify, BuildIfMissing AOT, and strict RequirePrebuilt AOT agree exactly in
the reported solution-difference metric (`0.000000e0`). The build row selects
`AotCompiled`, leaves a resolver snapshot, and reports a real compile/link
interval (`12.019 ms`). Both strict prebuilt rows stay `AotCompiled`, use
`RequirePrebuilt`, and have blank `compile_link`, so there is no hidden rebuild
and no hidden Lambdify fallback.

The apparently very slow Lambdify row (`2175 ms`, `symbolic_ms=2000 ms`) should
not be interpreted as hot Lambdify callback cost. The detailed handoff table
shows the actual generated handoff work is tiny for that row:
`initial_generate=5.692 ms` and `initial_sym_jac=0.571 ms`. This is a single
first-row cold-start artifact: logger/runtime initialization, one-time caches,
allocator/OS noise, and possibly already-warm AOT path differences dominate the
broad solver-level timer. The test is therefore a lifecycle/correctness gate,
not a Lambdify-vs-AOT performance ranking. If we want performance data for this
small polynomial problem, the next test should alternate Lambdify and prebuilt
AOT for several repetitions with cooldown, exactly like the combustion warm
story.
```

## AtomView Typed AOT Stage Parity (2026-09-21, instrumentation pass)

The historical `ExprLegacy` rows above remain the immutable AOT oracle. The
AtomView route now emits a typed stage report with the same canonical buckets:

```text
symbolic_prepare = validation + Atom preparation + Jacobian preparation
fixture_generation = lowering + source emission + materialization
compile = external compiler/build process
link = runtime registration/publication
```

The first three buckets are owned by the prepared AtomView plan. The last two
are recorded by the production BVP handoff, not by a test-only stopwatch. The
compatibility diagnostics projection uses `generated.aot.typed.*` keys so the
existing solver reports and the typed snapshot can be compared in one story
record. Warm residual/Jacobian calls and their integer counters remain separate
from cold preparation.

The acceptance criterion is deliberately conservative: AtomView AOT may not
replace ExprLegacy merely because its source or module generation is faster.
The dated release gate must use the same fixture, grid, matrix backend,
toolchain and profile, and must show residual/Jacobian/solution parity together
with no regression in the important cold stages, warm runtime and lifecycle
reliability. Until that gate is rerun, ExprLegacy AOT stays as a required
oracle and compatibility route.

The diagnostic command is:

```powershell
cargo test --release --lib --no-default-features symbolic_assembly_backends_report_combustion_aot_crate_build_table -- --ignored --nocapture --test-threads=1
```

The test report is written separately under `test_reports/BVP_Damp_AOT/` and
does not overwrite older dated baseline records.

## Canonical defaults and stage-table contract (2026-09-21)

The public `GeneratedBackendConfig::default()` contract is now explicit:

```text
Lambdify frontend: AtomView
AOT compiler:      C/tcc
matrix default:    selected by the high-level solver preset
```

`ExprLegacy`, Rust AOT, gcc, and Zig remain explicit compatibility or comparison
routes. They must not silently become the default merely because a diagnostic
test uses them. The canonical combustion end-to-end story therefore labels its
Lambdify row `AtomView` and compares it with the explicit C-gcc, C-tcc, and Zig
rows.

The end-to-end diagnostic prints two related tables. The first is the solver
correctness/wall-clock table with iterations, linear solves, Jacobian rebuilds,
and solution drift. The second is the cold AOT lifecycle table:

```text
source | variant | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms | residual_calls | jacobian_calls | status
```

Missing stages are reported as `0`, not `NaN`. AOT rows enable detailed typed
telemetry for this diagnostic only, so preparation, fixture, compile/link and
callback counters are real. This telemetry is not enabled in production unless
the caller requests it.

The Rust compile-preset table remains a separate Rust-only diagnostic. It is not
the canonical stage table and must not replace the cross-toolchain correctness
matrix or the AtomView+tcc default route.

### 2026-09-21: AtomView typed stage rerun and Banded callback repair

The following debug reruns were performed after the AtomView Banded callback
fix. Historical release records are intentionally retained; these numbers are
not a replacement release baseline because the rerun used the debug profile
and a single cold execution where noted.

Commands:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::symbolic_assembly_backends_report_combustion_aot_crate_build_table -- --ignored --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::combustion_1000_compiled_banded_zig_bootstrap_smoke -- --ignored --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::combustion_1000_end_to_end_banded_lapack_refine_statistics -- --ignored --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy -- --ignored --nocapture --test-threads=1
```

The canonical text reports are stored under `test_reports/BVP_Damp_AOT/` and
were refreshed on `2026-09-21` for all four tests. The direct crate-build
report now contains real typed cold-stage buckets:

```text
frontend | n_steps | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms
AtomView |     200 |              15.380 |                160.661 |   2678.995 |   0.000
AtomView |     300 |              23.411 |                220.912 |   3731.755 |   0.000
```

`link_ms=0` is expected in this test because it stops after a generated Rust
crate build and does not load/register a runtime library. The full Banded
end-to-end debug rerun completed with five Newton iterations, ten linear
solves and one Jacobian rebuild for each route:

```text
Lambdify ExprLegacy | total 1325.601 ms | solve 590.188 ms
C-gcc               | total 6104.024 ms | solve 3063.584 ms | diff 8.882e-16
C-tcc               | total 1756.982 ms | solve 1253.723 ms | diff 8.990e-16
Zig                 | total 24181.681 ms | solve 12470.267 ms | diff 8.333e-16
```

The Zig Banded bootstrap smoke also passed. The oscillator comparison passed
with equal reported maximum solution values and
`max_diff_lambdify_vs_atomview_aot=3.271975e-7`. These are correctness and
telemetry checks, not release performance conclusions. A release rerun with
the same fixture, toolchain, profile and repetitions is still required before
using the rows as a new baseline.

The original `0 values; expected 20988` failure was not a compiler failure.
Two lifecycle mismatches were corrected: compact Banded ABI selection is now
restricted to whole-Jacobian plans, while chunked plans use the explicit-entry
ABI; and the linked Banded callback derives its structure from AtomView sparse
entries instead of the intentionally empty Expr compatibility vectors.



## AOT Build, Artifact, and Runtime Stories


### `symbolic_assembly_backends_report_combustion_aot_crate_build_table`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Reports AOT crate emission/materialization/build behavior for combustion symbolic
backends. Use it when changing codegen module structure, artifact naming, or build
profiles.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_aot_crate_build_table -- --ignored --nocapture --test-threads=1
```

Result:
╰─────────────────────────────┴────────────────────╯
[BVP symbolic assembly AOT crate build] combustion ExprLegacy vs AtomView
backend      | n_steps | jac_prep_ms |   lookup_ms |      jac_ms |      nnz | finalize_ms |  module_ms |      source_ms | materialize_ms |   build_ms | source_kb |   blocks |    instr |    temps |  max_blk |  outputs | status            
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ExprLegacy   |     200 |       0.000 |       0.000 |       0.000 |        0 |       0.000 |     40.408 |         12.121 |      3.382 |   8421.773 |  549.4 |       32 |    13751 |    13751 |      598 |     5388 | ok                
AtomView     |     200 |       6.934 |       1.477 |       4.705 |     4188 |       0.664 |      3.502 |         12.930 |      3.041 |   7142.342 |  560.1 |       32 |    14020 |    14020 |      612 |     5388 | ok                
ExprLegacy   |     300 |       0.000 |       0.000 |       0.000 |        0 |       0.000 |     59.248 |         18.430 |      3.423 |  13272.824 |  815.3 |       32 |    20393 |    20393 |      892 |     8088 | ok                
AtomView     |     300 |      10.910 |       2.332 |       7.479 |     6288 |       0.968 |      3.863 |         17.945 |      3.782 |  12516.252 |  830.0 |       32 |    20765 |    20765 |      912 |     8088 | ok                

[BVP symbolic assembly AOT crate build] atom module pass breakdown
backend      | n_steps |  res_view_ms | res_lower_ms |    res_ph_ms | res_reuse_ms |   sp_view_ms |  sp_lower_ms |     sp_ph_ms |  sp_reuse_ms
-----------------------------------------------------------------------------------------------------------------------------------------------
AtomView     |     200 |        0.039 |        6.354 |        0.349 |        0.296 |        0.029 |        3.085 |        0.221 |        0.026
AtomView     |     300 |        0.011 |        6.829 |        0.183 |        0.063 |        0.066 |        3.864 |        0.104 |        0.032
ok
```text
Date:
Conclusion:
This table is a Rust/generated-crate build diagnostic for ExprLegacy vs AtomView,
not a cross-language compiler matrix. It answers whether symbolic frontend changes
move module/source/materialize/build costs. On the current combustion-200/300
fixtures, `build_ms` dominates this table by orders of magnitude: roughly 7-8.4 s
for n=200 and 12.5-13.3 s for n=300, while AtomView Jacobian preparation, module
lowering, source emission, and materialization stay in the millisecond-to-tens-of-
milliseconds range. AtomView is not the bottleneck here; the generated crate build is.

Do not use this table to conclude anything about Zig vs gcc vs tcc. For that, use
`bvp_generated_backend_pipeline_comparison_table`, which explicitly separates
`artifact_ms`, `materialize_ms`, `build_ms`, `link_ms`, and first callback issue for
Rust, C-gcc, C-tcc, and Zig.
```

### 2026-09-21: canonical default and live AOT telemetry smoke

The public defaults are now explicit and tested: Lambdify selects `AtomView`,
while the default compiled route selects C with `tcc`. The Rust compile-preset
table above remains intentionally Rust-only and is not a replacement for the
cross-toolchain stage table.

Debug smoke command:

```powershell
$env:BVP_BANDED_LAPACK_REPETITIONS='1'
$env:BVP_BANDED_LAPACK_COOLDOWN_MS='0'
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::combustion_1000_end_to_end_banded_lapack_refine_statistics -- --ignored --nocapture --test-threads=1
```

The dated report is
`test_reports/BVP_Damp_AOT/RustedSciThe__numerical__BVP_Damp__test_aot_diagnostics__tests__combustion_1000_end_to_end_banded_lapack_refine_statistics.md`.
The one-run debug result was:

```text
route       | total_ms | bootstrap_ms | solve_ms  | iterations | linear_solves | jac_rebuilds | max_diff
AtomView    |  868.797 |      485.027 |  383.241  |          5 |            10 |            1 | baseline
C-gcc       | 6215.059 |     3066.698 | 3148.222  |          5 |            10 |            1 | 0.0e0
C-tcc       | 1899.737 |      544.066 | 1355.544  |          5 |            10 |            1 | 2.22e-16
Zig         |25744.860 |    12594.838 |13149.905  |          5 |            10 |            1 | 4.44e-16
```

The cold lifecycle table now also contains live callback counters:
`C-gcc=12/1`, `C-tcc=12/1`, and `Zig=12/1` for residual/Jacobian calls;
Lambdify correctly reports zero AOT callbacks. The earlier zero-counter rows
were stale build-time snapshots, not a lack of callback execution. These are
debug correctness/schema numbers only; release comparisons must use the same
fixture with repeated runs and must preserve the historical records.

### 2026-09-21: combustion-1000 end-to-end rerun after AOT diagnostic hardening

This is a new one-repetition debug record for the same test and the same
combustion-1000 Banded/Lapack-style setup. It is placed next to the previous
record intentionally; the previous numbers remain the historical comparison
point and are not overwritten.

Command:

```powershell
$env:BVP_BANDED_LAPACK_REPETITIONS='1'
$env:BVP_BANDED_LAPACK_COOLDOWN_MS='0'
cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics::tests::combustion_1000_end_to_end_banded_lapack_refine_statistics -- --ignored --nocapture --test-threads=1
```

Report:

`test_reports/BVP_Damp_AOT/RustedSciThe__numerical__BVP_Damp__test_aot_diagnostics__tests__combustion_1000_end_to_end_banded_lapack_refine_statistics.md`

```text
route       | total_ms | bootstrap_ms | solve_ms  | iterations | linear_solves | jac_rebuilds | max_diff_vs_AtomView
AtomView    |  878.849 |      487.163 |  391.032  |          5 |            10 |            1 | baseline
C-gcc       | 6302.631 |     3165.596 | 3136.906  |          5 |            10 |            1 | 0.000e0
C-tcc       | 1871.495 |      544.493 | 1326.886  |          5 |            10 |            1 | 2.220e-16
Zig         |25519.542 |    12390.748 |13128.655  |          5 |            10 |            1 | 4.441e-16
```

The lifecycle rows were:

```text
route       | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms | residual_calls | jacobian_calls
AtomView    |               0.000 |                 0.000 |      0.000 |   0.000 |              0 |              0
C-gcc       |              77.324 |               622.046 |   1861.449 |  17.693 |             12 |              1
C-tcc       |              74.687 |               593.366 |     48.027 |  10.347 |             12 |              1
Zig         |              74.023 |               622.867 |  11832.074 |   2.538 |             12 |              1
```

Interpretation against the preceding one-run debug record:

- The integer solver path is identical for every successful route: five
  iterations, ten linear solves and one Jacobian rebuild. This supports that
  the routes followed the same numerical trajectory.
- Cross-route correctness remains at machine precision: `C-gcc` is exact at
  the displayed precision, while `C-tcc` and Zig are below `5e-16` in the
  reported Linf solution difference.
- AtomView Lambdify moved from `868.797 ms` to `878.849 ms` total and from
  `383.241 ms` to `391.032 ms` solve time. This is approximately `+1.2%`
  total and `+2.0%` solve, but one debug repetition cannot distinguish noise
  from a systematic regression.
- C/gcc moved from `6215.059 ms` to `6302.631 ms` total, while its solve time
  moved from `3148.222 ms` to `3136.906 ms`. C/tcc improved from `1899.737 ms`
  to `1871.495 ms` total and from `1355.544 ms` to `1326.886 ms` solve.
  Zig improved slightly in both total and solve time.
- The new lifecycle counters are live post-solve counters (`12` residual and
  `1` Jacobian call for each compiled route), not stale build-time snapshots.
  Their presence is a diagnostics improvement, not a numerical-performance
  change.

Conclusion: no production performance verdict is changed by this rerun. The
AtomView Lambdify row shows a small possible debug fluctuation, while tcc and
Zig are slightly faster and gcc has mixed stage changes. A release baseline
must use multiple repetitions and the same cooldown/toolchain policy before a
regression or improvement is declared.

### 2026-09-22: release AOT corpus rerun and callback-only terminology

This fresh local run produced 33 dated reports under
`test_reports/BVP_Damp_AOT/`. The report timestamps are UTC and therefore fall
on 2026-09-21, while the local file timestamps start shortly after midnight on
2026-09-22. Historical STORY entries above remain unchanged.

The main cold end-to-end combustion-1000 Banded/LAPACK comparison was:

```text
source     | variant  | total_ms mean+/-std | bootstrap_ms mean+/-std | solve_ms mean+/-std | iters | linsys | jac_rebuilds | max solution diff
Lambdify   | AtomView |  267.615 +/- 37.254 | 121.471 +/- 8.189       |  146.051 +/- 31.692 |   5.0 |   10.0 |          1.0 | baseline
Compiled   | C-gcc    | 4244.480 +/- 21.735 | 2088.159 +/- 1.721       | 2156.225 +/- 20.108 |   5.0 |   10.0 |          1.0 | 0.000e0
Compiled   | C-tcc    |  546.959 +/- 11.538 |  233.303 +/- 5.576       |  313.556 +/-  8.195 |   5.0 |   10.0 |          1.0 | 2.220e-16
Compiled   | Zig      |23398.759 +/-714.630 |11579.727 +/-208.827       |11818.930 +/-510.493 |   5.0 |   10.0 |          1.0 | 4.441e-16
```

The corresponding cold lifecycle buckets were:

```text
route    | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms
Lambdify |               0.000 |                 0.000 |      0.000 |   0.000
C-gcc    |               7.3   |                69-74  | 1870-1884 |  16-17
C-tcc    |               7.3   |                69-70  |    48-49  |   8-9
Zig      |               7.5   |                87-93  |11173-12255|   1.1-1.2
```

Every successful route followed the same integer trajectory: five Newton
iterations, ten linear solves and one Jacobian rebuild. This confirms lifecycle
and numerical parity, but not a cold performance win. In this protocol
Lambdify remains the cold winner; TCC is the practical compiled route, while
GCC and Zig are dominated by compilation.

#### What `callback-only` means

The callback-throughput story excludes symbolic preparation, fixture
generation, compilation, linking, dynamic loading, matrix factorization, RHS
solves and Newton control logic. It prepares one generated runtime and then
repeatedly calls only the residual and Jacobian callbacks at a fixed callback
state. It answers: "how expensive is evaluation after the artifact already
exists?"

For `n_steps=200` and 20 repeated callback iterations the fresh table was:

```text
backend       | residual_ms | jacobian_ms | total_ms | speedup_vs_Lambdify | residual_diff | jacobian_diff
Lambdify      |       0.919 |       7.070 |    7.990 |              1.000x |      0.000e0  |     0.000e0
AtomView+Rust |       0.347 |       1.486 |    1.833 |              4.358x |      3.553e-15|     1.084e-19
AtomView+GCC  |       0.440 |       1.047 |    1.487 |              5.374x |      3.553e-15|     1.084e-19
AtomView+TCC  |       0.345 |       0.983 |    1.328 |              6.019x |      3.553e-15|     1.084e-19
AtomView+Zig  |       0.157 |       0.788 |    0.945 |              8.457x |      3.553e-15|     1.084e-19
```

This is evaluator-throughput speedup, not solver speedup. The cold and full
solver tables include costs that callback-only intentionally removes. The
difference between these tables is why cold E2E, warm solve and callback-only
must remain separate metrics.

#### Correctness qualification: block-tridiagonal diagnostic

The fresh sparse-vs-banded linear-system report contains a negative diagnostic:
`block_tridiagonal_lu_consistent` returned `final_rr=1.244e31` and
`max_diff=2.190e31`, yet some rows were printed with `status=ok`. The same
failure appears for Lambdify and the derived control path, so it is not evidence
of an AtomView or AOT callback defect. It is an experimental linear-solver and
reporting issue. LAPACK-style Banded LU on the same matrix has relative
residual around `1e-15`.

Block-tridiagonal rows therefore remain diagnostic only and must not count as
AOT correctness passes. The Zig bootstrap report has the same weakness: it says
`status=ok` while reporting `direct_rr=final_rr=1.244e31`. The next test pass
must classify such rows as `diag`/`failed` using a residual threshold, or state
explicitly that the test is bootstrap-only.

Primary fresh reports:

- `test_reports/BVP_Damp_AOT/*combustion_1000_end_to_end_banded_lapack_refine_statistics.md`
- `test_reports/BVP_Damp_AOT/*combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000.md`
- `test_reports/BVP_Damp_AOT/*combustion_1000_linear_system_story_sparse_vs_banded_consistent.md`
- `test_reports/BVP_Damp_AOT/*combustion_1000_compiled_banded_zig_bootstrap_smoke.md`

## 2026-09-22: release Sparse/Banded toolchain, chunking and Frozen lifecycle

The fresh release reports are dated `2026-09-22T15:03:01Z` through
`2026-09-22T15:14:41Z` and are preserved under
`test_reports/BVP_Damp_AOT_Race/` and `test_reports/BVP_Damp_AOT_Frozen/`.
All rows in these reports completed successfully; this section records the
architecture and correctness interpretation, while detailed timing conclusions
remain in `BVP_DAMP_STORY_PERFORMANCE.md`.

The combustion-1000 Sparse/Banded race used five cold repetitions and five
warm repetitions. The C-tcc rows were the most useful cold reference:

```text
matrix | toolchain | total_ms mean | solve_diff
Sparse | C-tcc     |       222.878 | 2.220e-16
Banded | C-tcc     |       299.807 | 8.731e-15
```

The same race passed for C-gcc and Zig. The large GCC and Zig totals are
toolchain/build dominated and must not be interpreted as callback throughput.
All routes kept the same five iterations, ten linear solves and one Jacobian
rebuild.

The two-repetition release matrix also passed for Lambdify plus AOT Rust/GCC,
TCC and Zig on Sparse and Banded, with whole and chunk4 policies. The
combustion-200 matrix strengthened this with three repetitions per row and
componentwise solution differences no larger than approximately `2.915e-15`.
It also confirmed that chunk4 rows expose four residual and four Jacobian jobs
without numerical drift.

The combustion-3000 stress reports provide the frontend split explicitly:

```text
frontend  | total_ms mean | AOT tcc/whole | AOT tcc/chunk4 | max_solution_diff
ExprLegacy|      1087.283 |      3065.615 |       3353.459 | 1.110e-16
AtomView  |       481.740 |      2382.519 |       3211.508 | 2.220e-16
```

Both AOT routes used real four-way callback jobs in chunk4 and reported no
fallback. The stage tables also distinguish symbolic preparation, artifact
generation, compile/link and solve work; these values must not be collapsed
into a single “AOT speed” number.

The newly captured Frozen reports close the missing report-file gap for the
three combustion-1000 stories. The Banded cold route, Banded
BuildIfMissing -> RequirePrebuilt lifecycle and Sparse
BuildIfMissing -> RequirePrebuilt lifecycle all preserve backend selection,
integer solver trajectory and roundoff-level solution parity. Prebuilt rows
have no compile/link interval, and their artifact policy is explicitly
`RequirePrebuilt`, so fallback compilation is not being hidden.

## 2026-09-22: apple-to-apple AOT and parallel tuning rerun

The fresh release reports use the same `n_steps=1000` Banded fixture and keep
cold user E2E, runtime solve and callback/lifecycle stages separate. The raw
records are:

- `test_reports/BVP_Damp_AOT/*aot_banded_apple_to_apple_release_protocol.md`
- `test_reports/BVP_Damp_AOT/*aot_combustion_parallel_tuning_reports_runtime_table.md`

The apple-to-apple protocol used four cold and four warm repetitions and fresh
child solves for honest E2E. All successful routes followed five Newton
iterations, ten linear solves and one Jacobian rebuild; solution differences
were zero for Rust/GCC and `2.22e-16` for TCC and Zig.

Representative cold E2E means were:

```text
route       | honest_e2e_ms | runtime_solve_ms
Lambdify    |        277.287 |          130.068
tcc/seq     |       2464.387 |          162.083
gcc/par-8   |       4327.755 |          164.488
rust/seq    |       6408.197 |          192.726
zig/seq     |      14182.721 |          163.087
```

The tuning report confirms the same qualitative result: cold AOT is dominated
by compilation (about `49 ms` for TCC, `1.86 s` for GCC and more than `11 s`
for Zig in the measured rows), while the Newton trajectory is stable. TCC
parallel rows change callback job counts to `4`, `8` or `16`, but the runtime
solve gain is small and not monotonic. The row-reserved variant has the best
cold TCC E2E in this tuning map, but it is not a universal default.

The callback/lifecycle counters are now live and comparable: compiled rows
report `12` residual calls, `1` Jacobian call and actual `res_jobs`/`jac_jobs`;
Lambdify rows correctly leave AOT callback counters empty.

The same report batch also refreshed the combustion Lapack statistics, sparse
versus Banded linear-system comparison, sparse callback chunking, TCC honest
wall-clock, compiler diagnostic, callback throughput, DevFastest toolchain and
Zig bootstrap reports. The Zig structured block-tridiagonal diagnostic remains
classified as `diag` when it reports `final_rr=1.244e31`; it must not count as
a correctness pass. The Lapack-style Banded path in the same report remains
the valid correctness route.

### 2026-09-22: AOT prepared-runtime contract reports

The AOT runtime-contract slice was also refreshed and wrote five dated reports
at `2026-09-22T10:46Z`:

- `aot_artifact_failure_contract_preserves_state_and_quarantine_diagnostics`;
- `aot_story_protocol_is_explicit_and_validated`;
- `atomview_aot_contract_rejects_missing_parameter_binding_before_abi`;
- `atomview_banded_aot_contract_is_self_contained_and_typed`;
- `atomview_compact_banded_owned_callback_preserves_slots_and_invalidation`.

These are debug lifecycle/correctness gates, not timing measurements. Together
they cover typed failure state and quarantine diagnostics, explicit story
protocol validation, pre-ABI parameter binding errors, self-contained AtomView
Banded callbacks, compact slot ownership and invalidation. Their raw files are
kept in `test_reports/BVP_Damp_AOT/` and are intentionally not collapsed into
the cold performance tables.

## 2026-09-22: investigation of false-green block solve and Lane-Emden drift

### Block-tridiagonal status correction

The earlier Zig bootstrap row was misleading: the structured solver returned
`Ok(report)`, and the test translated that API success directly into
`status=ok`, even though the numerical report contained
`direct_rr=final_rr=1.244e31` and `max|x|=2.190e31`. The failure is shared by
the Lambdify/derived/compiled block-tridiagonal paths; LAPACK-style Banded on
the same matrix remains around `1e-15`. This is therefore not an AOT callback
parity failure.

The diagnostic now applies `final_relative_residual <= 1e-8` before assigning a
green status. The focused Zig report is consequently:

```text
solver=block_tridiagonal_lu_consistent[g=2,refine=1]
direct_rr=1.244e31 | final_rr=1.244e31 | max|x|=2.190e31 | status=diag
```

The same Zig-generated Banded matrix and RHS were then solved through the
faithful production-default LAPACK Banded route in the same debug test. It produced
`status=ok, solve_rr=6.725e-16`. This isolates the anomaly to
`block_tridiagonal_lu_consistent[g=2,refine=1]`; it is not a Zig codegen or
callback numerical defect.

The smoke test still passes as an artifact/bootstrap test because it asserts
only finite output and explicitly expects the known structured-solver status
`diag`; it is not counted as a numerical correctness pass. This prevents a
future report from silently turning the same failure green.

### Lane-Emden symbolic parity correction

The previous `lane-emden-48` drift (`5.039e-8` residual and `2.399e-7`
Jacobian) had two precision-loss sources in the Atom path:

1. `f64` constants falling back to a six-significant-digit decimal rational;
2. overflowed rational coefficients being reduced with a fixed `1e6`
   fixed-point scale.

Both conversions now use high-precision decimal input and an adaptive
fixed-point scale. The new ignored localization gate compares two callback
states and reports the largest rows/cells. Its maximum drift is now
approximately `2.11e-11` for residuals and `3.00e-11` for Jacobian entries,
and the test enforces a `1e-9` componentwise gate.

The corrected report is preserved at:

- `test_reports/BVP_Damp_AOT/*lane_emden_symbolic_parity_localization.md`
- `test_reports/BVP_Damp_AOT/*combustion_1000_compiled_banded_zig_bootstrap_smoke.md`

### 2026-09-22: release AOT runtime-contract rerun at 19:49

The release command reran the focused AtomView AOT runtime-contract suite and
generated five reports: self-contained Banded contract, missing parameter
binding rejection before ABI use, compact Banded slot invalidation, explicit
story-protocol validation, and stale/partial artifact quarantine diagnostics.
The reports were produced before automatic status insertion and therefore
contain only headers/timestamps; they are execution artifacts, not sufficient
pass/fail evidence by themselves. The reporting utility now writes an explicit
status on every capture drop. These tests do not compile external artifacts
and provide lifecycle and typed-error evidence only; they must not be mixed
with cold AOT timings.
The five raw reports are retained under `test_reports/BVP_Damp_AOT/` with UTC
timestamps around `2026-09-22T16:49:14Z`.

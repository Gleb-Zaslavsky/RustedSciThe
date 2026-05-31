# BVP Damped/Frozen Story Test Registry

This file is the release-run notebook for the Damped/Frozen BVP solver story tests.
The unit tests protect individual pieces of code. The story tests answer larger
questions: which backend is actually selected, where time is spent, whether AtomView
and ExprLegacy agree numerically, whether AOT artifacts survive the build/runtime
handoff, and whether sparse and banded matrix routes behave like the same numerical
solver in different linear algebra clothes.

The file is intentionally practical. When a release run is performed, add the date,
machine/toolchain notes, the important numbers, and the conclusion under the relevant
entry. This keeps the tests from turning into a collection of impressive tables with
unclear hypotheses.

## Running Policy

Heavy story tests should normally be run one at a time in release mode with one test
thread. This avoids mixing AOT file-system effects, dynamic library loading, and
benchmark noise.

```powershell
cargo test --release <test_name> -- --ignored --nocapture --test-threads=1
```

The `--test-threads=1` flag belongs to Rust's test harness. It only prevents several
test functions from running at the same time. It does not disable the solver's own
parallelism: symbolic differentiation, Rayon execution, generated residual/Jacobian
chunking, and AOT runtime parallel callbacks still use the policies configured by
the test itself.

For non-ignored acceptance/story-like tests:

```powershell
cargo test --release <test_name> -- --nocapture --test-threads=1
```

When interpreting timings, separate build/prepare time from runtime solve time. AOT
compile cost is part of the end-to-end user experience for a cold artifact, but it is
not the same phenomenon as residual/Jacobian callback throughput inside Newton's loop.

Recent BVP Damp story tables also print callback-stage timings for linked AOT
routes.  These split the broad solver-level `Jacobian` bucket into generated
Jacobian value evaluation and matrix assembly.  Lambdify rows may leave these
columns blank because the legacy trait object exposes only a single Jacobian call.
This split is intentionally there to avoid comparing apples to oranges: codegen
hot-callback benchmarks measure generated values, while solver-level `jac_ms`
also includes assembly and trait/matrix handoff costs.

## Result Note Template

Use this compact form after each meaningful release run.

```text
Date:
Command:
Machine/toolchain:
Status:
Important numbers:
Conclusion:
Follow-up:
```

## Backend Vocabulary

`ExprLegacy` is the older symbolic assembly path. `AtomView` is the newer atom-based
assembly path intended to be faster and easier to feed into generated backends. A
story test comparing them asks whether two symbolic frontends produce equivalent
runtime residuals/Jacobians and equivalent solutions.

`Lambdify` evaluates generated symbolic expressions inside Rust without compiling an
external artifact. It is a good correctness baseline because it avoids compiler and
dynamic-loader noise.

`AOT` builds a generated residual/Jacobian artifact and then calls that compiled
runtime path. `BuildIfMissing` means "if the artifact is not already usable, build it
and then use the compiled backend immediately". `RequirePrebuilt` means "do not fall
back to Lambdify; fail if the compiled artifact is not available and callable".

`Sparse` and `Banded` are different linear algebra backends for the same discretized
Newton system. Sparse is the general production baseline. Banded is the structured
route that should win on narrow-band BVP/PDE-like systems when the bandwidth metadata
is correct.

## AOT Chunking and Parallel Execution Source of Truth

Chunking has two separate knobs and they should not be mentally merged. `AotChunkingPolicy`
is a code-generation layout choice: it decides whether generated residual and Jacobian
callbacks are emitted as one large function (`Whole`) or as several chunk functions.
`AotExecutionPolicy` is a runtime scheduling choice: it decides whether those generated
chunks are called sequentially or through the parallel executor. Chunking without
parallel runtime is mostly a structure/diagnostic choice. Parallel runtime without
enough coarse chunks has little useful work to schedule.

The user-facing starting point for production experiments is:

```rust
GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
    .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
        Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
    ))
    .with_aot_execution_policy(AotExecutionPolicy::Parallel(ParallelExecutorConfig {
        jobs_per_worker: 1,
        max_residual_jobs: Some(4),
        max_sparse_jobs: Some(4),
        fallback_policy: ParallelFallbackPolicy::Never,
    }));
```

This is deliberately conservative: four coarse chunks are large enough to amortize
scheduler overhead on medium and large BVPs, and small enough to avoid creating a
forest of tiny generated functions. The tuning test below shows that larger settings
can win on large sparse problems. On one combustion-1000 run, `par-8x8-jobs8` was
the fastest (`solve_ms` about 8351 ms, speedup about 1.67x over sequential), while
`par-16x16-jobs16` regressed and row-count sparse chunking was almost neutral. Treat
that as the current empirical rule: start with `4x4`, try `8x8` for large sparse
systems, and do not assume that more chunks always means faster.

For Banded problems, the linear solve is often already much cheaper than Sparse, so
callback chunking may not dominate the total solve. Use the end-to-end matrices to
check whether chunking actually moves the total runtime. If `whole` and `chunk4`
agree numerically but timing is flat, prefer the simpler `Whole` or `Auto` policy
unless the larger-grid release matrix shows a measurable benefit.

The tests are intentionally split by question. `debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000`
is the correctness microscope for generated callbacks. `combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix`
is the practical medium-grid end-to-end matrix. `aot_combustion_parallel_tuning_reports_runtime_table`
is the sparse runtime tuning map for choosing `target_chunks` and job caps after
correctness has already been established.

## Release Matrix: Sparse/Banded AOT Toolchain and Chunking

These are the most important story runs for the current backend work. They compare
the real production-weight cases across sparse and banded matrix backends, Lambdify
baseline, AOT, and AOT chunking/toolchain options.

### `combustion_1000_lambdify_sparse_vs_banded_end_to_end_race`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

Hypothesis: for the same combustion-1000 problem, Lambdify Sparse and Lambdify Banded
should solve to the same solution quality, while exposing the pure linear-backend
performance difference without AOT compiler noise.

Instrumentation note: after the sparse-first Jacobian change this table also prints the
symbolic handoff and internal Jacobian-stage breakdown. Both `Sparse` (`FaerSparseCol`)
and `Banded` are expected to report `dense_cache` near zero; a material nonzero value
would mean that a quadratic dense symbolic cache has leaked back into a sparse route.

Command:

```powershell
cargo test --release combustion_1000_lambdify_sparse_vs_banded_end_to_end_race -- --ignored --nocapture --test-threads=1
```

Result:
CPU 4 Core
[BVP Damp race] finished source=Lambdify matrix=Banded variant=ExprLegacy status=ok
[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded end-to-end
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 904.174 +/- 296.365 [736.232, 1496.208] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
Lambdify | Banded | ExprLegacy |  5/5  | 735.021 +/- 38.833 [701.336, 810.123] | 7.876e-15 +/- 0.0e0 | 7.864e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify        | 841.400 +/- 80.639 | 727.200 +/- 136.563   | 38.400 +/- 5.499   | 8.400 +/- 6.800    | 2.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | ExprLegacy | symbolic+lambdify        | 779.600 +/- 39.823 | 668.000 +/- 39.360    | 16.000 +/- 1.549   | 0.200 +/- 0.400    | 2.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp race] combustion-1000 Lambdify callback stage breakdown
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
Lambdify | Banded | ExprLegacy | symbolic+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           

[BVP Damp race] combustion-1000 Lambdify lifecycle/refinement breakdown
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 727.200 +/- 136.563
Lambdify | Banded | ExprLegacy | symbolic+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 668.000 +/- 39.360

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 593.149 +/- 20.029 | 203.979 +/- 4.243  | 230.187 +/- 14.543 | 16.021 +/- 0.958 | 72.468 +/- 2.475 | -                | -               | -            | -            | -            | -           
Lambdify | Banded | ExprLegacy | 596.544 +/- 39.422 | 213.062 +/- 23.104 | 241.078 +/- 20.526 | 16.032 +/- 0.355 | 57.111 +/- 1.976 | -                | -               | -            | -            | -            | -           

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded internal symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 230.187 +/- 14.543 | 2.203 +/- 0.210 | 226.238 +/- 14.321 | 0.000 +/- 0.000 | 1.177 +/- 0.386
Lambdify | Banded | ExprLegacy | 241.078 +/- 20.526 | 2.550 +/- 0.437 | 236.802 +/- 20.752 | 0.000 +/- 0.000 | 1.235 +/- 0.345

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 72.468 +/- 2.475 | 24.484 +/- 1.523 | 47.976 +/- 2.112
Lambdify | Banded | ExprLegacy | 57.111 +/- 1.976 | 17.257 +/- 1.044 | 39.849 +/- 1.711
ok

CPU 12 Core

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded end-to-end
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 629.489 +/- 836.140 [190.615, 2301.626] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
Lambdify | Banded | ExprLegacy |  5/5  | 192.346 +/- 17.037 [174.110, 216.556] | 7.876e-15 +/- 0.0e0 | 7.864e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify        | 594.600 +/- 702.779 | 525.400 +/- 737.371   | 13.200 +/- 2.926   | 5.800 +/- 0.400    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | ExprLegacy | symbolic+lambdify        | 228.400 +/- 23.771 | 147.000 +/- 13.784    | 4.200 +/- 0.400    | 0.200 +/- 0.400    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp race] combustion-1000 Lambdify callback stage breakdown
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
Lambdify | Banded | ExprLegacy | symbolic+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           

[BVP Damp race] combustion-1000 Lambdify lifecycle/refinement breakdown
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | symbolic+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 525.400 +/- 737.371
Lambdify | Banded | ExprLegacy | symbolic+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 147.000 +/- 13.784

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 112.184 +/- 7.579 | 24.370 +/- 2.049   | 24.172 +/- 0.911 | 6.656 +/- 0.569 | 16.689 +/- 1.644 | -                | -               | -            | -            | -            | -           
Lambdify | Banded | ExprLegacy | 102.215 +/- 8.016 | 24.731 +/- 1.482   | 23.699 +/- 0.209 | 6.827 +/- 0.442 | 9.274 +/- 0.184 | -                | -               | -            | -            | -            | -           

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded internal symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 24.172 +/- 0.911 | 1.173 +/- 0.102 | 22.270 +/- 0.862 | 0.000 +/- 0.000 | 0.530 +/- 0.088
Lambdify | Banded | ExprLegacy | 23.699 +/- 0.209 | 1.104 +/- 0.078 | 21.823 +/- 0.230 | 0.000 +/- 0.000 | 0.530 +/- 0.056

[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 16.689 +/- 1.644 | 8.175 +/- 1.468  | 8.510 +/- 0.259 
Lambdify | Banded | ExprLegacy | 9.274 +/- 0.184 | 3.800 +/- 0.110  | 5.470 +/- 0.087 
ok


```text
Date: 2026-05-26
Status: ok; sparse-first storage and the Banded Lambdify binding fix are confirmed on the real solve.
Important numbers:
  Both rows completed `5/5` with solution disagreement only at roundoff scale
  (`7.876e-15` for Banded against the Sparse baseline).
  Sparse and Banded both report `dense_cache = 0.000 ms`.
  Their symbolic Jacobian costs are now in the same range:
  Sparse `230.187 +/- 14.543 ms`, Banded `241.078 +/- 20.526 ms`;
  in both cases almost all of that time is `row_diff`.
  Banded performs the numerical linear work much faster
  (`16.000 +/- 1.549 ms` versus Sparse `38.400 +/- 5.499 ms`).
  After removing repeated Banded callback-factory preparation, Banded
  `initial_bind` is `57.111 +/- 1.976 ms`, versus Sparse
  `72.468 +/- 2.475 ms` and far below the former `~372 ms` pathology.
Conclusion:
  Both sparse-first routes are genuine, and the Banded callback-construction
  defect has been removed. Banded now wins the end-to-end Lambdify comparison:
  `735.021 +/- 38.833 ms` versus Sparse `904.174 +/- 296.365 ms`.
Follow-up:
  The former Banded binding asymmetry is closed: Jacobian callback compilation
  is now `17.257 +/- 1.044 ms` against Sparse `24.484 +/- 1.523 ms`, and
  residual compilation is `39.849 +/- 1.711 ms` against Sparse
  `47.976 +/- 2.112 ms`. The next shared performance target is not a Banded
  callback special case; it is `row_diff`, which dominates symbolic Jacobian
  construction on both matrix routes.
Implementation follow-up (confirmed by release rerun):
  The remaining Banded callback factory rebuilt the same `Vec<&str>` runtime
  argument view separately for every compiled Jacobian or residual scalar.
  Sparse builds that view once per compilation pass. Banded now follows the
  same design and reuses one immutable argument view across all parallel
  closure builds. The production Banded entry also no longer infers its
  structural layout twice, allocates temporary base-variable strings during
  that inference, or creates a disposable zero-filled `BandedAssembly` merely
  to obtain diagonal lengths for its compiled plan. The numerical evaluator
  layout and runtime equations are unchanged. The rerun confirms the intended
  effect in `jacobian_compile`, `residual_compile`, and end-to-end wall time.
```

### `combustion_1000_aot_sparse_vs_banded_end_to_end_race`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

Hypothesis: for the same combustion-1000 problem, AOT Sparse and AOT Banded should
produce equivalent solutions; timing differences should be attributable to matrix
structure, generated callback path, artifact reuse, and chunking policy rather than
different mathematical systems.

Instrumentation note: the release rerun should inspect the newly printed
`internal symbolic-Jacobian stages` table. The `Sparse` rows are now a direct
regression check for the same dense-cache removal already proved on `Banded`.

Command:

```powershell
cargo test --release combustion_1000_aot_sparse_vs_banded_end_to_end_race -- --ignored --nocapture --test-threads=1
```
CPU 4 Core
[BVP Damp race] finished source=Compiled matrix=Banded variant=Zig status=ok
[BVP Damp race] combustion-1000 AOT Sparse vs Banded end-to-end
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc    |  5/5  | 6195.708 +/- 419.368 [5774.386, 6740.635] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | C-gcc    |  5/5  | 6125.567 +/- 288.347 [5773.368, 6456.782] | 2.220e-15 +/- 0.0e0 | 2.217e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Sparse | C-tcc    |  5/5  | 960.802 +/- 42.171 [913.437, 1026.507] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | C-tcc    |  5/5  | 880.917 +/- 37.481 [838.300, 943.702] | 1.998e-15 +/- 0.0e0 | 1.995e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Sparse | Zig      |  5/5  | 45326.797 +/- 3014.489 [41208.920, 48784.828] | 4.441e-16 +/- 0.0e0 | 4.434e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | Zig      |  5/5  | 45173.647 +/- 1063.148 [43884.225, 46508.523] | 3.553e-15 +/- 0.0e0 | 3.547e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc    | symbolic+aot-build+link  | 5400.000 +/- 489.898 | 5400.000 +/- 489.898  | 35.800 +/- 8.158   | 2.400 +/- 0.490    | 1.200 +/- 0.400    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | C-gcc    | symbolic+aot-build+link  | 5600.000 +/- 489.898 | 5600.000 +/- 489.898  | 15.400 +/- 1.855   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Sparse | C-tcc    | symbolic+aot-build+link  | 983.200 +/- 20.585 | 875.200 +/- 38.649    | 33.800 +/- 1.327   | 2.000 +/- 0.000    | 1.400 +/- 0.490    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | C-tcc    | symbolic+aot-build+link  | 922.800 +/- 40.425 | 816.200 +/- 36.521    | 14.800 +/- 1.327   | 1.000 +/- 0.000    | 1.200 +/- 0.400    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Sparse | Zig      | symbolic+aot-build+link  | 44800.000 +/- 2925.748 | 44800.000 +/- 2925.748 | 34.200 +/- 2.400   | 1.200 +/- 0.400    | 0.400 +/- 0.490    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | Zig      | symbolic+aot-build+link  | 44400.000 +/- 1200.000 | 44400.000 +/- 1200.000 | 15.200 +/- 0.980   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp race] combustion-1000 AOT callback stage breakdown
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | symbolic+aot-build+link | 3.049 +/- 0.231    | 1.056 +/- 0.184    | 1.412 +/- 0.345      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | C-gcc      | symbolic+aot-build+link | 2.651 +/- 0.151    | 1.177 +/- 0.267    | 0.118 +/- 0.070      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Sparse | C-tcc      | symbolic+aot-build+link | 3.185 +/- 0.227    | 0.976 +/- 0.033    | 1.349 +/- 0.122      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | C-tcc      | symbolic+aot-build+link | 2.957 +/- 0.222    | 1.055 +/- 0.082    | 0.128 +/- 0.059      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Sparse | Zig        | symbolic+aot-build+link | 1.743 +/- 0.684    | 0.266 +/- 0.022    | 1.375 +/- 0.179      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | Zig        | symbolic+aot-build+link | 1.619 +/- 0.909    | 0.355 +/- 0.016    | 0.147 +/- 0.087      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job

[BVP Damp race] combustion-1000 AOT lifecycle/refinement breakdown
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 5400.000 +/- 489.898
Compiled | Banded | C-gcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 5600.000 +/- 489.898
Compiled | Sparse | C-tcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 875.200 +/- 38.649
Compiled | Banded | C-tcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 816.200 +/- 36.521
Compiled | Sparse | Zig        | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 44800.000 +/- 2925.748
Compiled | Banded | Zig        | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 44400.000 +/- 1200.000

[BVP Damp race] combustion-1000 AOT Sparse vs Banded symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | 491.673 +/- 60.357 | 94.623 +/- 38.144  | 245.984 +/- 29.685 | 30.122 +/- 4.375 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 63.912 +/- 6.029
Compiled | Banded | C-gcc      | 459.353 +/- 29.941 | 72.293 +/- 8.529   | 242.406 +/- 22.942 | 28.120 +/- 1.852 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 66.351 +/- 6.109
Compiled | Sparse | C-tcc      | 443.974 +/- 31.555 | 68.409 +/- 3.635   | 234.245 +/- 26.447 | 27.557 +/- 2.109 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 63.970 +/- 4.939
Compiled | Banded | C-tcc      | 431.064 +/- 27.580 | 70.824 +/- 8.296   | 220.893 +/- 19.030 | 26.887 +/- 0.859 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 63.558 +/- 1.968
Compiled | Sparse | Zig        | 428.938 +/- 18.800 | 69.492 +/- 2.595   | 218.103 +/- 16.582 | 27.040 +/- 0.998 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 64.186 +/- 2.203
Compiled | Banded | Zig        | 424.330 +/- 17.988 | 70.796 +/- 8.815   | 215.834 +/- 10.541 | 27.118 +/- 1.699 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 63.785 +/- 3.394

[BVP Damp race] combustion-1000 AOT Sparse vs Banded internal symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | 245.984 +/- 29.685 | 2.277 +/- 0.264 | 242.205 +/- 29.413 | 0.001 +/- 0.001 | 1.036 +/- 0.294
Compiled | Banded | C-gcc      | 242.406 +/- 22.942 | 2.319 +/- 0.154 | 238.662 +/- 23.126 | 0.000 +/- 0.000 | 0.986 +/- 0.180
Compiled | Sparse | C-tcc      | 234.245 +/- 26.447 | 2.127 +/- 0.171 | 230.593 +/- 26.351 | 0.000 +/- 0.000 | 1.083 +/- 0.174
Compiled | Banded | C-tcc      | 220.893 +/- 19.030 | 1.990 +/- 0.126 | 217.526 +/- 19.238 | 0.000 +/- 0.000 | 1.030 +/- 0.246
Compiled | Sparse | Zig        | 218.103 +/- 16.582 | 2.040 +/- 0.159 | 214.754 +/- 16.748 | 0.000 +/- 0.000 | 0.950 +/- 0.114
Compiled | Banded | Zig        | 215.834 +/- 10.541 | 2.111 +/- 0.230 | 212.475 +/- 10.609 | 0.000 +/- 0.000 | 0.905 +/- 0.115
ok


CPU 12 Core
BVP Damp race] combustion-1000 AOT Sparse vs Banded end-to-end
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc    |  5/5  | 2464.054 +/- 781.613 [2059.052, 4026.919] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | C-gcc    |  5/5  | 2074.642 +/- 40.950 [2024.927, 2148.779] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Sparse | C-tcc    |  5/5  | 343.565 +/- 25.438 [319.661, 390.622] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | C-tcc    |  5/5  | 346.313 +/- 16.717 [325.718, 373.081] | 2.926e-15 +/- 0.0e0 | 2.921e-15 +/- 3.9e-31 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Sparse | Zig      |  5/5  | 13048.019 +/- 127.318 [12922.752, 13272.369] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5
Compiled | Banded | Zig      |  5/5  | 13105.019 +/- 81.321 [12962.536, 13192.804] | 2.928e-15 +/- 0.0e0 | 2.923e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 5/5

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc    | symbolic+aot-build+link  | 2400.000 +/- 800.000 | 2200.000 +/- 400.000  | 11.200 +/- 0.400   | 1.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | C-gcc    | symbolic+aot-build+link  | 2000.000 +/- 0.000 | 1800.000 +/- 400.000  | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.400 +/- 0.490    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Sparse | C-tcc    | symbolic+aot-build+link  | 378.200 +/- 29.614 | 290.000 +/- 19.738    | 11.000 +/- 0.000   | 1.000 +/- 0.000    | 0.600 +/- 0.490    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | C-tcc    | symbolic+aot-build+link  | 378.600 +/- 15.603 | 304.200 +/- 15.854    | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.400 +/- 0.490    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Sparse | Zig      | symbolic+aot-build+link  | 12600.000 +/- 489.898 | 12400.000 +/- 489.898 | 11.200 +/- 0.400   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Compiled | Banded | Zig      | symbolic+aot-build+link  | 12800.000 +/- 400.000 | 12800.000 +/- 400.000 | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp race] combustion-1000 AOT callback stage breakdown
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | symbolic+aot-build+link | 1.642 +/- 0.128    | 0.618 +/- 0.007    | 0.507 +/- 0.057      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | C-gcc      | symbolic+aot-build+link | 1.715 +/- 0.363    | 0.586 +/- 0.016    | 0.078 +/- 0.039      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Sparse | C-tcc      | symbolic+aot-build+link | 1.767 +/- 0.186    | 0.620 +/- 0.025    | 0.494 +/- 0.038      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | C-tcc      | symbolic+aot-build+link | 1.861 +/- 0.329    | 0.562 +/- 0.016    | 0.075 +/- 0.038      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Sparse | Zig        | symbolic+aot-build+link | 0.606 +/- 0.061    | 0.174 +/- 0.004    | 0.481 +/- 0.037      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job
Compiled | Banded | Zig        | symbolic+aot-build+link | 0.538 +/- 0.009    | 0.179 +/- 0.001    | 0.047 +/- 0.002      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 6000.000 +/- 0.000 | 20988.000 +/- 0.000 | single_requested_job | single_requested_job

[BVP Damp race] combustion-1000 AOT lifecycle/refinement breakdown
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 2200.000 +/- 400.000
Compiled | Banded | C-gcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1800.000 +/- 400.000
Compiled | Sparse | C-tcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 290.000 +/- 19.738
Compiled | Banded | C-tcc      | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 304.200 +/- 15.854
Compiled | Sparse | Zig        | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 12400.000 +/- 489.898
Compiled | Banded | Zig        | symbolic+aot-build+link | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 12800.000 +/- 400.000

[BVP Damp race] combustion-1000 AOT Sparse vs Banded symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | 97.839 +/- 11.389 | 38.312 +/- 0.722   | 7.410 +/- 1.124 | 9.916 +/- 1.019 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 27.012 +/- 7.913
Compiled | Banded | C-gcc      | 105.403 +/- 7.817 | 39.624 +/- 1.854   | 7.634 +/- 0.365 | 11.087 +/- 1.236 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 28.019 +/- 7.138
Compiled | Sparse | C-tcc      | 99.626 +/- 7.676 | 38.653 +/- 0.329   | 7.443 +/- 0.586 | 10.405 +/- 1.218 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 24.482 +/- 1.814
Compiled | Banded | C-tcc      | 95.726 +/- 6.179 | 38.342 +/- 0.685   | 7.413 +/- 0.798 | 9.821 +/- 1.205 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 24.869 +/- 3.224
Compiled | Sparse | Zig        | 94.757 +/- 5.221 | 38.299 +/- 0.516   | 7.320 +/- 0.544 | 9.847 +/- 0.938 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 23.723 +/- 0.761
Compiled | Banded | Zig        | 99.466 +/- 5.913 | 39.145 +/- 1.525   | 7.759 +/- 0.541 | 11.163 +/- 1.483 | 0.002 +/- 0.001 | -                | -               | -            | -            | -            | 25.293 +/- 2.875

[BVP Damp race] combustion-1000 AOT Sparse vs Banded internal symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Compiled | Sparse | C-gcc      | 7.410 +/- 1.124 | 2.519 +/- 0.110 | 3.566 +/- 0.754 | 0.000 +/- 0.000 | 0.926 +/- 0.191
Compiled | Banded | C-gcc      | 7.634 +/- 0.365 | 2.561 +/- 0.082 | 3.724 +/- 0.242 | 0.000 +/- 0.000 | 0.996 +/- 0.071
Compiled | Sparse | C-tcc      | 7.443 +/- 0.586 | 2.610 +/- 0.247 | 3.565 +/- 0.296 | 0.000 +/- 0.000 | 0.926 +/- 0.081
Compiled | Banded | C-tcc      | 7.413 +/- 0.798 | 2.603 +/- 0.196 | 3.512 +/- 0.506 | 0.000 +/- 0.000 | 0.956 +/- 0.106
Compiled | Sparse | Zig        | 7.320 +/- 0.544 | 2.577 +/- 0.134 | 3.448 +/- 0.341 | 0.000 +/- 0.000 | 0.946 +/- 0.082
Compiled | Banded | Zig        | 7.759 +/- 0.541 | 2.890 +/- 0.438 | 3.578 +/- 0.323 | 0.000 +/- 0.000 | 0.943 +/- 0.048
ok
```text
Date: 2026-05-26
Status: ok; full unchunked AOT language/matrix comparison is numerically clean after sparse-first storage.
Important numbers:
  All six AOT rows completed 5/5 and agree at roundoff scale
  (`solve_diff <= 3.553e-15`, identical `max_abs_sol = 1.002`).
  C-tcc is the clear cold end-to-end leader at this size:
  Sparse `0.961 +/- 0.042 s`, Banded `0.881 +/- 0.037 s`.
  C-gcc is about `6.1-6.2 s`; Zig is about `45.2-45.3 s`.
  Every Sparse and Banded AOT row now reports `dense_cache` at zero scale
  (`0.000-0.001 ms`), while `row_diff` accounts for virtually all
  `initial_sym_jac` (`~212-242 ms`).
  Banded reduces linear-solve cost from roughly `34-36 ms` (Sparse) to
  `15 ms`, although toolchain/bootstrap cost still dominates gcc and Zig.
Conclusion:
  Correctness and both sparse-first numerical routes are healthy. The removal
  of dense symbolic caching is confirmed for Sparse as well as Banded; there
  is no remaining sparse-specific materialization penalty in this experiment.
  For a cold `combustion-1000` AOT solve, toolchain choice dominates matrix
  choice, and `tcc` is the useful default candidate. Zig's generated callbacks
  are fast once loaded, but its cold route is far too expensive here.
Follow-up:
  The next symbolic-performance target shared by Sparse and Banded is
  `row_diff`. This test does not exercise `chunk4`; use the explicit chunking
  matrix below after clearing `BVP_AOT_MATRIX_FILTER` before drawing
  parallelism conclusions.
```

### `combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

Hypothesis: for combustion-1000, Sparse and Banded AOT routes should stay
numerically consistent across supported AOT emitters (`gcc`, `tcc`, `zig`, Rust)
and across two explicit chunking/runtime policies: whole sequential callbacks and
four-way generated chunks with forced parallel runtime execution. This is the
main release matrix for toolchain/chunking behavior.

The table includes Sparse and Banded Lambdify/AtomView baselines. They are not
competing AOT variants; they are the sanity check that the BVP formulation and
Newton path are solvable before interpreting AOT rows. When
`BVP_AOT_MATRIX_FILTER` is used, these baselines are kept in the run
automatically.

This test intentionally uses `RebuildAlways` so toolchain/build behavior is visible.
It is therefore much heavier than a warm-artifact solve benchmark.

The test prints `source`, `matrix`, `variant`, and `bootstrap_hint` before every
variant and flushes stdout. This is deliberate: if a generated backend fails
inside the numerical solve, the row should normally end with `solve_failed(...)`
or `solve_panicked(...)`. Rust AOT callbacks are guarded at the FFI boundary, so
a generated Rust panic should be reported as a failed callback instead of aborting
the whole process. If the process still aborts, the last printed "running ..."
line is the variant that should be isolated first with `BVP_AOT_MATRIX_FILTER`.

Command:

```powershell
cargo test --release combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix -- --ignored --nocapture --test-threads=1
```

Optional variant filter. The filter is a case-insensitive substring matched against
`source matrix variant bootstrap_hint`. This is useful after a full matrix run points
to a suspicious toolchain/chunking pair.

```powershell
$env:BVP_AOT_MATRIX_FILTER="Banded rust/chunk4"; cargo test --release combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix -- --ignored --nocapture --test-threads=1
```

Result:
CPU 4 Core
[BVP Damp race] combustion-1000 AOT toolchain/chunking release matrix
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView |  2/2  | 4333.220 +/- 203.711 [4129.510, 4536.931] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
Lambdify | Banded | AtomView |  2/2  | 2838.295 +/- 153.644 [2684.651, 2991.940] | 1.776e-15 +/- 0.0e0 | 1.774e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | tcc/whole |  2/2  | 2677.880 +/- 64.358 [2613.521, 2742.238] | 1.332e-15 +/- 0.0e0 | 1.330e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | tcc/whole |  2/2  | 2455.313 +/- 29.603 [2425.710, 2484.916] | 1.776e-15 +/- 0.0e0 | 1.774e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView | baseline+symbolic+lambdify | 4000.000 +/- 0.000 | 4000.000 +/- 0.000    | 39.000 +/- 4.000   | 5.000 +/- 0.000    | 6.000 +/- 1.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | AtomView | baseline+symbolic+lambdify | 2500.000 +/- 500.000 | 2000.000 +/- 0.000    | 23.500 +/- 1.500   | 1.000 +/- 0.000    | 6.000 +/- 1.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc/whole | rebuild+seq+whole        | 2000.000 +/- 0.000 | 2000.000 +/- 0.000    | 37.000 +/- 1.000   | 2.000 +/- 0.000    | 2.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole | rebuild+seq+whole        | 2000.000 +/- 0.000 | 2000.000 +/- 0.000    | 16.000 +/- 1.000   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
ok


test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2294 filtered out; finished in 1205.83s

CPU 12 Core
[BVP Damp race] combustion-1000 AOT toolchain/chunking release matrix
[BVP Damp race] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView |  2/2  | 1251.021 +/- 989.469 [261.552, 2240.490] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
Lambdify | Banded | AtomView |  2/2  | 240.476 +/- 6.789 [233.686, 247.265] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | gcc/whole |  2/2  | 2391.218 +/- 2.517 [2388.702, 2393.735] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | gcc/chunk4 |  2/2  | 2031.400 +/- 5.521 [2025.879, 2036.921] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | gcc/whole |  2/2  | 2244.969 +/- 10.823 [2234.146, 2255.792] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | gcc/chunk4 |  2/2  | 1999.187 +/- 36.198 [1962.989, 2035.385] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | tcc/whole |  2/2  | 301.822 +/- 13.405 [288.417, 315.227] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | tcc/chunk4 |  2/2  | 294.648 +/- 6.208 [288.440, 300.856] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | tcc/whole |  2/2  | 289.553 +/- 5.388 [284.164, 294.941] | 2.926e-15 +/- 0.0e0 | 2.921e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | tcc/chunk4 |  2/2  | 283.638 +/- 7.165 [276.472, 290.803] | 2.926e-15 +/- 0.0e0 | 2.921e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | zig/whole |  2/2  | 32371.548 +/- 142.126 [32229.422, 32513.675] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | zig/chunk4 |  2/2  | 29795.796 +/- 47.145 [29748.651, 29842.942] | 6.661e-16 +/- 0.0e0 | 6.651e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | zig/whole |  2/2  | 31841.566 +/- 17.113 [31824.453, 31858.679] | 2.928e-15 +/- 0.0e0 | 2.923e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | zig/chunk4 |  2/2  | 29498.752 +/- 236.641 [29262.111, 29735.393] | 2.928e-15 +/- 0.0e0 | 2.923e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | rust/whole |  2/2  | 5327.877 +/- 87.499 [5240.377, 5415.376] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Sparse | rust/chunk4 |  2/2  | 4405.365 +/- 11.448 [4393.917, 4416.813] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | rust/whole |  2/2  | 5134.168 +/- 4.292 [5129.876, 5138.460] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2
AOT      | Banded | rust/chunk4 |  2/2  | 4350.468 +/- 9.912 [4340.556, 4360.380] | 2.932e-15 +/- 0.0e0 | 2.927e-15 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 2/2

[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts.
source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView | baseline+symbolic+lambdify | 1154.000 +/- 846.000 | 1095.000 +/- 905.000  | 14.500 +/- 1.500   | 7.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | AtomView | baseline+symbolic+lambdify | 281.500 +/- 10.500 | 189.500 +/- 2.500     | 4.500 +/- 0.500    | 0.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | gcc/whole | rebuild+seq+whole        | 2000.000 +/- 0.000 | 2000.000 +/- 0.000    | 11.500 +/- 0.500   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | gcc/chunk4 | rebuild+par+chunk4       | 2000.000 +/- 0.000 | 1000.000 +/- 0.000    | 12.500 +/- 0.500   | 1.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | gcc/whole | rebuild+seq+whole        | 2000.000 +/- 0.000 | 2000.000 +/- 0.000    | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | gcc/chunk4 | rebuild+par+chunk4       | 1500.000 +/- 500.000 | 1000.000 +/- 0.000    | 4.500 +/- 0.500    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc/whole | rebuild+seq+whole        | 338.000 +/- 17.000 | 248.000 +/- 7.000     | 11.000 +/- 0.000   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc/chunk4 | rebuild+par+chunk4       | 332.000 +/- 10.000 | 239.000 +/- 2.000     | 13.000 +/- 1.000   | 0.500 +/- 0.500    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole | rebuild+seq+whole        | 325.500 +/- 8.500  | 244.000 +/- 1.000     | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4       | 318.500 +/- 9.500  | 239.000 +/- 4.000     | 5.000 +/- 0.000    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | zig/whole | rebuild+seq+whole        | 32000.000 +/- 0.000 | 32000.000 +/- 0.000   | 11.000 +/- 0.000   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | zig/chunk4 | rebuild+par+chunk4       | 29000.000 +/- 0.000 | 29000.000 +/- 0.000   | 11.500 +/- 0.500   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | zig/whole | rebuild+seq+whole        | 31000.000 +/- 0.000 | 31000.000 +/- 0.000   | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | zig/chunk4 | rebuild+par+chunk4       | 29000.000 +/- 0.000 | 29000.000 +/- 0.000   | 4.000 +/- 0.000    | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | rust/whole | rebuild+seq+whole        | 5000.000 +/- 0.000 | 5000.000 +/- 0.000    | 11.000 +/- 0.000   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | rust/chunk4 | rebuild+par+chunk4       | 4000.000 +/- 0.000 | 4000.000 +/- 0.000    | 13.500 +/- 0.500   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | rust/whole | rebuild+seq+whole        | 5000.000 +/- 0.000 | 5000.000 +/- 0.000    | 4.000 +/- 0.000    | 1.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | rust/chunk4 | rebuild+par+chunk4       | 4000.000 +/- 0.000 | 4000.000 +/- 0.000    | 5.000 +/- 0.000    | 0.500 +/- 0.500    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
ok
```text
Date: 2026-05-26
Status: ok, but focused `tcc/whole` evidence only; not a completed matrix run.
Important numbers:
  AOT Banded `tcc/whole` completes in `2.455 +/- 0.030 s` against the
  Banded Lambdify baseline at `2.838 +/- 0.154 s`.
  AOT Sparse `tcc/whole` completes in `2.678 +/- 0.064 s` against the
  Sparse Lambdify baseline at `4.333 +/- 0.204 s`.
  All four shown rows agree at roundoff scale.
Conclusion:
  The focused result is good evidence that cold `tcc/whole` AOT can outperform
  Lambdify at `n_steps=1000`, especially through the Banded route. It is not
  evidence about chunking or gcc/zig/Rust: only `tcc/whole` rows appear, which
  means a persisted `BVP_AOT_MATRIX_FILTER` selected a subset.
Follow-up:
  Clear the PowerShell environment filter before a full matrix rerun:
  `Remove-Item Env:\BVP_AOT_MATRIX_FILTER -ErrorAction SilentlyContinue`.
```

Known interpretation rule: if `whole` variants solve but `chunk4` variants fail
across several languages, treat it as a chunk orchestration/equivalence issue
until proven otherwise. The next diagnostic step is an elementwise
whole-vs-chunk residual/Jacobian equivalence gate before Newton solve.

The dedicated gate for that local investigation is
`debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000`.
It is deliberately not classified as a story/performance test: it does not solve
the BVP and it does not interpret timings. It only checks that `Sparse + AtomView
+ AOT` whole and chunked generated callbacks produce identical residual and
Jacobian values on the same initial vector. Use it when the release matrix points
at a sparse chunking bug.

```powershell
cargo test debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000 -- --ignored --nocapture
```

### `combustion_3000_banded_lambdify_vs_aot_end_to_end_stress`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

Hypothesis: the previous `n_steps=200` and `n_steps=1000` matrices show that the
Banded route is the practical route for structured combustion BVPs, and that C
`tcc` is often the most competitive cold-build AOT toolchain. This stress test
pushes the same question to `n_steps=3000`: does Banded AOT still solve to the
Lambdify baseline, and does the larger grid finally make AOT/compiled callbacks
worthwhile in honest wall-clock end-to-end time?

This is intentionally a narrow stress matrix, not a full toolchain zoo. The
original version ran Banded AtomView Lambdify against Banded AtomView AOT and
revealed a large preparation anomaly. The current version is a matched-assembly
control: it runs Banded `ExprLegacy` Lambdify against Banded `ExprLegacy` AOT
with `tcc/whole` and `tcc/chunk4`, matching the symbolic assembly backend used
by `bvp_generated_backend_pipeline_comparison_table`. Toolchain ranking is not
the purpose of this already expensive test; `gcc`, `zig`, and Rust AOT are
covered by smaller toolchain matrices.

The test uses `RebuildAlways` and `Release`, so `total_ms` is the honest "press
the button and wait for the answer" time: symbolic assembly, AOT
generation/build/link, and the Newton solve. A single in-process list of rows
proved untrustworthy for timing because large symbolic preparations interfere
with later rows. The current test therefore launches every selected row in a
fresh child process and repeats the same `Lambdify -> tcc/whole -> tcc/chunk4`
sequence twice. This isolates Rust-side callback registries and loaded AOT
libraries, but intentionally leaves machine-level drift visible: sustained
CPU load, memory pressure, disk/antivirus activity, and persistent on-disk AOT
files are part of the observed environment. The saved AtomView result below
is historical evidence for the separate AtomView investigation; new runs
should show `assembly=ExprLegacy`.

Command:

```powershell
Remove-Item Env:\BVP_AOT_MATRIX_FILTER -ErrorAction SilentlyContinue
cargo test --release combustion_3000_banded_lambdify_vs_aot_end_to_end_stress -- --ignored --nocapture --test-threads=1
```
CPU 4 Core
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_3000_banded_lambdify_vs_aot_end_to_end_stress ... [BVP Damp isolated cold] protocol cooldown_ms=5000, cleanup_child_artifacts=false
[BVP Damp isolated cold] launching repetition 1/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=4621.540 symbolic_ms=4000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=5189.730 symbolic_ms=5000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=4444.232 symbolic_ms=4000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=4796.873 symbolic_ms=4000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=4830.796 symbolic_ms=4000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=4503.061 symbolic_ms=4000.000 status=ok
[BVP Damp stress] combustion-3000 raw process-isolated cold observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | ExprLegacy | 4621.540 |    4000.000 |           2183.403 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | AOT      | tcc/whole  | 5189.730 |    5000.000 |           2441.148 |         380.123 |             10.625 |              3.227 |    1.000 |    1.000 | ok
  1 |   3 | AOT      | tcc/chunk4 | 4444.232 |    4000.000 |           1948.739 |         320.278 |              6.067 |              1.566 |    4.000 |    4.000 | ok
  2 |   1 | Lambdify | ExprLegacy | 4796.873 |    4000.000 |           1914.411 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | AOT      | tcc/whole  | 4830.796 |    4000.000 |           2058.435 |         496.166 |             10.706 |              2.617 |    1.000 |    1.000 | ok
  2 |   3 | AOT      | tcc/chunk4 | 4503.061 |    4000.000 |           1978.357 |         320.369 |              5.579 |              1.526 |    4.000 |    4.000 | ok

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify |  2/2   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/whole  | rebuild+seq+whole |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT timing/counters
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | 4709.207 +/- 87.666 [4621.540, 4796.873] | 4000.000 +/- 0.000 | 4000.000 +/- 0.000 | 66.000 +/- 11.000 | 1.500 +/- 0.500 | 9.000 +/- 3.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 5010.263 +/- 179.467 [4830.796, 5189.730] | 4500.000 +/- 500.000 | 4500.000 +/- 500.000 | 60.000 +/- 5.000 | 3.500 +/- 0.500 | 7.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 4473.646 +/- 29.415 [4444.232, 4503.061] | 4000.000 +/- 0.000 | 4000.000 +/- 0.000 | 55.500 +/- 0.500 | 2.500 +/- 0.500 | 5.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT callback stages
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 10.665 +/- 0.040   | 2.922 +/- 0.305    | 0.858 +/- 0.084      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 18000.000 +/- 0.000 | 62988.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 5.823 +/- 0.244    | 1.546 +/- 0.020    | 0.915 +/- 0.153      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 4500.000 +/- 0.000 | 15747.000 +/- 0.000 | none         | none        

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT lifecycle/refinement stages
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 4000.000 +/- 0.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 4500.000 +/- 500.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 4000.000 +/- 0.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT symbolic handoff passes
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 3863.257 +/- 14.942 | 593.747 +/- 18.354 | 2048.907 +/- 134.496 | 54.644 +/- 4.312 | 950.145 +/- 135.327 | -                | -               | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | 3186.344 +/- 246.633 | 674.420 +/- 46.195 | 2249.792 +/- 191.357 | 56.949 +/- 6.577 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 142.339 +/- 9.463
AOT      | Banded | tcc/chunk4 | 2892.245 +/- 7.430 | 676.440 +/- 24.594 | 1963.548 +/- 14.809 | 49.091 +/- 0.562 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 139.218 +/- 0.326

[BVP Damp stress] combustion-3000 internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 2048.907 +/- 134.496 | 7.969 +/- 0.002 | 2035.442 +/- 133.929 | 0.000 +/- 0.000 | 3.732 +/- 0.411
AOT      | Banded | tcc/whole  | 2249.792 +/- 191.357 | 8.136 +/- 0.215 | 2236.350 +/- 190.843 | 0.000 +/- 0.000 | 3.930 +/- 0.307
AOT      | Banded | tcc/chunk4 | 1963.548 +/- 14.809 | 8.066 +/- 0.103 | 1950.572 +/- 14.647 | 0.000 +/- 0.000 | 3.413 +/- 0.046

[BVP Damp stress] combustion-3000 Lambdify callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 950.145 +/- 135.327 | 517.666 +/- 114.880 | 432.473 +/- 20.447
AOT      | Banded | tcc/whole  | 0.002 +/- 0.000 | -                | -               
AOT      | Banded | tcc/chunk4 | 0.002 +/- 0.000 | -                | -               

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | tcc/whole  | 177.967 +/- 6.833 | 116.356 +/- 3.522 | 116.286 +/- 3.515 | 71.837 +/- 0.374 | 33.205 +/- 1.599 | 14.166 +/- 1.649 | 11.604 +/- 4.254 | 438.144 +/- 58.022 | 107.898 +/- 26.116
AOT      | Banded | tcc/chunk4 | 163.202 +/- 0.138 | 101.422 +/- 0.542 | 101.319 +/- 0.547 | 61.948 +/- 0.278 | 30.185 +/- 0.207 | 10.384 +/- 0.008 | 7.885 +/- 0.120 | 320.323 +/- 0.046 | 69.515 +/- 0.158
ok

CPU 12 Core
╰────────────────────────────────────────────────┴────────────────────────╯
[BVP Damp debug] Banded rust chunk4: residual_len=6000, jacobian=6000x6000
[BVP Damp debug] AtomView AOT whole-vs-chunk4 callback correctness matrix
matrix | toolchain | comparison           | residual_diff | jacobian_diff | status
------------------------------------------------------------------------------------------------------------
Sparse | gcc       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Sparse | gcc       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Sparse | gcc       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Sparse | tcc       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Sparse | tcc       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Sparse | tcc       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Sparse | zig       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Sparse | zig       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Sparse | zig       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Sparse | rust      | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Sparse | rust      | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Sparse | rust      | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Banded | gcc       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Banded | gcc       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Banded | gcc       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Banded | tcc       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Banded | tcc       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Banded | tcc       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Banded | zig       | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Banded | zig       | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Banded | zig       | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok
Banded | rust      | lambdify-vs-whole    |    0.000000e0 |    0.000000e0 | ok
Banded | rust      | whole-vs-chunk4      |    0.000000e0 |    0.000000e0 | ok
Banded | rust      | lambdify-vs-chunk4   |    0.000000e0 |    0.000000e0 | ok

[BVP Damp debug] AtomView AOT callback probe statistics; all time columns are milliseconds
matrix | toolchain | mode     | total_probe_ms | prepare_ms | residual_ms | jacobian_ms | res_calls | jac_calls | residual_len | jacobian_shape | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | baseline  | lambdify |       4134.880 |   1340.441 |       1.887 |    2791.912 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | gcc       | whole    |       5618.246 |   2874.586 |       1.292 |    2742.102 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | gcc       | chunk4   |       5122.765 |   2405.702 |       1.298 |    2715.487 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | tcc       | whole    |       3484.730 |    765.902 |       1.209 |    2717.321 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | tcc       | chunk4   |       3406.889 |    646.935 |       1.175 |    2758.501 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | zig       | whole    |      35804.647 |  33058.836 |       0.689 |    2744.848 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | zig       | chunk4   |      32992.672 |  30272.788 |       0.961 |    2718.660 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | rust      | whole    |       8592.354 |   5813.192 |       1.314 |    2777.595 |         1 |         1 |         6000 |   6000x6000   | ok
Sparse | rust      | chunk4   |       7563.373 |   4776.222 |       1.563 |    2785.335 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | baseline  | lambdify |        552.799 |    402.985 |       0.507 |     149.275 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | gcc       | whole    |       2684.754 |   2531.466 |       1.075 |     152.187 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | gcc       | chunk4   |       2425.290 |   2268.654 |       1.146 |     155.462 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | tcc       | whole    |        778.630 |    629.144 |       1.070 |     148.390 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | tcc       | chunk4   |        775.524 |    620.655 |       1.125 |     153.716 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | zig       | whole    |      32288.706 |  32140.361 |       0.591 |     147.725 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | zig       | chunk4   |      30034.642 |  29885.292 |       0.935 |     148.388 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | rust      | whole    |       5525.589 |   5374.459 |       1.241 |     149.861 |         1 |         1 |         6000 |   6000x6000   | ok
Banded | rust      | chunk4   |       4753.809 |   4596.489 |       1.171 |     156.120 |         1 |         1 |         6000 |   6000x6000   | ok
test numerical::BVP_Damp::BVP_Damp_tests4::tests::debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000 ... ok

Interpretation after the Banded sparse-first Jacobian change:

This run confirms a real architectural improvement rather than a favorable
ordering accident. All three rows still follow the same nonlinear path
(`5` iterations, `10` linear solves, `1` Jacobian rebuild), and both AOT
rows agree with the Lambdify reference to `1.110e-16`. The optimization did
not alter the solution.

The cold end-to-end result is now stable and interpretable. `Lambdify`
completes in `4.709 +/- 0.088 s`; `tcc/whole` takes
`5.010 +/- 0.179 s`, while `tcc/chunk4` takes
`4.474 +/- 0.029 s`. Thus whole sequential cold AOT is not a win on this
sample, but explicit four-way chunking is about `5%` faster than Lambdify
and about `11%` faster than `tcc/whole` in the observed runs. This is an
honest wall-clock observation, not yet an attribution of the entire difference
to callback parallelism: `initial_sym_jac` and `compile_link` also differ
materially between the AOT rows.

The diagnostic table identifies why earlier measurements were pathological.
Before the sparse-first change, the recorded `initial_sym_jac` values for the
same large Banded family were measured in tens of seconds (`~48-56 s` for the
AOT rows in focused runs). They are now `2.250 +/- 0.191 s` for `tcc/whole`
and `1.964 +/- 0.015 s` for `tcc/chunk4`. The new internal table reports
`dense_cache = 0.000 ms` in every Banded row, exactly as required by the new
storage route. The removed full zero-filled dense Jacobian was therefore a
genuine large-grid performance defect, not merely noisy instrumentation.

What remains is now visible and much cleaner. Almost all remaining
`initial_sym_jac` cost is `row_diff` (`~1.95-2.24 s`); variable-set setup
and sparse flattening are only a few milliseconds. Any next optimization
should target symbolic differentiation itself, its expression construction,
or safe reuse across equivalent local stencil rows, rather than matrix
materialization or AOT compilation.

Chunking is nevertheless confirmed to execute. `tcc/chunk4` reports four
actual jobs with no fallback and reduces callback residual/Jacobian values
from `10.665/2.922 ms` to `5.823/1.546 ms`. This proves a hot-callback
benefit. The larger full-wall-clock advantage of the chunked row cannot yet
be assigned to that benefit alone, because the same row also recorded a
smaller symbolic-Jacobian interval and a shorter compiler/linker interval.

Optional focused run:

```powershell
$env:BVP_AOT_MATRIX_FILTER="tcc/whole"
cargo test --release combustion_3000_banded_lambdify_vs_aot_end_to_end_stress -- --ignored --nocapture --test-threads=1
```

A focused filter is useful for debugging one row. The current implementation
runs each selected `Lambdify`, `tcc/whole`, and `tcc/chunk4` row in its own
fresh child process and reconstructs the same correctness/timing tables in the
parent. Internal AOT callback parallelism is not disabled: the `chunk4` row
asserts multiple executed jobs and no sequential fallback. For full
comparisons, run with `BVP_AOT_MATRIX_FILTER` cleared.

The isolated runner accepts two test-protocol environment variables.
`BVP_AOT_COLD_COOLDOWN_MS` inserts a pause after every completed child row;
this is meant to expose or reduce thermal, paging, and filesystem-scanning
drift, not to improve solver performance. With
`BVP_AOT_COLD_CLEAN_ARTIFACTS=1`, the parent removes only generated
`build-<child-pid>-...` directories after that child process has exited. It
does not clear live runtime registries or unload a DLL from a running solver.

Controlled cooldown run:

```powershell
$env:BVP_AOT_COLD_COOLDOWN_MS="10000"
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
cargo test --release combustion_3000_banded_lambdify_vs_aot_end_to_end_stress -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_AOT_COLD_COOLDOWN_MS -ErrorAction SilentlyContinue
Remove-Item Env:\BVP_AOT_COLD_CLEAN_ARTIFACTS -ErrorAction SilentlyContinue
12 Core
ld_artifacts=false
[BVP Damp isolated cold] launching repetition 1/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=2856.057 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2831.846 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2809.510 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=2660.386 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2948.038 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2868.862 symbolic_ms=2000.000 status=ok
[BVP Damp stress] combustion-3000 ExprLegacy raw process-isolated cold observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | ExprLegacy | 2856.057 |    2000.000 |            237.753 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | AOT      | tcc/whole  | 2831.846 |    2000.000 |            241.448 |         123.875 |              7.212 |              1.623 |    1.000 |    1.000 | ok
  1 |   3 | AOT      | tcc/chunk4 | 2809.510 |    2000.000 |            238.606 |         105.221 |              4.554 |              1.086 |    4.000 |    4.000 | ok
  2 |   1 | Lambdify | ExprLegacy | 2660.386 |    2000.000 |            242.504 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | AOT      | tcc/whole  | 2948.038 |    2000.000 |            249.019 |         127.966 |              7.351 |              1.662 |    1.000 |    1.000 | ok
  2 |   3 | AOT      | tcc/chunk4 | 2868.862 |    2000.000 |            235.386 |         105.231 |              4.725 |              0.989 |    4.000 |    4.000 | ok

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify |  2/2   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/whole  | rebuild+seq+whole |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT timing/counters
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | 2758.222 +/- 97.836 [2660.386, 2856.057] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 21.500 +/- 0.500 | 1.500 +/- 0.500 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 2889.942 +/- 58.096 [2831.846, 2948.038] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 19.500 +/- 0.500 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 2839.186 +/- 29.676 [2809.510, 2868.862] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 22.000 +/- 1.000 | 1.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT callback stages
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 7.282 +/- 0.070    | 1.643 +/- 0.020    | 0.357 +/- 0.046      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 18000.000 +/- 0.000 | 62988.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 4.640 +/- 0.086    | 1.038 +/- 0.048    | 0.401 +/- 0.054      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 4500.000 +/- 0.000 | 15747.000 +/- 0.000 | none         | none        

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT lifecycle/refinement stages
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT symbolic handoff passes
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 523.850 +/- 2.307 | 71.458 +/- 1.638   | 240.128 +/- 2.375 | 30.024 +/- 2.494 | 52.258 +/- 0.389 | -                | -               | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | 438.918 +/- 9.119 | 70.486 +/- 1.515   | 245.234 +/- 3.785 | 28.579 +/- 1.900 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 61.111 +/- 0.263
AOT      | Banded | tcc/chunk4 | 435.960 +/- 5.544 | 71.777 +/- 0.242   | 236.996 +/- 1.610 | 27.632 +/- 0.463 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 61.680 +/- 0.305

[BVP Damp stress] combustion-3000 ExprLegacy internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 240.128 +/- 2.375 | 4.104 +/- 0.046 | 232.700 +/- 3.206 | 0.000 +/- 0.000 | 2.437 +/- 0.626
AOT      | Banded | tcc/whole  | 245.234 +/- 3.785 | 4.594 +/- 0.226 | 237.513 +/- 4.099 | 0.000 +/- 0.000 | 2.209 +/- 0.375
AOT      | Banded | tcc/chunk4 | 236.996 +/- 1.610 | 4.323 +/- 0.054 | 229.889 +/- 1.740 | 0.000 +/- 0.000 | 1.962 +/- 0.080

[BVP Damp stress] combustion-3000 ExprLegacy Lambdify callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 52.258 +/- 0.389 | 15.588 +/- 0.313 | 36.665 +/- 0.077
AOT      | Banded | tcc/whole  | 0.001 +/- 0.000 | -                | -               
AOT      | Banded | tcc/chunk4 | 0.001 +/- 0.000 | -                | -               

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | tcc/whole  | 84.497 +/- 0.186 | 60.298 +/- 0.130 | 60.246 +/- 0.124 | 37.253 +/- 1.341 | 11.986 +/- 0.146 | 7.097 +/- 0.369 | 4.740 +/- 0.017 | 125.920 +/- 2.045 | 11.019 +/- 0.342
AOT      | Banded | tcc/chunk4 | 74.255 +/- 0.489 | 48.107 +/- 0.279 | 48.055 +/- 0.278 | 29.848 +/- 0.108 | 11.837 +/- 0.080 | 6.710 +/- 0.019 | 4.063 +/- 0.378 | 105.226 +/- 0.005 | 10.957 +/- 0.025
ok

```

### `combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

This is the production-facing companion to the `ExprLegacy` control above.
The older stress row must remain available because it documents the removal
of the dense-cache and repeated-handoff defects. Backend recommendations,
however, should now be based on the corrected `AtomView` path: at
`n_steps=1000` it reduced `initial_sym_jac` to about `39 ms` and reduced the
frontend solution delta to `4.079e-12`.

The new test uses the identical process-isolated protocol and identical
assertions as the ExprLegacy control. It compares Banded `AtomView +
Lambdify` with Banded `AtomView + TCC AOT` in `whole` and forced `chunk4`
modes, performs two cold repetitions, verifies solution agreement with the
Lambdify baseline, and requires the chunked callback route to report actual
parallel jobs with no fallback. Its timing tables therefore answer the
practical question: once the recommended symbolic frontend is used, does
compiled TCC and/or four-way callback chunking improve honest wall-clock time
on the large combustion solve?

Controlled release command:

```powershell
Remove-Item Env:\BVP_AOT_MATRIX_FILTER -ErrorAction SilentlyContinue
$env:BVP_AOT_COLD_COOLDOWN_MS="5000"
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
cargo test --release combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_AOT_COLD_COOLDOWN_MS -ErrorAction SilentlyContinue
Remove-Item Env:\BVP_AOT_COLD_CLEAN_ARTIFACTS -ErrorAction SilentlyContinue
```
4 Core
child_artifacts=true
[BVP Damp isolated cold] launching repetition 1/2 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=4922.683 symbolic_ms=4000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2754.956 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2638.131 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=2732.643 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2829.871 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2983.275 symbolic_ms=2000.000 status=ok
[BVP Damp stress] combustion-3000 AtomView raw process-isolated cold observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | AtomView   | 4922.683 |    4000.000 |            147.964 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | AOT      | tcc/whole  | 2754.956 |    2000.000 |            115.185 |         400.981 |             13.615 |              2.676 |    1.000 |    1.000 | ok
  1 |   3 | AOT      | tcc/chunk4 | 2638.131 |    2000.000 |            110.859 |         326.381 |              7.623 |              1.880 |    4.000 |    4.000 | ok
  2 |   1 | Lambdify | AtomView   | 2732.643 |    2000.000 |            119.814 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | AOT      | tcc/whole  | 2829.871 |    2000.000 |            125.323 |         370.132 |             13.897 |             10.720 |    1.000 |    1.000 | ok
  2 |   3 | AOT      | tcc/chunk4 | 2983.275 |    2000.000 |            114.089 |         410.489 |              6.620 |              1.576 |    4.000 |    4.000 | ok

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify |  2/2   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole |  2/2   | 4.319e-16 +/- 0.0e0  | 4.313e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 |  2/2   | 4.319e-16 +/- 0.0e0  | 4.313e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT timing/counters
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | 3827.663 +/- 1095.020 [2732.643, 4922.683] | 3000.000 +/- 1000.000 | 3000.000 +/- 1000.000 | 65.500 +/- 1.500 | 5.000 +/- 1.000 | 12.000 +/- 1.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | 2792.414 +/- 37.458 [2754.956, 2829.871] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 64.000 +/- 2.000 | 7.000 +/- 4.000 | 9.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | 2810.703 +/- 172.572 [2638.131, 2983.275] | 2500.000 +/- 500.000 | 2000.000 +/- 0.000 | 62.000 +/- 1.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT callback stages
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | 13.756 +/- 0.141   | 6.698 +/- 4.022    | 0.810 +/- 0.007      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 18000.000 +/- 0.000 | 62988.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | 7.121 +/- 0.501    | 1.728 +/- 0.152    | 0.824 +/- 0.047      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 4500.000 +/- 0.000 | 15747.000 +/- 0.000 | none         | none        

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT lifecycle/refinement stages
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 3000.000 +/- 1000.000
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT symbolic handoff passes
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 1831.410 +/- 207.654 | 366.330 +/- 110.240 | 133.889 +/- 14.075 | 89.107 +/- 0.893 | 749.495 +/- 0.026 | -                | -               | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | 823.089 +/- 11.022 | 240.840 +/- 16.062 | 120.254 +/- 5.069 | 84.497 +/- 1.002 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 210.891 +/- 0.915
AOT      | Banded | tcc/chunk4 | 787.827 +/- 19.975 | 226.922 +/- 8.030  | 112.474 +/- 1.615 | 85.212 +/- 0.766 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 236.442 +/- 17.209

[BVP Damp stress] combustion-3000 AtomView internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 133.889 +/- 14.075 | 28.366 +/- 3.327 | 88.389 +/- 9.944 | 0.007 +/- 0.003 | 13.820 +/- 0.574
AOT      | Banded | tcc/whole  | 120.254 +/- 5.069 | 23.870 +/- 0.398 | 81.107 +/- 4.894 | 0.009 +/- 0.000 | 12.891 +/- 0.242
AOT      | Banded | tcc/chunk4 | 112.474 +/- 1.615 | 24.723 +/- 1.277 | 72.590 +/- 0.703 | 0.004 +/- 0.000 | 12.153 +/- 0.111

[BVP Damp stress] combustion-3000 AtomView Lambdify callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 749.495 +/- 0.026 | 144.078 +/- 5.779 | 605.410 +/- 5.805
AOT      | Banded | tcc/whole  | 0.002 +/- 0.000 | -                | -               
AOT      | Banded | tcc/chunk4 | 0.002 +/- 0.000 | -                | -               

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | tcc/whole  | 207.938 +/- 12.552 | 147.497 +/- 13.462 | 147.440 +/- 13.465 | 89.218 +/- 10.243 | 31.005 +/- 1.422 | 14.782 +/- 0.508 | 5.858 +/- 0.374 | 385.557 +/- 15.424 | 91.281 +/- 8.263
AOT      | Banded | tcc/chunk4 | 199.327 +/- 14.492 | 125.570 +/- 11.392 | 125.485 +/- 11.375 | 81.131 +/- 9.201 | 32.246 +/- 1.583 | 16.434 +/- 0.233 | 6.870 +/- 0.464 | 368.435 +/- 42.054 | 107.194 +/- 14.128
ok

12   Core

test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress ... [BVP Damp isolated cold] protocol cooldown_ms=5000, cleanup_child_artifacts=true
[BVP Damp isolated cold] launching repetition 1/2 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=2536.791 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2782.190 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2832.848 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=2646.221 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=2758.553 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=2772.996 symbolic_ms=2000.000 status=ok
[BVP Damp stress] combustion-3000 AtomView raw process-isolated cold observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | AtomView   | 2536.791 |    2000.000 |             26.013 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | AOT      | tcc/whole  | 2782.190 |    2000.000 |             27.197 |         116.047 |              8.526 |              1.501 |    1.000 |    1.000 | ok
  1 |   3 | AOT      | tcc/chunk4 | 2832.848 |    2000.000 |             26.778 |         104.782 |              3.953 |              0.790 |    4.000 |    4.000 | ok
  2 |   1 | Lambdify | AtomView   | 2646.221 |    2000.000 |             25.796 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | AOT      | tcc/whole  | 2758.553 |    2000.000 |             25.063 |         116.312 |              6.749 |              1.510 |    1.000 |    1.000 | ok
  2 |   3 | AOT      | tcc/chunk4 | 2772.996 |    2000.000 |             25.764 |         104.247 |              4.607 |              0.783 |    4.000 |    4.000 | ok

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify |  2/2   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole |  2/2   | 8.882e-16 +/- 0.0e0  | 8.868e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 |  2/2   | 8.882e-16 +/- 0.0e0  | 8.868e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT timing/counters
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | 2591.506 +/- 54.715 [2536.791, 2646.221] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 21.000 +/- 1.000 | 2.500 +/- 0.500 | 2.500 +/- 0.500 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | 2770.371 +/- 11.819 [2758.553, 2782.190] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 19.500 +/- 0.500 | 1.000 +/- 0.000 | 5.000 +/- 1.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | 2802.922 +/- 29.926 [2772.996, 2832.848] | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 22.000 +/- 0.000 | 1.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT callback stages
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | 7.637 +/- 0.888    | 1.506 +/- 0.005    | 0.276 +/- 0.024      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 18000.000 +/- 0.000 | 62988.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | 4.280 +/- 0.327    | 0.787 +/- 0.003    | 0.336 +/- 0.020      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 4500.000 +/- 0.000 | 15747.000 +/- 0.000 | none         | none        

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT lifecycle/refinement stages
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | atomview+lambdify | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000
AOT      | Banded | tcc/whole  | atomview+rebuild+seq+whole | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | atomview+rebuild+par+chunk4 | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 2000.000 +/- 0.000

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT symbolic handoff passes
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 419.573 +/- 3.959 | 129.354 +/- 4.704  | 25.905 +/- 0.109 | 38.210 +/- 0.435 | 52.026 +/- 0.622 | -                | -               | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | 336.252 +/- 9.281 | 131.792 +/- 0.372  | 26.130 +/- 1.067 | 39.053 +/- 1.805 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 89.188 +/- 3.023
AOT      | Banded | tcc/chunk4 | 328.116 +/- 3.983 | 122.997 +/- 0.946  | 26.271 +/- 0.507 | 38.491 +/- 0.206 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 96.582 +/- 4.091

[BVP Damp stress] combustion-3000 AtomView internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 25.905 +/- 0.109 | 10.339 +/- 0.031 | 10.997 +/- 0.106 | 0.000 +/- 0.000 | 3.221 +/- 0.048
AOT      | Banded | tcc/whole  | 26.130 +/- 1.067 | 10.355 +/- 0.175 | 11.237 +/- 0.633 | 0.000 +/- 0.000 | 3.116 +/- 0.211
AOT      | Banded | tcc/chunk4 | 26.271 +/- 0.507 | 10.165 +/- 0.238 | 11.297 +/- 0.284 | 0.000 +/- 0.000 | 3.075 +/- 0.099

[BVP Damp stress] combustion-3000 AtomView Lambdify callback compilation stages
[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank.
source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile
--------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | 52.026 +/- 0.622 | 15.389 +/- 0.071 | 36.631 +/- 0.550
AOT      | Banded | tcc/whole  | 0.001 +/- 0.000 | -                | -               
AOT      | Banded | tcc/chunk4 | 0.001 +/- 0.000 | -                | -               

[BVP Damp stress] combustion-3000 AtomView Banded Lambdify vs AOT cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | AtomView   | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | tcc/whole  | 88.273 +/- 3.183 | 62.930 +/- 2.100 | 62.873 +/- 2.113 | 40.603 +/- 0.573 | 10.801 +/- 0.030 | 9.828 +/- 1.230 | 4.023 +/- 0.928 | 116.180 +/- 0.133 | 10.907 +/- 0.444
AOT      | Banded | tcc/chunk4 | 80.222 +/- 1.343 | 52.582 +/- 0.734 | 52.516 +/- 0.717 | 35.049 +/- 0.419 | 11.492 +/- 0.246 | 8.707 +/- 0.487 | 5.136 +/- 0.153 | 104.514 +/- 0.267 | 11.355 +/- 0.157
ok



Release interpretation for the AtomView production stress:

Correctness is closed at this scale. Both TCC AOT routes agree with the
AtomView Lambdify baseline to `4.319e-16`, and all three routes use the same
Newton choreography: five nonlinear iterations, ten linear solves and one
Jacobian rebuild. The large-grid AOT and chunked callback routes therefore do
not introduce a detectable solution drift.

The key result is the disappearance of the expensive symbolic-Jacobian
frontier. In the preceding ExprLegacy control at the same grid size,
`initial_sym_jac` was approximately `2.0-2.25 s`; with AtomView it is now
`133.889 +/- 14.075 ms` for Lambdify, `120.254 +/- 5.069 ms` for TCC
`whole`, and `112.474 +/- 1.615 ms` for TCC `chunk4`. The remaining
`row_diff` work is only `72.6-88.4 ms`, and `dense_cache` remains effectively
zero. Thus the optimized AtomView route scales correctly to the 3000-point
combustion problem and should replace ExprLegacy as the production symbolic
frontend for this class of Banded solves.

Parallel callback execution is real, but this run does not show a cold-start
wall-clock win for it. `tcc/chunk4` uses four residual and four Jacobian jobs
without fallback and reduces the hot residual-values interval from
`13.756` to `7.121 ms`; Jacobian-values evaluation also falls from
`6.698` to `1.728 ms`, although the sequential measurement is visibly noisy.
Those callback savings are small compared with the roughly `2.8 s` complete
cold solve. Consequently `tcc/whole = 2792.414 +/- 37.458 ms` and
`tcc/chunk4 = 2810.703 +/- 172.572 ms` are a statistical tie, not evidence
that forced chunking is either a cold-start improvement or a regression.

The Lambdify comparison is informative but noisy: its two cold samples are
`4922.683` and `2732.643 ms`, while the measured `initial_sym_jac` remains
small in both. The first-row excess is therefore outside the now-fixed
symbolic Jacobian stage and makes the aggregate Lambdify mean unsuitable for
a precise TCC ranking. A conservative production conclusion is: choose
`AtomView + Banded`; use TCC AOT when an artifact can be reused or the solve
is repeated; treat forced `chunk4` as a demonstrated hot-callback option, not
as the default cold policy until a rotated multi-run table separates its
small runtime gain from setup and machine noise.

Archived interpretation from earlier ExprLegacy stress transcripts (retained
for chronology; it does not describe the AtomView release rows above):

This run settles one architectural question and opens a benchmark-methodology
one. The old in-process AOT-registry hypothesis cannot explain the remaining
variance: the second fresh-process Lambdify baseline is already almost twice
as slow as the first (`43.312 s -> 85.863 s`), with the increase concentrated
in symbolic Jacobian construction (`30.756 s -> 67.530 s`). AOT rows then
degrade in the same direction. For `tcc/whole`, total time rises from
`34.399 s` to `173.085 s`, while `initial_sym_jac` and `compile_link` rise
from `22.819/0.820 s` to `86.448/37.777 s`. For `tcc/chunk4`, they rise from
`20.127/4.711 s` to `123.716/22.560 s`.

The generated runtime remains correct and the parallel callback is real:
`tcc/chunk4` reports four residual and four Jacobian jobs without fallback,
while its hot residual/Jacobian-value intervals are lower than the sequential
row. Thus this output must not be used to conclude that chunking is slower.
It shows that an `n_steps=3000` cold benchmark is strongly affected by
run-to-run machine drift outside the process: sustained symbolic workloads,
compiler/file activity, memory pressure or paging, thermal/power behavior, or
filesystem scanning are all plausible contributors.

For correctness and lifecycle, the test is sound: all solutions agree to
`1.110e-16`, direct rebinding remains in force, and no refinement rebuild is
present. For performance ranking, two repetitions in a fixed order are not
enough. A defensible follow-up is to keep this large test as a stress gate,
and use a smaller repeated matrix for ranking with rotated row order plus
sentinel Lambdify observations that quantify ambient drift during the run.

Follow-up process-isolated rerun:

The newer `combustion-3000` transcript confirms the earlier diagnosis while
removing one suspected culprit. The two Lambdify rows still drift strongly
(`45.224 s -> 64.878 s`), and the AOT routes still swing in opposite
directions: `tcc/whole = 43.503 s -> 94.832 s`, while `tcc/chunk4 =
64.204 s -> 31.113 s`. However, AOT compilation is no longer the expensive
or unstable part: `compile_link` remains below `1.52 s` in all four AOT rows.
The spread follows the symbolic stage, especially `initial_sym_jac`:
`tcc/whole = 33.598 s -> 73.303 s` and `tcc/chunk4 =
43.395 s -> 19.580 s`.

This is stronger than merely observing "machine slowdown". Because the last
row becomes the fastest row in repetition two, the behavior is not a simple
monotone heating/order penalty. The unstable quantity is the large symbolic
Jacobian construction pass itself, possibly amplified by allocation,
paging/cache state, or other system effects. Runtime chunking remains valid:
`tcc/chunk4` executes four jobs without fallback and reduces the Jacobian
values callback from about `3.0 ms` to `1.78 ms`.

The captured stress transcript reports artifact cleanup as disabled, so it is
not yet the controlled clean/cooldown comparison described above. Its correct
conclusion is narrower: AOT lifecycle and parallel callback correctness are
healthy; `combustion-3000` wall-clock ranking is dominated by unstable
symbolic preparation and must not be used to select `whole` versus `chunk4`.

```text
Date: 2026-05-26
Status: historical pre-isolation interpretation; superseded by the process-isolated analysis above.
Important numbers:
  Correctness is clean for both AOT routes: `solve_diff = 1.110e-16`, 2/2.
  Direct rebinding remains validated: every `post_build_*` column is blank and
  `initial_bind` stays below `0.25 ms` for both AOT variants.
  Aggregate totals cannot rank the strategies reliably:
  `tcc/whole = 47.612 +/- 8.290 s`,
  `tcc/chunk4 = 76.823 +/- 40.830 s`.
  The raw rows explain the large `chunk4` variance. When it follows `whole`,
  it takes `117.653 s` with `initial_sym_jac = 89.279 s`; when it is the first
  AOT row, it takes `35.993 s` with `initial_sym_jac = 22.035 s`.
  `whole` is much less affected in the reverse sequence: `39.322 s` as the
  first AOT row and `55.902 s` after `chunk4`, with symbolic Jacobian times
  of `29.012 s` and `26.220 s`.
  The compiler/linker interval also grows on third-position AOT rows:
  `chunk4` after `whole` has `13.542 s`, while `chunk4` first has `4.466 s`;
  `whole` after `chunk4` similarly has `13.690 s`, while `whole` first has
  `1.509 s`.
  In the hot compiled callback portion, parallel execution is real and useful:
  `chunk4` reports four jobs without fallback and reduces residual evaluation
  from `11.760 ms` to `7.383 ms`, Jacobian values from `3.136 ms` to
  `2.139 ms`, and matrix assembly from `1.196 ms` to `1.030 ms`.
Conclusion:
  The order-resolved run definitively rules out an intrinsic four-times
  slowdown of `chunk4`: `chunk4` is the fastest AOT row when executed before
  `whole`. The poor aggregate value is caused by its one disastrous run after
  `whole`. The callback implementation is correctly parallel and faster in
  its measured hot stages. The remaining gap is process-lifetime interference
  during heavy cold preparation: a previous AOT build appears capable of
  distorting the next symbolic/build pass, especially for the
  `whole -> chunk4` order.
Follow-up:
  Do not optimize callback chunking on the basis of these totals. Investigate
  retained process state between AOT rows: loaded linked backends/registries,
  artifact lifecycle and memory pressure around very large symbolic objects.
  One concrete implementation candidate is the linked AOT runtime registry:
  the manifest key includes generated chunk metadata, so `tcc/whole` and
  `tcc/chunk4` are different registered backends, while `RebuildAlways`
  unregisters only the current key. A differently chunked DLL can therefore
  remain loaded during the next row. This warrants an isolation experiment;
  it does not yet prove that registry retention alone caused the outlier.
  A confirmation run should repeat each order at least twice or execute each
  cold row in a fresh child process so that end-to-end rankings are not
  polluted by previous generated artifacts.
```

The raw table is now part of the test output and must be kept with future
results: without it, the aggregate `chunk4` mean would incorrectly suggest a
slow parallel callback rather than a sequence-sensitive cold-preparation
anomaly.

Toolchain interpretation note: these tables do not measure "bare C compiler
speed". They measure the RustedSciThe AOT route as a user experiences it:
symbolic assembly, code generation, file materialization, compiler/linker work,
dynamic loading, callback handoff, and the Newton solve. In that full pipeline,
the current release data supports a practical `tcc` advantage over `gcc`, but the
observed factor is usually about 1.5-2x for cold end-to-end rows, not 5-10x. A
larger multiplier may be true for isolated tinycc-vs-gcc compilation of a small C
translation unit, especially when gcc is asked to optimize heavily, but that is a
different benchmark. If we need that number, add a separate codegen-only test that
times compile/link of the same already-generated C source without symbolic
assembly, dynamic callback setup, or Newton iterations.

Interpretation of the recorded AtomView `combustion-3000` end-to-end run:

The poor AOT wall-clock result is not explained by generated callback execution
or by the banded linear solver. For `tcc/whole`, `linear_ms`, `jac_ms`, and
`fun_ms` are only tens of milliseconds, while the gap against Lambdify is about
`107 s`; almost the entire gap is reported in `symbolic_ms` (`197 s` versus
`94 s`). The parallel `tcc/chunk4` callback is actually faster than
`tcc/whole` in its hot residual/Jacobian timings, so this run must not be read
as evidence that generated callbacks are slow.

This does not directly contradict
`bvp_generated_backend_pipeline_comparison_table`: that pipeline measurement
currently exercises `ExprLegacy` and one generated problem, while this real
solver stress uses `AtomView` and may regenerate the discretized problem after
adaptive mesh refinement. In `NRBVP`, every non-empty refined mesh invokes
`try_eq_generate(...)` again. With an AOT build policy that creates a new
artifact for a changed mesh, one solve can therefore contain multiple large
code-generation/build cycles.

The lifecycle/refinement table subsequently ruled out adaptive-grid rebuilding:
both rows report `refinements=0` and `final_grid_points=3001`. Therefore the
remaining suspect is the AtomView preparation route itself. The test has now
been changed to an `ExprLegacy` matched-assembly control. If its AOT row follows
the favorable codegen pipeline result, the next task is a focused AtomView
preparation-stage breakdown rather than further AOT runtime tuning.

Update from the `ExprLegacy` matched-assembly control:

The hypothesis that AtomView alone caused the real-solve gap is disproved.
With `ExprLegacy` on both rows, Lambdify completed in `58.560 s`, whereas
`AOT C-tcc/whole` required `169.285 s`. Correctness remains excellent
(`solve_diff = 1.110e-16`), both routes use the same final grid with zero
refinements, and the hot AOT callback/linear-solver work is still only tens of
milliseconds. The unexplained interval is again entirely inside the solver's
one-time generated-backend preparation (`symbolic_ms = 164 s` for AOT against
`57 s` for Lambdify).

This is stronger evidence than the earlier AtomView result: the separate
pipeline table and the real solver now use the same assembly family but give
opposite performance stories. Inspection of the production handoff found a
likely mechanism. A `RebuildAlways` AOT solve first generates an unresolved
symbolic bundle, builds and registers the compiled artifact, and then calls
`regenerate_sparse_bundle_after_aot_build(...)` to regenerate the bundle under
`RequirePrebuilt` so compiled callbacks can be bound. A large symbolic system
may therefore be prepared twice in one button-to-result solve.

The next version of this story table reports that lifecycle directly rather
than inferring it from the coarse `Symbolic Operations` timer. It adds a
symbolic handoff-pass table with `initial_*` and `post_build_*` stages
(`discretization`, symbolic Jacobian, sparse preparation, and
`runtime_binding`). The binding columns are intentional: they show whether
the initial unresolved AOT pass has paid for a Lambdify callback preparation
that is discarded after the compiled artifact is built. It is followed by an
AOT cold-build table with artifact/module/lowering/source emission, file
materialization, external compile/link, and runtime registration times. These
values come from the production handoff and existing codegen breakdown rather
than a test-side wrapper.

The debug run with this breakdown confirmed the lifecycle defect. In the AOT
row, `initial_total = 93.992 s` is already essentially a complete Lambdify
preparation, including `initial_bind = 26.693 s`. After compilation, the
solver paid a second generated pass of `80.768 s`, dominated by a repeated
symbolic Jacobian construction (`post_sym_jac = 67.642 s`). Compilation itself
was not the cause (`compile_link = 3.096 s`). The `169.285 s` release result
and the `183.585 s` debug result are therefore explained by duplicated
preparation, not by slow AOT callback execution.

The production handoff has now been changed accordingly. For
`BuildIfMissing` and `RebuildAlways`, an AOT-targeting request prepares an AOT
bundle directly instead of first compiling Lambdify callbacks. Once the
artifact has been built and registered, the already prepared bundle is
promoted to `AotCompiled` and its runtime callbacks are rebound in place;
`regenerate_sparse_bundle_after_aot_build(...)` is no longer used. On the
next run, `initial_bind` should be negligible, the `post_build_*` symbolic
columns should be blank, and only the new `rebind_ms` value should be present.

Post-fix debug run:

The rerun validates the fix in the real combustion solve. Correctness is
unchanged (`solve_diff = 1.110e-16`), while `AOT C-tcc/whole` now completes
in `75.737 s` against the Lambdify baseline's `92.575 s`. Thus, even in a
debug build, AOT is about `18%` faster end to end on this case after removal
of the duplicated preparation.

The stage table is decisive. For AOT, `initial_bind` fell from `26.693 s` to
`0.226 ms`; every `post_build_*` symbolic regeneration field is blank; the
new direct callback attachment costs only `rebind_ms = 376.299 ms`. The
remaining AOT preparation is a single real symbolic pass
(`initial_sym_jac = 58.031 s`), plus cold build costs including
`compile_link = 4.394 s`. In contrast, Lambdify pays
`initial_bind = 23.784 s` to create callable lambdified evaluators.

The cold-build codegen fields are diagnostic sub-stages rather than a
sum-of-parts timing equation: for example, lowering work contributes inside
module/artifact construction. They should be used to locate hot areas, not
added to reconstruct total wall-clock time.

Latest multi-variant rerun after direct AOT rebinding:

The lifecycle correction remains effective: all AOT rows have blank
`post_build_*` symbolic columns and sub-millisecond initial binding, so the
old "prepare the same symbolic system twice" failure is no longer present.
Correctness also remains clean for `tcc/whole`, `tcc/chunk4`, and
`gcc/whole`.

The new bad result is narrower and must not be misdiagnosed. `tcc/chunk4`
completed in `188.804 s`, versus `57.538 s` for `tcc/whole`, and the excess
is reported almost entirely as `initial_sym_jac` (`156.027 s` versus
`44.433 s`). Inspection of the production path shows that both variants call
the same `prepare_symbolic_bvp_stage_with_params(...)`; the chunking
strategies are consumed only afterwards by
`prepare_sparse_backend_execution_timed(...)`. Therefore this single run does
not prove that chunk planning rebuilt the symbolic Jacobian. It demonstrates a
severe symbolic-stage outlier correlated with the chunk4 row in this ordering.

The runtime part supplies a separate, useful fact: parallel binding is real
(`res_jobs=4`, `jac_jobs=4`, no fallback), but it is not beneficial here.
Residual evaluation falls from `12.795 ms` to `6.964 ms`, while Jacobian
values rise from `3.029 ms` to `33.541 ms`; either way, these hot callback
costs are negligible compared with the unexplained symbolic-stage excess.

Next diagnostic step: repeat `tcc/whole` and `tcc/chunk4` in isolated focused
runs and in reversed order, with at least two repetitions per row, before
changing symbolic or chunking code. If only the chunk4-focused runs reproduce
the `initial_sym_jac` inflation, add instrumentation inside
`calc_jacobian_parallel_smart_optimized()` for row count, derivative count,
and rayon wall time. If the inflation follows run order instead, the culprit
is resource/cache interference in the heavy sequential test rather than
chunking semantics.

Isolated focused reruns, 2026-05-26:

The focused experiment rejects the hypothesis that `tcc/chunk4` inherently
multiplies symbolic construction work. Run alone, `tcc/whole` completes in
`75.352 s`, with `initial_sym_jac = 56.402 s`. Run alone, `tcc/chunk4`
completes in `63.915 s`, with `initial_sym_jac = 48.375 s`. Both solutions
agree with the Lambdify baseline to `1.110e-16`, both have blank
`post_build_*` fields, and both therefore confirm that direct AOT rebinding
has removed the former double-preparation defect.

The parallel callback is also genuinely active in the isolated `chunk4` run:
`res_jobs=4`, `jac_jobs=4`, with no fallback. Its linked callback interval is
not the source of the old disaster: although callback timings remain noisy
(`jacobian_values_ms = 35.971 ms` for `chunk4`, versus `128.415 ms` for this
particular `whole` run), these are millisecond-scale contributions inside
minute-scale symbolic preparation.

Consequently, the earlier full-matrix row in which `tcc/chunk4` took
`188.804 s` must be treated as contaminated wall-clock evidence, most likely
from order/resource interference among several very heavy symbolic builds
within one process. The full stress matrix remains valuable for correctness
and lifecycle assertions, but it is not a trustworthy performance ranking
unless variants are isolated or the experiment is redesigned with
order-balanced repetitions.

Practical conclusion from isolated evidence: for this `combustion-3000` Banded case, both cold AOT
routes outperform their local Lambdify baselines in isolated runs. The
measured improvements are approximately `18%` for `tcc/whole`
(`75.352 s` versus `92.132 s`) and `30%` for `tcc/chunk4`
(`63.915 s` versus `90.677 s`). One isolated run per row is enough to refute
the four-times-slower claim, but not enough to declare `chunk4` universally
faster. The test is now implemented as a process-isolated two-repetition
comparison; append its next output below before making a final strategy
recommendation.

### `combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix`

File: `src/numerical/BVP_Damp/BVP_Damp_tests4.rs`

Hypothesis: on a real combustion BVP solve with a moderate grid (`n_steps=200`),
Sparse and Banded Lambdify baselines and AOT variants should produce the same
solution while exposing toolchain/chunking performance differences. This is the
main practical end-to-end matrix to run before paying for the combustion-1000
release matrix. It covers `Sparse` and `Banded`, `gcc`, `tcc`, and `zig`, and two
runtime callback layouts: `whole` sequential callbacks and `chunk4` generated
callbacks with forced parallel runtime execution.

The test runs each row three times and prints separate tables. The correctness
table compares every row against the first successful Lambdify baseline in the same
repetition. The timing/counter table reports total wall time, solver timer time,
symbolic/build/bootstrap time, linear-system time, Jacobian time, residual/function
time, Newton iterations, linear solves, and Jacobian rebuilds. The callback/runtime
table reports linked AOT hot callback timings plus `res_jobs`, `jac_jobs`,
`res_work/job`, `jac_work/job`, and fallback reasons read from solver statistics.
This keeps correctness, solver-level performance, and actual parallel runtime
behavior separate.

Full-matrix command. Clear `BVP_AOT_MATRIX_FILTER` first because PowerShell keeps
environment variables in the current session; otherwise an earlier focused run will
continue to select only one AOT row plus the Lambdify baselines.

```powershell
Remove-Item Env:\BVP_AOT_MATRIX_FILTER -ErrorAction SilentlyContinue
cargo test --release combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix -- --ignored --nocapture --test-threads=1
```

Optional focused run:

```powershell
$env:BVP_AOT_MATRIX_FILTER="Sparse gcc/chunk4"
cargo test --release combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix -- --ignored --nocapture --test-threads=1
```

Result:
CPU 4 Core
[BVP Damp race] finished source=AOT matrix=Banded variant=tcc status=ok
[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT correctness matrix
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | Banded | AtomView   | baseline         |  3/3   | 6.883e-15 +/- 0.0e0  | 6.873e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | tcc        | whole            |  3/3   | 1.110e-15 +/- 0.0e0  | 1.109e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc        | whole            |  3/3   | 6.442e-15 +/- 0.0e0  | 6.432e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT timing/counter matrix
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | 477.545 +/- 267.930 [277.687, 856.260] | 523.000 +/- 268.188 | 423.333 +/- 265.775 | 7.000 +/- 0.000 | 1.000 +/- 0.000 | 0.333 +/- 0.471 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | AtomView   | baseline         | 246.067 +/- 4.398 [239.908, 249.899] | 286.333 +/- 5.735 | 196.000 +/- 1.414 | 3.000 +/- 0.000 | 0.000 +/- 0.000 | 0.667 +/- 0.471 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc        | whole            | 304.711 +/- 12.854 [287.611, 318.603] | 348.667 +/- 13.816 | 252.667 +/- 9.877 | 6.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc        | whole            | 301.327 +/- 39.218 [271.416, 356.731] | 343.000 +/- 41.045 | 255.333 +/- 38.690 | 2.333 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT callback stage matrix
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
Lambdify | Banded | AtomView   | baseline         | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Sparse | tcc        | whole            | 0.589 +/- 0.016    | 0.192 +/- 0.005    | 0.283 +/- 0.056      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc        | whole            | 0.529 +/- 0.011    | 0.212 +/- 0.006    | 0.032 +/- 0.022      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT lifecycle matrix
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 423.333 +/- 265.775
Lambdify | Banded | AtomView   | baseline         | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 196.000 +/- 1.414
AOT      | Sparse | tcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 252.667 +/- 9.877
AOT      | Banded | tcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 255.333 +/- 38.690
ok

12 Core
BVP Damp race] finished source=AOT matrix=Banded variant=zig status=ok
[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT correctness matrix
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | Banded | AtomView   | baseline         |  3/3   | 7.994e-15 +/- 0.0e0  | 7.981e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | gcc        | whole            |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | gcc        | chunk4           |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | tcc        | whole            |  3/3   | 4.441e-16 +/- 0.0e0  | 4.434e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | tcc        | chunk4           |  3/3   | 4.441e-16 +/- 0.0e0  | 4.434e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | zig        | whole            |  3/3   | 4.441e-16 +/- 0.0e0  | 4.434e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | zig        | chunk4           |  3/3   | 4.441e-16 +/- 0.0e0  | 4.434e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | gcc        | whole            |  3/3   | 7.994e-15 +/- 0.0e0  | 7.981e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | gcc        | chunk4           |  3/3   | 7.994e-15 +/- 0.0e0  | 7.981e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc        | whole            |  3/3   | 7.994e-15 +/- 0.0e0  | 7.981e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc        | chunk4           |  3/3   | 7.994e-15 +/- 0.0e0  | 7.981e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | zig        | whole            |  3/3   | 8.216e-15 +/- 0.0e0  | 8.203e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | zig        | chunk4           |  3/3   | 8.216e-15 +/- 0.0e0  | 8.203e-15 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT timing/counter matrix
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | 774.975 +/- 949.808 [103.132, 2118.206] | 757.667 +/- 878.462 | 706.333 +/- 914.761 | 2.000 +/- 0.000 | 1.000 +/- 0.000 | 0.667 +/- 0.471 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | AtomView   | baseline         | 100.539 +/- 4.762 [93.975, 105.126] | 132.333 +/- 5.907 | 62.000 +/- 4.243 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | gcc        | whole            | 570.721 +/- 6.286 [561.831, 575.215] | 602.667 +/- 6.944 | 534.000 +/- 6.481 | 2.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | gcc        | chunk4           | 556.808 +/- 3.131 [552.631, 560.170] | 588.333 +/- 3.859 | 518.000 +/- 2.449 | 2.333 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc        | whole            | 125.605 +/- 2.252 [122.588, 127.996] | 155.000 +/- 2.449 | 89.667 +/- 1.247 | 2.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | tcc        | chunk4           | 126.820 +/- 5.877 [122.520, 135.129] | 156.333 +/- 4.714 | 89.000 +/- 2.828 | 2.333 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | zig        | whole            | 12089.778 +/- 48.640 [12044.161, 12157.174] | 12000.000 +/- 0.000 | 12000.000 +/- 0.000 | 2.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | zig        | chunk4           | 10543.611 +/- 68.868 [10468.495, 10634.856] | 10000.000 +/- 0.000 | 10000.000 +/- 0.000 | 2.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | gcc        | whole            | 556.296 +/- 13.813 [537.611, 570.571] | 586.667 +/- 11.954 | 522.000 +/- 13.367 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | gcc        | chunk4           | 530.533 +/- 2.351 [528.303, 533.784] | 562.333 +/- 3.300 | 494.667 +/- 1.700 | 0.667 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc        | whole            | 119.682 +/- 2.605 [116.683, 123.035] | 149.000 +/- 4.082 | 86.000 +/- 2.160 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc        | chunk4           | 124.379 +/- 4.583 [119.259, 130.380] | 155.000 +/- 5.888 | 87.667 +/- 0.471 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | zig        | whole            | 11982.277 +/- 68.672 [11898.393, 12066.604] | 11666.667 +/- 471.405 | 11333.333 +/- 471.405 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | zig        | chunk4           | 10491.143 +/- 47.210 [10435.721, 10551.095] | 10000.000 +/- 0.000 | 10000.000 +/- 0.000 | 0.667 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT callback stage matrix
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
Lambdify | Banded | AtomView   | baseline         | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Sparse | gcc        | whole            | 0.267 +/- 0.008    | 0.115 +/- 0.000    | 0.107 +/- 0.009      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Sparse | gcc        | chunk4           | 0.592 +/- 0.014    | 0.156 +/- 0.049    | 0.183 +/- 0.015      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        
AOT      | Sparse | tcc        | whole            | 0.297 +/- 0.038    | 0.110 +/- 0.003    | 0.128 +/- 0.042      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Sparse | tcc        | chunk4           | 0.627 +/- 0.028    | 0.136 +/- 0.003    | 0.203 +/- 0.051      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        
AOT      | Sparse | zig        | whole            | 0.117 +/- 0.004    | 0.033 +/- 0.001    | 0.111 +/- 0.016      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Sparse | zig        | chunk4           | 0.499 +/- 0.033    | 0.063 +/- 0.005    | 0.154 +/- 0.037      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        
AOT      | Banded | gcc        | whole            | 0.259 +/- 0.045    | 0.148 +/- 0.059    | 0.014 +/- 0.007      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | gcc        | chunk4           | 0.612 +/- 0.027    | 0.148 +/- 0.006    | 0.028 +/- 0.018      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        
AOT      | Banded | tcc        | whole            | 0.353 +/- 0.150    | 0.098 +/- 0.001    | 0.013 +/- 0.008      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc        | chunk4           | 0.611 +/- 0.053    | 0.126 +/- 0.004    | 0.025 +/- 0.016      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        
AOT      | Banded | zig        | whole            | 0.107 +/- 0.005    | 0.034 +/- 0.000    | 0.012 +/- 0.007      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1200.000 +/- 0.000 | 4188.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | zig        | chunk4           | 0.503 +/- 0.028    | 0.054 +/- 0.004    | 0.015 +/- 0.004      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 300.000 +/- 0.000 | 1047.000 +/- 0.000 | none         | none        

[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT lifecycle matrix
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | AtomView   | baseline         | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 706.333 +/- 914.761
Lambdify | Banded | AtomView   | baseline         | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 62.000 +/- 4.243
AOT      | Sparse | gcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 534.000 +/- 6.481
AOT      | Sparse | gcc        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 518.000 +/- 2.449
AOT      | Sparse | tcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 89.667 +/- 1.247
AOT      | Sparse | tcc        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 89.000 +/- 2.828
AOT      | Sparse | zig        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 12000.000 +/- 0.000
AOT      | Sparse | zig        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 10000.000 +/- 0.000
AOT      | Banded | gcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 522.000 +/- 13.367
AOT      | Banded | gcc        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 494.667 +/- 1.700
AOT      | Banded | tcc        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 86.000 +/- 2.160
AOT      | Banded | tcc        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 87.667 +/- 0.471
AOT      | Banded | zig        | whole            | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 11333.333 +/- 471.405
AOT      | Banded | zig        | chunk4           | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 201.000 +/- 0.000  | 10000.000 +/- 0.000
ok
```text
Date: 2026-05-26
Status: ok, focused `tcc/whole` rerun only; the text below replaces an older full-matrix interpretation.
Important numbers:
  All four displayed rows passed 3/3 and agree at roundoff scale
  (`solve_diff <= 6.883e-15`, `max_abs_sol = 1.002`).
  Banded Lambdify is cheapest for this small cold solve: `246.067 +/- 4.398 ms`.
  Banded `tcc/whole` AOT costs `301.327 +/- 39.218 ms`; its cold bootstrap is
  still larger than the callback work saved at `n_steps=200`.
  Banded again reduces linear time (`2.333 ms` AOT, `3.000 ms` Lambdify)
  compared with Sparse (`6.000 ms` AOT, `7.000 ms` Lambdify).

Conclusion:
  The displayed result is a useful small-problem baseline: correctness is good,
  Banded helps the numerical solve, but cold AOT does not yet pay for itself.
  This rerun does not exercise `chunk4`, gcc, or zig, because only `tcc/whole`
  appears in the recorded rows. Do not use it as renewed proof of chunking
  correctness or cross-language performance.

Follow-up:
  Clear `BVP_AOT_MATRIX_FILTER` and rerun this inexpensive matrix if a fresh
  whole-versus-chunk confirmation is required.
```

### `combustion_1000_end_to_end_banded_lapack_refine_statistics`

File: `src/numerical/BVP_Damp/BVP_Damp_tests3.rs`

Hypothesis: the Banded route using the LAPACK-style banded LU backend remains stable
under the heavy combustion-1000 end-to-end solve, including refinement and statistics
collection. This is the main "does the production banded path hold together?" story.

Command:

```powershell
cargo test --release combustion_1000_end_to_end_banded_lapack_refine_statistics -- --ignored --nocapture --test-threads=1
```

Result:
[BVP Damp end-to-end] combustion-1000 full solve with banded backends using lapack_style_banded_lu
source     | variant    | bootstrap_ms |   solve_ms |   total_ms |  max_abs_sol |   solve_diff |   rel_x_diff |   iters |  linsys |  jac_re | linear_timer       | jac_timer          | fun_timer          | status  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify   | ExprLegacy |     3014.410 |   2658.656 |   5673.293 |   1.001534e0 |      0.000e0 |      0.000e0 |       5 |      10 |       1 | 0.702, 18          | 0.024, 0           | 0.114, 3           | ok      
Compiled   | C-gcc      |     7309.295 |   8379.927 |  15689.300 |   1.001534e0 |     3.683e-8 |     3.677e-8 |       5 |      10 |       1 | 0.186, 15          | 0.016, 1           | 0.02, 1            | ok      
Compiled   | C-tcc      |     2184.684 |   2308.611 |   4493.345 |   1.001534e0 |     3.683e-8 |     3.677e-8 |       5 |      10 |       1 | 0.667, 15          | 0.061, 1           | 0.084, 1           | ok      
Compiled   | Zig        |    44614.908 |  46109.995 |  90724.982 |   1.001534e0 |     3.683e-8 |     3.677e-8 |       5 |      10 |       1 | 0.035, 16          | 0.001, 0           | 0.002, 0           | ok      
ok

```text
Date:
Status:
Important numbers:
Conclusion:
Follow-up:
```

### `oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy`

File: `src/numerical/BVP_Damp/BVP_Damp_tests3.rs`

Hypothesis: a smaller but complete oscillator BVP should agree between Banded
Lambdify and Banded AtomView+AOT. This is a useful system-level AOT check that is
less domain-specific than combustion.

Command:

```powershell
cargo test --release oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Status:
Important numbers:
Conclusion:
Follow-up:
```

## Symbolic Assembly Parity Stories

These tests are not primarily speed benchmarks. They answer whether `ExprLegacy` and
`AtomView` are still mathematically equivalent after changes to discretization,
boundary condition handling, bandwidth metadata, or code generation.

### `symbolic_assembly_backends_report_representative_fixture_table`

Compares symbolic assembly stage timings on representative small/medium fixtures.
Use it when changing symbolic discretization or AtomView lowering.

```powershell
cargo test --release symbolic_assembly_backends_report_representative_fixture_table -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_representative_solver_table`

Runs solver-level comparisons for representative exact-like BVP fixtures. Use it to
check that equivalent assembly leads to equivalent Newton solves, not merely matching
matrix entries.

```powershell
cargo test --release symbolic_assembly_backends_report_representative_solver_table -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_build_sparse_bundles_for_representative_examples`

Builds sparse bundles from representative examples and compares residual/Jacobian
callbacks numerically. This is a focused "bundle correctness" gate before full solves.

```powershell
cargo test --release symbolic_assembly_backends_build_sparse_bundles_for_representative_examples -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_sparse_bundle_timings`

Checks ExprLegacy vs AtomView on the real combustion sparse bundle. Use this after
changes to sparse symbolic assembly or combustion fixture construction.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_sparse_bundle_timings -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_discretized_row_diagnostics`

Row-level diagnostic for combustion residual assembly. This is for localizing a
disagreement, not for routine performance comparison.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_discretized_row_diagnostics -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_solver_stress_table`

Solver-level stress comparison on combustion. Use it when a bundle-level comparison
passes but end-to-end solve behavior still looks different.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_solver_stress_table -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_solver_stress_stats_1000`

Repeated combustion stress comparison with aggregate statistics. This is the better
choice when a single run is too noisy.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_solver_stress_stats_1000 -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Runs:
Conclusion:
```

## AOT Build, Artifact, and Runtime Stories

These tests distinguish "symbolic math is wrong" from "the generated artifact was not
built, loaded, or reused correctly". This separation matters because AOT failures can
look like backend regressions while actually being file-lock, compiler, or resolver
problems.

### `symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000`

Compares Rust AOT compile presets on a compact combustion fixture and separates build
cost from runtime cost. The function name is historical; the current guard uses a
moderate `n_steps=200` grid so it stays focused on rustc preset behavior rather than
becoming a large generated-crate stress test. This is intentionally Rust-only because
these presets describe rustc profile/compile behavior; C and Zig emitter comparisons
are covered by the toolchain-oriented end-to-end and callback-throughput matrices
below.

```powershell
cargo test --release symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000 -- --ignored --nocapture --test-threads=1
```

Result:
note: this is intentionally Rust-AOT only; C/Zig toolchain comparisons live in the end-to-end and callback-throughput matrices.
backend      | preset      | bootstrap_ms |   solve_ms | max_abs_solution
--------------------------------------------------------------------------
ExprLegacy   | Production  |    10955.500 |    403.895 |       1.001542e0
ExprLegacy   | FastBuild   |    12171.674 |    364.030 |       1.001542e0
ExprLegacy   | DevFastest  |     3714.862 |    461.780 |       1.001542e0
AtomView     | Production  |    12238.889 |    346.974 |       1.001542e0
AtomView     | FastBuild   |    10037.169 |    349.112 |       1.001542e0
AtomView     | DevFastest  |     3320.997 |    431.823 |       1.001542e0
ok
```text
Date:
Toolchains:
Conclusion:
```

### `combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000`

End-to-end comparison between Lambdify and AtomView DevFastest AOT. The function name
is historical; the current matrix uses `n_steps=200` because it is meant to compare
toolchain behavior without turning Rust AOT into a compiler stack/deep-crate stress
test. Use it to ensure the practical compiled route does not change the solution.
The AOT side is now a toolchain matrix over Rust, `gcc`, `tcc`, and Zig, so this test
answers "which generated language is practical for this real solver path?" rather
than silently treating Rust AOT as the only AOT implementation.

```powershell
cargo test --release combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000 -- --ignored --nocapture --test-threads=1
```
CPU 4 Core
Result:
[AOT measure] combustion-atomview-zig-devfastest-vs-lambdify-200: bootstrap=27782.330 ms, solve=221.131 ms
[BVP end-to-end compare] combustion Lambdify vs AtomView DevFastest toolchain matrix, n_steps=200
backend          |     setup_ms |   solve_ms |   total_ms |   diff_vs_base | max_abs_solution
-----------------------------------------------------------------------------------------------
Lambdify         |     1153.448 |    330.791 |   1484.239 |     0.000000e0 |       1.001542e0
AtomView+rust    |     4566.821 |    244.948 |   4811.769 |    3.550755e-8 |       1.001542e0
AtomView+gcc     |     2354.096 |    247.888 |   2601.983 |    3.550755e-8 |       1.001542e0
AtomView+tcc     |      732.615 |    224.767 |    957.382 |    3.550755e-8 |       1.001542e0
AtomView+zig     |    27782.330 |    221.131 |  28003.461 |    3.550755e-8 |       1.001542e0
ok
CPU 12 Core
[AOT measure] combustion-atomview-zig-devfastest-vs-lambdify-200: bootstrap=9119.630 ms, solve=99.342 ms
[BVP end-to-end compare] combustion Lambdify vs AtomView DevFastest toolchain matrix, n_steps=200
backend          |     setup_ms |   solve_ms |   total_ms |   diff_vs_base | max_abs_solution
-----------------------------------------------------------------------------------------------
Lambdify         |     1979.350 |    105.916 |   2085.266 |     0.000000e0 |       1.001542e0
AtomView+rust    |     1125.377 |    153.000 |   1278.377 |   9.517298e-11 |       1.001542e0
AtomView+gcc     |      858.897 |    124.244 |    983.141 |   9.517298e-11 |       1.001542e0
AtomView+tcc     |      207.873 |    115.084 |    322.956 |   9.517298e-11 |       1.001542e0
AtomView+zig     |     9119.630 |     99.342 |   9218.972 |   9.517298e-11 |       1.001542e0
test numerical::BVP_Damp::BVP_Damp_tests3::tests::combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000 ... ok
ok
```text
Date:
Conclusion:
```

### `combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000`

Measures runtime callback throughput after the generated backend is linked into the
current process. This is not a compile benchmark; it asks how expensive the callback
path is once the artifact already exists. The table includes Lambdify as the baseline
and AtomView AOT rows for Rust, `gcc`, `tcc`, and Zig, because callback throughput is
one of the places where backend language can matter most. The function name is
historical; the current diagnostic uses `n_steps=200` to avoid mixing callback
throughput with large generated-crate compiler stress.

```powershell
cargo test --release combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000 -- --ignored --nocapture --test-threads=1
```

Result:
[BVP callback throughput] combustion Lambdify vs AtomView linked-runtime toolchain matrix, n_steps=200, iters=20
[BVP callback throughput] note: AtomView+Linked now measures the generated cdylib runtime path loaded into the current process
backend          |  residual_ms |  jacobian_ms |     total_ms | speedup_vs_lambdify |      res_diff |      jac_diff
----------------------------------------------------------------------------------------------------------------
Lambdify         |        0.832 |       17.824 |       18.656 |              1.000x |    0.000000e0 |    0.000000e0
AtomView+rust    |        0.549 |        5.254 |        5.804 |              3.215x |   1.100000e-6 |   1.111111e-6
AtomView+gcc     |        0.863 |        7.877 |        8.740 |              2.135x |   1.100000e-6 |   1.111111e-6
AtomView+tcc     |        0.871 |        6.943 |        7.814 |              2.388x |   1.100000e-6 |   1.111111e-6
AtomView+zig     |        0.516 |        4.548 |        5.064 |              3.684x |   1.100000e-6 |   1.111111e-6
ok

```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_aot_crate_build_table`

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

### `combustion_1000_compiled_banded_zig_bootstrap_smoke`

Focused Zig banded AOT bootstrap diagnostic for combustion-1000. This is a toolchain
health smoke test, not a broad solver benchmark.

```powershell
cargo test --release combustion_1000_compiled_banded_zig_bootstrap_smoke -- --ignored --nocapture --test-threads=1
```

Result:
[BVP symbolic assembly AOT crate build] combustion ExprLegacy vs AtomView
backend      | n_steps | jac_prep_ms |   lookup_ms |      jac_ms |      nnz | finalize_ms |  module_ms |      source_ms | materialize_ms |   build_ms | source_kb |   blocks |    instr |    temps |  max_blk |  outputs | status            
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ExprLegacy   |     200 |       0.000 |       0.000 |       0.000 |        0 |       0.000 |     41.374 |         12.281 |      3.685 |   7333.758 |  549.4 |       32 |    13751 |    13751 |      598 |     5388 | ok                
AtomView     |     200 |       6.706 |       1.375 |       4.598 |     4188 |       0.652 |      3.756 |         12.567 |      2.982 |   7129.178 |  560.1 |       32 |    14020 |    14020 |      612 |     5388 | ok                
ExprLegacy   |     300 |       0.000 |       0.000 |       0.000 |        0 |       0.000 |     61.154 |         18.207 |      3.885 |  12858.105 |  815.3 |       32 |    20393 |    20393 |      892 |     8088 | ok                
AtomView     |     300 |      10.737 |       2.060 |       7.618 |     6288 |       0.932 |      4.023 |         18.695 |      3.813 |  13071.811 |  830.0 |       32 |    20765 |    20765 |      912 |     8088 | ok                

[BVP symbolic assembly AOT crate build] atom module pass breakdown
backend      | n_steps |  res_view_ms | res_lower_ms |    res_ph_ms | res_reuse_ms |   sp_view_ms |  sp_lower_ms |     sp_ph_ms |  sp_reuse_ms
-----------------------------------------------------------------------------------------------------------------------------------------------
AtomView     |     200 |        0.008 |        6.852 |        0.572 |        0.058 |        0.047 |        3.188 |        0.112 |        0.043
AtomView     |     300 |        0.024 |        7.095 |        0.194 |        0.083 |        0.042 |        3.825 |        0.090 |        0.046
ok
```text
Date:
Status:
Conclusion:
```

## Linear Algebra Stories

These tests isolate the matrix backend from the rest of the BVP pipeline as much as
possible.

### `combustion_1000_linear_system_story_sparse_vs_banded_consistent`

Checks a combustion-1000 linear-system story for Sparse baseline vs consistent
superblock/banded solve. Use this when touching banded storage, LAPACK-style banded
LU, or sparse-to-banded conversion logic.

```powershell
cargo test --release combustion_1000_linear_system_story_sparse_vs_banded_consistent -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

### `diagnose_combustion_1000_lapack_style_banded_path`

Focused diagnostic for LAPACK-style banded storage/factor/solve on combustion-1000.
Use this when the end-to-end banded story fails and the suspected source is linear
algebra rather than symbolic assembly.

```powershell
cargo test --release diagnose_combustion_1000_lapack_style_banded_path -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```

## Generate-Breakdown and IR Diagnostics

These are diagnostic tools. They should not be the first tests to run in routine
release comparisons, but they are valuable when a larger story points at a specific
stage.

### `symbolic_assembly_backends_report_combustion_generate_breakdown_table`

Breaks `generate_ms` into symbolic/backend stages for combustion. Use it to locate
which part of generation regressed.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_generate_breakdown_table -- --ignored --nocapture --test-threads=1
```

Result:
[BVP symbolic assembly generate breakdown] combustion ExprLegacy vs AtomView
backend      | n_steps | discretization_ms | symbolic_jacobian_ms |  sparse_aot_prep_ms |   total_ms
--------------------------------------------------------------------------------------------------------
ExprLegacy   |     200 |            59.580 |               69.732 |               3.660 |    210.464
AtomView     |     200 |            16.122 |               66.169 |               6.105 |    174.209
ExprLegacy   |     300 |            67.751 |              133.474 |               5.397 |    366.429
AtomView     |     300 |            22.043 |              130.285 |               9.226 |    333.475
ok
```text
Date:
Conclusion:
```

### `symbolic_assembly_backends_report_combustion_chunk_ir_table`

Compares chunk-level IR lowering for ExprLegacy vs AtomView combustion. Use it when
callbacks disagree and the issue appears before compilation/runtime linking.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_chunk_ir_table -- --ignored --nocapture --test-threads=1
```

Result:
[BVP symbolic assembly chunk IR compare] combustion, n_steps=300, legacy_instr_total=20393, atom_instr_total=20765, legacy_temps_total=20393, atom_temps_total=20765
fn_name                              | outputs | legacy_instr | atom_instr | legacy_temps | atom_temps
-------------------------------------------------------------------------------------------------------
eval_bvp_residual_chunk_0            |     113 |          868 |        908 |          868 |        908
eval_bvp_residual_chunk_1            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_2            |     113 |          888 |        908 |          888 |        908
eval_bvp_residual_chunk_3            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_4            |     113 |          883 |        902 |          883 |        902
eval_bvp_residual_chunk_5            |     113 |          881 |        906 |          881 |        906
eval_bvp_residual_chunk_6            |     113 |          882 |        901 |          882 |        901
eval_bvp_residual_chunk_7            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_8            |     113 |          888 |        908 |          888 |        908
eval_bvp_residual_chunk_9            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_10           |     113 |          883 |        902 |          883 |        902
eval_bvp_residual_chunk_11           |     113 |          887 |        906 |          887 |        906
ok
```text
Date:
Conclusion:
```

## Non-Ignored Acceptance and Tuning Checks

These are not the heavy release matrix, but they are useful quick gates before paying
for combustion-1000 release runs.

### `aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps`

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

Default Rust AOT acceptance coverage for combustion across sequential/parallel
execution and varied grids. This is a smaller preflight before the full 1000-step
release stories.

```powershell
cargo test --release aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids -- --nocapture --test-threads=1
```

### `aot_tcc_smoke_exact_two_point_small_grid_solves`

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

### `aot_combustion_parallel_tuning_reports_runtime_table`

Reports runtime behavior for combustion parallel tuning. This is the source-of-truth
test for the meaning of `target_chunks`, `max_*_jobs`, and AOT toolchain choice on
the sparse AOT callback path. It keeps the mathematical problem fixed, uses the
Lambdify route as the correctness reference, and then runs the same sequential and
explicit parallel/chunking policies for each supported AOT emitter: Rust, `gcc`,
`tcc`, and Zig. Row labels have the form `toolchain/chunking`, for example
`tcc/seq`, `gcc/par-4x4-jobs4`, or `zig/par-res16-row32-jobs8`.

The table reports both user-facing and diagnostic timings. In the current
implementation, `honest_user_e2e_ms` is measured by running one variant in a
fresh child process around `solver.try_solve()`: for AOT rows it includes
symbolic work, code generation, compilation, linking, and the Newton solve,
without carrying a linked artifact or runtime registry state from the
preceding row. This is the closest number to "I pressed Enter and waited for
the answer." `honest_speedup` is computed against the sequential row of the
same toolchain using that full wall-clock number.

The diagnostic columns remain useful but answer narrower questions. `solve_ms` is
the Newton solve after callbacks are already prepared, `manual_bootstrap_ms` is the
test runner's explicit callback preparation/linking stage, and `speedup_vs_seq`
uses `solve_ms`, not full end-to-end time. The table also reports `max_diff_vs_ref`,
`linear_ms`, `jac_ms`, `fun_ms`, and Newton/linear-solve counters. Same-toolchain
baselines matter because compiler/runtime overhead differs substantially between
Rust, C, and Zig artifacts.

The current version uses `n_steps=1000` and four repetitions per row. This is still
heavy, but it gives enough averaging to make the toolchain/chunking matrix useful
without turning the release run into an all-night ritual.
The prepared-runtime branch remains in the parent process so it can answer
questions about hot callback evaluation. Read the output as two joined but
separate experiments: first look at process-isolated `honest_user_e2e_ms` to
decide whether a toolchain/chunking strategy helps a real cold calculation,
then look at `solve_ms` and callback stage columns to understand whether the
benefit or loss comes from runtime evaluation rather than compile/link overhead.

Interpret this test only after correctness has already been established by the
callback-equivalence gate and the end-to-end matrix. A fast tuning row with nonzero
solution drift would be a bug, not an optimization.

```powershell
cargo test --release aot_combustion_parallel_tuning_reports_runtime_table -- --ignored --nocapture --test-threads=1
```

New runs contain `isolated cold stage breakdown` and `isolated cold
numerical/runtime stages` tables. Unlike the prepared-runtime tables below
them, every column in these cold tables comes from the same child solve as
`honest_user_e2e_ms`; use those rows to diagnose an anomalous cold run.

Controlled clean/cooldown run:

```powershell
$env:BVP_AOT_COLD_COOLDOWN_MS="5000"
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
cargo test --release aot_combustion_parallel_tuning_reports_runtime_table -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_AOT_COLD_COOLDOWN_MS -ErrorAction SilentlyContinue
Remove-Item Env:\BVP_AOT_COLD_CLEAN_ARTIFACTS -ErrorAction SilentlyContinue
```

Result:
4 Core
Note: the result block immediately below uses the new isolated cold
wall-clock route. Its `honest_user_e2e_ms` values are measured in fresh child
processes. The nearby `solve_ms`, `manual_bootstrap_ms`, `solver_total_ms`,
`symbolic_ms`, and callback-stage values still come from the separate
prepared-runtime branch in the parent process; they diagnose hot callbacks,
not the internals of the isolated cold row.

[AOT combustion tuning map] scenario=medium-grid-multi-toolchain, n_steps=1000, repetitions=4
[AOT combustion tuning map] runner=manual_prelinked_runtime_tuning; solve_ms is the runtime Newton solve after callbacks are prepared.
[AOT combustion tuning map] manual_bootstrap_ms is diagnostic setup for this test only; do not read it as normal solver end-to-end time.
[AOT combustion tuning map] honest_user_e2e_ms is measured in a fresh child process around solver.try_solve(); AOT rows force RebuildAlways Release so codegen/build/link/Newton are included without registry/DLL carry-over.
[AOT combustion tuning map] correctness summary
config                         | n_steps | runs | honest_user_e2e_ms | solve_ms           | speedup_vs_seq     | max_diff_vs_ref   
-----------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              |    1000 |    4 | 4279.960 +/- 502.303 | 5888.742 +/- 1149.197 | -                  | 0.000e0 +/- 0.0e0 
rust/seq                       |    1000 |    4 | 9839.490 +/- 254.343 | 2780.567 +/- 512.826 | 1.000 +/- 0.000    | 1.332e-15 +/- 0.0e0
rust/par-4x4-jobs4             |    1000 |    4 | 11059.898 +/- 147.695 | 2737.770 +/- 394.019 | 1.011 +/- 0.082    | 1.332e-15 +/- 0.0e0
rust/par-8x8-jobs8             |    1000 |    4 | 10784.267 +/- 217.173 | 2787.694 +/- 363.005 | 0.991 +/- 0.079    | 1.332e-15 +/- 0.0e0
rust/par-16x16-jobs16          |    1000 |    4 | 9765.439 +/- 66.197 | 2867.258 +/- 453.752 | 0.966 +/- 0.052    | 1.332e-15 +/- 0.0e0
rust/par-res16-row32-jobs8     |    1000 |    4 | 10522.154 +/- 205.259 | 2930.631 +/- 407.563 | 0.943 +/- 0.063    | 1.332e-15 +/- 0.0e0
gcc/seq                        |    1000 |    4 | 7165.707 +/- 34.647 | 2874.009 +/- 389.318 | 1.000 +/- 0.000    | 1.332e-15 +/- 0.0e0
gcc/par-4x4-jobs4              |    1000 |    4 | 7406.996 +/- 32.830 | 2974.814 +/- 397.017 | 0.969 +/- 0.074    | 1.332e-15 +/- 0.0e0
gcc/par-8x8-jobs8              |    1000 |    4 | 2528500.765 +/- 4366873.977 | 2887.109 +/- 368.771 | 0.995 +/- 0.017    | 1.332e-15 +/- 0.0e0
gcc/par-16x16-jobs16           |    1000 |    4 | 7456.883 +/- 397.108 | 3308.578 +/- 910.536 | 0.900 +/- 0.125    | 1.332e-15 +/- 0.0e0
gcc/par-res16-row32-jobs8      |    1000 |    4 | 7776.052 +/- 132.766 | 3042.555 +/- 332.564 | 0.942 +/- 0.035    | 1.332e-15 +/- 0.0e0
tcc/seq                        |    1000 |    4 | 3018.766 +/- 524.588 | 3829.074 +/- 1695.094 | 1.000 +/- 0.000    | 1.332e-15 +/- 0.0e0
tcc/par-4x4-jobs4              |    1000 |    4 | 3047.160 +/- 626.735 | 3103.900 +/- 599.433 | 1.183 +/- 0.274    | 1.332e-15 +/- 0.0e0
tcc/par-8x8-jobs8              |    1000 |    4 | 2792.135 +/- 196.184 | 3097.631 +/- 510.044 | 1.187 +/- 0.307    | 1.332e-15 +/- 0.0e0
tcc/par-16x16-jobs16           |    1000 |    4 | 2844.886 +/- 339.077 | 3192.128 +/- 494.101 | 1.156 +/- 0.324    | 1.332e-15 +/- 0.0e0
tcc/par-res16-row32-jobs8      |    1000 |    4 | 2986.163 +/- 95.073 | 3131.296 +/- 437.765 | 1.179 +/- 0.344    | 1.332e-15 +/- 0.0e0
zig/seq                        |    1000 |    4 | 39523.312 +/- 272.746 | 3012.142 +/- 344.320 | 1.000 +/- 0.000    | 1.110e-15 +/- 0.0e0
zig/par-4x4-jobs4              |    1000 |    4 | 63809.719 +/- 1440.263 | 3103.238 +/- 312.503 | 0.970 +/- 0.039    | 1.110e-15 +/- 0.0e0
zig/par-8x8-jobs8              |    1000 |    4 | 47705.400 +/- 641.149 | 2998.033 +/- 310.797 | 1.004 +/- 0.013    | 1.110e-15 +/- 0.0e0
zig/par-16x16-jobs16           |    1000 |    4 | 42046.726 +/- 3909.314 | 3069.044 +/- 217.246 | 0.979 +/- 0.050    | 1.110e-15 +/- 0.0e0
zig/par-res16-row32-jobs8      |    1000 |    4 | 39387.851 +/- 2538.496 | 3278.190 +/- 363.739 | 0.920 +/- 0.063    | 1.110e-15 +/- 0.0e0

[AOT combustion tuning map] honest wall-clock summary; all time columns are milliseconds
note: this is the closest table to stopwatch timing from button press to finished result.
config                         | honest_user_e2e_ms | honest_speedup     | runtime_solve_ms   | manual_bootstrap_ms mean+/-std [min,max]
------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 4279.960 +/- 502.303 | -                  | 5888.742 +/- 1149.197 | 5764.379 +/- 1178.866 [4231.719, 7010.757]
rust/seq                       | 9839.490 +/- 254.343 | 1.000 +/- 0.000    | 2780.567 +/- 512.826 | 15319.471 +/- 1107.395 [13571.252, 16409.112]
rust/par-4x4-jobs4             | 11059.898 +/- 147.695 | 0.890 +/- 0.011    | 2737.770 +/- 394.019 | 16521.235 +/- 1187.201 [14843.401, 17743.683]
rust/par-8x8-jobs8             | 10784.267 +/- 217.173 | 0.912 +/- 0.016    | 2787.694 +/- 363.005 | 16231.877 +/- 1081.096 [14539.662, 17300.082]
rust/par-16x16-jobs16          | 9765.439 +/- 66.197 | 1.008 +/- 0.027    | 2867.258 +/- 453.752 | 15461.456 +/- 1171.619 [13653.238, 16724.856]
rust/par-res16-row32-jobs8     | 10522.154 +/- 205.259 | 0.936 +/- 0.043    | 2930.631 +/- 407.563 | 16235.451 +/- 1043.294 [15047.506, 17372.607]
gcc/seq                        | 7165.707 +/- 34.647 | 1.000 +/- 0.000    | 2874.009 +/- 389.318 | 12973.820 +/- 919.254 [11489.877, 13854.639]
gcc/par-4x4-jobs4              | 7406.996 +/- 32.830 | 0.967 +/- 0.006    | 2974.814 +/- 397.017 | 12857.572 +/- 833.210 [11509.808, 13590.602]
gcc/par-8x8-jobs8              | 2528500.765 +/- 4366873.977 | 0.737 +/- 0.425    | 2887.109 +/- 368.771 | 12900.272 +/- 1000.442 [11381.785, 13884.834]
gcc/par-16x16-jobs16           | 7456.883 +/- 397.108 | 0.963 +/- 0.048    | 3308.578 +/- 910.536 | 13759.631 +/- 2092.800 [11466.736, 17151.671]
gcc/par-res16-row32-jobs8      | 7776.052 +/- 132.766 | 0.922 +/- 0.015    | 3042.555 +/- 332.564 | 13682.581 +/- 1167.274 [12011.995, 15298.967]
tcc/seq                        | 3018.766 +/- 524.588 | 1.000 +/- 0.000    | 3829.074 +/- 1695.094 | 8805.500 +/- 1037.699 [7239.428, 10058.263]
tcc/par-4x4-jobs4              | 3047.160 +/- 626.735 | 0.996 +/- 0.027    | 3103.900 +/- 599.433 | 8782.910 +/- 1352.249 [7095.684, 10874.171]
tcc/par-8x8-jobs8              | 2792.135 +/- 196.184 | 1.074 +/- 0.105    | 3097.631 +/- 510.044 | 9329.589 +/- 1828.618 [7284.489, 12276.863]
tcc/par-16x16-jobs16           | 2844.886 +/- 339.077 | 1.055 +/- 0.053    | 3192.128 +/- 494.101 | 9358.565 +/- 1931.975 [7157.543, 12427.560]
tcc/par-res16-row32-jobs8      | 2986.163 +/- 95.073 | 1.007 +/- 0.140    | 3131.296 +/- 437.765 | 9489.414 +/- 1529.700 [7726.985, 11936.813]
zig/seq                        | 39523.312 +/- 272.746 | 1.000 +/- 0.000    | 3012.142 +/- 344.320 | 46207.615 +/- 760.615 [45343.338, 47418.462]
zig/par-4x4-jobs4              | 63809.719 +/- 1440.263 | 0.620 +/- 0.010    | 3103.238 +/- 312.503 | 69045.543 +/- 1346.544 [67479.619, 71153.718]
zig/par-8x8-jobs8              | 47705.400 +/- 641.149 | 0.829 +/- 0.006    | 2998.033 +/- 310.797 | 54521.796 +/- 1945.923 [52809.609, 57696.149]
zig/par-16x16-jobs16           | 42046.726 +/- 3909.314 | 0.947 +/- 0.074    | 3069.044 +/- 217.246 | 46337.629 +/- 1133.498 [45114.702, 48193.295]
zig/par-res16-row32-jobs8      | 39387.851 +/- 2538.496 | 1.007 +/- 0.054    | 3278.190 +/- 363.739 | 45991.535 +/- 4019.271 [43395.620, 52946.763]

[AOT combustion tuning map] runtime timing/counter summary; all time columns are milliseconds
note: manual_bootstrap_ms is callback preparation/linking performed by this diagnostic runner. Use solve_ms, callback stages, and counters for chunking decisions.
config                         | solve_ms           | manual_bootstrap_ms mean+/-std [min,max] | solver_total_ms    | symbolic_ms        | linear_ms          | jac_ms             | fun_ms             | iters          | linsys         | jac_re        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 5888.742 +/- 1149.197 | 5764.379 +/- 1178.866 [4231.719, 7010.757] | 5250.000 +/- 829.156 | 5000.000 +/- 1224.745 | 39.500 +/- 2.958   | 5.750 +/- 0.433    | 5.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/seq                       | 2780.567 +/- 512.826 | 15319.471 +/- 1107.395 [13571.252, 16409.112] | 12250.000 +/- 433.013 | 2000.000 +/- 707.107 | 38.250 +/- 3.112   | 3.000 +/- 0.000    | 1.750 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-4x4-jobs4             | 2737.770 +/- 394.019 | 16521.235 +/- 1187.201 [14843.401, 17743.683] | 13500.000 +/- 500.000 | 2250.000 +/- 433.013 | 37.250 +/- 1.299   | 2.000 +/- 0.000    | 1.750 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-8x8-jobs8             | 2787.694 +/- 363.005 | 16231.877 +/- 1081.096 [14539.662, 17300.082] | 13250.000 +/- 433.013 | 2250.000 +/- 433.013 | 39.250 +/- 1.479   | 2.250 +/- 0.433    | 1.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-16x16-jobs16          | 2867.258 +/- 453.752 | 15461.456 +/- 1171.619 [13653.238, 16724.856] | 12250.000 +/- 433.013 | 2500.000 +/- 500.000 | 38.000 +/- 1.581   | 2.500 +/- 0.500    | 1.750 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-res16-row32-jobs8     | 2930.631 +/- 407.563 | 16235.451 +/- 1043.294 [15047.506, 17372.607] | 13000.000 +/- 707.107 | 2500.000 +/- 500.000 | 39.000 +/- 2.739   | 2.500 +/- 0.500    | 1.750 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/seq                        | 2874.009 +/- 389.318 | 12973.820 +/- 919.254 [11489.877, 13854.639] | 9750.000 +/- 433.013 | 2500.000 +/- 500.000 | 38.250 +/- 2.046   | 2.250 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-4x4-jobs4              | 2974.814 +/- 397.017 | 12857.572 +/- 833.210 [11509.808, 13590.602] | 9750.000 +/- 433.013 | 2750.000 +/- 433.013 | 39.000 +/- 1.225   | 2.000 +/- 0.000    | 1.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-8x8-jobs8              | 2887.109 +/- 368.771 | 12900.272 +/- 1000.442 [11381.785, 13884.834] | 2531000.000 +/- 4367077.455 | 2250.000 +/- 433.013 | 37.500 +/- 2.062   | 2.000 +/- 0.000    | 1.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-16x16-jobs16           | 3308.578 +/- 910.536 | 13759.631 +/- 2092.800 [11466.736, 17151.671] | 10500.000 +/- 1500.000 | 2750.000 +/- 829.156 | 42.000 +/- 6.519   | 2.000 +/- 0.000    | 1.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-res16-row32-jobs8      | 3042.555 +/- 332.564 | 13682.581 +/- 1167.274 [12011.995, 15298.967] | 10500.000 +/- 500.000 | 2500.000 +/- 500.000 | 40.000 +/- 2.550   | 2.000 +/- 0.000    | 1.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/seq                        | 3829.074 +/- 1695.094 | 8805.500 +/- 1037.699 [7239.428, 10058.263] | 6250.000 +/- 2165.064 | 3250.000 +/- 1639.360 | 43.750 +/- 10.779  | 2.250 +/- 0.433    | 2.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-4x4-jobs4              | 3103.900 +/- 599.433 | 8782.910 +/- 1352.249 [7095.684, 10874.171] | 5750.000 +/- 1299.038 | 2250.000 +/- 433.013 | 42.750 +/- 8.955   | 2.000 +/- 0.000    | 1.750 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-8x8-jobs8              | 3097.631 +/- 510.044 | 9329.589 +/- 1828.618 [7284.489, 12276.863] | 5500.000 +/- 866.025 | 2250.000 +/- 433.013 | 43.000 +/- 5.612   | 2.250 +/- 0.433    | 1.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-16x16-jobs16           | 3192.128 +/- 494.101 | 9358.565 +/- 1931.975 [7157.543, 12427.560] | 5750.000 +/- 829.156 | 2500.000 +/- 500.000 | 41.750 +/- 5.117   | 2.000 +/- 0.000    | 1.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-res16-row32-jobs8      | 3131.296 +/- 437.765 | 9489.414 +/- 1529.700 [7726.985, 11936.813] | 5750.000 +/- 829.156 | 2500.000 +/- 500.000 | 38.250 +/- 1.479   | 2.000 +/- 0.000    | 1.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/seq                        | 3012.142 +/- 344.320 | 46207.615 +/- 760.615 [45343.338, 47418.462] | 42000.000 +/- 707.107 | 2500.000 +/- 500.000 | 37.500 +/- 0.866   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-4x4-jobs4              | 3103.238 +/- 312.503 | 69045.543 +/- 1346.544 [67479.619, 71153.718] | 66250.000 +/- 1639.360 | 2250.000 +/- 433.013 | 38.750 +/- 1.639   | 1.750 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-8x8-jobs8              | 2998.033 +/- 310.797 | 54521.796 +/- 1945.923 [52809.609, 57696.149] | 50250.000 +/- 1089.725 | 2500.000 +/- 500.000 | 39.000 +/- 2.550   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-16x16-jobs16           | 3069.044 +/- 217.246 | 46337.629 +/- 1133.498 [45114.702, 48193.295] | 44750.000 +/- 4205.651 | 2500.000 +/- 500.000 | 38.000 +/- 0.707   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-res16-row32-jobs8      | 3278.190 +/- 363.739 | 45991.535 +/- 4019.271 [43395.620, 52946.763] | 42250.000 +/- 2772.634 | 2500.000 +/- 500.000 | 41.250 +/- 3.700   | 1.750 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[AOT combustion tuning map] linked callback stage summary; all time columns are milliseconds
note: these columns are populated by linked AOT callbacks; Lambdify rows may be blank.
config                         | residual_values    | jacobian_values    | jacobian_assembly   
--------------------------------------------------------------------------------------------------
lambdify-baseline              | -                  | -                  | -                   
rust/seq                       | 3.341 +/- 0.106    | 1.756 +/- 0.042    | 1.524 +/- 0.134     
rust/par-4x4-jobs4             | 2.785 +/- 0.236    | 1.218 +/- 0.113    | 1.323 +/- 0.040     
rust/par-8x8-jobs8             | 2.668 +/- 0.285    | 1.167 +/- 0.046    | 1.518 +/- 0.182     
rust/par-16x16-jobs16          | 2.564 +/- 0.196    | 1.192 +/- 0.058    | 1.543 +/- 0.257     
rust/par-res16-row32-jobs8     | 2.690 +/- 0.241    | 1.349 +/- 0.079    | 1.465 +/- 0.155     
gcc/seq                        | 2.695 +/- 0.086    | 0.989 +/- 0.071    | 1.468 +/- 0.197     
gcc/par-4x4-jobs4              | 2.493 +/- 0.262    | 0.750 +/- 0.034    | 1.706 +/- 0.054     
gcc/par-8x8-jobs8              | 2.433 +/- 0.078    | 0.746 +/- 0.025    | 1.326 +/- 0.022     
gcc/par-16x16-jobs16           | 2.323 +/- 0.216    | 0.750 +/- 0.033    | 1.658 +/- 0.219     
gcc/par-res16-row32-jobs8      | 2.324 +/- 0.155    | 0.767 +/- 0.013    | 1.555 +/- 0.136     
tcc/seq                        | 3.232 +/- 0.281    | 0.954 +/- 0.028    | 1.574 +/- 0.227     
tcc/par-4x4-jobs4              | 2.540 +/- 0.122    | 0.676 +/- 0.040    | 1.429 +/- 0.172     
tcc/par-8x8-jobs8              | 2.281 +/- 0.121    | 0.696 +/- 0.037    | 1.736 +/- 0.211     
tcc/par-16x16-jobs16           | 2.285 +/- 0.214    | 0.687 +/- 0.026    | 1.526 +/- 0.276     
tcc/par-res16-row32-jobs8      | 2.341 +/- 0.133    | 0.791 +/- 0.034    | 1.325 +/- 0.036     
zig/seq                        | 1.049 +/- 0.033    | 0.248 +/- 0.004    | 1.332 +/- 0.102     
zig/par-4x4-jobs4              | 1.492 +/- 0.058    | 0.265 +/- 0.012    | 1.599 +/- 0.185     
zig/par-8x8-jobs8              | 1.388 +/- 0.133    | 0.245 +/- 0.018    | 1.340 +/- 0.086     
zig/par-16x16-jobs16           | 1.471 +/- 0.099    | 0.229 +/- 0.004    | 1.396 +/- 0.129     
zig/par-res16-row32-jobs8      | 1.500 +/- 0.044    | 0.276 +/- 0.021    | 1.626 +/- 0.056     
[AOT runtime tuning winner] scenario=medium-grid-multi-toolchain, config=rust/par-4x4-jobs4, n_steps=1000, runs=4, solve_ms_mean=2737.770, speedup_vs_seq_mean=1.011, manual_bootstrap_ms_mean=16521.235
[AOT isolated cold wall-clock winner] scenario=medium-grid-multi-toolchain, config=tcc/par-8x8-jobs8, n_steps=1000, runs=4, honest_user_e2e_ms_mean=2792.135
ok
  CPU 12 Core

[AOT combustion tuning map] scenario=medium-grid-multi-toolchain, n_steps=1000, repetitions=4
[AOT combustion tuning map] runner=manual_prelinked_runtime_tuning; solve_ms is the runtime Newton solve after callbacks are prepared.
[AOT combustion tuning map] manual_bootstrap_ms is diagnostic setup for this test only; do not read it as normal solver end-to-end time.
[AOT combustion tuning map] honest_user_e2e_ms is measured in a fresh child process around solver.try_solve(); AOT rows force RebuildAlways Release so codegen/build/link/Newton are included without registry/DLL carry-over.
[AOT combustion tuning map] correctness summary
config                         | n_steps | runs | honest_user_e2e_ms | solve_ms           | speedup_vs_seq     | max_diff_vs_ref   
-----------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              |    1000 |    4 | 2221.342 +/- 61.433 | 254.516 +/- 8.453  | -                  | 0.000e0 +/- 0.0e0 
rust/seq                       |    1000 |    4 | 5102.487 +/- 221.786 | 273.526 +/- 8.957  | 1.000 +/- 0.000    | 0.000e0 +/- 0.0e0 
rust/par-4x4-jobs4             |    1000 |    4 | 6119.064 +/- 103.374 | 241.578 +/- 10.278 | 1.135 +/- 0.074    | 0.000e0 +/- 0.0e0 
rust/par-8x8-jobs8             |    1000 |    4 | 5716.109 +/- 63.543 | 219.232 +/- 4.542  | 1.249 +/- 0.059    | 0.000e0 +/- 0.0e0 
rust/par-16x16-jobs16          |    1000 |    4 | 5332.123 +/- 98.920 | 243.983 +/- 4.934  | 1.122 +/- 0.058    | 0.000e0 +/- 0.0e0 
rust/par-res16-row32-jobs8     |    1000 |    4 | 5436.328 +/- 137.308 | 259.640 +/- 8.332  | 1.055 +/- 0.063    | 0.000e0 +/- 0.0e0 
gcc/seq                        |    1000 |    4 | 4100.413 +/- 61.910 | 209.734 +/- 5.191  | 1.000 +/- 0.000    | 0.000e0 +/- 0.0e0 
gcc/par-4x4-jobs4              |    1000 |    4 | 4010.559 +/- 59.611 | 192.126 +/- 2.583  | 1.092 +/- 0.039    | 0.000e0 +/- 0.0e0 
gcc/par-8x8-jobs8              |    1000 |    4 | 3971.941 +/- 39.999 | 202.417 +/- 7.978  | 1.038 +/- 0.058    | 0.000e0 +/- 0.0e0 
gcc/par-16x16-jobs16           |    1000 |    4 | 3908.012 +/- 78.537 | 194.636 +/- 4.392  | 1.078 +/- 0.028    | 0.000e0 +/- 0.0e0 
gcc/par-res16-row32-jobs8      |    1000 |    4 | 4222.290 +/- 45.525 | 217.868 +/- 6.264  | 0.964 +/- 0.044    | 0.000e0 +/- 0.0e0 
tcc/seq                        |    1000 |    4 | 2347.863 +/- 23.883 | 205.861 +/- 3.376  | 1.000 +/- 0.000    | 6.661e-16 +/- 0.0e0
tcc/par-4x4-jobs4              |    1000 |    4 | 2285.397 +/- 34.680 | 202.013 +/- 9.210  | 1.021 +/- 0.038    | 6.661e-16 +/- 0.0e0
tcc/par-8x8-jobs8              |    1000 |    4 | 2284.489 +/- 28.781 | 194.197 +/- 3.823  | 1.061 +/- 0.028    | 6.661e-16 +/- 0.0e0
tcc/par-16x16-jobs16           |    1000 |    4 | 2318.514 +/- 88.005 | 197.940 +/- 6.917  | 1.042 +/- 0.046    | 6.661e-16 +/- 0.0e0
tcc/par-res16-row32-jobs8      |    1000 |    4 | 2368.917 +/- 57.392 | 215.223 +/- 7.643  | 0.957 +/- 0.026    | 6.661e-16 +/- 0.0e0
zig/seq                        |    1000 |    4 | 15119.454 +/- 45.275 | 206.369 +/- 5.645  | 1.000 +/- 0.000    | 6.661e-16 +/- 0.0e0
zig/par-4x4-jobs4              |    1000 |    4 | 31801.728 +/- 325.416 | 189.085 +/- 5.388  | 1.092 +/- 0.013    | 6.661e-16 +/- 0.0e0
zig/par-8x8-jobs8              |    1000 |    4 | 28887.835 +/- 382.594 | 194.222 +/- 4.253  | 1.063 +/- 0.021    | 6.661e-16 +/- 0.0e0
zig/par-16x16-jobs16           |    1000 |    4 | 23018.939 +/- 165.037 | 196.351 +/- 4.972  | 1.052 +/- 0.034    | 6.661e-16 +/- 0.0e0
zig/par-res16-row32-jobs8      |    1000 |    4 | 20788.874 +/- 165.923 | 208.175 +/- 2.080  | 0.992 +/- 0.034    | 6.661e-16 +/- 0.0e0

[AOT combustion tuning map] honest wall-clock summary; all time columns are milliseconds
note: this is the closest table to stopwatch timing from button press to finished result.
config                         | honest_user_e2e_ms | honest_speedup     | runtime_solve_ms   | manual_bootstrap_ms mean+/-std [min,max]
------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 2221.342 +/- 61.433 | -                  | 254.516 +/- 8.453  | 679.375 +/- 844.589 [188.359, 2142.232]
rust/seq                       | 5102.487 +/- 221.786 | 1.000 +/- 0.000    | 273.526 +/- 8.957  | 3737.675 +/- 67.350 [3641.919, 3831.364]
rust/par-4x4-jobs4             | 6119.064 +/- 103.374 | 0.834 +/- 0.041    | 241.578 +/- 10.278 | 4634.628 +/- 31.285 [4593.620, 4672.015]
rust/par-8x8-jobs8             | 5716.109 +/- 63.543 | 0.893 +/- 0.043    | 219.232 +/- 4.542  | 4179.569 +/- 55.751 [4121.368, 4270.131]
rust/par-16x16-jobs16          | 5332.123 +/- 98.920 | 0.957 +/- 0.035    | 243.983 +/- 4.934  | 3814.850 +/- 46.463 [3743.549, 3866.863]
rust/par-res16-row32-jobs8     | 5436.328 +/- 137.308 | 0.939 +/- 0.037    | 259.640 +/- 8.332  | 3863.213 +/- 24.773 [3823.372, 3889.545]
gcc/seq                        | 4100.413 +/- 61.910 | 1.000 +/- 0.000    | 209.734 +/- 5.191  | 2323.285 +/- 16.145 [2299.863, 2345.396]
gcc/par-4x4-jobs4              | 4010.559 +/- 59.611 | 1.023 +/- 0.025    | 192.126 +/- 2.583  | 2247.517 +/- 23.246 [2231.626, 2287.683]
gcc/par-8x8-jobs8              | 3971.941 +/- 39.999 | 1.032 +/- 0.016    | 202.417 +/- 7.978  | 2202.471 +/- 20.836 [2182.041, 2237.209]
gcc/par-16x16-jobs16           | 3908.012 +/- 78.537 | 1.050 +/- 0.034    | 194.636 +/- 4.392  | 2160.851 +/- 18.594 [2143.427, 2188.352]
gcc/par-res16-row32-jobs8      | 4222.290 +/- 45.525 | 0.971 +/- 0.018    | 217.868 +/- 6.264  | 2366.202 +/- 20.447 [2339.474, 2396.345]
tcc/seq                        | 2347.863 +/- 23.883 | 1.000 +/- 0.000    | 205.861 +/- 3.376  | 578.805 +/- 12.925 [566.678, 597.811]
tcc/par-4x4-jobs4              | 2285.397 +/- 34.680 | 1.028 +/- 0.016    | 202.013 +/- 9.210  | 485.574 +/- 8.685 [477.518, 499.076]
tcc/par-8x8-jobs8              | 2284.489 +/- 28.781 | 1.028 +/- 0.021    | 194.197 +/- 3.823  | 489.338 +/- 4.386 [484.143, 493.762]
tcc/par-16x16-jobs16           | 2318.514 +/- 88.005 | 1.014 +/- 0.043    | 197.940 +/- 6.917  | 492.527 +/- 9.771 [481.768, 502.286]
tcc/par-res16-row32-jobs8      | 2368.917 +/- 57.392 | 0.992 +/- 0.023    | 215.223 +/- 7.643  | 560.602 +/- 6.477 [551.217, 568.966]
zig/seq                        | 15119.454 +/- 45.275 | 1.000 +/- 0.000    | 206.369 +/- 5.645  | 13271.845 +/- 52.942 [13215.413, 13358.746]
zig/par-4x4-jobs4              | 31801.728 +/- 325.416 | 0.475 +/- 0.006    | 189.085 +/- 5.388  | 30223.051 +/- 289.188 [29993.720, 30714.438]
zig/par-8x8-jobs8              | 28887.835 +/- 382.594 | 0.523 +/- 0.008    | 194.222 +/- 4.253  | 26892.655 +/- 230.331 [26633.971, 27164.836]
zig/par-16x16-jobs16           | 23018.939 +/- 165.037 | 0.657 +/- 0.005    | 196.351 +/- 4.972  | 21300.287 +/- 142.817 [21127.682, 21460.036]
zig/par-res16-row32-jobs8      | 20788.874 +/- 165.923 | 0.727 +/- 0.006    | 208.175 +/- 2.080  | 19131.950 +/- 166.016 [18852.343, 19259.560]

[AOT combustion tuning map] isolated cold stage breakdown; every column in this table comes from the same child solve as honest_user_e2e_ms.
config                         | honest_e2e_ms      | solver_total_ms    | symbolic_ms        | initial_sym_jac    | artifact_ms        | materialize_ms     | compile_link_ms   
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 2221.342 +/- 61.433 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.408 +/- 0.274    | -                  | -                  | -                 
rust/seq                       | 5102.487 +/- 221.786 | 4750.000 +/- 433.013 | 4750.000 +/- 433.013 | 9.871 +/- 0.230    | 73.643 +/- 2.500   | 3.171 +/- 0.467    | 3168.637 +/- 17.314
rust/par-4x4-jobs4             | 6119.064 +/- 103.374 | 5750.000 +/- 433.013 | 5750.000 +/- 433.013 | 9.576 +/- 0.525    | 39.582 +/- 1.042   | 3.488 +/- 0.350    | 4170.910 +/- 21.426
rust/par-8x8-jobs8             | 5716.109 +/- 63.543 | 5000.000 +/- 0.000 | 5000.000 +/- 0.000 | 10.436 +/- 0.935   | 42.570 +/- 0.691   | 4.518 +/- 0.933    | 3722.285 +/- 46.239
rust/par-16x16-jobs16          | 5332.123 +/- 98.920 | 5000.000 +/- 0.000 | 5000.000 +/- 0.000 | 9.891 +/- 0.254    | 45.669 +/- 0.996   | 3.439 +/- 0.433    | 3339.094 +/- 52.831
rust/par-res16-row32-jobs8     | 5436.328 +/- 137.308 | 5000.000 +/- 0.000 | 5000.000 +/- 0.000 | 9.707 +/- 0.340    | 73.757 +/- 0.232   | 3.539 +/- 0.676    | 3387.769 +/- 87.670
gcc/seq                        | 4100.413 +/- 61.910 | 4000.000 +/- 0.000 | 3750.000 +/- 433.013 | 9.269 +/- 0.089    | 55.757 +/- 0.674   | 2.927 +/- 0.661    | 1783.624 +/- 10.971
gcc/par-4x4-jobs4              | 4010.559 +/- 59.611 | 3750.000 +/- 433.013 | 3250.000 +/- 433.013 | 9.194 +/- 0.198    | 25.583 +/- 0.279   | 3.278 +/- 0.625    | 1797.181 +/- 6.033
gcc/par-8x8-jobs8              | 3971.941 +/- 39.999 | 3250.000 +/- 433.013 | 3000.000 +/- 0.000 | 9.336 +/- 0.138    | 26.276 +/- 0.967   | 3.120 +/- 0.668    | 1731.185 +/- 23.253
gcc/par-16x16-jobs16           | 3908.012 +/- 78.537 | 3250.000 +/- 433.013 | 3000.000 +/- 0.000 | 9.427 +/- 0.259    | 29.768 +/- 0.574   | 3.345 +/- 0.549    | 1709.231 +/- 7.534
gcc/par-res16-row32-jobs8      | 4222.290 +/- 45.525 | 4000.000 +/- 0.000 | 4000.000 +/- 0.000 | 9.343 +/- 0.187    | 58.705 +/- 0.406   | 3.946 +/- 0.342    | 1828.323 +/- 27.525
tcc/seq                        | 2347.863 +/- 23.883 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.377 +/- 0.263    | 55.015 +/- 1.040   | 3.150 +/- 0.667    | 43.808 +/- 0.431  
tcc/par-4x4-jobs4              | 2285.397 +/- 34.680 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.526 +/- 0.417    | 24.979 +/- 0.796   | 3.538 +/- 0.850    | 40.982 +/- 0.450  
tcc/par-8x8-jobs8              | 2284.489 +/- 28.781 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.537 +/- 0.279    | 27.067 +/- 0.772   | 2.792 +/- 0.536    | 40.260 +/- 0.762  
tcc/par-16x16-jobs16           | 2318.514 +/- 88.005 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.511 +/- 0.306    | 29.433 +/- 0.592   | 2.809 +/- 0.613    | 40.725 +/- 0.337  
tcc/par-res16-row32-jobs8      | 2368.917 +/- 57.392 | 2000.000 +/- 0.000 | 2000.000 +/- 0.000 | 9.555 +/- 0.201    | 58.199 +/- 0.829   | 2.661 +/- 0.512    | 43.794 +/- 0.657  
zig/seq                        | 15119.454 +/- 45.275 | 15000.000 +/- 0.000 | 15000.000 +/- 0.000 | 9.143 +/- 0.212    | 72.041 +/- 0.735   | 2.570 +/- 0.482    | 12753.707 +/- 67.918
zig/par-4x4-jobs4              | 31801.728 +/- 325.416 | 31500.000 +/- 500.000 | 31500.000 +/- 500.000 | 9.320 +/- 0.105    | 41.127 +/- 0.386   | 2.258 +/- 0.728    | 29554.372 +/- 336.136
zig/par-8x8-jobs8              | 28887.835 +/- 382.594 | 28250.000 +/- 433.013 | 28250.000 +/- 433.013 | 9.339 +/- 0.225    | 47.465 +/- 3.814   | 2.642 +/- 0.602    | 26699.650 +/- 316.456
zig/par-16x16-jobs16           | 23018.939 +/- 165.037 | 22500.000 +/- 500.000 | 22500.000 +/- 500.000 | 9.295 +/- 0.257    | 46.338 +/- 0.351   | 3.201 +/- 0.137    | 20834.006 +/- 163.770
zig/par-res16-row32-jobs8      | 20788.874 +/- 165.923 | 20250.000 +/- 433.013 | 20000.000 +/- 0.000 | 9.352 +/- 0.120    | 76.045 +/- 0.733   | 2.504 +/- 0.615    | 18595.053 +/- 123.769

[AOT combustion tuning map] isolated cold numerical/runtime stages; all columns are from the fresh child solve.
config                         | linear_ms          | jac_ms             | fun_ms             | residual_values    | jacobian_values    | jacobian_assembly  | res_jobs     | jac_jobs     | rebind_ms          | register_link_ms  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 16.500 +/- 0.866   | 6.250 +/- 0.433    | 1.500 +/- 0.500    | -                  | -                  | -                  | -            | -            | -                  | -                 
rust/seq                       | 13.750 +/- 1.639   | 1.000 +/- 0.000    | 1.250 +/- 0.433    | 2.883 +/- 0.499    | 1.131 +/- 0.022    | 0.574 +/- 0.015    | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 53.058 +/- 1.426   | 0.882 +/- 0.033   
rust/par-4x4-jobs4             | 20.750 +/- 0.829   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 1.664 +/- 0.108    | 0.925 +/- 0.606    | 0.599 +/- 0.026    | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 25.423 +/- 0.959   | 0.709 +/- 0.026   
rust/par-8x8-jobs8             | 19.250 +/- 0.829   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 2.248 +/- 0.079    | 1.087 +/- 0.475    | 0.598 +/- 0.013    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 30.027 +/- 1.686   | 0.756 +/- 0.031   
rust/par-16x16-jobs16          | 20.250 +/- 0.433   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 2.542 +/- 0.115    | 0.880 +/- 0.048    | 0.622 +/- 0.017    | 16.000 +/- 0.000 | 16.000 +/- 0.000 | 35.686 +/- 4.534   | 0.777 +/- 0.026   
rust/par-res16-row32-jobs8     | 16.250 +/- 1.090   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 2.096 +/- 0.088    | 0.790 +/- 0.049    | 0.583 +/- 0.021    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 51.106 +/- 1.951   | 0.918 +/- 0.064   
gcc/seq                        | 14.000 +/- 1.225   | 1.000 +/- 0.000    | 0.250 +/- 0.433    | 1.627 +/- 0.213    | 0.647 +/- 0.074    | 0.581 +/- 0.028    | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 45.436 +/- 2.819   | 20.649 +/- 0.696  
gcc/par-4x4-jobs4              | 19.750 +/- 0.433   | 1.250 +/- 0.433    | 0.000 +/- 0.000    | 1.370 +/- 0.046    | 0.780 +/- 0.655    | 0.668 +/- 0.118    | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 22.659 +/- 0.393   | 14.794 +/- 0.253  
gcc/par-8x8-jobs8              | 19.750 +/- 0.829   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 1.917 +/- 0.071    | 0.845 +/- 0.592    | 0.603 +/- 0.011    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 24.596 +/- 0.803   | 16.078 +/- 0.994  
gcc/par-16x16-jobs16           | 19.750 +/- 1.479   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 2.327 +/- 0.106    | 0.532 +/- 0.020    | 0.602 +/- 0.022    | 16.000 +/- 0.000 | 16.000 +/- 0.000 | 26.150 +/- 0.692   | 16.981 +/- 0.301  
gcc/par-res16-row32-jobs8      | 16.750 +/- 1.299   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 1.820 +/- 0.062    | 0.871 +/- 0.573    | 0.626 +/- 0.046    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 42.756 +/- 1.085   | 22.259 +/- 0.340  
tcc/seq                        | 13.500 +/- 2.291   | 1.000 +/- 0.000    | 0.250 +/- 0.433    | 1.566 +/- 0.139    | 0.584 +/- 0.007    | 0.585 +/- 0.045    | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 42.896 +/- 1.219   | 10.704 +/- 0.094  
tcc/par-4x4-jobs4              | 19.250 +/- 0.433   | 1.000 +/- 1.225    | 0.500 +/- 0.500    | 1.421 +/- 0.117    | 0.714 +/- 0.682    | 0.755 +/- 0.250    | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 22.681 +/- 1.019   | 9.884 +/- 0.111   
tcc/par-8x8-jobs8              | 21.000 +/- 1.871   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 1.948 +/- 0.082    | 0.506 +/- 0.022    | 0.601 +/- 0.009    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 24.756 +/- 0.898   | 10.238 +/- 0.354  
tcc/par-16x16-jobs16           | 20.500 +/- 1.118   | 1.000 +/- 0.000    | 1.250 +/- 0.433    | 2.429 +/- 0.208    | 0.503 +/- 0.013    | 0.667 +/- 0.058    | 16.000 +/- 0.000 | 16.000 +/- 0.000 | 25.081 +/- 0.321   | 10.544 +/- 0.555  
tcc/par-res16-row32-jobs8      | 15.500 +/- 0.500   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 1.696 +/- 0.120    | 0.829 +/- 0.611    | 0.650 +/- 0.127    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 43.594 +/- 1.428   | 10.621 +/- 0.085  
zig/seq                        | 14.500 +/- 2.598   | 0.250 +/- 0.433    | 0.000 +/- 0.000    | 0.668 +/- 0.189    | 0.171 +/- 0.003    | 0.622 +/- 0.118    | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 42.889 +/- 1.018   | 1.016 +/- 0.145   
zig/par-4x4-jobs4              | 19.000 +/- 0.707   | 0.750 +/- 0.433    | 0.000 +/- 0.000    | 0.889 +/- 0.058    | 0.174 +/- 0.018    | 0.719 +/- 0.114    | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 22.773 +/- 0.647   | 0.765 +/- 0.020   
zig/par-8x8-jobs8              | 19.750 +/- 1.090   | 0.750 +/- 0.433    | 1.000 +/- 0.000    | 1.327 +/- 0.083    | 0.233 +/- 0.014    | 0.717 +/- 0.072    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 24.133 +/- 0.770   | 0.820 +/- 0.054   
zig/par-16x16-jobs16           | 19.500 +/- 0.866   | 0.500 +/- 0.500    | 1.000 +/- 0.000    | 1.669 +/- 0.151    | 0.269 +/- 0.106    | 0.657 +/- 0.071    | 16.000 +/- 0.000 | 16.000 +/- 0.000 | 26.119 +/- 0.262   | 0.836 +/- 0.102   
zig/par-res16-row32-jobs8      | 17.000 +/- 1.225   | 0.500 +/- 0.500    | 1.000 +/- 0.000    | 1.298 +/- 0.055    | 0.255 +/- 0.019    | 0.699 +/- 0.133    | 8.000 +/- 0.000 | 8.000 +/- 0.000 | 42.555 +/- 0.979   | 0.956 +/- 0.120   

[AOT combustion tuning map] runtime timing/counter summary; all time columns are milliseconds
note: manual_bootstrap_ms is callback preparation/linking performed by this diagnostic runner. Use solve_ms, callback stages, and counters for chunking decisions.
config                         | solve_ms           | manual_bootstrap_ms mean+/-std [min,max] | solver_total_ms    | symbolic_ms        | linear_ms          | jac_ms             | fun_ms             | iters          | linsys         | jac_re        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline              | 254.516 +/- 8.453  | 679.375 +/- 844.589 [188.359, 2142.232] | 289.000 +/- 8.396  | 194.500 +/- 7.665  | 13.500 +/- 0.500   | 6.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/seq                       | 273.526 +/- 8.957  | 3737.675 +/- 67.350 [3641.919, 3831.364] | 10000.000 +/- 0.000 | 201.750 +/- 12.930 | 11.750 +/- 0.433   | 1.000 +/- 0.000    | 2.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-4x4-jobs4             | 241.578 +/- 10.278 | 4634.628 +/- 31.285 [4593.620, 4672.015] | 11000.000 +/- 0.000 | 180.250 +/- 9.444  | 14.750 +/- 0.433   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-8x8-jobs8             | 219.232 +/- 4.542  | 4179.569 +/- 55.751 [4121.368, 4270.131] | 10750.000 +/- 433.013 | 156.500 +/- 3.571  | 15.750 +/- 0.433   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-16x16-jobs16          | 243.983 +/- 4.934  | 3814.850 +/- 46.463 [3743.549, 3866.863] | 10000.000 +/- 0.000 | 174.500 +/- 5.590  | 13.500 +/- 1.118   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
rust/par-res16-row32-jobs8     | 259.640 +/- 8.332  | 3863.213 +/- 24.773 [3823.372, 3889.545] | 10000.000 +/- 0.000 | 193.000 +/- 3.674  | 13.500 +/- 0.500   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/seq                        | 209.734 +/- 5.191  | 2323.285 +/- 16.145 [2299.863, 2345.396] | 9000.000 +/- 0.000 | 158.250 +/- 6.016  | 11.000 +/- 0.000   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-4x4-jobs4              | 192.126 +/- 2.583  | 2247.517 +/- 23.246 [2231.626, 2287.683] | 9000.000 +/- 0.000 | 138.500 +/- 2.693  | 13.750 +/- 2.046   | 0.750 +/- 0.433    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-8x8-jobs8              | 202.417 +/- 7.978  | 2202.471 +/- 20.836 [2182.041, 2237.209] | 9000.000 +/- 0.000 | 145.750 +/- 6.759  | 13.500 +/- 0.500   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-16x16-jobs16           | 194.636 +/- 4.392  | 2160.851 +/- 18.594 [2143.427, 2188.352] | 9000.000 +/- 0.000 | 141.500 +/- 4.500  | 13.250 +/- 0.433   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
gcc/par-res16-row32-jobs8      | 217.868 +/- 6.264  | 2366.202 +/- 20.447 [2339.474, 2396.345] | 9000.000 +/- 0.000 | 161.250 +/- 7.155  | 14.500 +/- 0.866   | 1.000 +/- 0.000    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/seq                        | 205.861 +/- 3.376  | 578.805 +/- 12.925 [566.678, 597.811] | 7000.000 +/- 0.000 | 154.000 +/- 1.871  | 11.250 +/- 0.433   | 1.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-4x4-jobs4              | 202.013 +/- 9.210  | 485.574 +/- 8.685 [477.518, 499.076] | 7000.000 +/- 0.000 | 147.250 +/- 10.401 | 13.750 +/- 1.479   | 0.750 +/- 0.433    | 0.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-8x8-jobs8              | 194.197 +/- 3.823  | 489.338 +/- 4.386 [484.143, 493.762] | 7000.000 +/- 0.000 | 140.000 +/- 4.183  | 14.250 +/- 0.829   | 1.250 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-16x16-jobs16           | 197.940 +/- 6.917  | 492.527 +/- 9.771 [481.768, 502.286] | 7000.000 +/- 0.000 | 143.250 +/- 6.457  | 13.000 +/- 1.225   | 1.000 +/- 0.707    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-res16-row32-jobs8      | 215.223 +/- 7.643  | 560.602 +/- 6.477 [551.217, 568.966] | 7000.000 +/- 0.000 | 160.250 +/- 7.361  | 14.000 +/- 0.707   | 1.750 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/seq                        | 206.369 +/- 5.645  | 13271.845 +/- 52.942 [13215.413, 13358.746] | 20000.000 +/- 0.000 | 157.000 +/- 4.183  | 11.000 +/- 0.000   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-4x4-jobs4              | 189.085 +/- 5.388  | 30223.051 +/- 289.188 [29993.720, 30714.438] | 36500.000 +/- 500.000 | 139.000 +/- 3.937  | 12.250 +/- 0.433   | 0.000 +/- 0.000    | 0.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-8x8-jobs8              | 194.222 +/- 4.253  | 26892.655 +/- 230.331 [26633.971, 27164.836] | 33750.000 +/- 433.013 | 141.250 +/- 4.657  | 14.250 +/- 1.090   | 0.000 +/- 0.000    | 0.500 +/- 0.500    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-16x16-jobs16           | 196.351 +/- 4.972  | 21300.287 +/- 142.817 [21127.682, 21460.036] | 28000.000 +/- 0.000 | 142.250 +/- 2.947  | 13.750 +/- 0.829   | 0.250 +/- 0.433    | 1.000 +/- 0.000    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
zig/par-res16-row32-jobs8      | 208.175 +/- 2.080  | 19131.950 +/- 166.016 [18852.343, 19259.560] | 25750.000 +/- 433.013 | 155.000 +/- 0.707  | 13.250 +/- 1.639   | 0.250 +/- 0.433    | 0.250 +/- 0.433    | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[AOT combustion tuning map] linked callback stage summary; all time columns are milliseconds
note: these columns are populated by linked AOT callbacks; Lambdify rows may be blank.
config                         | residual_values    | jacobian_values    | jacobian_assembly   
--------------------------------------------------------------------------------------------------
lambdify-baseline              | -                  | -                  | -                   
rust/seq                       | 3.815 +/- 0.344    | 1.202 +/- 0.025    | 0.528 +/- 0.034     
rust/par-4x4-jobs4             | 1.773 +/- 0.139    | 0.867 +/- 0.153    | 0.637 +/- 0.116     
rust/par-8x8-jobs8             | 2.239 +/- 0.059    | 0.950 +/- 0.049    | 0.558 +/- 0.055     
rust/par-16x16-jobs16          | 2.367 +/- 0.106    | 0.951 +/- 0.042    | 0.555 +/- 0.029     
rust/par-res16-row32-jobs8     | 2.323 +/- 0.101    | 0.943 +/- 0.058    | 0.617 +/- 0.067     
gcc/seq                        | 1.899 +/- 0.125    | 0.617 +/- 0.010    | 0.473 +/- 0.006     
gcc/par-4x4-jobs4              | 1.330 +/- 0.103    | 0.424 +/- 0.037    | 0.746 +/- 0.142     
gcc/par-8x8-jobs8              | 1.801 +/- 0.108    | 0.655 +/- 0.150    | 0.650 +/- 0.191     
gcc/par-16x16-jobs16           | 2.054 +/- 0.129    | 0.595 +/- 0.064    | 0.502 +/- 0.021     
gcc/par-res16-row32-jobs8      | 1.819 +/- 0.069    | 0.560 +/- 0.023    | 0.535 +/- 0.053     
tcc/seq                        | 1.833 +/- 0.203    | 0.614 +/- 0.044    | 0.501 +/- 0.045     
tcc/par-4x4-jobs4              | 1.402 +/- 0.099    | 0.706 +/- 0.258    | 0.532 +/- 0.038     
tcc/par-8x8-jobs8              | 1.758 +/- 0.073    | 0.861 +/- 0.648    | 0.614 +/- 0.191     
tcc/par-16x16-jobs16           | 2.098 +/- 0.170    | 0.821 +/- 0.535    | 0.504 +/- 0.021     
tcc/par-res16-row32-jobs8      | 1.818 +/- 0.044    | 1.423 +/- 0.560    | 0.604 +/- 0.171     
zig/seq                        | 0.612 +/- 0.072    | 0.178 +/- 0.005    | 0.551 +/- 0.094     
zig/par-4x4-jobs4              | 0.858 +/- 0.078    | 0.169 +/- 0.004    | 0.632 +/- 0.085     
zig/par-8x8-jobs8              | 1.264 +/- 0.157    | 0.221 +/- 0.011    | 0.528 +/- 0.070     
zig/par-16x16-jobs16           | 1.594 +/- 0.130    | 0.222 +/- 0.013    | 0.602 +/- 0.111     
zig/par-res16-row32-jobs8      | 1.223 +/- 0.099    | 0.214 +/- 0.013    | 0.582 +/- 0.108     
[AOT runtime tuning winner] scenario=medium-grid-multi-toolchain, config=zig/par-4x4-jobs4, n_steps=1000, runs=4, solve_ms_mean=189.085, speedup_vs_seq_mean=1.092, manual_bootstrap_ms_mean=30223.051
[AOT isolated cold wall-clock winner] scenario=medium-grid-multi-toolchain, config=lambdify-baseline, n_steps=1000, runs=4, honest_user_e2e_ms_mean=2221.342
ok  
```text
Date: 2026-05-26
Status: numerically clean and process-isolated for cold wall clock; one cold row is diagnostically unresolved.
Important numbers:
  All rows retain the same numerical trajectory: `max_diff_vs_ref` is
  `1.110e-15..1.332e-15`, with five Newton iterations, ten linear solves,
  and one Jacobian rebuild throughout.
  In isolated cold wall-clock time, tcc is the practical leader at this size:
  `tcc/seq = 3.019 +/- 0.525 s` and `tcc/par-8x8-jobs8 =
  2.792 +/- 0.196 s`, versus Lambdify at `4.280 +/- 0.502 s`.
  The tcc chunking gain is modest and noisy; it supports trying chunking, not
  declaring a universal policy.
  Hot callback stages confirm actual local parallel work. For tcc, residual
  values fall from `3.232 ms` sequential to about `2.28 ms` for the 8/16-job
  variants, and Jacobian values fall from `0.954 ms` to about `0.69 ms`.
  Zig has fast callback stages but poor cold wall clock (`39.5 s` sequential)
  because preparation/build dominates this one-off solve.
  `gcc/par-8x8-jobs8` is a severe outlier:
  `honest_user_e2e_ms = 2,528,501 +/- 4,366,874 ms`, while its hot
  `solve_ms` and callback timings are ordinary. Because the cold child
  currently reports only elapsed time and its solution, the table cannot yet
  attribute that stall to symbolic assembly, code generation, external
  compile/link, loading, or the numerical solve.
Conclusion:
  The current data establish that C-tcc AOT is competitive with and, in this
  sample, faster than Lambdify on combustion-1000, while parallel callback
  execution is genuinely active. They do not establish a reliable ranking of
  every gcc chunking policy because one isolated cold row suffered an
  unexplained multi-minute/hour-scale stall.
Follow-up:
  Extend the isolated-child payload with the same lifecycle/build breakdown
  used by the end-to-end stress test, and print raw cold observations before
  aggregation. Until that exists, keep `honest_user_e2e_ms` for end-to-end
  ranking and use parent-side callback columns only for hot-runtime behavior;
  never combine them to explain the gcc outlier.
```

### `combustion_tcc_chunking_honest_wall_clock_table`

File: `src/numerical/BVP_Damp/BVP_Damp_tests3.rs`

This is the compact practical companion to the full multi-toolchain tuning
map. It keeps only the Lambdify reference and the C-tcc AOT rows
(`seq` plus the available explicit chunking policies), because those are the
routes that have been competitive in real cold calculations. Each observation
is a fresh-process `solver.try_solve()` measurement and carries its own
symbolic/build/link/runtime breakdown; three repetitions keep the cost
manageable while still exposing severe drift.

Normal run:

```powershell
cargo test --release combustion_tcc_chunking_honest_wall_clock_table -- --ignored --nocapture --test-threads=1
```
=false
[BVP Damp isolated cold] launching repetition 1/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=45224.163 symbolic_ms=45000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=43502.825 symbolic_ms=43000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=64203.958 symbolic_ms=63000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=64878.376 symbolic_ms=64000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/whole
[BVP Damp isolated cold] finished source=AOT variant=tcc/whole total_ms=94831.516 symbolic_ms=93000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/2 source=AOT variant=tcc/chunk4
[BVP Damp isolated cold] finished source=AOT variant=tcc/chunk4 total_ms=31113.214 symbolic_ms=30000.000 status=ok
[BVP Damp stress] combustion-3000 raw process-isolated cold observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | ExprLegacy | 45224.163 |   45000.000 |          22946.688 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | AOT      | tcc/whole  | 43502.825 |   43000.000 |          33598.463 |         780.821 |              9.855 |              2.889 |    1.000 |    1.000 | ok
  1 |   3 | AOT      | tcc/chunk4 | 64203.958 |   63000.000 |          43394.932 |        1516.724 |             19.795 |              1.780 |    4.000 |    4.000 | ok
  2 |   1 | Lambdify | ExprLegacy | 64878.376 |   64000.000 |          46690.807 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | AOT      | tcc/whole  | 94831.516 |   93000.000 |          73303.311 |         674.572 |             10.455 |              3.108 |    1.000 |    1.000 | ok
  2 |   3 | AOT      | tcc/chunk4 | 31113.214 |   30000.000 |          19580.109 |         585.381 |              7.667 |              1.780 |    4.000 |    4.000 | ok

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify |  2/2   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/whole  | rebuild+seq+whole |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 |  2/2   | 1.110e-16 +/- 0.0e0  | 1.109e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 2/2

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT timing/counters
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | 55051.269 +/- 9827.106 [45224.163, 64878.376] | 54500.000 +/- 9500.000 | 54500.000 +/- 9500.000 | 70.000 +/- 9.000 | 5.000 +/- 2.000 | 13.500 +/- 5.500 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 69167.170 +/- 25664.346 [43502.825, 94831.516] | 68500.000 +/- 25500.000 | 68000.000 +/- 25000.000 | 56.000 +/- 1.000 | 4.500 +/- 0.500 | 6.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 47658.586 +/- 16545.372 [31113.214, 64203.958] | 47500.000 +/- 16500.000 | 46500.000 +/- 16500.000 | 66.000 +/- 10.000 | 4.000 +/- 0.000 | 11.500 +/- 6.500 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT callback stages
[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank.
source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | -                  | -                  | -                    | -        | -        | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | rebuild+seq+whole | 10.155 +/- 0.300   | 2.998 +/- 0.110    | 1.152 +/- 0.034      | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 18000.000 +/- 0.000 | 62988.000 +/- 0.000 | single_chunk | single_chunk
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | 13.731 +/- 6.064   | 1.780 +/- 0.000    | 1.045 +/- 0.013      | 4.000 +/- 0.000 | 4.000 +/- 0.000 | 4500.000 +/- 0.000 | 15747.000 +/- 0.000 | none         | none        

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT lifecycle/refinement stages
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 54500.000 +/- 9500.000
AOT      | Banded | tcc/whole  | rebuild+seq+whole | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 68000.000 +/- 25000.000
AOT      | Banded | tcc/chunk4 | rebuild+par+chunk4 | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 3001.000 +/- 0.000 | 46500.000 +/- 16500.000

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT symbolic handoff passes
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 48303.884 +/- 9228.416 | 622.045 +/- 10.298 | 34818.748 +/- 11872.060 | 75.225 +/- 5.454 | 3852.987 +/- 587.363 | -                | -               | -            | -            | -            | -           
AOT      | Banded | tcc/whole  | 62335.106 +/- 21502.626 | 864.139 +/- 262.478 | 53450.887 +/- 19852.424 | 85.781 +/- 4.576 | 0.096 +/- 0.092 | -                | -               | -            | -            | -            | 186.551 +/- 32.749
AOT      | Banded | tcc/chunk4 | 41913.796 +/- 14851.625 | 594.264 +/- 10.085 | 31487.521 +/- 11907.411 | 73.787 +/- 19.773 | 0.252 +/- 0.037 | -                | -               | -            | -            | -            | 215.235 +/- 2.163

[BVP Damp stress] combustion-3000 ExprLegacy Banded Lambdify vs AOT cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | tcc/whole  | 185.456 +/- 6.021 | 120.089 +/- 2.654 | 119.308 +/- 2.714 | 73.614 +/- 1.756 | 32.483 +/- 1.210 | 13.214 +/- 1.107 | 179.309 +/- 58.294 | 727.697 +/- 53.125 | 135.051 +/- 68.518
AOT      | Banded | tcc/chunk4 | 197.476 +/- 6.569 | 116.271 +/- 6.655 | 115.380 +/- 6.662 | 78.579 +/- 3.993 | 31.925 +/- 0.173 | 12.332 +/- 0.074 | 894.744 +/- 611.831 | 1051.052 +/- 465.671 | 892.917 +/- 867.478
ok

Controlled run for investigating machine-level drift:

```powershell
$env:BVP_AOT_COLD_COOLDOWN_MS="5000"
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
cargo test --release combustion_tcc_chunking_honest_wall_clock_table -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_AOT_COLD_COOLDOWN_MS -ErrorAction SilentlyContinue
Remove-Item Env:\BVP_AOT_COLD_CLEAN_ARTIFACTS -ErrorAction SilentlyContinue
```

Result:
running 1 test
test numerical::BVP_Damp::BVP_Damp_tests3::tests::combustion_tcc_chunking_honest_wall_clock_table ... [AOT tcc practical cold map] n_steps=1000, repetitions=3, cooldown_ms=5000, cleanup_child_artifacts=true
[AOT isolated cold raw] rep=1 config=lambdify-baseline total_ms=4904.111 symbolic_ms=4000.000 initial_sym_jac_ms=1358.737 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=1 config=tcc/seq total_ms=3007.796 symbolic_ms=2000.000 initial_sym_jac_ms=1318.643 materialize_ms=5.843 compile_link_ms=123.468 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=1 config=tcc/par-4x4-jobs4 total_ms=2891.278 symbolic_ms=2000.000 initial_sym_jac_ms=1269.241 materialize_ms=5.141 compile_link_ms=119.697 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=1 config=tcc/par-8x8-jobs8 total_ms=2881.390 symbolic_ms=2000.000 initial_sym_jac_ms=1263.639 materialize_ms=5.674 compile_link_ms=117.616 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=1 config=tcc/par-16x16-jobs16 total_ms=2947.096 symbolic_ms=2000.000 initial_sym_jac_ms=1292.017 materialize_ms=5.181 compile_link_ms=118.358 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=1 config=tcc/par-res16-row32-jobs8 total_ms=3179.979 symbolic_ms=3000.000 initial_sym_jac_ms=1288.898 materialize_ms=6.309 compile_link_ms=124.869 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=2 config=lambdify-baseline total_ms=4414.911 symbolic_ms=4000.000 initial_sym_jac_ms=1257.088 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=2 config=tcc/seq total_ms=2967.503 symbolic_ms=2000.000 initial_sym_jac_ms=1322.125 materialize_ms=5.132 compile_link_ms=121.703 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=2 config=tcc/par-4x4-jobs4 total_ms=2962.507 symbolic_ms=2000.000 initial_sym_jac_ms=1290.189 materialize_ms=5.461 compile_link_ms=122.295 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=2 config=tcc/par-8x8-jobs8 total_ms=2985.518 symbolic_ms=2000.000 initial_sym_jac_ms=1334.517 materialize_ms=5.029 compile_link_ms=119.091 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=2 config=tcc/par-16x16-jobs16 total_ms=2753.008 symbolic_ms=2000.000 initial_sym_jac_ms=1187.961 materialize_ms=5.705 compile_link_ms=117.945 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=2 config=tcc/par-res16-row32-jobs8 total_ms=2967.354 symbolic_ms=2000.000 initial_sym_jac_ms=1176.121 materialize_ms=5.986 compile_link_ms=128.984 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=3 config=lambdify-baseline total_ms=4099.897 symbolic_ms=4000.000 initial_sym_jac_ms=1235.880 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=3 config=tcc/seq total_ms=2746.082 symbolic_ms=2000.000 initial_sym_jac_ms=1178.417 materialize_ms=5.912 compile_link_ms=119.776 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=3 config=tcc/par-4x4-jobs4 total_ms=2934.066 symbolic_ms=2000.000 initial_sym_jac_ms=1314.463 materialize_ms=5.109 compile_link_ms=122.963 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=3 config=tcc/par-8x8-jobs8 total_ms=2969.902 symbolic_ms=2000.000 initial_sym_jac_ms=1323.255 materialize_ms=4.957 compile_link_ms=120.622 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=3 config=tcc/par-16x16-jobs16 total_ms=2948.806 symbolic_ms=2000.000 initial_sym_jac_ms=1286.298 materialize_ms=5.037 compile_link_ms=116.105 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=3 config=tcc/par-res16-row32-jobs8 total_ms=3219.608 symbolic_ms=3000.000 initial_sym_jac_ms=1319.656 materialize_ms=5.552 compile_link_ms=125.498 res_jobs=8.000 jac_jobs=8.000

[AOT tcc practical cold map] correctness and wall-clock table
config                     | honest_e2e_ms [min,max] | max_diff           | symbolic_ms        | initial_sym_jac   
------------------------------------------------------------------------------------------------------------------
lambdify-baseline          | 4472.973 +/- 330.876 [4099.897, 4904.111] | 0.000e0 +/- 0.0e0  | 4000.000 +/- 0.000 | 1283.902 +/- 53.620
tcc/seq                    | 2907.127 +/- 115.058 [2746.082, 3007.796] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1273.062 +/- 66.939
tcc/par-4x4-jobs4          | 2929.284 +/- 29.275 [2891.278, 2962.507] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1291.298 +/- 18.478
tcc/par-8x8-jobs8          | 2945.603 +/- 45.851 [2881.390, 2985.518] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1307.137 +/- 31.100
tcc/par-16x16-jobs16       | 2882.970 +/- 91.900 [2753.008, 2948.806] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1255.425 +/- 47.762
tcc/par-res16-row32-jobs8  | 3122.314 +/- 110.761 [2967.354, 3219.608] | 1.332e-15 +/- 0.0e0 | 2666.667 +/- 471.405 | 1261.558 +/- 61.704

[AOT tcc practical cold map] build and callback stages from the same child solves
config                     | materialize_ms     | compile_link_ms    | residual_values    | jacobian_values    | res_jobs     | jac_jobs    
------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline          | -                  | -                  | -                  | -                  | -            | -           
tcc/seq                    | 5.629 +/- 0.353    | 121.649 +/- 1.508  | 3.937 +/- 0.916    | 0.984 +/- 0.014    | 1.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-4x4-jobs4          | 5.237 +/- 0.159    | 121.652 +/- 1.409  | 2.356 +/- 0.155    | 0.773 +/- 0.008    | 4.000 +/- 0.000 | 4.000 +/- 0.000
tcc/par-8x8-jobs8          | 5.220 +/- 0.322    | 119.110 +/- 1.227  | 2.377 +/- 0.090    | 0.705 +/- 0.097    | 8.000 +/- 0.000 | 8.000 +/- 0.000
tcc/par-16x16-jobs16       | 5.308 +/- 0.287    | 117.469 +/- 0.979  | 2.154 +/- 0.174    | 0.606 +/- 0.011    | 16.000 +/- 0.000 | 16.000 +/- 0.000
tcc/par-res16-row32-jobs8  | 5.949 +/- 0.310    | 126.450 +/- 1.810  | 2.103 +/- 0.046    | 0.637 +/- 0.012    | 8.000 +/- 0.000 | 8.000 +/- 0.000
okrunning 1 test
test numerical::BVP_Damp::BVP_Damp_tests3::tests::combustion_tcc_chunking_honest_wall_clock_table ... [AOT tcc practical cold map] n_steps=1000, repetitions=3, cooldown_ms=5000, cleanup_child_artifacts=true
[AOT isolated cold raw] rep=1 config=lambdify-baseline total_ms=4904.111 symbolic_ms=4000.000 initial_sym_jac_ms=1358.737 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=1 config=tcc/seq total_ms=3007.796 symbolic_ms=2000.000 initial_sym_jac_ms=1318.643 materialize_ms=5.843 compile_link_ms=123.468 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=1 config=tcc/par-4x4-jobs4 total_ms=2891.278 symbolic_ms=2000.000 initial_sym_jac_ms=1269.241 materialize_ms=5.141 compile_link_ms=119.697 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=1 config=tcc/par-8x8-jobs8 total_ms=2881.390 symbolic_ms=2000.000 initial_sym_jac_ms=1263.639 materialize_ms=5.674 compile_link_ms=117.616 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=1 config=tcc/par-16x16-jobs16 total_ms=2947.096 symbolic_ms=2000.000 initial_sym_jac_ms=1292.017 materialize_ms=5.181 compile_link_ms=118.358 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=1 config=tcc/par-res16-row32-jobs8 total_ms=3179.979 symbolic_ms=3000.000 initial_sym_jac_ms=1288.898 materialize_ms=6.309 compile_link_ms=124.869 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=2 config=lambdify-baseline total_ms=4414.911 symbolic_ms=4000.000 initial_sym_jac_ms=1257.088 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=2 config=tcc/seq total_ms=2967.503 symbolic_ms=2000.000 initial_sym_jac_ms=1322.125 materialize_ms=5.132 compile_link_ms=121.703 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=2 config=tcc/par-4x4-jobs4 total_ms=2962.507 symbolic_ms=2000.000 initial_sym_jac_ms=1290.189 materialize_ms=5.461 compile_link_ms=122.295 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=2 config=tcc/par-8x8-jobs8 total_ms=2985.518 symbolic_ms=2000.000 initial_sym_jac_ms=1334.517 materialize_ms=5.029 compile_link_ms=119.091 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=2 config=tcc/par-16x16-jobs16 total_ms=2753.008 symbolic_ms=2000.000 initial_sym_jac_ms=1187.961 materialize_ms=5.705 compile_link_ms=117.945 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=2 config=tcc/par-res16-row32-jobs8 total_ms=2967.354 symbolic_ms=2000.000 initial_sym_jac_ms=1176.121 materialize_ms=5.986 compile_link_ms=128.984 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=3 config=lambdify-baseline total_ms=4099.897 symbolic_ms=4000.000 initial_sym_jac_ms=1235.880 materialize_ms=NaN compile_link_ms=NaN res_jobs=NaN jac_jobs=NaN
[AOT isolated cold raw] rep=3 config=tcc/seq total_ms=2746.082 symbolic_ms=2000.000 initial_sym_jac_ms=1178.417 materialize_ms=5.912 compile_link_ms=119.776 res_jobs=1.000 jac_jobs=1.000
[AOT isolated cold raw] rep=3 config=tcc/par-4x4-jobs4 total_ms=2934.066 symbolic_ms=2000.000 initial_sym_jac_ms=1314.463 materialize_ms=5.109 compile_link_ms=122.963 res_jobs=4.000 jac_jobs=4.000
[AOT isolated cold raw] rep=3 config=tcc/par-8x8-jobs8 total_ms=2969.902 symbolic_ms=2000.000 initial_sym_jac_ms=1323.255 materialize_ms=4.957 compile_link_ms=120.622 res_jobs=8.000 jac_jobs=8.000
[AOT isolated cold raw] rep=3 config=tcc/par-16x16-jobs16 total_ms=2948.806 symbolic_ms=2000.000 initial_sym_jac_ms=1286.298 materialize_ms=5.037 compile_link_ms=116.105 res_jobs=16.000 jac_jobs=16.000
[AOT isolated cold raw] rep=3 config=tcc/par-res16-row32-jobs8 total_ms=3219.608 symbolic_ms=3000.000 initial_sym_jac_ms=1319.656 materialize_ms=5.552 compile_link_ms=125.498 res_jobs=8.000 jac_jobs=8.000

[AOT tcc practical cold map] correctness and wall-clock table
config                     | honest_e2e_ms [min,max] | max_diff           | symbolic_ms        | initial_sym_jac   
------------------------------------------------------------------------------------------------------------------
lambdify-baseline          | 4472.973 +/- 330.876 [4099.897, 4904.111] | 0.000e0 +/- 0.0e0  | 4000.000 +/- 0.000 | 1283.902 +/- 53.620
tcc/seq                    | 2907.127 +/- 115.058 [2746.082, 3007.796] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1273.062 +/- 66.939
tcc/par-4x4-jobs4          | 2929.284 +/- 29.275 [2891.278, 2962.507] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1291.298 +/- 18.478
tcc/par-8x8-jobs8          | 2945.603 +/- 45.851 [2881.390, 2985.518] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1307.137 +/- 31.100
tcc/par-16x16-jobs16       | 2882.970 +/- 91.900 [2753.008, 2948.806] | 1.332e-15 +/- 0.0e0 | 2000.000 +/- 0.000 | 1255.425 +/- 47.762
tcc/par-res16-row32-jobs8  | 3122.314 +/- 110.761 [2967.354, 3219.608] | 1.332e-15 +/- 0.0e0 | 2666.667 +/- 471.405 | 1261.558 +/- 61.704

[AOT tcc practical cold map] build and callback stages from the same child solves
config                     | materialize_ms     | compile_link_ms    | residual_values    | jacobian_values    | res_jobs     | jac_jobs    
------------------------------------------------------------------------------------------------------------------------------------
lambdify-baseline          | -                  | -                  | -                  | -                  | -            | -           
tcc/seq                    | 5.629 +/- 0.353    | 121.649 +/- 1.508  | 3.937 +/- 0.916    | 0.984 +/- 0.014    | 1.000 +/- 0.000 | 1.000 +/- 0.000
tcc/par-4x4-jobs4          | 5.237 +/- 0.159    | 121.652 +/- 1.409  | 2.356 +/- 0.155    | 0.773 +/- 0.008    | 4.000 +/- 0.000 | 4.000 +/- 0.000
tcc/par-8x8-jobs8          | 5.220 +/- 0.322    | 119.110 +/- 1.227  | 2.377 +/- 0.090    | 0.705 +/- 0.097    | 8.000 +/- 0.000 | 8.000 +/- 0.000
tcc/par-16x16-jobs16       | 5.308 +/- 0.287    | 117.469 +/- 0.979  | 2.154 +/- 0.174    | 0.606 +/- 0.011    | 16.000 +/- 0.000 | 16.000 +/- 0.000
tcc/par-res16-row32-jobs8  | 5.949 +/- 0.310    | 126.450 +/- 1.810  | 2.103 +/- 0.046    | 0.637 +/- 0.012    | 8.000 +/- 0.000 | 8.000 +/- 0.000
ok

Interpretation:

This controlled run is stable enough to answer the practical question. With
fresh child processes, `5 s` cooldown and per-child artifact cleanup, every
TCC route agrees with Lambdify to `1.332e-15`, while the full cold solve is
substantially faster than Lambdify: `4.473 +/- 0.331 s` for Lambdify versus
`2.907 +/- 0.115 s` for `tcc/seq` and `2.883 +/- 0.092 s` for
`tcc/par-16x16-jobs16`. The reduction is approximately `35%`.

Parallel execution is unambiguously real: the table reports `4`, `8`, or
`16` actual jobs as configured, and hot callback work falls from
`3.937/0.984 ms` for sequential residual/Jacobian values to
`2.154/0.606 ms` for `par-16x16-jobs16`. Nevertheless, at this grid size
the runtime saving is small compared with the full cold preparation: the
parallel winners differ from `tcc/seq` by less than its run-to-run spread.
The conservative practical default therefore remains `tcc/seq` or future
`Auto`; explicit chunking is justified when callback throughput matters over
repeated solves, not from this one-off cold difference alone.

The outlier among TCC policies is `par-res16-row32-jobs8`, which rises to
`3.122 +/- 0.111 s` and occasionally reports a coarser `symbolic_ms` bucket.
It remains correct, but is not an attractive default for this problem.

### `combustion_1000_banded_symbolic_frontend_honest_wall_clock_table`

This is the missing symbolic-frontend control for the cold wall-clock story.
The previous controlled TCC table proves that compiled callbacks and chunking
work; the very large `combustion-3000` stress run shows that
`initial_sym_jac` can dominate and fluctuate. This test holds the matrix
backend fixed at `Banded`, uses sequential whole-module TCC for the AOT rows,
and compares `ExprLegacy` against `AtomView` through both `Lambdify` and
`AOT`. It therefore answers whether the expensive symbolic-Jacobian stage is
specific to one frontend or shared by both.

The test runs every row in a fresh child process, prints full solution
correctness, wall-clock, symbolic handoff and TCC cold-build tables, and
requires AOT direct rebinding to leave every post-build symbolic regeneration
field blank. Use the controlled invocation when recording performance:

```powershell
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
$env:BVP_AOT_COLD_COOLDOWN_MS="5000"
cargo test --release combustion_1000_banded_symbolic_frontend_honest_wall_clock_table -- --ignored --nocapture --test-threads=1
Remove-Item Env:BVP_AOT_COLD_CLEAN_ARTIFACTS
Remove-Item Env:BVP_AOT_COLD_COOLDOWN_MS
```
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_banded_symbolic_frontend_honest_wall_clock_table ... [BVP Damp symbolic frontend cold] protocol cooldown_ms=5000, cleanup_child_artifacts=true
[BVP Damp isolated cold] launching repetition 1/3 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=2194.362 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1048.084 symbolic_ms=969.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=1784.155 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1438.708 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=1509.576 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1228.450 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=1585.647 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1309.582 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=1359.016 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1060.524 symbolic_ms=986.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=1548.773 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1315.717 symbolic_ms=1000.000 status=ok
[BVP Damp symbolic frontend cold] combustion-1000 raw process-isolated observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | ExprLegacy | 2194.362 |    2000.000 |            240.103 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | Lambdify | AtomView   | 1048.084 |     969.000 |             42.618 |               - |                  - |                  - |        - |        - | ok
  1 |   3 | AOT      | ExprLegacy+tcc | 1784.155 |    1000.000 |            258.248 |         151.360 |              3.136 |              0.939 |    1.000 |    1.000 | ok
  1 |   4 | AOT      | AtomView+tcc | 1438.708 |    1000.000 |             36.221 |         127.855 |              3.336 |              0.877 |    1.000 |    1.000 | ok
  2 |   1 | Lambdify | ExprLegacy | 1509.576 |    1000.000 |            300.099 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | Lambdify | AtomView   | 1228.450 |    1000.000 |             40.810 |               - |                  - |                  - |        - |        - | ok
  2 |   3 | AOT      | ExprLegacy+tcc | 1585.647 |    1000.000 |            238.290 |         126.677 |              3.124 |              0.873 |    1.000 |    1.000 | ok
  2 |   4 | AOT      | AtomView+tcc | 1309.582 |    1000.000 |             43.006 |         119.552 |              3.789 |              0.879 |    1.000 |    1.000 | ok
  3 |   1 | Lambdify | ExprLegacy | 1359.016 |    1000.000 |            256.351 |               - |                  - |                  - |        - |        - | ok
  3 |   2 | Lambdify | AtomView   | 1060.524 |     986.000 |             34.375 |               - |                  - |                  - |        - |        - | ok
  3 |   3 | AOT      | ExprLegacy+tcc | 1548.773 |    1000.000 |            257.358 |         127.889 |              3.048 |              0.934 |    1.000 |    1.000 | ok
  3 |   4 | AOT      | AtomView+tcc | 1315.717 |    1000.000 |             37.717 |         120.505 |              3.407 |              0.876 |    1.000 |    1.000 | ok

[BVP Damp symbolic frontend cold] combustion-1000 Banded ExprLegacy/AtomView correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | Banded | AtomView   | atomview+lambdify |  3/3   | 4.079e-12 +/- 0.0e0  | 4.073e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole |  3/3   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | AtomView+tcc | atomview+rebuild+seq+whole |  3/3   | 4.079e-12 +/- 0.0e0  | 4.073e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

[BVP Damp symbolic frontend cold] combustion-1000 Banded ExprLegacy/AtomView wall-clock and solver stages
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | 1687.652 +/- 363.532 [1359.016, 2194.362] | 1333.333 +/- 471.405 | 1333.333 +/- 471.405 | 22.333 +/- 0.943 | 0.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Banded | AtomView   | atomview+lambdify | 1112.353 +/- 82.250 [1048.084, 1228.450] | 1000.000 +/- 0.000 | 985.000 +/- 12.675 | 24.000 +/- 0.000 | 0.333 +/- 0.471 | 4.333 +/- 1.886 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole | 1639.525 +/- 103.371 [1548.773, 1784.155] | 1000.000 +/- 0.000 | 1000.000 +/- 0.000 | 24.000 +/- 0.816 | 1.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Banded | AtomView+tcc | atomview+rebuild+seq+whole | 1354.669 +/- 59.477 [1309.582, 1438.708] | 1000.000 +/- 0.000 | 1000.000 +/- 0.000 | 23.667 +/- 0.471 | 1.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp symbolic frontend cold] combustion-1000 backend selection and symbolic totals
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | exprlegacy+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1333.333 +/- 471.405
Lambdify | Banded | AtomView   | atomview+lambdify | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 985.000 +/- 12.675
AOT      | Banded | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1000.000 +/- 0.000
AOT      | Banded | AtomView+tcc | atomview+rebuild+seq+whole | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1000.000 +/- 0.000

[BVP Damp symbolic frontend cold] combustion-1000 symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 650.718 +/- 65.771 | 224.498 +/- 18.315 | 265.518 +/- 25.337 | 17.104 +/- 0.631 | 71.979 +/- 22.726 | -                | -               | -            | -            | -            | -           
Lambdify | Banded | AtomView   | 343.242 +/- 31.015 | 90.984 +/- 20.298  | 39.268 +/- 3.538 | 27.620 +/- 1.199 | 70.329 +/- 8.513 | -                | -               | -            | -            | -            | -           
AOT      | Banded | ExprLegacy+tcc | 571.272 +/- 25.598 | 232.928 +/- 22.858 | 251.299 +/- 9.206 | 18.439 +/- 1.347 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 49.867 +/- 4.732
AOT      | Banded | AtomView+tcc | 258.108 +/- 13.746 | 82.883 +/- 11.792  | 38.982 +/- 2.911 | 27.043 +/- 0.521 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 69.736 +/- 0.715

[BVP Damp symbolic frontend cold] combustion-1000 internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | 265.518 +/- 25.337 | 2.593 +/- 0.049 | 260.760 +/- 25.402 | 0.000 +/- 0.000 | 1.659 +/- 0.141
Lambdify | Banded | AtomView   | 39.268 +/- 3.538 | 7.173 +/- 0.146 | 26.076 +/- 2.649 | 0.003 +/- 0.000 | 5.465 +/- 1.298
AOT      | Banded | ExprLegacy+tcc | 251.299 +/- 9.206 | 2.529 +/- 0.041 | 246.559 +/- 9.291 | 0.000 +/- 0.000 | 1.674 +/- 0.130
AOT      | Banded | AtomView+tcc | 38.982 +/- 2.911 | 7.175 +/- 0.047 | 26.807 +/- 3.013 | 0.003 +/- 0.000 | 4.288 +/- 0.353

[BVP Damp symbolic frontend cold] combustion-1000 tcc cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Banded | ExprLegacy | -                | -            | -              | -              | -            | -            | -            | -             | -            
Lambdify | Banded | AtomView   | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Banded | ExprLegacy+tcc | 50.253 +/- 0.792 | 30.573 +/- 1.353 | 30.451 +/- 1.379 | 20.177 +/- 0.289 | 10.466 +/- 0.280 | 3.470 +/- 0.007 | 5.139 +/- 2.143 | 135.309 +/- 11.361 | 63.256 +/- 66.006
AOT      | Banded | AtomView+tcc | 57.497 +/- 1.188 | 37.604 +/- 1.101 | 37.512 +/- 1.118 | 24.500 +/- 0.691 | 9.910 +/- 0.119 | 5.241 +/- 0.327 | 7.033 +/- 4.341 | 122.638 +/- 3.710 | 65.431 +/- 62.712
ok

Interpretation after the AtomView Jacobian and coefficient-parity corrections:

This controlled release rerun closes the numerical question that remained
open in the older table. `ExprLegacy + TCC` still agrees with the
`ExprLegacy + Lambdify` baseline at machine precision (`2.220e-16`), while
both AtomView routes now differ by only `4.079e-12`. The former AtomView
delta was `3.683e-8`, so repairing rational conversion of discretization
coefficients reduced it by roughly four orders of magnitude. The identical
Newton counters in all four rows confirm that the frontends follow the same
nonlinear solve path.

The performance result is equally useful. For Lambdify, AtomView lowers cold
wall-clock from `1687.652 +/- 363.532 ms` to
`1112.353 +/- 82.250 ms`, a reduction of about `34%`. For TCC AOT it lowers
the same end-to-end measurement from `1639.525 +/- 103.371 ms` to
`1354.669 +/- 59.477 ms`, about `17%`. The relevant stage is now clearly
visible: `initial_sym_jac` falls from `265.518` to `39.268 ms` in Lambdify
and from `251.299` to `38.982 ms` in TCC AOT. Within that stage,
`row_diff` changes from roughly `247-261 ms` to only `26-27 ms`.

AtomView also reduces discretization itself: `initial_discretize` changes
from `224.498` to `90.984 ms` for Lambdify and from `232.928` to
`82.883 ms` for AOT. The AOT cold-build cost is well behaved:
`compile_link` is `122.638 +/- 3.710 ms` for `AtomView + TCC`, and direct
rebinding remains correct because every post-build symbolic-regeneration
column is blank.

At `n_steps=1000`, cold `AtomView + Lambdify` is still faster than cold
`AtomView + TCC` (`1112.353` versus `1354.669 ms`). This is not a regression:
the hot AOT callback work is already very small, while a one-shot cold run
must additionally pay artifact construction, compilation, registration and
rebinding. Larger grids or repeated solves remain the meaningful cases for
an AOT wall-clock advantage.

Implementation follow-up (confirmed by this release rerun):

Inspection showed that the recorded `AtomView` row differentiation was not
atom-native: after packed discretization, `install_atom_discretized_system`
materialized residuals back into `Expr`, and the shared preparation stage
called `Expr::diff` for both frontends. The solver now uses the existing
`PreparedSparseAtomSystem::calc_sparse_jacobian_with_bandwidth` path whenever
`AtomView` is paired with `Sparse` or `Banded`. Only the differentiated
nonzero entries are converted to `Expr` afterwards, because the existing
Lambdify and banded AOT callback bridges consume that sparse compatibility
cache. Dense remains deliberately unchanged.

This is a backend-completeness correction rather than a timing-only shortcut:
the packed frontend now owns its Jacobian differentiation stage in the
production sparse/banded solve path. The release rows above confirm the
expected reduction in `initial_sym_jac` and `row_diff` without changing the
Newton choreography.

API decision after this evidence:

The Banded presets used by both Damped and Frozen solvers now select
`AtomView` as their default symbolic frontend, for Lambdify as well as AOT
policy modes. This is intentionally scoped to Banded presets: the general
Sparse/default configuration is unchanged. `ExprLegacy` remains available as
an explicit override for compatibility baselines and frontend parity studies.
The default switch is covered by configuration tests and by small Damped and
Frozen Banded end-to-end solves, so a later refactor cannot silently restore
the expensive legacy frontend on the normal banded route.

Debug validation after the atom-native Jacobian routing:

The post-change debug run completed all twelve rows (`3 repetitions x 4
frontends/toolchains`) without a solver failure or an AOT handoff failure.
This run is deliberately not used as a precise performance benchmark, but it
does establish that the newly active route is exercised in the real
end-to-end solve. In the same process, `initial_sym_jac` fell from
`3473.384 +/- 614.139 ms` (`Lambdify + ExprLegacy`) to
`382.975 +/- 68.778 ms` (`Lambdify + AtomView`), and from
`3581.609 +/- 230.583 ms` (`AOT TCC + ExprLegacy`) to
`318.339 +/- 12.070 ms` (`AOT TCC + AtomView`). The internal breakdown
attributes that change to `row_diff`: approximately `3.45 s -> 0.288 s` for
Lambdify and `3.56 s -> 0.239 s` for TCC AOT. This is the expected signature
of atom-native differentiation rather than an accidental callback shortcut.

The remaining `3.683e-8` solution delta was then localized before Newton or
AOT execution. In the combustion row diagnostic the AtomView conversion had
represented the grid coefficient `34.72222222222222` as
`1736111 / 50000`, introducing a residual difference of
`5.666667e-7`. The cause was the conversion of floating constants into Atom
rationals with only six decimal significant digits. The converter now first
recovers accurate compact fractions with a bounded continued-fraction pass
and falls back to the existing decimal representation when a small physical
constant, such as the reaction scale `9e-8`, cannot safely be represented
under that denominator cap.

The debug diagnostic after this correction reduces the combustion
discretized-row maximum difference to `1.196106e-11`. At callback-bundle
level, `residual_max_diff` drops from `6.888889e-7` to `1.196106e-11`,
while `jacobian_max_diff` drops from `2.222222e-6` to `4.163336e-17`.
Dedicated regressions cover both the repeating flux coefficient and the
small nonzero combustion scale. The release table above is the replacement
measurement: its AtomView `solve_diff = 4.079e-12` confirms the correction
in a real cold end-to-end solve.

### `combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table`

This is the Sparse counterpart of the Banded frontend decision test. It uses
the same process-isolated cold protocol, the same combustion-1000 system, the
same four frontend/toolchain rows (`ExprLegacy` and `AtomView`, each through
Lambdify and sequential whole-module TCC AOT), and the same stage tables. The
purpose is deliberately narrow: determine whether `Sparse` deserves the same
`AtomView` default as `Banded`, rather than assuming the result transfers
between matrix routes.

Run it before changing the Sparse default:

```powershell
$env:BVP_AOT_COLD_CLEAN_ARTIFACTS="1"
$env:BVP_AOT_COLD_COOLDOWN_MS="5000"
cargo test --release combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table -- --ignored --nocapture
Remove-Item Env:BVP_AOT_COLD_CLEAN_ARTIFACTS
Remove-Item Env:BVP_AOT_COLD_COOLDOWN_MS
```
BVP Damp symbolic frontend cold] protocol cooldown_ms=5000, cleanup_child_artifacts=true
[BVP Damp isolated cold] launching repetition 1/3 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=1595.984 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1802.397 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=1680.484 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 1/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1466.507 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=Lambdify variant=ExprLegacy
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=2309.126 symbolic_ms=2000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1262.295 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=1763.495 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 2/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1469.465 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=Lambdify variant=ExprLegacy
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table has been running for over 60 seconds
[BVP Damp isolated cold] finished source=Lambdify variant=ExprLegacy total_ms=1352.938 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=Lambdify variant=AtomView
[BVP Damp isolated cold] finished source=Lambdify variant=AtomView total_ms=1708.836 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=AOT variant=ExprLegacy+tcc
[BVP Damp isolated cold] finished source=AOT variant=ExprLegacy+tcc total_ms=2098.678 symbolic_ms=1000.000 status=ok
[BVP Damp isolated cold] launching repetition 3/3 source=AOT variant=AtomView+tcc
[BVP Damp isolated cold] finished source=AOT variant=AtomView+tcc total_ms=1562.688 symbolic_ms=1000.000 status=ok
[BVP Damp symbolic frontend cold] combustion-1000 Sparse raw process-isolated observations
[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled.
rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | Lambdify | ExprLegacy | 1595.984 |    1000.000 |            231.478 |               - |                  - |                  - |        - |        - | ok
  1 |   2 | Lambdify | AtomView   | 1802.397 |    1000.000 |             42.794 |               - |                  - |                  - |        - |        - | ok
  1 |   3 | AOT      | ExprLegacy+tcc | 1680.484 |    1000.000 |            246.987 |         140.949 |              3.135 |              0.859 |    1.000 |    1.000 | ok
  1 |   4 | AOT      | AtomView+tcc | 1466.507 |    1000.000 |             65.390 |         130.607 |              3.782 |              0.943 |    1.000 |    1.000 | ok
  2 |   1 | Lambdify | ExprLegacy | 2309.126 |    2000.000 |            493.282 |               - |                  - |                  - |        - |        - | ok
  2 |   2 | Lambdify | AtomView   | 1262.295 |    1000.000 |             37.290 |               - |                  - |                  - |        - |        - | ok
  2 |   3 | AOT      | ExprLegacy+tcc | 1763.495 |    1000.000 |            250.384 |         132.061 |              3.394 |              1.043 |    1.000 |    1.000 | ok
  2 |   4 | AOT      | AtomView+tcc | 1469.465 |    1000.000 |             36.910 |         146.272 |              3.738 |              0.953 |    1.000 |    1.000 | ok
  3 |   1 | Lambdify | ExprLegacy | 1352.938 |    1000.000 |            261.347 |               - |                  - |                  - |        - |        - | ok
  3 |   2 | Lambdify | AtomView   | 1708.836 |    1000.000 |             38.225 |               - |                  - |                  - |        - |        - | ok
  3 |   3 | AOT      | ExprLegacy+tcc | 2098.678 |    1000.000 |            345.030 |         141.670 |              3.068 |              0.928 |    1.000 |    1.000 | ok
  3 |   4 | AOT      | AtomView+tcc | 1562.688 |    1000.000 |             39.768 |         150.547 |              3.895 |              0.966 |    1.000 |    1.000 | ok

[BVP Damp symbolic frontend cold] combustion-1000 Sparse ExprLegacy/AtomView correctness
[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | exprlegacy+lambdify |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | Sparse | AtomView   | atomview+lambdify |  3/3   | 4.079e-12 +/- 0.0e0  | 4.073e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole |  3/3   | 4.441e-16 +/- 0.0e0  | 4.434e-16 +/- 4.9e-32 | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | AtomView+tcc | atomview+rebuild+seq+whole |  3/3   | 4.079e-12 +/- 0.0e0  | 4.073e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

[BVP Damp symbolic frontend cold] combustion-1000 Sparse ExprLegacy/AtomView wall-clock and solver stages
[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | exprlegacy+lambdify | 1752.682 +/- 405.783 [1352.938, 2309.126] | 1333.333 +/- 471.405 | 1333.333 +/- 471.405 | 49.000 +/- 4.243 | 5.000 +/- 0.000 | 2.667 +/- 0.943 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
Lambdify | Sparse | AtomView   | atomview+lambdify | 1591.176 +/- 235.670 [1262.295, 1802.397] | 1000.000 +/- 0.000 | 1000.000 +/- 0.000 | 50.000 +/- 1.414 | 4.667 +/- 0.471 | 5.333 +/- 0.471 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole | 1847.553 +/- 180.778 [1680.484, 2098.678] | 1333.333 +/- 471.405 | 1000.000 +/- 0.000 | 48.667 +/- 2.357 | 2.000 +/- 0.000 | 1.333 +/- 0.471 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000
AOT      | Sparse | AtomView+tcc | atomview+rebuild+seq+whole | 1499.553 +/- 44.659 [1466.507, 1562.688] | 1000.000 +/- 0.000 | 1000.000 +/- 0.000 | 51.000 +/- 3.266 | 2.000 +/- 0.000 | 2.000 +/- 0.000 | 5.000 +/- 0.000 | 10.000 +/- 0.000 | 1.000 +/- 0.000

[BVP Damp symbolic frontend cold] combustion-1000 Sparse backend selection and symbolic totals
[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time.
source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | exprlegacy+lambdify | ExprLegacy | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1333.333 +/- 471.405
Lambdify | Sparse | AtomView   | atomview+lambdify | AtomView | Lambdify         | UseIfAvailable | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1000.000 +/- 0.000
AOT      | Sparse | ExprLegacy+tcc | exprlegacy+rebuild+seq+whole | ExprLegacy | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1000.000 +/- 0.000
AOT      | Sparse | AtomView+tcc | atomview+rebuild+seq+whole | AtomView | AotCompiled      | RebuildAlways | 0.000 +/- 0.000    | 1001.000 +/- 0.000 | 1000.000 +/- 0.000

[BVP Damp symbolic frontend cold] combustion-1000 Sparse symbolic handoff stages
[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact.
source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 860.994 +/- 303.859 | 341.941 +/- 165.865 | 328.702 +/- 117.012 | 17.058 +/- 0.501 | 96.686 +/- 16.209 | -                | -               | -            | -            | -            | -           
Lambdify | Sparse | AtomView   | 396.944 +/- 26.661 | 86.224 +/- 4.956   | 39.436 +/- 2.405 | 27.631 +/- 0.928 | 123.416 +/- 24.225 | -                | -               | -            | -            | -            | -           
AOT      | Sparse | ExprLegacy+tcc | 640.927 +/- 86.747 | 272.302 +/- 40.562 | 280.801 +/- 45.438 | 16.979 +/- 0.255 | 0.001 +/- 0.000 | -                | -               | -            | -            | -            | 43.926 +/- 0.526
AOT      | Sparse | AtomView+tcc | 283.323 +/- 8.158 | 95.354 +/- 3.513   | 47.356 +/- 12.805 | 27.291 +/- 0.978 | 0.002 +/- 0.000 | -                | -               | -            | -            | -            | 71.303 +/- 9.337

[BVP Damp symbolic frontend cold] combustion-1000 Sparse internal initial symbolic-Jacobian stages
[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes.
source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | 328.702 +/- 117.012 | 2.530 +/- 0.067 | 323.905 +/- 116.678 | 0.000 +/- 0.000 | 1.692 +/- 0.222
Lambdify | Sparse | AtomView   | 39.436 +/- 2.405 | 7.527 +/- 0.366 | 25.398 +/- 1.071 | 0.000 +/- 0.000 | 5.611 +/- 1.395
AOT      | Sparse | ExprLegacy+tcc | 280.801 +/- 45.438 | 2.855 +/- 0.424 | 275.817 +/- 45.634 | 0.000 +/- 0.000 | 1.605 +/- 0.099
AOT      | Sparse | AtomView+tcc | 47.356 +/- 12.805 | 14.140 +/- 9.871 | 27.512 +/- 2.218 | 0.001 +/- 0.000 | 4.712 +/- 0.734

[BVP Damp symbolic frontend cold] combustion-1000 Sparse tcc cold-build stages
[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank.
source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | -                | -            | -              | -              | -            | -            | -            | -             | -            
Lambdify | Sparse | AtomView   | -                | -            | -              | -              | -            | -            | -            | -             | -            
AOT      | Sparse | ExprLegacy+tcc | 50.738 +/- 1.907 | 32.027 +/- 0.843 | 31.945 +/- 0.827 | 20.555 +/- 0.550 | 10.335 +/- 0.293 | 4.311 +/- 0.479 | 5.758 +/- 0.326 | 138.226 +/- 4.370 | 55.227 +/- 33.142
AOT      | Sparse | AtomView+tcc | 120.283 +/- 8.598 | 48.320 +/- 6.546 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 11.148 +/- 0.446 | 5.242 +/- 0.268 | 6.627 +/- 0.827 | 142.475 +/- 8.571 | 38.075 +/- 7.037
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table ... ok

Decision after the release run:

The Sparse gate satisfies the same conditions that justified the Banded
default change. All four rows solve with identical Newton choreography and
the AtomView rows differ from the ExprLegacy Lambdify reference by only
`4.079e-12`. For Lambdify, `initial_sym_jac` falls from
`328.702 +/- 117.012 ms` to `39.436 +/- 2.405 ms`, with `row_diff`
falling from `323.905` to `25.398 ms`. For TCC AOT, `initial_sym_jac`
falls from `280.801 +/- 45.438 ms` to `47.356 +/- 12.805 ms`, with
`row_diff` falling from `275.817` to `27.512 ms`. Direct AOT rebinding
remains valid: all post-build symbolic regeneration columns are blank.

The cold wall-clock result is less dramatic than the internal-stage result,
but it is favorable rather than neutral: Lambdify changes from
`1752.682 +/- 405.783 ms` to `1591.176 +/- 235.670 ms`, while TCC AOT
changes from `1847.553 +/- 180.778 ms` to
`1499.553 +/- 44.659 ms`. Consequently the Sparse production presets now
select `AtomView` by default as well. `ExprLegacy` remains an explicit
compatibility/control override, not a hidden fallback.

Debug wiring validation after adding the test:

The non-release run completed all twelve process-isolated observations
successfully. Both AtomView rows agree with the ExprLegacy Lambdify reference
to `4.079e-12`, and every row retains the same five Newton iterations, ten
linear solves and one Jacobian rebuild. The diagnostic signal is already
large: `initial_sym_jac` changes from `3719.070 +/- 403.033 ms` to
`457.717 +/- 125.161 ms` on Lambdify and from
`3854.957 +/- 830.024 ms` to `393.557 +/- 94.731 ms` on TCC AOT.
Corresponding `row_diff` changes from roughly `3.7-3.8 s` to
`0.31-0.34 s`. This validates the intended Sparse AtomView path and its
instrumentation. It is retained as the pre-release validation record; the
release decision above supersedes its former "default not yet changed"
qualification.

Release-run analysis from the two scaling points above:

The correctness signal is clean. Both `n_steps=200` and `n_steps=1000` keep every AOT
configuration within roughly `3.6e-8` of the Lambdify reference, while the Newton
counters stay identical across rows: five nonlinear iterations, ten linear solves,
one Jacobian rebuild. That means the chunking policies are not changing the numerical
path. For this test, the earlier sparse chunking bug is not resurfacing.

The runtime signal is less favorable for chunking. At `n_steps=200`, the best parallel
row is only a statistical tie with sequential AOT: `par-4x4-jobs4` reports about
`214.5 ms` solve time versus `216.0 ms` sequential, which is inside the observed noise.
At `n_steps=1000`, sequential AOT is the winner: about `2945 ms`, while all explicit
parallel/chunking policies are slower or statistically indistinguishable. In other
words, for this sparse combustion workload and this machine, the explicit chunking
policies do not currently provide a reliable solve-loop speedup. The likely reason is
that the callback evaluation is not the dominant cost once the sparse linear solve and
Newton orchestration are included; the extra scheduling overhead consumes the expected
gain.

The AOT-vs-Lambdify solve-loop comparison is still meaningful. At `n_steps=1000`, AOT
solve time is about `2x` faster than Lambdify (`~2.95 s` versus `~6.01 s`). At
`n_steps=200`, AOT is also faster in the solve loop, but the margin is much smaller.
However, end-to-end time tells the opposite story when artifacts are built or
bootstrapped inside the run: Lambdify is much cheaper to prepare, while AOT spends
roughly `23-24 s` in `prepare/bootstrap_ms` for `n_steps=1000`. Therefore these rows
should not be used to claim “AOT is faster end-to-end” unless the artifact is reused
across many solves or the benchmark is changed to strict prebuilt/runtime-only mode.

There is one telemetry anomaly that should be investigated before using the internal
timers as authoritative performance counters. For some rows, especially Lambdify,
outer `solve_ms` and internal `solver_total_ms` differ by large factors. For example,
`n_steps=200` reports Lambdify `solve_ms ~= 264 ms`, but `solver_total_ms ~= 2800 ms`.
That cannot be interpreted as a simple wall-clock subset relationship. Possible
causes are unit parsing of the solver timer strings, cumulative timers with overlapping
sections, or timers that include repeated symbolic/bootstrap work not covered by the
outer solve stopwatch. Until this is fixed or documented, `solve_ms` and
`end_to_end_ms` are the reliable wall-clock columns, while `linear_ms`, `jac_ms`,
`fun_ms`, and `symbolic_ms` should be treated as diagnostic hints rather than final
accounting.

The callback-stage split confirms the same diagnosis from another angle. For the
`n_steps=1000` run, explicit parallel policies reduce linked callback value time:
residual values drop from about `4.8 ms` to roughly `2.7-2.9 ms`, and Jacobian values
drop from about `1.82 ms` to roughly `1.18-1.31 ms`. Matrix assembly stays around
`1.5 ms`. That is a real local win, not a fake fallback. But the local win is only a
few milliseconds across the whole solve, while the Newton loop still spends seconds
in total and the AOT prepare/bootstrap path spends tens of seconds when artifacts are
built or linked inside the run. This is the key explanation for the apparent
contradiction: codegen callback benchmarks can show large relative speedups while
end-to-end BVP solve time barely moves.

Practical conclusion: the safe current recommendation is `Lambdify` for cheap
one-off solves, sequential AOT for repeated large sparse solves with reusable
artifacts, and no default recommendation for explicit chunking on this workload.
Chunking remains useful as a tunable option, but the source-of-truth table currently
says "measure before enabling", not "enable by default".

Additional anomaly check from the repeated release runs:

The near equality of `sequential-baseline` and the explicit `par-*` rows should not
be read as proof that the solver secretly rebuilt a sequential backend. The lower-level
callback diagnostic below is the direct guard for that question, and it reports real
linked chunks plus real runtime jobs. Therefore the more likely interpretation is
economics, not fallback: callback work is too small compared with scheduling, FFI,
temporary-buffer, and copy-back overhead, while the full Newton solve also includes
linear solves and orchestration that chunking does not accelerate.

The zero or near-zero `jac_ms` / `fun_ms` entries in some end-to-end tables are a
separate instrumentation artifact. The current `CustomTimer::elapsed_time` stores
sub-second timings through integer milliseconds, so operations below `1 ms` are
reported as `0 ms`; for longer sections it stores whole seconds, losing fractional
precision. Thus `jac_ms=0` or `fun_ms=0` in compiled AOT rows means "below the current
timer resolution" or "rounded away", not "Jacobian/residual callback was not called".
For callback-level performance, prefer the explicit `residual_ms` / `jacobian_ms`
columns from `combustion_sparse_aot_callback_chunking_parallelism_diagnostic`, which
are measured directly with `Instant` around repeated callback calls.

### `combustion_sparse_aot_callback_chunking_parallelism_diagnostic`

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
CPU 4 Core
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
CPU 12 Core

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
  On this `n_steps=200` callback-only fixture, gcc is the only clear modest
  winner: callback total falls from `8.905 ms` whole to `7.769..7.881 ms`,
  approximately `1.13x..1.15x`.
  Rust parallel rows lose to whole (`8.334 ms` whole versus
  `9.197..10.065 ms`), and tcc is also flat/slower (`8.057 ms` whole versus
  `8.211..8.635 ms`).
  Zig is effectively a tie at best (`8.765 ms` whole versus `8.710 ms` for
  `par-8x8-jobs8`), with other chunk layouts slower.
Conclusion:
  This rerun closes the binary correctness question: chunk functions are
  exported, bound, and executed through a genuinely parallel runtime route.
  It also shows why `Auto` must remain conservative. At this medium callback
  size the scheduling/FFI cost is already comparable to the arithmetic saved,
  so explicit parallelism is not a default performance win even after removal
  of temporary-buffer copy-back overhead.
Follow-up:
  Use this test as a binding/correctness guard, not as a universal performance
  recommendation. Use the xlarge compiled fixture for positive break-even
  evidence and the full solve stories for application-level economics.
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

### `diagnose_eval_fraction_of_solve_time`

Estimates how much of the solve time is spent in callback evaluation. Use it when
deciding whether AOT/chunking work can plausibly move the total runtime.

```powershell
cargo test --release diagnose_eval_fraction_of_solve_time -- --nocapture --test-threads=1
```

```text
Date:
Conclusion:
```
## bvp_generated_backend_pipeline_comparison_table  

Latest multi-run pipeline interpretation:

Note for the next run: the pipeline test now prints two fine-grained internal
AOT stage tables. The first separates sparse lookup, symbolic Jacobian build,
chunk-plan finalization, Atom lowering, peephole optimization, temporary reuse,
and module push. The second separates module construction, preliminary source
probe, final language source emission, C header emission, artifact packaging,
and the remaining `artifact_other_ms`. This turns the large `artifact_ms` bucket
from a black box into actionable timings.

For `combustion-1000`, the cold bootstrap cost is now separated into the common
symbolic stage and the route-specific callable preparation stage. The common
symbolic work is about `2.01 s` for both Lambdify and AOT. Lambdify then spends
about `0.36 s` preparing callable residual/Jacobian functions and reaches first
outputs in about `2.37 s`.

`C-tcc` is still the fastest cold AOT route, but the table shows why it does not
beat Lambdify in one-shot bootstrap latency. Its compiler build is only about
`0.16 s`, yet AOT callable preparation is about `1.14 s`. The dominant AOT-only
piece is `artifact_ms` at about `0.81 s`; `materialize_ms` is about `0.12 s`,
`link_ms` about `0.05 s`, and the actual `build_ms` is relatively small. In
other words, tcc is not losing because C compilation is slow. It is losing
because generating, packaging, materializing, and linking the generated artifact
still costs much more than Lambdify callable preparation on this problem.

The current cold-bootstrap ranking is therefore: Lambdify first (`~2.37 s`),
then `C-tcc` (`~3.16 s`), then `C-gcc` (`~9.48 s`), Rust (`~15.20 s`), and Zig
(`~48.43 s`). This does not contradict runtime callback tables where compiled
callbacks can be faster per call. It says that for a one-shot "press button and
solve" BVP with only a small number of Newton/Jacobian refreshes, cold AOT must
pay back a sizeable preparation cost. The next optimization target is not tcc
itself; it is reducing or reusing the AOT artifact-generation/materialization
path, and making the warm/prebuilt artifact story explicit.

## New Heavy Lifecycle and Frozen Stories

### `combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`

This Damped production-lifecycle story is intentionally different from the cold
toolchain races. It solves the combustion problem through the Banded Lambdify
baseline, then constructs both Sparse and Banded AtomView `tcc` AOT routes with
`BuildIfMissing`. The resolver snapshots returned by those successful builds are
fed into three strict `RequirePrebuilt` solves for each matrix backend. A prebuilt
row is not allowed to rebuild or fall back to Lambdify: it must still report
`AotCompiled` and `RequirePrebuilt`.

The table reports solution difference against Lambdify, wall-clock, symbolic and
linear-system time, initial handoff time, direct rebind time, and compile/link
time. In healthy `prebuilt` rows, compiler/linker time should be blank while
correctness remains within `1e-5`.

```powershell
cargo test --release --lib combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture
```

```text
Date: release run recorded below.
Result: Both Sparse and Banded AtomView+tcc rows remained `AotCompiled` through
        `BuildIfMissing` and all three strict `RequirePrebuilt` reuses.  Solution
        differences against the Banded Lambdify baseline were `2.73e-15` for
        Sparse and `2.22e-16` for Banded.  No prebuilt row reported compile/link
        time.
Conclusion: The production-scale resolver/artifact lifecycle is healthy for
            Damped combustion-1000 on both native matrix paths.  Warm Banded
            prebuilt solves averaged about `411 ms`, versus about `442 ms` for
            Sparse prebuilt solves, so the Banded linear-algebra advantage is
            visible even after compilation is removed from the comparison.
            The one Lambdify baseline (`2424 ms`) is sufficient for correctness,
            but not sufficient by itself for a precise speedup claim.
```

### `frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story`

This is the first heavy end-to-end combustion story for Frozen Newton. It solves
the six-component problem through Frozen itself, using Banded AtomView Lambdify
as baseline and freshly rebuilt `tcc` AOT callbacks in both `whole` and forced
`chunk4` modes. Two repetitions per route expose gross instability without
turning this first coverage gate into an all-night benchmark.

The test asserts numerical agreement with Lambdify, compiled backend selection
for AOT rows, and real multi-job callback execution for `chunk4`. Its tables
include full wall-clock, symbolic preparation, linear/Jacobian/residual times,
handoff and compile/link stages, callback job counts, and Frozen Newton counters.
If Frozen itself cannot converge this hard initial guess, the failure is evidence
about method coverage rather than a tolerance to weaken.

```powershell
cargo test --release --lib frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story -- --ignored --nocapture
```

```text
Date: release run recorded below.
Result: Frozen solved combustion-1000 in all six cold rows.  Both AOT variants
        agreed with Lambdify to `2.22e-16`.  `tcc/chunk4` reported
        `res_jobs=4` and `jac_jobs=4` on both repetitions, proving actual
        callback-level parallel execution.  Mean total times were approximately
        `468 ms` for Lambdify and `690 ms` for both `tcc/whole` and
        `tcc/chunk4`.
Conclusion: Frozen is now covered by a real heavy symbolic/AOT correctness
            route, and chunking is operational rather than silently sequential.
            For a cold single solve at this scale, AOT does not pay for itself:
            the entire numerical Newton phase is small compared with bootstrap,
            so chunking cannot move end-to-end time materially.
```

### `frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story`

This complementary Frozen story checks artifact lifecycle rather than cold
chunking economics. It runs a Lambdify baseline, one Banded AtomView `tcc`
`BuildIfMissing` solve, and three `RequirePrebuilt` solves using the resolver
snapshot created by the build row. The strict reuse rows must remain
`AotCompiled`, preserve the baseline solution, and avoid a compiler/linker build.

```powershell
cargo test --release --lib frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture
```

```text
Date: release run recorded below.
Result: The build row and all three strict prebuilt rows remained
        `AotCompiled`; each matched the Lambdify solution to `2.22e-16`.
        `BuildIfMissing` took about `745 ms`, while three `RequirePrebuilt`
        solves averaged `361 +/- 10 ms` with blank compile/link time.
        The single Lambdify baseline took `537 ms`.
Conclusion: Frozen's compiled artifact is reusable through the public lifecycle,
            and warm/prebuilt AOT is materially useful here.  On this run the
            build cost exceeds one Lambdify solve, but approximately two
            subsequent prebuilt solves are already enough to amortize that
            startup penalty.  This supports documenting AtomView+tcc AOT as a
            repeated-solve route, not a guaranteed one-shot win.
```

### `frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story`

This is the Sparse companion to the verified Frozen Banded lifecycle story. It
uses the same combustion-1000 equations and AtomView frontend, but runs through
the real Sparse solver route: a Sparse Lambdify baseline, one Sparse `tcc`
`BuildIfMissing` solve, and three strict `RequirePrebuilt` solves using the
resolver snapshot produced by the build row. It closes the remaining matrix-route
breadth question without mixing in cold chunking economics.

The gate requires every AOT row to select `AotCompiled`; every prebuilt row must
remain `RequirePrebuilt` and leave `compile_link` blank. Numerical agreement with
the Sparse Lambdify baseline must remain within `1e-5`.

```powershell
cargo test --release --lib frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture
```

```text
Date: 2026-05-27 (release).
Result: The Sparse Lambdify baseline solved in `585.910 ms`.  The tcc
        `BuildIfMissing` row selected `AotCompiled`, solved in `754.842 ms`,
        and reported the expected real `compile_link=157.091 ms`.  All three
        subsequent `RequirePrebuilt` rows remained `AotCompiled`, left
        `compile_link` blank, and averaged `393.085 +/- 7.184 ms`.
[BVP Frozen story] combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: correctness/backend selection
source   | variant    | selected_backend | build_policy    | solve_diff
----------------------------------------------------------------------------------
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | build      | AotCompiled      | BuildIfMissing  | 4.440892e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 4.440892e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 4.440892e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 4.440892e-16

[BVP Frozen story] combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: wall-clock and Newton stages; milliseconds
source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |  585.910 |     525.000 |    44.000 |  5.000 |  5.000 |     9 |      9 |      1
AOT      | build      |  754.842 |     684.000 |    46.000 |  4.000 |  6.000 |     9 |      9 |      1
AOT      | prebuilt   |  383.283 |     321.000 |    48.000 |  2.000 |  3.000 |     9 |      9 |      1
AOT      | prebuilt   |  400.300 |     338.000 |    48.000 |  2.000 |  2.000 |     9 |      9 |      1
AOT      | prebuilt   |  395.671 |     333.000 |    46.000 |  2.000 |  2.000 |     9 |      9 |      1

[BVP Frozen story] combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: generated handoff and compiled callback stages; milliseconds
source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs
------------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |          352.057 |          45.630 |       NaN |          NaN |      NaN |      NaN
AOT      | build      |          255.459 |          45.846 |    67.400 |      157.091 |    1.000 |    1.000
AOT      | prebuilt   |          268.449 |          36.269 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |          288.374 |          35.879 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |          266.026 |          35.957 |       NaN |          NaN |    1.000 |    1.000

[BVP Frozen story] combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: repeated-run summary; milliseconds
source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------
AOT      | build      |   754.842 +/- 0.000     |      684.000 +/- 0.000     |     46.000 +/- 0.000     | 4.440892e-16
AOT      | prebuilt   |   393.085 +/- 7.184     |      330.667 +/- 7.134     |     47.333 +/- 0.943     | 4.440892e-16
Lambdify | AtomView   |   585.910 +/- 0.000     |      525.000 +/- 0.000     |     44.000 +/- 0.000     | 0.000000e0
test numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story ... ok

Conclusion: Frozen AtomView+tcc artifact lifecycle is now release-confirmed
            for both Sparse and Banded matrix routes.  The Sparse compiled
            callback agrees with its Lambdify baseline to `4.440892e-16`, and
            strict prebuilt reuse performs no hidden compilation or fallback.
            Its warm total is promising relative to the single Lambdify row,
            but this test closes lifecycle/correctness coverage rather than
            establishing a precise Sparse warm speedup.
```

### `combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story`

The lifecycle test above proves reuse, but its single Lambdify baseline is not a
fair timing sample: in the recorded run that one row was unexpectedly slow.
This follow-up is the timing experiment for the repeated-solve claim. It creates
one Banded AtomView `tcc` artifact with `BuildIfMissing`, excludes that setup row
from the performance summary, then measures five Lambdify solves and five strict
`RequirePrebuilt` solves. The execution order alternates per repetition so neither
route always runs first. A pause before every measured row lets external compiler
processes, memory pressure, and CPU boost/thermal transients settle without
destroying the warm artifact being measured.

By default the pause is `5000 ms`. It can be changed without editing the test:

```powershell
$env:BVP_AOT_WARM_COOLDOWN_MS="5000"
cargo test --release --lib combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story -- --ignored --nocapture
Remove-Item Env:BVP_AOT_WARM_COOLDOWN_MS
```

Healthy interpretation requires every `prebuilt` row to report `AotCompiled`,
`RequirePrebuilt`, and blank `compile_link`, with solution differences at roundoff
level. Only after this paired summary should we state a warm AOT speedup over
Lambdify for Damped combustion-1000.

```text
Date: 2026-05-27 (release, `BVP_AOT_WARM_COOLDOWN_MS=5000`).
Result: The setup `BuildIfMissing` solve selected `AotCompiled`, finished in
        `733.905 ms`, and reported a real `compile_link=133.364 ms`.  All five
        measured `RequirePrebuilt` rows stayed `AotCompiled`, had blank
        `compile_link`, and agreed with Lambdify to `2.220446e-16`.
        With alternating order and the same cooldown, Lambdify measured
        `468.337 +/- 12.471 ms`, while prebuilt tcc measured
        `431.397 +/- 10.850 ms`.
Conclusion: The former `2424 ms` Lambdify row was a timing outlier, not a
            representative baseline.  In a controlled warm repeated-solve
            comparison, Banded AtomView+tcc `RequirePrebuilt` is about
            `36.94 ms` faster per solve than Lambdify (`~7.9%`, or `1.09x`).
            The difference comes predominantly from symbolic/setup work
            (`397.8 ms` versus `358.8 ms`), not from the linear solve
            (`16.0 ms` versus `16.4 ms`).  Relative to one Lambdify solve, the
            first AOT build costs about `265.6 ms` extra, so this observed run
            pays back after approximately eight subsequent prebuilt reuses.
            This closes the Damped warm timing question for combustion-1000:
            tcc AOT is a justified repeated-solve route, while a one-shot
            solve should still prefer Lambdify unless the generated artifact
            is otherwise needed.
```

## Closed Production Findings

- AtomView is the production symbolic frontend for the measured combustion
  family: it preserves solution quality while removing the severe ExprLegacy
  symbolic-Jacobian cost on both Sparse and Banded routes.
- Damped artifact stability is closed for AtomView+tcc combustion-1000 on both
  Sparse and Banded paths: both stayed
   `AotCompiled` through `BuildIfMissing -> RequirePrebuilt` release runs with
   roundoff-level solution agreement and no compile/link stage in prebuilt rows.
- Frozen Sparse and Banded lifecycle coverage is closed for combustion-1000:
  both routes preserve Lambdify parity and strict prebuilt reuse in release;
  Banded additionally has heavy cold whole/chunk4 evidence for actual four-job
  callback execution.
- The paired cooldown-controlled Damped warm comparison closes the earlier timing
  ambiguity: strict prebuilt Banded `tcc` is consistently about `7.9%` faster
  than Lambdify on the measured repeated-solve route, while cold and warm
  measurements must continue to be reported separately.

## Remaining Story Work

1. Confirm that `combustion_1000_aot_sparse_vs_banded_end_to_end_race` and
   `combustion_1000_end_to_end_banded_lapack_refine_statistics` still agree on
   solution quality and expose the expected banded speedup after the handoff and
   frontend optimizations.
2. Toolchain stability: determine whether remaining `gcc`, `zig`, or Rust AOT failures
   are real codegen
   issues, missing toolchains, or file-lock/dynamic-loader effects?
3. Rust AOT artifact scalability: very large generated Rust cdylibs can still fail
   during compilation before runtime evaluation begins. Keep this separate from
   C/Zig AOT runtime chunking and from the callback-parallelism guard.
4. Broaden Frozen beyond combustion, for example
   with one qualitatively different hard BVP. This is coverage breadth rather than
   a backend/lifecycle blocker.

## Auto Chunking Policy Note

The AOT story should no longer require users to infer manually whether `whole`
or chunked callbacks are appropriate from several release tables.  The codegen
layer now exposes a structured `Auto` plan with the quantities that matter:
worker count, minimum useful work/job derived from measured Rayon overhead,
residual and Jacobian chunk counts, selected grouped jobs, work/job,
work/chunk, and a reason label.

For solver-level BVP Damp/Frozen execution this means:

- If the user explicitly requests `whole`, `chunk4`, `chunk8`, or another
  chunking strategy, the story/perf test remains forced and diagnostic.
- If the user leaves chunking on `Auto` and uses `BuildIfMissing` or
  `RebuildAlways`, the handoff may rebuild the prepared bundle with coarse
  chunking only when the workload plan says that parallel execution is
  justified.
- Runtime statistics now include both the planned Auto decision (`aot.auto.*`)
  and the actual linked runtime behavior (`aot.runtime.*`).  Future story tables
  should prefer these native diagnostics over hand-written wrappers.



cargo test --release --lib combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture

Result
[BVP Damp lifecycle] combustion-1000 AtomView tcc BuildIfMissing -> RequirePrebuilt correctness
matrix | phase      | selected_backend | build_policy    | solve_diff
------------------------------------------------------------------------------------
Banded | baseline   | Lambdify         | UseIfAvailable  | 0.000000e0
Sparse | build      | AotCompiled      | BuildIfMissing  | 2.730672e-15
Sparse | prebuilt   | AotCompiled      | RequirePrebuilt | 2.730672e-15
Sparse | prebuilt   | AotCompiled      | RequirePrebuilt | 2.730672e-15
Sparse | prebuilt   | AotCompiled      | RequirePrebuilt | 2.730672e-15
Banded | build      | AotCompiled      | BuildIfMissing  | 2.220446e-16
Banded | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16
Banded | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16
Banded | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16

[BVP Damp lifecycle] wall-clock and artifact stages; all time columns are milliseconds
matrix | phase      | total_ms | symbolic_ms | linear_ms | initial_generate | rebind_ms | compile_link
----------------------------------------------------------------------------------------------------------------
Banded | baseline   | 2424.496 |    2000.000 |    23.000 |          310.658 |       NaN |          NaN
Sparse | build      |  783.624 |     676.000 |    45.000 |          226.818 |    61.406 |      136.771
Sparse | prebuilt   |  467.590 |     354.000 |    47.000 |          256.670 |       NaN |          NaN
Sparse | prebuilt   |  437.642 |     338.000 |    43.000 |          241.951 |       NaN |          NaN
Sparse | prebuilt   |  422.194 |     325.000 |    43.000 |          238.324 |       NaN |          NaN
Banded | build      |  654.426 |     588.000 |    15.000 |          224.482 |    59.661 |      117.124
Banded | prebuilt   |  437.881 |     340.000 |    22.000 |          243.534 |       NaN |          NaN
Banded | prebuilt   |  396.882 |     331.000 |    15.000 |          240.987 |       NaN |          NaN
Banded | prebuilt   |  399.590 |     330.000 |    15.000 |          238.324 |       NaN |          NaN
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story ... ok


cargo test --release --lib frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story -- --ignored --nocapture


[BVP Frozen story] combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes: correctness/backend selection
source   | variant    | selected_backend | build_policy    | solve_diff
----------------------------------------------------------------------------------
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | tcc/whole  | AotCompiled      | RebuildAlways   | 2.220446e-16
AOT      | tcc/chunk4 | AotCompiled      | RebuildAlways   | 2.220446e-16
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | tcc/whole  | AotCompiled      | RebuildAlways   | 2.220446e-16
AOT      | tcc/chunk4 | AotCompiled      | RebuildAlways   | 2.220446e-16

[BVP Frozen story] combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes: wall-clock and Newton stages; milliseconds
source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |  551.174 |     518.000 |    21.000 |  1.000 |  4.000 |     9 |      9 |      1
AOT      | tcc/whole  |  648.996 |     606.000 |    25.000 |  1.000 |  3.000 |     9 |      9 |      1
AOT      | tcc/chunk4 |  724.202 |     682.000 |    22.000 |  1.000 |  4.000 |     9 |      9 |      1
Lambdify | AtomView   |  384.430 |     352.000 |    19.000 |  1.000 |  3.000 |     9 |      9 |      1
AOT      | tcc/whole  |  731.726 |     691.000 |    21.000 |  1.000 |  3.000 |     9 |      9 |      1
AOT      | tcc/chunk4 |  656.908 |     622.000 |    19.000 |  1.000 |  2.000 |     9 |      9 |      1

[BVP Frozen story] combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes: generated handoff and compiled callback stages; milliseconds
source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs
------------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |          333.860 |          44.775 |       NaN |          NaN |      NaN |      NaN
AOT      | tcc/whole  |          246.211 |          35.751 |    71.478 |      125.007 |    1.000 |    1.000
AOT      | tcc/chunk4 |          246.389 |          37.216 |    70.880 |      118.963 |    4.000 |    4.000
Lambdify | AtomView   |          313.779 |          35.852 |       NaN |          NaN |      NaN |      NaN
AOT      | tcc/whole  |          240.288 |          38.187 |    66.017 |      193.398 |    1.000 |    1.000
AOT      | tcc/chunk4 |          265.347 |          41.705 |    68.624 |      115.381 |    4.000 |    4.000

[BVP Frozen story] combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes: repeated-run summary; milliseconds
source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------
AOT      | tcc/chunk4 |   690.555 +/- 33.647    |      652.000 +/- 30.000    |     20.500 +/- 1.500     | 2.220446e-16
AOT      | tcc/whole  |   690.361 +/- 41.365    |      648.500 +/- 42.500    |     23.000 +/- 2.000     | 2.220446e-16
Lambdify | AtomView   |   467.802 +/- 83.372    |      435.000 +/- 83.000    |     20.000 +/- 1.000     | 0.000000e0
test numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story ... ok


cargo test --release --lib frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture


[BVP Frozen story] combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: correctness/backend selection
source   | variant    | selected_backend | build_policy    | solve_diff
----------------------------------------------------------------------------------
Lambdify | AtomView   | Lambdify         | UseIfAvailable  | 0.000000e0
AOT      | build      | AotCompiled      | BuildIfMissing  | 2.220446e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16
AOT      | prebuilt   | AotCompiled      | RequirePrebuilt | 2.220446e-16

[BVP Frozen story] combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: wall-clock and Newton stages; milliseconds
source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re
----------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |  537.323 |     495.000 |    28.000 |  1.000 |  3.000 |     9 |      9 |      1
AOT      | build      |  744.947 |     708.000 |    22.000 |  1.000 |  3.000 |     9 |      9 |      1
AOT      | prebuilt   |  372.814 |     329.000 |    28.000 |  0.000 |  4.000 |     9 |      9 |      1
AOT      | prebuilt   |  348.924 |     310.000 |    23.000 |  1.000 |  2.000 |     9 |      9 |      1
AOT      | prebuilt   |  361.511 |     322.000 |    22.000 |  1.000 |  2.000 |     9 |      9 |      1

[BVP Frozen story] combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: generated handoff and compiled callback stages; milliseconds
source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs
------------------------------------------------------------------------------------------------------------------------
Lambdify | AtomView   |          325.909 |          38.282 |       NaN |          NaN |      NaN |      NaN
AOT      | build      |          270.829 |          46.461 |    68.943 |      135.139 |    1.000 |    1.000
AOT      | prebuilt   |          271.590 |          37.085 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |          256.659 |          36.019 |       NaN |          NaN |    1.000 |    1.000
AOT      | prebuilt   |          268.125 |          35.897 |       NaN |          NaN |    1.000 |    1.000

[BVP Frozen story] combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: repeated-run summary; milliseconds
source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------
AOT      | build      |   744.947 +/- 0.000     |      708.000 +/- 0.000     |     22.000 +/- 0.000     | 2.220446e-16
AOT      | prebuilt   |   361.083 +/- 9.758     |      320.333 +/- 7.846     |     24.333 +/- 2.625     | 2.220446e-16
Lambdify | AtomView   |   537.323 +/- 0.000     |      495.000 +/- 0.000     |     28.000 +/- 0.000     | 0.000000e0
test numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story ... ok






cargo test --release --lib combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story -- --ignored --nocapture


[BVP Damp warm] Banded AtomView Lambdify vs tcc RequirePrebuilt; setup build row
phase | selected_backend | build_policy    | total_ms | symbolic_ms | rebind_ms | compile_link | solve_diff
------------------------------------------------------------------------------------------------------------------------
build | AotCompiled      | BuildIfMissing  |  733.905 |     653.000 |    69.242 |      133.364 | 2.220446e-16

[BVP Damp warm] measured rows after cooldown_ms=5000; milliseconds
rep | pos | phase      | selected_backend | build_policy    | total_ms | symbolic_ms | linear_ms | initial_generate | compile_link | solve_diff
----------------------------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | lambdify   | Lambdify         | UseIfAvailable  |  482.297 |     406.000 |    18.000 |          319.857 |          NaN | 0.000000e0
  1 |   2 | prebuilt   | AotCompiled      | RequirePrebuilt |  431.213 |     356.000 |    18.000 |          254.684 |          NaN | 2.220446e-16
  2 |   1 | prebuilt   | AotCompiled      | RequirePrebuilt |  452.425 |     382.000 |    17.000 |          273.741 |          NaN | 2.220446e-16
  2 |   2 | lambdify   | Lambdify         | UseIfAvailable  |  462.140 |     397.000 |    15.000 |          302.744 |          NaN | 0.000000e0
  3 |   1 | lambdify   | Lambdify         | UseIfAvailable  |  484.443 |     415.000 |    16.000 |          316.599 |          NaN | 0.000000e0
  3 |   2 | prebuilt   | AotCompiled      | RequirePrebuilt |  424.810 |     353.000 |    15.000 |          253.830 |          NaN | 2.220446e-16
  4 |   1 | prebuilt   | AotCompiled      | RequirePrebuilt |  425.099 |     352.000 |    17.000 |          252.196 |          NaN | 2.220446e-16
  4 |   2 | lambdify   | Lambdify         | UseIfAvailable  |  456.626 |     387.000 |    16.000 |          304.847 |          NaN | 0.000000e0
  5 |   1 | lambdify   | Lambdify         | UseIfAvailable  |  456.181 |     384.000 |    15.000 |          300.296 |          NaN | 0.000000e0
  5 |   2 | prebuilt   | AotCompiled      | RequirePrebuilt |  423.439 |     351.000 |    15.000 |          243.429 |          NaN | 2.220446e-16

[BVP Damp warm] paired summary: build row excluded; each route has the same cooldown and alternating order
phase      | runs | total_ms mean+/-std [min,max] | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff
------------------------------------------------------------------------------------------------------------------------------------------------------
lambdify   |    5 | 468.337 +/- 12.471 [456.181, 484.443] | 397.800 +/- 11.583    | 16.000 +/- 1.095      | 0.000000e0
prebuilt   |    5 | 431.397 +/- 10.850 [423.439, 452.425] | 358.800 +/- 11.720    | 16.400 +/- 1.200      | 2.220446e-16
test numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story ... ok

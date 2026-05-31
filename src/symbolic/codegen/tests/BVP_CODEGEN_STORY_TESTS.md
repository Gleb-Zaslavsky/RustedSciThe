# BVP Codegen Correctness And Performance Story Tests

This document is a map of the existing BVP code generation diagnostics.  It is
not a replacement for the tests themselves.  Its purpose is to keep the
questions, commands, and release-run conclusions in one place, so we do not
rebuild the same diagnostic machinery in solver-level story tests.

The codegen tests answer a different question than the end-to-end BVP damped
solver stories.  Here we mostly isolate generated residual and Jacobian
callbacks, build latency, runtime throughput, chunking, and parallel executor
economics.  Solver-level tests such as `BVP_DAMP_STORY_TESTS.md` then answer how
these callbacks behave inside Newton iterations and linear solves.

## How To Read The Layers

There are three layers in the current BVP codegen test suite.

The first layer is semantic correctness: lambdified residuals and Jacobians,
IR-level AOT evaluation, compiled fixture callbacks, sparse values, and banded
values must agree on the same inputs.  If this layer fails, performance numbers
are meaningless.

The second layer is callable economics: symbolic preparation, IR lowering,
source generation, compiler build/link, runtime linking, and callback execution
are measured separately.  This is where we decide when AOT pays back its
bootstrap cost.

The third layer is parallel-executor economics: chunk count, requested jobs,
actual jobs, fallback policy, Rayon overhead, and work per job are examined
without the full Newton solve in the way.  This is the cleanest place to ask
whether chunking should help at all.

## Running Notes

Performance diagnostics should be interpreted from release builds.  Debug builds
are useful for smoke checks and diagnostics, but they are not evidence for a
break-even point.

`--test-threads=1` only serializes the Rust test harness.  It does not disable
Rayon parallelism inside the codegen executors.  Use it when comparing noisy
tables or when AOT artifacts are being built in the same target directory.

The default Rayon pool size is controlled by the process environment.  To check
scaling explicitly, run the same command with different `RAYON_NUM_THREADS`
values.

```powershell
$env:RAYON_NUM_THREADS="4"
cargo test --release diagnose_chunk_granularity_and_fallback -- --nocapture
Remove-Item Env:RAYON_NUM_THREADS
```

Toolchain discovery for generated backends uses the normal executable lookup and
also honors these environment variables when present: `RUSTEDSCITHE_TCC`,
`RUSTEDSCITHE_GCC`, and `RUSTEDSCITHE_ZIG`.

## Correctness Gates

### `bvp_sparse_banded_correctness_matrix_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test bvp_sparse_banded_correctness_matrix_table -- --nocapture
```

This is the first gate for sparse/banded generated values.  It compares the
current BVP residual and Jacobian outputs across lambdified and generated paths
and should stay non-ignored because it is a correctness test, not a performance
story.

Current result:

```text
TODO: paste latest debug/release summary here.
```

Conclusion:

```text
TODO: record whether sparse/banded generated values match the lambdify baseline.
```

### `real_bvp_parallel2_and_generated_aot_match_exactly_on_same_inputs`

File: `src/symbolic/codegen/tests/codegen_bvp_performance_tests.rs`

Command:

```powershell
cargo test real_bvp_parallel2_and_generated_aot_match_exactly_on_same_inputs -- --nocapture
```

This gate checks that the generated AOT path and the legacy `parallel2`
lambdified path agree on the same real BVP fixture.  It is a semantic bridge
between the old and new execution machinery.

Current result:

```text
TODO: paste latest result here.
```

Conclusion:

```text
TODO: record whether this remains a safe semantic baseline.
```

## Stage And Runtime Tables

### `benchmark_real_bvp_pipeline_stage_timings`

File: `src/symbolic/codegen/tests/codegen_bvp_performance_tests.rs`

Command:

```powershell
cargo test --release benchmark_real_bvp_pipeline_stage_timings -- --nocapture
```

This is a small real-BVP stage table.  It separates symbolic construction,
lambdify work, AOT planning, AOT assembly, and emitted-source cost.  Use it when
we need a quick regression check that preparation stages did not unexpectedly
move.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: summarize which stage dominates on the small real BVP case.
```

### `benchmark_bvp_sparse_vs_banded_lambdify_and_ir_table`

File: `src/symbolic/codegen/tests/codegen_bvp_performance_tests.rs`

Command:

```powershell
cargo test --release benchmark_bvp_sparse_vs_banded_lambdify_and_ir_table -- --nocapture
```

This table compares sparse and banded callback evaluation at the lambdify/IR
level.  It is useful before entering the full solver because it isolates value
generation from linear solves.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record whether sparse or banded value generation is the bottleneck.
```

### `benchmark_bvp_sparse_vs_banded_linear_solve_table`

File: `src/symbolic/codegen/tests/codegen_bvp_performance_tests.rs`

Command:

```powershell
cargo test --release benchmark_bvp_sparse_vs_banded_linear_solve_table -- --nocapture
```

This table compares the downstream linear algebra side.  It should be read
together with the callback table above: a faster Jacobian callback does not
matter if the linear solve dominates the full Newton step.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: summarize when banded linear algebra beats sparse on the current fixtures.
```

## Generated Backend Toolchain Matrix

### `bvp_atomview_aot_optimization_profile_bootstrap_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
$env:BVP_PIPELINE_RUNS="4"
cargo test --release bvp_atomview_aot_optimization_profile_bootstrap_table -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_PIPELINE_RUNS
```

This table isolates the second AOT bootstrap stage: `AtomView -> IR -> emitted
source/artifact`. It deliberately stops before materialization, compiler build,
runtime linking, and first callback evaluation. The goal is to answer a narrower
question than the full pipeline table: do IR cleanup passes pay for themselves
when the user cares about time-to-first-solve?

The default `Full` profile is the historical behavior. `FastBootstrap` skips
the optional peephole/temporary-reuse cleanup passes during Atom lowering, while
`NoPeephole` and `NoTempReuse` are diagnostic splits used to see which pass is
actually expensive. At the time this section was added, BVP `Auto` temporary
reuse was already conservative for residual vectors and sparse values, so the
main expected delta is `peephole_ms`.

Current result:

```text
test symbolic::codegen::tests::codegen_bvp_backend_comparison_tests::bvp_generated_backend_pipeline_comparison_table ... [BVP backend compare] starting pipeline comparison test

[BVP backend pipeline compare] scenario=small-damp1-24, residuals=48, vars=48, nnz=118, multi-run bootstrap summary
route    | assembly   | variant        | preset      | ok/runs | symbolic_ms mean+/-std [min,max] | callable_prep_ms mean+/-std [min,max] | artifact_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms mean+/-std [min,max] | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         |  10/10  | 2.158+/-0.424 [1.653,3.338]        | 0.125+/-0.013 [0.112,0.148]             | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.005+/-0.000  | 2.289+/-0.419 [1.786,3.454]                 | ok 10/10
AOT      | ExprLegacy | Rust           | FastBuild   |  10/10  | 2.158+/-0.424 [1.653,3.338]        | 599.088+/-49.227 [547.565,680.555]      | 0.678+/-0.043 | 2.120+/-0.191  | 595.753+/-49.265 | 0.537+/-0.027 | 0.018+/-0.001  | 601.264+/-49.311 [549.613,682.812]          | ok 10/10
AOT      | ExprLegacy | C-gcc          | Production  |  10/10  | 2.158+/-0.424 [1.653,3.338]        | 417.044+/-52.383 [375.408,535.749]      | 0.387+/-0.033 | 2.882+/-0.088  | 401.849+/-52.026 | 11.926+/-15.805 | 0.004+/-0.001  | 419.207+/-52.644 [377.524,539.095]          | ok 10/10
AOT      | ExprLegacy | C-tcc          | Production  |  10/10  | 2.158+/-0.424 [1.653,3.338]        | 24.635+/-5.320 [21.233,39.905]          | 0.314+/-0.033 | 3.606+/-1.879  | 19.957+/-3.288 | 0.757+/-0.319 | 0.014+/-0.002  | 26.807+/-5.434 [23.328,42.137]              | ok 10/10
AOT      | ExprLegacy | Zig            | Production  |  10/10  | 2.158+/-0.424 [1.653,3.338]        | 18536.645+/-842.059 [17539.327,20261.117] | 0.592+/-0.043 | 2.012+/-0.744  | 18532.635+/-842.160 | 1.406+/-0.215 | 0.011+/-0.002  | 18538.814+/-842.225 [17541.367,20263.129]   | ok 10/10
AtomView-only planning stages. ExprLegacy rows are expected to be zero here; use the module/source table below for the active legacy module-build cost.
route    | assembly   | variant        | preset      | jac_prepare | lookup | jac_build | chunk_plan | lower | peephole | temp_reuse | module_push
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | FastBuild   | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-gcc          | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-tcc          | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Zig            | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
route    | assembly   | variant        | preset      | module_ms | module_init | residual_lower | jacobian_lower | source_probe | source_emit | c_header | packaging | artifact_other | source_kb
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | FastBuild   | 0.276+/-0.032 | 0.001+/-0.002 | 0.237+/-0.024  | 0.130+/-0.044  | 0.000+/-0.000 | 0.332+/-0.016 | 0.000+/-0.000 | 0.023+/-0.007 | 0.047+/-0.021  | 13.685+/-0.000
AOT      | ExprLegacy | C-gcc          | Production  | 0.253+/-0.034 | 0.001+/-0.000 | 0.224+/-0.030  | 0.095+/-0.027  | 0.000+/-0.000 | 0.074+/-0.007 | 0.007+/-0.005 | 0.028+/-0.005 | 0.025+/-0.010  | 12.638+/-0.000
AOT      | ExprLegacy | C-tcc          | Production  | 0.199+/-0.033 | 0.001+/-0.000 | 0.162+/-0.021  | 0.069+/-0.019  | 0.000+/-0.000 | 0.067+/-0.005 | 0.003+/-0.000 | 0.022+/-0.002 | 0.023+/-0.005  | 12.638+/-0.000
AOT      | ExprLegacy | Zig            | Production  | 0.187+/-0.018 | 0.001+/-0.000 | 0.163+/-0.016  | 0.064+/-0.012  | 0.000+/-0.000 | 0.361+/-0.031 | 0.000+/-0.000 | 0.019+/-0.002 | 0.025+/-0.004  | 14.971+/-0.000
route    | assembly   | variant        | preset      | residual_diff | jacobian_diff | status
---------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | Rust           | FastBuild   | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-gcc          | Production  | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-tcc          | Production  | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | Zig            | Production  | 5.551115e-17  | 0.000000e0    | ok 10/10
[BVP backend compare] pipeline comparison finished scenario `small-damp1-24`
[BVP backend compare] pipeline comparison entering scenario `combustion-100`

[BVP backend pipeline compare] scenario=combustion-100, residuals=600, vars=600, nnz=2088, multi-run bootstrap summary
route    | assembly   | variant        | preset      | ok/runs | symbolic_ms mean+/-std [min,max] | callable_prep_ms mean+/-std [min,max] | artifact_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms mean+/-std [min,max] | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         |  10/10  | 71.733+/-6.740 [63.389,85.435]     | 4.851+/-0.157 [4.696,5.229]             | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.161+/-0.018  | 76.746+/-6.779 [68.264,90.400]              | ok 10/10
AOT      | ExprLegacy | Rust           | FastBuild   |  10/10  | 71.733+/-6.740 [63.389,85.435]     | 10245.334+/-490.843 [9846.877,11570.242] | 13.576+/-1.088 | 3.830+/-1.262  | 10227.323+/-490.861 | 0.605+/-0.013 | 0.184+/-0.093  | 10317.252+/-492.674 [9916.384,11644.439]    | ok 10/10
AOT      | ExprLegacy | C-gcc          | Production  |  10/10  | 71.733+/-6.740 [63.389,85.435]     | 5216.773+/-449.272 [4893.689,6486.652]  | 6.837+/-0.096 | 3.790+/-0.787  | 5159.535+/-457.344 | 46.610+/-75.011 | 0.103+/-0.006  | 5288.609+/-447.611 [4959.407,6552.078]      | ok 10/10
AOT      | ExprLegacy | C-tcc          | Production  |  10/10  | 71.733+/-6.740 [63.389,85.435]     | 52.804+/-28.039 [41.010,136.594]        | 7.132+/-0.889 | 3.860+/-0.628  | 30.694+/-1.614 | 11.118+/-27.401 | 0.249+/-0.012  | 124.786+/-32.879 [104.634,222.274]          | ok 10/10
AOT      | ExprLegacy | Zig            | Production  |  10/10  | 71.733+/-6.740 [63.389,85.435]     | 19819.203+/-514.907 [19312.987,21301.961] | 13.825+/-0.776 | 2.401+/-0.309  | 19801.119+/-514.393 | 1.858+/-0.817 | 0.115+/-0.018  | 19891.051+/-512.516 [19393.382,21367.399]   | ok 10/10
AtomView-only planning stages. ExprLegacy rows are expected to be zero here; use the module/source table below for the active legacy module-build cost.
route    | assembly   | variant        | preset      | jac_prepare | lookup | jac_build | chunk_plan | lower | peephole | temp_reuse | module_push
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | FastBuild   | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-gcc          | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-tcc          | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Zig            | Production  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
route    | assembly   | variant        | preset      | module_ms | module_init | residual_lower | jacobian_lower | source_probe | source_emit | c_header | packaging | artifact_other | source_kb
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | FastBuild   | 4.881+/-0.810 | 0.001+/-0.000 | 4.813+/-0.710  | 1.979+/-0.148  | 0.000+/-0.000 | 7.617+/-0.676 | 0.000+/-0.000 | 0.494+/-0.037 | 0.583+/-0.079  | 316.604+/-0.000
AOT      | ExprLegacy | C-gcc          | Production  | 4.660+/-0.097 | 0.001+/-0.000 | 4.612+/-0.100  | 1.968+/-0.132  | 0.000+/-0.000 | 1.332+/-0.026 | 0.009+/-0.005 | 0.523+/-0.059 | 0.314+/-0.046  | 307.663+/-0.000
AOT      | ExprLegacy | C-tcc          | Production  | 4.999+/-0.891 | 0.001+/-0.000 | 4.810+/-0.754  | 2.109+/-0.699  | 0.000+/-0.000 | 1.349+/-0.066 | 0.006+/-0.002 | 0.482+/-0.035 | 0.297+/-0.074  | 307.663+/-0.000
AOT      | ExprLegacy | Zig            | Production  | 4.566+/-0.242 | 0.001+/-0.000 | 4.515+/-0.243  | 1.961+/-0.232  | 0.000+/-0.000 | 8.367+/-0.615 | 0.000+/-0.000 | 0.468+/-0.015 | 0.424+/-0.070  | 341.867+/-0.000
route    | assembly   | variant        | preset      | residual_diff | jacobian_diff | status
---------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | Rust           | FastBuild   | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-gcc          | Production  | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-tcc          | Production  | 5.551115e-17  | 1.110223e-16  | ok 10/10
AOT      | ExprLegacy | Zig            | Production  | 5.551115e-17  | 1.110223e-16  | ok 10/10
[BVP backend compare] pipeline comparison finished scenario `combustion-100`
[BVP backend compare] pipeline comparison entering scenario `combustion-1000`

[BVP backend pipeline compare] scenario=combustion-1000, residuals=6000, vars=6000, nnz=20988, multi-run bootstrap summary
route    | assembly   | variant        | preset      | ok/runs | symbolic_ms mean+/-std [min,max] | callable_prep_ms mean+/-std [min,max] | artifact_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms mean+/-std [min,max] | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         |  10/10  | 2137.908+/-287.723 [1772.195,2695.938] | 350.529+/-143.590 [271.656,772.735]     | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 2.270+/-0.283  | 2490.706+/-402.817 [2056.421,3471.703]      | ok 10/10
AOT      | ExprLegacy | Rust           | DevFastest  |  10/10  | 2137.908+/-287.723 [1772.195,2695.938] | 11759.294+/-1215.859 [10408.027,14321.918] | 142.565+/-32.022 | 8.221+/-2.062  | 11607.355+/-1190.768 | 1.153+/-0.124 | 5.563+/-1.810  | 13902.765+/-1470.211 [12336.405,17022.248]  | ok 10/10
AOT      | ExprLegacy | C-gcc          | DevFastest  |  10/10  | 2137.908+/-287.723 [1772.195,2695.938] | 6390.915+/-447.515 [5901.821,7290.051]  | 69.863+/-4.743 | 5.857+/-0.796  | 6269.952+/-413.716 | 45.243+/-45.267 | 4.176+/-1.804  | 8532.999+/-605.280 [7857.144,9656.449]      | ok 10/10
AOT      | ExprLegacy | C-tcc          | DevFastest  |  10/10  | 2137.908+/-287.723 [1772.195,2695.938] | 236.255+/-15.290 [215.254,270.294]      | 70.663+/-5.219 | 6.348+/-1.237  | 147.085+/-12.506 | 12.158+/-2.602 | 2.538+/-0.461  | 2376.700+/-295.540 [2018.570,2934.241]      | ok 10/10
AOT      | ExprLegacy | Zig            | DevFastest  |  10/10  | 2137.908+/-287.723 [1772.195,2695.938] | 46298.748+/-2561.034 [43789.276,52018.183] | 147.581+/-14.036 | 4.458+/-0.767  | 46134.870+/-2559.826 | 11.840+/-2.415 | 4.949+/-2.720  | 48441.605+/-2529.404 [45713.656,53792.682]  | ok 10/10
AtomView-only planning stages. ExprLegacy rows are expected to be zero here; use the module/source table below for the active legacy module-build cost.
route    | assembly   | variant        | preset      | jac_prepare | lookup | jac_build | chunk_plan | lower | peephole | temp_reuse | module_push
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-gcc          | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-tcc          | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Zig            | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
route    | assembly   | variant        | preset      | module_ms | module_init | residual_lower | jacobian_lower | source_probe | source_emit | c_header | packaging | artifact_other | source_kb
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | DevFastest  | 51.201+/-7.885 | 0.001+/-0.000 | 51.136+/-7.895 | 22.752+/-3.112 | 0.000+/-0.000 | 83.021+/-22.552 | 0.000+/-0.000 | 5.554+/-1.043 | 2.788+/-1.219  | 3376.271+/-0.000
AOT      | ExprLegacy | C-gcc          | DevFastest  | 49.662+/-4.090 | 0.001+/-0.000 | 49.573+/-4.068 | 23.207+/-3.873 | 0.000+/-0.000 | 13.099+/-0.369 | 0.010+/-0.002 | 4.892+/-0.156 | 2.199+/-0.575  | 3367.721+/-0.000
AOT      | ExprLegacy | C-tcc          | DevFastest  | 50.237+/-4.381 | 0.001+/-0.000 | 50.186+/-4.388 | 22.940+/-3.010 | 0.000+/-0.000 | 13.016+/-0.477 | 0.009+/-0.002 | 5.307+/-1.091 | 2.094+/-0.566  | 3367.721+/-0.000
AOT      | ExprLegacy | Zig            | DevFastest  | 52.762+/-8.835 | 0.001+/-0.000 | 52.696+/-8.820 | 24.379+/-5.921 | 0.000+/-0.000 | 87.218+/-5.523 | 0.000+/-0.000 | 5.090+/-0.584 | 2.511+/-0.818  | 3633.540+/-0.000
route    | assembly   | variant        | preset      | residual_diff | jacobian_diff | status
---------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | Rust           | DevFastest  | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-gcc          | DevFastest  | 0.000000e0    | 0.000000e0    | ok 10/10
AOT      | ExprLegacy | C-tcc          | DevFastest  | 4.440892e-16  | 6.661338e-16  | ok 10/10
AOT      | ExprLegacy | Zig            | DevFastest  | 4.440892e-16  | 2.220446e-16  | ok 10/10
[BVP backend compare] pipeline comparison finished scenario `combustion-1000`

ok
```text
Latest release table is pasted above.  The combustion-1000 run is the current
best source for cold-bootstrap economics because it includes Lambdify, Rust,
C-gcc, C-tcc, and Zig with ten runs each.
```

Conclusion:

```text
For combustion-1000, the common symbolic stage costs about 1.90 s for every
route. Lambdify then needs about 0.35 s of callable preparation and reaches
first callable outputs in about 2.26 s. This means a cold AOT route must keep
its whole artifact/materialize/build/link overhead very small to beat Lambdify
on a one-shot solve.

C-tcc is the only AOT route currently close to Lambdify for cold one-shot use:
about 2.82 s total-to-outputs. The stage that pulls it down is not the external
compiler. The measured C-tcc split is: artifact/source/module work ~0.64 s,
materialization ~0.11 s, tcc build ~0.15 s, dynamic link ~0.009 s, and first
callback issue ~0.002 s. In other words, tcc itself is already fast; the main
AOT tax is the pre-compiler factory that constructs and emits the generated
artifact.

Inside the C-tcc artifact block, the largest visible substage is module
construction (~0.41 s), followed by source-size probe plus final source emission
(~0.18 s total), then packaging (~0.045 s). `artifact_other` is now tiny
(~0.002 s), so the former opaque container is mostly explained. The practical
optimization target is therefore module construction/source emission, not
peephole/temp-reuse toggles and not tcc compilation.

Implementation note: the artifact path has now been changed so the source-size
probe is not paid during real AOT artifact creation.  Module-only diagnostics
can still measure `source_probe`, but generated artifacts now emit language
source once and fill `source_kb` from that real source string.  Re-running this
table should therefore show `source_probe ~= 0` for AOT artifact rows and should
move roughly the old probe cost out of `artifact_ms`/`callable_prep_ms`.

The source emitter also now reserves a conservative output buffer based on
module size before pushing generated code into the final `String`.  This is a
low-risk allocation optimization; it should not change generated source, but it
may shave part of `source_emit` noise on large BVP modules.

The next diagnostic refinement is also wired: the module/source table now splits
`module_ms` into `module_init`, `residual_lower`, and `jacobian_lower` for the
ExprLegacy prepared-problem driver.  This should show whether the remaining
~0.4 s module stage is dominated by residual lowering or sparse-Jacobian
lowering before we optimize it.

The prepared-problem driver now builds residual blocks and Jacobian blocks as
independent Rayon branches and then appends them to the module in the original
deterministic order.  This targets the visible `residual_lower + jacobian_lower`
sum without changing emitted source semantics.  In the next release run,
`module_ms` should move closer to `max(residual_lower, jacobian_lower)` than to
their sum.

The C/Rust/Zig materialization paths also stopped rebuilding a second giant
generated-source string just to add file prologues.  They now stream the small
prologue and the already-emitted source/header body directly into the file.
This targets `materialize_ms` without changing the generated file layout.

The C source path now also writes generated functions directly into the final
module source buffer instead of building a temporary `String` per block and
copying it afterwards.  The C emitter additionally writes temporary names
(`t0`, `t1`, ...) directly into the output buffer instead of allocating one
small `String` per IR operand.  These are source-emission optimizations only:
`generated_aot_artifact_skips_source_probe_without_changing_c_source` verifies
that the generated C source remains byte-for-byte unchanged.

C-gcc, Rust, and Zig are dominated by compiler time on this fixture:
~6.47 s for gcc, ~11.43 s for Rust, and ~47.09 s for Zig. They may still be
runtime-throughput candidates for repeated solves, but they are not attractive
for cold one-shot BVP bootstrap here.

Diagnostic caveat: this pipeline table is currently an ExprLegacy pipeline, so
the Atom/codegen planning table is intentionally zero. The `assembly` column now
makes this explicit. AtomView-specific planning costs should be read from
`bvp_atomview_aot_optimization_profile_bootstrap_table`; this table is the
source of truth for Lambdify-vs-AOT cold-start economics.
```

### `bvp_generated_backend_runtime_comparison_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test --release bvp_generated_backend_runtime_comparison_table -- --ignored --nocapture --test-threads=1
```

This isolates runtime callback throughput after bootstrap has already been paid.
It compares lambdify with generated backends across available compilers.

Current result:
test symbolic::codegen::tests::codegen_bvp_backend_comparison_tests::bvp_generated_backend_runtime_comparison_table ... [BVP backend compare] starting runtime comparison test
[BVP backend compare] probing available generated backends
[BVP backend compare] probing C compiler `gcc`
[BVP backend compare] probing C compiler `tcc`
[BVP backend compare] probing Zig compiler `zig`
[BVP backend compare] finished backend probe: 4 variant(s)
[BVP backend compare] detected variants: Rust, C-gcc, C-tcc, Zig
[BVP backend compare] probes: gcc=detected, tcc=detected, zig=detected
[BVP backend compare] runtime comparison entering scenario `small-damp1-24`
[BVP backend compare] building symbolic scenario `small-damp1-24`
creating discretization equations
╭─────────────────────────────┬─────╮
│ bc handling                 │ 0   │
├─────────────────────────────┼─────┤
│ consistency test            │ 0   │
├─────────────────────────────┼─────┤
│ flat list creation          │ 0   │
├─────────────────────────────┼─────┤
│ discretization of equations │ 100 │
├─────────────────────────────┼─────┤
│ bounds and tolerances       │ 0   │
├─────────────────────────────┼─────┤
│ BC application              │ 0   │
├─────────────────────────────┼─────┤
│ total time, sec             │ 1   │
╰─────────────────────────────┴─────╯
[BVP backend compare] scenario `small-damp1-24` ready: residuals=48, vars=48, nnz=118
[BVP backend compare] measuring lambdify baseline for scenario `small-damp1-24`
[BVP backend compare] lambdify baseline for scenario `small-damp1-24` finished: total 0.770 ms
[BVP backend compare] building backend `Rust` for scenario `small-damp1-24`
[BVP backend compare] backend `Rust` for scenario `small-damp1-24` finished with status `ok` (build 898.716 ms, total 912.307 ms)
[BVP backend compare] measuring linked runtime for `Rust` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Rust` on scenario `small-damp1-24` finished: total 0.265 ms
[BVP backend compare] building backend `C-gcc` for scenario `small-damp1-24`
[BVP backend compare] backend `C-gcc` for scenario `small-damp1-24` finished with status `ok` (build 622.578 ms, total 816.335 ms)
[BVP backend compare] measuring linked runtime for `C-gcc` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-gcc` on scenario `small-damp1-24` finished: total 0.212 ms
[BVP backend compare] building backend `C-tcc` for scenario `small-damp1-24`
[BVP backend compare] backend `C-tcc` for scenario `small-damp1-24` finished with status `ok` (build 31.207 ms, total 286.790 ms)
[BVP backend compare] measuring linked runtime for `C-tcc` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-tcc` on scenario `small-damp1-24` finished: total 0.485 ms
[BVP backend compare] building backend `Zig` for scenario `small-damp1-24`
[BVP backend compare] backend `Zig` for scenario `small-damp1-24` finished with status `ok` (build 24017.927 ms, total 24040.940 ms)
[BVP backend compare] measuring linked runtime for `Zig` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Zig` on scenario `small-damp1-24` finished: total 0.198 ms
[BVP backend runtime compare] scenario=small-damp1-24, residuals=48, vars=48, nnz=118, iters=200, samples=5
variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       |       0.444 |       0.326 |    0.770 |              1.000x |    0.000000e0 |    0.000000e0 | ok
Rust           |       0.220 |       0.045 |    0.265 |              2.904x |    0.000000e0 |    0.000000e0 | ok
C-gcc          |       0.186 |       0.026 |    0.212 |              3.639x |    0.000000e0 |    0.000000e0 | ok
C-tcc          |       0.252 |       0.233 |    0.485 |              1.589x |    0.000000e0 |    0.000000e0 | ok
Zig            |       0.102 |       0.096 |    0.198 |              3.888x |  5.551115e-17 |    0.000000e0 | ok
[BVP backend compare] runtime comparison finished scenario `small-damp1-24`
[BVP backend compare] runtime comparison entering scenario `combustion-100`
[BVP backend compare] building symbolic scenario `combustion-100`
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0                  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 58                 │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 44.827586206896555 │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 53.44827586206897  │
╰─────────────────────────────┴────────────────────╯
[BVP backend compare] scenario `combustion-100` ready: residuals=600, vars=600, nnz=2088
[BVP backend compare] measuring lambdify baseline for scenario `combustion-100`
[BVP backend compare] lambdify baseline for scenario `combustion-100` finished: total 4.632 ms
[BVP backend compare] building backend `Rust` for scenario `combustion-100`
[BVP backend compare] backend `Rust` for scenario `combustion-100` finished with status `ok` (build 13960.327 ms, total 14163.942 ms)
[BVP backend compare] measuring linked runtime for `Rust` on scenario `combustion-100`
[BVP backend compare] runtime for `Rust` on scenario `combustion-100` finished: total 0.191 ms
[BVP backend compare] building backend `C-gcc` for scenario `combustion-100`
[BVP backend compare] backend `C-gcc` for scenario `combustion-100` finished with status `ok` (build 5236.955 ms, total 5558.624 ms)
[BVP backend compare] measuring linked runtime for `C-gcc` on scenario `combustion-100`
[BVP backend compare] runtime for `C-gcc` on scenario `combustion-100` finished: total 0.179 ms
[BVP backend compare] building backend `C-tcc` for scenario `combustion-100`
[BVP backend compare] backend `C-tcc` for scenario `combustion-100` finished with status `ok` (build 28.597 ms, total 323.886 ms)
[BVP backend compare] measuring linked runtime for `C-tcc` on scenario `combustion-100`
[BVP backend compare] runtime for `C-tcc` on scenario `combustion-100` finished: total 0.752 ms
[BVP backend compare] building backend `Zig` for scenario `combustion-100`
[BVP backend compare] backend `Zig` for scenario `combustion-100` finished with status `ok` (build 21915.371 ms, total 22067.113 ms)
[BVP backend compare] measuring linked runtime for `Zig` on scenario `combustion-100`
[BVP backend compare] runtime for `Zig` on scenario `combustion-100` finished: total 0.370 ms
[BVP backend runtime compare] scenario=combustion-100, residuals=600, vars=600, nnz=2088, iters=40, samples=5
variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       |       2.344 |       2.288 |    4.632 |              1.000x |    0.000000e0 |    0.000000e0 | ok
Rust           |       0.092 |       0.099 |    0.191 |             24.305x |    0.000000e0 |    0.000000e0 | ok
C-gcc          |       0.087 |       0.092 |    0.179 |             25.849x |    0.000000e0 |    0.000000e0 | ok
C-tcc          |       0.353 |       0.399 |    0.752 |              6.156x |  5.551115e-17 |  1.110223e-16 | ok
Zig            |       0.144 |       0.226 |    0.370 |             12.518x |  5.551115e-17 |  1.110223e-16 | ok
[BVP backend compare] runtime comparison finished scenario `combustion-100`
[BVP backend compare] runtime comparison entering scenario `combustion-1000`
[BVP backend compare] building symbolic scenario `combustion-1000`
creating discretization equations
╭─────────────────────────────┬─────────────────────╮
│ bounds and tolerances       │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ bc handling                 │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ discretization of equations │ 59.96131528046422   │
├─────────────────────────────┼─────────────────────┤
│ flat list creation          │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ BC application              │ 38.87814313346228   │
├─────────────────────────────┼─────────────────────┤
│ consistency test            │ 0.19342359767891684 │
├─────────────────────────────┼─────────────────────┤
│ total time, sec             │ 517                 │
╰─────────────────────────────┴─────────────────────╯
[BVP backend compare] scenario `combustion-1000` ready: residuals=6000, vars=6000, nnz=20988
[BVP backend compare] measuring lambdify baseline for scenario `combustion-1000`
[BVP backend compare] lambdify baseline for scenario `combustion-1000` finished: total 8.775 ms
[BVP backend compare] building backend `Rust` for scenario `combustion-1000`
[BVP backend compare] backend `Rust` for scenario `combustion-1000` finished with status `ok` (build 13084.143 ms, total 15642.523 ms)
[BVP backend compare] measuring linked runtime for `Rust` on scenario `combustion-1000`
[BVP backend compare] runtime for `Rust` on scenario `combustion-1000` finished: total 1.599 ms
[BVP backend compare] building backend `C-gcc` for scenario `combustion-1000`
[BVP backend compare] backend `C-gcc` for scenario `combustion-1000` finished with status `ok` (build 5510.517 ms, total 8507.106 ms)
[BVP backend compare] measuring linked runtime for `C-gcc` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-gcc` on scenario `combustion-1000` finished: total 1.007 ms
[BVP backend compare] building backend `C-tcc` for scenario `combustion-1000`
[BVP backend compare] backend `C-tcc` for scenario `combustion-1000` finished with status `ok` (build 137.233 ms, total 2945.861 ms)
[BVP backend compare] measuring linked runtime for `C-tcc` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-tcc` on scenario `combustion-1000` finished: total 1.476 ms
[BVP backend compare] building backend `Zig` for scenario `combustion-1000`
[BVP backend compare] backend `Zig` for scenario `combustion-1000` finished with status `ok` (build 39258.592 ms, total 41898.787 ms)
[BVP backend compare] measuring linked runtime for `Zig` on scenario `combustion-1000`
[BVP backend compare] runtime for `Zig` on scenario `combustion-1000` finished: total 1.285 ms
[BVP backend runtime compare] scenario=combustion-1000, residuals=6000, vars=6000, nnz=20988, iters=6, samples=3
variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       |       4.672 |       4.103 |    8.775 |              1.000x |    0.000000e0 |    0.000000e0 | ok
Rust           |       0.586 |       1.013 |    1.599 |              5.486x |    0.000000e0 |    0.000000e0 | ok
C-gcc          |       0.476 |       0.531 |    1.007 |              8.717x |    0.000000e0 |    0.000000e0 | ok
C-tcc          |       0.665 |       0.811 |    1.476 |              5.944x |  4.440892e-16 |  6.661338e-16 | ok
Zig            |       0.471 |       0.814 |    1.285 |              6.830x |  4.440892e-16 |  2.220446e-16 | ok
[BVP backend compare] runtime comparison finished scenario `combustion-1000`
```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record residual/Jacobian callback speedups and correctness diffs.
```

### `bvp_lambdify_vs_atomview_callable_leaders_compare`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test --release bvp_lambdify_vs_atomview_callable_leaders_compare -- --ignored --nocapture --test-threads=1
```

This is the practical break-even table for the current leader routes:
`ExprLegacy + Lambdify` versus `AtomView + compiled native backend`.  It prints
bootstrap deltas, runtime gains per call, and the number of repeated calls needed
for AOT to repay its build cost.

Current result:
[BVP backend compare] probing available generated backends
[BVP backend compare] probing C compiler `gcc`
[BVP backend compare] probing C compiler `tcc`
[BVP backend compare] probing Zig compiler `zig`
[BVP backend compare] finished backend probe: 4 variant(s)
[BVP backend compare] detected variants: C-gcc, C-tcc
[BVP backend compare] probes: gcc=detected, tcc=detected, zig=detected
[BVP backend compare] callable leaders compare entering scenario `combustion-100`
[BVP backend compare] building symbolic scenario `combustion-100`
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 39.21568627450981 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 51                │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 60.78431372549019 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
╰─────────────────────────────┴───────────────────╯
[BVP backend compare] scenario `combustion-100` ready: residuals=600, vars=600, nnz=2088
[BVP backend compare] building symbolic scenario `combustion-100`
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 53                 │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 30.188679245283016 │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 67.9245283018868   │
╰─────────────────────────────┴────────────────────╯
[BVP backend compare] scenario `combustion-100` ready: residuals=600, vars=600, nnz=2088
[BVP backend compare] measuring lambdify end-to-end callable path for scenario `combustion-100`
[BVP backend compare] measuring lambdify baseline for scenario `combustion-100`
[BVP backend compare] lambdify baseline for scenario `combustion-100` finished: total 4.466 ms
[BVP backend compare] measuring linked runtime for `C-gcc` on scenario `combustion-100`
[BVP backend compare] runtime for `C-gcc` on scenario `combustion-100` finished: total 0.657 ms
[BVP backend compare] measuring linked runtime for `C-tcc` on scenario `combustion-100`
[BVP backend compare] runtime for `C-tcc` on scenario `combustion-100` finished: total 0.742 ms
[BVP end-to-end callable compare] scenario=combustion-100, residuals=600, vars=600, nnz=2088
variant        | sym_backend | preset      | symbolic_ms | callable_prep_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       | ExprLegacy  | n/a         |      71.580 |            5.188 |          0.000 |    0.000 |   0.000 |          0.209 |              76.977 | ok
C-gcc          | AtomView    | DevFastest  |      71.883 |           78.443 |        100.629 |  769.743 |  12.422 |          0.278 |            1033.397 | ok
C-tcc          | AtomView    | DevFastest  |      71.883 |           87.132 |        120.994 |   32.017 |  13.650 |          0.308 |             325.984 | ok
variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       |            2.257 |            2.209 |         4.466 |                1.000x |    0.000000e0 |    0.000000e0 | ok
C-gcc          |            0.289 |            0.368 |         0.657 |                6.798x |    0.000000e0 |    0.000000e0 | ok
C-tcc          |            0.352 |            0.390 |         0.742 |                6.021x |  5.551115e-17 |  1.110223e-16 | ok
variant        | extra_bootstrap_ms_vs_lambdify | runtime_gain_ms_per_call | break_even_calls | status
------------------------------------------------------------------------------------------------------------
C-gcc          |                       956.420 |                    3.809 |          251.091 | ok
C-tcc          |                       249.007 |                    3.724 |           66.860 | ok
[BVP backend compare] callable leaders compare finished scenario `combustion-100`
[BVP backend compare] callable leaders compare entering scenario `combustion-1000`
[BVP backend compare] building symbolic scenario `combustion-1000`
creating discretization equations
╭─────────────────────────────┬─────────────────────╮
│ bc handling                 │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ consistency test            │ 0.20576131687242796 │
├─────────────────────────────┼─────────────────────┤
│ bounds and tolerances       │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ flat list creation          │ 0.20576131687242796 │
├─────────────────────────────┼─────────────────────┤
│ total time, sec             │ 486                 │
├─────────────────────────────┼─────────────────────┤
│ BC application              │ 34.5679012345679    │
├─────────────────────────────┼─────────────────────┤
│ discretization of equations │ 64.19753086419753   │
╰─────────────────────────────┴─────────────────────╯
[BVP backend compare] scenario `combustion-1000` ready: residuals=6000, vars=6000, nnz=20988
[BVP backend compare] building symbolic scenario `combustion-1000`
creating discretization equations
╭─────────────────────────────┬─────────────────────╮
│ consistency test            │ 0.18796992481203006 │
├─────────────────────────────┼─────────────────────┤
│ total time, sec             │ 532                 │
├─────────────────────────────┼─────────────────────┤
│ bc handling                 │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ flat list creation          │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ BC application              │ 42.857142857142854  │
├─────────────────────────────┼─────────────────────┤
│ bounds and tolerances       │ 0                   │
├─────────────────────────────┼─────────────────────┤
│ discretization of equations │ 56.20300751879699   │
╰─────────────────────────────┴─────────────────────╯
[BVP backend compare] scenario `combustion-1000` ready: residuals=6000, vars=6000, nnz=20988
[BVP backend compare] measuring lambdify end-to-end callable path for scenario `combustion-1000`
[BVP backend compare] measuring lambdify baseline for scenario `combustion-1000`
[BVP backend compare] lambdify baseline for scenario `combustion-1000` finished: total 14.932 ms
[BVP backend compare] measuring linked runtime for `C-gcc` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-gcc` on scenario `combustion-1000` finished: total 1.014 ms
[BVP backend compare] measuring linked runtime for `C-tcc` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-tcc` on scenario `combustion-1000` finished: total 1.603 ms
[BVP end-to-end callable compare] scenario=combustion-1000, residuals=6000, vars=6000, nnz=20988
variant        | sym_backend | preset      | symbolic_ms | callable_prep_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       | ExprLegacy  | n/a         |    1742.245 |          372.592 |          0.000 |    0.000 |   0.000 |          2.186 |            2117.023 | ok
C-gcc          | AtomView    | DevFastest  |    2138.805 |          813.098 |        110.177 | 5869.819 |  27.637 |          4.821 |            8964.356 | ok
C-tcc          | AtomView    | DevFastest  |    2138.805 |          748.227 |        106.413 |  147.074 |   9.342 |          2.633 |            3152.494 | ok
variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify       |            8.183 |            6.749 |        14.932 |                1.000x |    0.000000e0 |    0.000000e0 | ok
C-gcc          |            0.478 |            0.536 |         1.014 |               14.725x |    0.000000e0 |    0.000000e0 | ok
C-tcc          |            0.756 |            0.847 |         1.603 |                9.315x |  4.440892e-16 |  6.661338e-16 | ok
variant        | extra_bootstrap_ms_vs_lambdify | runtime_gain_ms_per_call | break_even_calls | status
------------------------------------------------------------------------------------------------------------
C-gcc          |                      6847.333 |                   13.918 |          491.976 | ok
C-tcc          |                      1035.471 |                   13.329 |           77.685 | ok
[BVP backend compare] callable leaders compare finished scenario `combustion-1000`
```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: write the current break-even call counts for combustion-100 and combustion-1000.
```

### `bvp_generated_backend_compile_preset_tradeoff_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test --release bvp_generated_backend_compile_preset_tradeoff_table -- --ignored --nocapture --test-threads=1
```

This test checks whether `Production`, `FastBuild`, or `DevFastest` is the right
compile profile for a given backend and problem size.  It should be used before
choosing defaults in public APIs or task documents.

Current result:
test symbolic::codegen::tests::codegen_bvp_backend_comparison_tests::bvp_generated_backend_compile_preset_tradeoff_table ... [BVP backend compare] starting compile-preset tradeoff test
[BVP backend compare] probing available generated backends
[BVP backend compare] probing C compiler `gcc`
[BVP backend compare] probing C compiler `tcc`
[BVP backend compare] probing Zig compiler `zig`
[BVP backend compare] finished backend probe: 4 variant(s)
[BVP backend compare] detected variants: Rust, C-gcc, C-tcc, Zig
[BVP backend compare] probes: gcc=detected, tcc=detected, zig=detected
[BVP backend compare] compile-preset comparison entering scenario `small-damp1-24`
[BVP backend compare] building symbolic scenario `small-damp1-24`
creating discretization equations
╭─────────────────────────────┬─────╮
│ bounds and tolerances       │ 0   │
├─────────────────────────────┼─────┤
│ BC application              │ 0   │
├─────────────────────────────┼─────┤
│ discretization of equations │ 100 │
├─────────────────────────────┼─────┤
│ bc handling                 │ 0   │
├─────────────────────────────┼─────┤
│ consistency test            │ 0   │
├─────────────────────────────┼─────┤
│ flat list creation          │ 0   │
├─────────────────────────────┼─────┤
│ total time, sec             │ 1   │
╰─────────────────────────────┴─────╯
[BVP backend compare] scenario `small-damp1-24` ready: residuals=48, vars=48, nnz=118
[BVP backend compare] measuring linked runtime for `Rust-FastBuild` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Rust-FastBuild` on scenario `small-damp1-24` finished: total 0.258 ms
[BVP backend compare] measuring linked runtime for `Rust-DevFastest` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Rust-DevFastest` on scenario `small-damp1-24` finished: total 0.479 ms
[BVP backend compare] measuring linked runtime for `C-gcc-Production` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-gcc-Production` on scenario `small-damp1-24` finished: total 0.216 ms
[BVP backend compare] measuring linked runtime for `C-gcc-FastBuild` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-gcc-FastBuild` on scenario `small-damp1-24` finished: total 0.203 ms
[BVP backend compare] measuring linked runtime for `C-gcc-DevFastest` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-gcc-DevFastest` on scenario `small-damp1-24` finished: total 0.406 ms
[BVP backend compare] measuring linked runtime for `C-tcc-Production` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-tcc-Production` on scenario `small-damp1-24` finished: total 0.477 ms
[BVP backend compare] measuring linked runtime for `C-tcc-FastBuild` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-tcc-FastBuild` on scenario `small-damp1-24` finished: total 0.484 ms
[BVP backend compare] measuring linked runtime for `C-tcc-DevFastest` on scenario `small-damp1-24`
[BVP backend compare] runtime for `C-tcc-DevFastest` on scenario `small-damp1-24` finished: total 0.513 ms
[BVP backend compare] measuring linked runtime for `Zig-Production` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Zig-Production` on scenario `small-damp1-24` finished: total 0.199 ms
[BVP backend compare] measuring linked runtime for `Zig-FastBuild` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Zig-FastBuild` on scenario `small-damp1-24` finished: total 0.179 ms
[BVP backend compare] measuring linked runtime for `Zig-DevFastest` on scenario `small-damp1-24`
[BVP backend compare] runtime for `Zig-DevFastest` on scenario `small-damp1-24` finished: total 0.799 ms
[BVP backend preset compare] scenario=small-damp1-24, residuals=48, vars=48, nnz=118
variant        | preset      | build_ms | total_to_outputs_ms | status
------------------------------------------------------------------------------
Rust           | FastBuild   |  539.671 |             550.123 | ok
Rust           | DevFastest  |  499.587 |             506.722 | ok
C-gcc          | Production  |  377.294 |             500.111 | ok
C-gcc          | FastBuild   |  324.411 |             455.485 | ok
C-gcc          | DevFastest  |  297.146 |             423.111 | ok
C-tcc          | Production  |   18.843 |             137.382 | ok
C-tcc          | FastBuild   |   17.932 |             319.901 | ok
C-tcc          | DevFastest  |   18.000 |             142.752 | ok
Zig            | Production  | 23493.450 |           23501.641 | ok
Zig            | FastBuild   | 16537.585 |           16546.447 | ok
Zig            | DevFastest  | 26790.812 |           26802.874 | ok
variant        | preset      | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_baseline | residual_diff | jacobian_diff
--------------------------------------------------------------------------------------------------------------------------------------------
Rust           | FastBuild   |            0.208 |            0.050 |         0.258 |                 1.000x |    0.000000e0 |    0.000000e0
Rust           | DevFastest  |            0.240 |            0.240 |         0.479 |                 0.537x |    0.000000e0 |    0.000000e0
C-gcc          | Production  |            0.191 |            0.025 |         0.216 |                 1.000x |    0.000000e0 |    0.000000e0
C-gcc          | FastBuild   |            0.178 |            0.025 |         0.203 |                 1.063x |    0.000000e0 |    0.000000e0
C-gcc          | DevFastest  |            0.207 |            0.199 |         0.406 |                 0.533x |    0.000000e0 |    0.000000e0
C-tcc          | Production  |            0.251 |            0.226 |         0.477 |                 1.000x |    0.000000e0 |    0.000000e0
C-tcc          | FastBuild   |            0.250 |            0.233 |         0.484 |                 0.986x |    0.000000e0 |    0.000000e0
C-tcc          | DevFastest  |            0.276 |            0.237 |         0.513 |                 0.929x |    0.000000e0 |    0.000000e0
Zig            | Production  |            0.099 |            0.100 |         0.199 |                 1.000x |  5.551115e-17 |    0.000000e0
Zig            | FastBuild   |            0.091 |            0.088 |         0.179 |                 1.108x |  5.551115e-17 |    0.000000e0
Zig            | DevFastest  |            0.406 |            0.394 |         0.799 |                 0.249x |  5.551115e-17 |    0.000000e0
[BVP backend compare] compile-preset comparison finished scenario `small-damp1-24`
[BVP backend compare] compile-preset comparison entering scenario `combustion-100`
[BVP backend compare] building symbolic scenario `combustion-100`
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 42.42424242424242 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 54.54545454545454 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 66                │
╰─────────────────────────────┴───────────────────╯
[BVP backend compare] scenario `combustion-100` ready: residuals=600, vars=600, nnz=2088
[BVP backend compare] measuring linked runtime for `Rust-FastBuild` on scenario `combustion-100`
[BVP backend compare] runtime for `Rust-FastBuild` on scenario `combustion-100` finished: total 0.205 ms
[BVP backend compare] measuring linked runtime for `Rust-DevFastest` on scenario `combustion-100`
[BVP backend compare] runtime for `Rust-DevFastest` on scenario `combustion-100` finished: total 0.575 ms
[BVP backend compare] measuring linked runtime for `C-gcc-Production` on scenario `combustion-100`
[BVP backend compare] runtime for `C-gcc-Production` on scenario `combustion-100` finished: total 0.177 ms
[BVP backend compare] measuring linked runtime for `C-gcc-FastBuild` on scenario `combustion-100`
[BVP backend compare] runtime for `C-gcc-FastBuild` on scenario `combustion-100` finished: total 0.188 ms
[BVP backend compare] measuring linked runtime for `C-gcc-DevFastest` on scenario `combustion-100`
[BVP backend compare] runtime for `C-gcc-DevFastest` on scenario `combustion-100` finished: total 0.606 ms
[BVP backend compare] measuring linked runtime for `C-tcc-Production` on scenario `combustion-100`
[BVP backend compare] runtime for `C-tcc-Production` on scenario `combustion-100` finished: total 0.702 ms
[BVP backend compare] measuring linked runtime for `C-tcc-FastBuild` on scenario `combustion-100`
[BVP backend compare] runtime for `C-tcc-FastBuild` on scenario `combustion-100` finished: total 0.767 ms
[BVP backend compare] measuring linked runtime for `C-tcc-DevFastest` on scenario `combustion-100`
[BVP backend compare] runtime for `C-tcc-DevFastest` on scenario `combustion-100` finished: total 0.696 ms
[BVP backend compare] measuring linked runtime for `Zig-Production` on scenario `combustion-100`
[BVP backend compare] runtime for `Zig-Production` on scenario `combustion-100` finished: total 0.353 ms
[BVP backend compare] measuring linked runtime for `Zig-FastBuild` on scenario `combustion-100`
[BVP backend compare] runtime for `Zig-FastBuild` on scenario `combustion-100` finished: total 0.363 ms
[BVP backend compare] measuring linked runtime for `Zig-DevFastest` on scenario `combustion-100`
[BVP backend compare] runtime for `Zig-DevFastest` on scenario `combustion-100` finished: total 0.733 ms
[BVP backend preset compare] scenario=combustion-100, residuals=600, vars=600, nnz=2088
variant        | preset      | build_ms | total_to_outputs_ms | status
------------------------------------------------------------------------------
Rust           | FastBuild   | 14032.513 |           14180.834 | ok
Rust           | DevFastest  | 1268.833 |            1420.158 | ok
C-gcc          | Production  | 6190.678 |            6521.824 | ok
C-gcc          | FastBuild   | 5083.912 |            5454.258 | ok
C-gcc          | DevFastest  |  902.599 |            1178.116 | ok
C-tcc          | Production  |   28.796 |             299.906 | ok
C-tcc          | FastBuild   |   38.577 |             331.343 | ok
C-tcc          | DevFastest  |   30.198 |             328.108 | ok
Zig            | Production  | 19817.530 |           19973.819 | ok
Zig            | FastBuild   | 17511.702 |           17666.195 | ok
Zig            | DevFastest  | 26111.000 |           26272.667 | ok
variant        | preset      | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_baseline | residual_diff | jacobian_diff
--------------------------------------------------------------------------------------------------------------------------------------------
Rust           | FastBuild   |            0.100 |            0.106 |         0.205 |                 1.000x |    0.000000e0 |    0.000000e0
Rust           | DevFastest  |            0.200 |            0.375 |         0.575 |                 0.357x |    0.000000e0 |    0.000000e0
C-gcc          | Production  |            0.085 |            0.092 |         0.177 |                 1.000x |    0.000000e0 |    0.000000e0
C-gcc          | FastBuild   |            0.094 |            0.094 |         0.188 |                 0.943x |    0.000000e0 |    0.000000e0
C-gcc          | DevFastest  |            0.280 |            0.326 |         0.606 |                 0.292x |    0.000000e0 |    0.000000e0
C-tcc          | Production  |            0.331 |            0.371 |         0.702 |                 1.000x |  5.551115e-17 |  1.110223e-16
C-tcc          | FastBuild   |            0.358 |            0.409 |         0.767 |                 0.915x |  5.551115e-17 |  1.110223e-16
C-tcc          | DevFastest  |            0.332 |            0.364 |         0.696 |                 1.009x |  5.551115e-17 |  1.110223e-16
Zig            | Production  |            0.138 |            0.214 |         0.353 |                 1.000x |  5.551115e-17 |  1.110223e-16
Zig            | FastBuild   |            0.141 |            0.223 |         0.363 |                 0.970x |  5.551115e-17 |  1.110223e-16
Zig            | DevFastest  |            0.265 |            0.469 |         0.733 |                 0.481x |  5.551115e-17 |  1.110223e-16
[BVP backend compare] compile-preset comparison finished scenario `combustion-100`
[BVP backend compare] compile-preset comparison entering scenario `combustion-1000`
[BVP backend compare] building symbolic scenario `combustion-1000`
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ discretization of equations │ 59.3984962406015   │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 39.34837092731829  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0.2506265664160401 │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 399                │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
╰─────────────────────────────┴────────────────────╯
[BVP backend compare] scenario `combustion-1000` ready: residuals=6000, vars=6000, nnz=20988
[BVP backend compare] measuring linked runtime for `Rust-DevFastest` on scenario `combustion-1000`
[BVP backend compare] runtime for `Rust-DevFastest` on scenario `combustion-1000` finished: total 1.503 ms
[BVP backend compare] measuring linked runtime for `C-gcc-DevFastest` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-gcc-DevFastest` on scenario `combustion-1000` finished: total 1.018 ms
[BVP backend compare] measuring linked runtime for `C-tcc-DevFastest` on scenario `combustion-1000`
[BVP backend compare] runtime for `C-tcc-DevFastest` on scenario `combustion-1000` finished: total 1.195 ms
[BVP backend compare] measuring linked runtime for `Zig-DevFastest` on scenario `combustion-1000`
[BVP backend compare] runtime for `Zig-DevFastest` on scenario `combustion-1000` finished: total 1.213 ms
[BVP backend preset compare] scenario=combustion-1000, residuals=6000, vars=6000, nnz=20988
variant        | preset      | build_ms | total_to_outputs_ms | status
------------------------------------------------------------------------------
Rust           | DevFastest  | 13370.742 |           15654.036 | ok
C-gcc          | DevFastest  | 5521.545 |            8286.545 | ok
C-tcc          | DevFastest  |  134.563 |            2631.116 | ok
Zig            | DevFastest  | 39632.651 |           42010.549 | ok
variant        | preset      | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_baseline | residual_diff | jacobian_diff
--------------------------------------------------------------------------------------------------------------------------------------------
Rust           | DevFastest  |            0.561 |            0.942 |         1.503 |                 1.000x |    0.000000e0 |    0.000000e0
C-gcc          | DevFastest  |            0.462 |            0.556 |         1.018 |                 1.000x |    0.000000e0 |    0.000000e0
C-tcc          | DevFastest  |            0.549 |            0.646 |         1.195 |                 1.000x |  4.440892e-16 |  6.661338e-16
Zig            | DevFastest  |            0.447 |            0.766 |         1.213 |                 1.000x |  4.440892e-16 |  2.220446e-16
[BVP backend compare] compile-preset comparison finished scenario `combustion-1000`
```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record which preset is preferred for short interactive runs and long repeated solves.
```

### `bvp_generated_compiled_sparse_banded_matrix_backend_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test --release bvp_generated_compiled_sparse_banded_matrix_backend_table -- --ignored --nocapture --test-threads=1
```

This is the compiled sparse/banded matrix-backend matrix across languages and
presets.  It belongs in codegen diagnostics because it answers whether compiled
callbacks themselves are correct and fast before the solver starts iterating.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record sparse vs banded callback behavior per toolchain.
```

### `bvp_callable_and_linear_solver_story_table`

File: `src/symbolic/codegen/tests/codegen_bvp_backend_comparison_tests.rs`

Command:

```powershell
cargo test --release bvp_callable_and_linear_solver_story_table -- --ignored --nocapture --test-threads=1
```

This is the closest codegen-level test to an end-to-end solver story.  It
contains callable preparation, runtime callbacks, and linear-solver comparison,
but still avoids Newton convergence noise.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record whether callable time or linear solve time dominates.
```

## Parallelism And Break-Even Diagnostics

### `diagnose_rayon_overhead_baseline`

File: `src/symbolic/codegen/tests/codegen_bvp_diagnostic_tests.rs`

Command:

```powershell
cargo test --release diagnose_rayon_overhead_baseline -- --nocapture
```

This measures raw Rayon scheduling overhead with no-op closures.  It is the
floor that every chunking strategy must beat.  If one sparse callback call is
only a few times larger than this number, parallelism cannot pay off.

Current result:
=== Rayon overhead baseline (workers=4, iters=10000) ===
no-op sequential:          0.4 ns/call
no-op join (2-way):        792.0 ns/call  (overhead +791.7 ns)
no-op join (4-way nested): 879.1 ns/call  (overhead +878.7 ns)

If sparse eval time is within ~5x of join overhead, parallelism
cannot help at this problem size regardless of chunk count.
```text
TODO: paste latest release result here.
```

Conclusion:

```text
TODO: record approximate join overhead on the current machine.
```

### `diagnose_chunk_granularity_and_fallback`

File: `src/symbolic/codegen/tests/codegen_bvp_diagnostic_tests.rs`

Command:

```powershell
cargo test --release diagnose_chunk_granularity_and_fallback -- --nocapture
```

This is the first test to run when "parallel chunking appears to do nothing".
It prints requested jobs, actual jobs, fallback status, work per job, and
nanoseconds per call on the compiled stress path.

Current historical result, before the overhead-derived Auto policy:

=== Chunk granularity / fallback diagnostic ===
fields=8, steps=96, nnz=3056, chunks=16
rayon_threads=4, samples=9, jacobian_iters=200
Auto fallback threshold: nnz < 256 triggers sequential fallback

requested_jobs   actual_jobs  fallback   nnz/job    ns/call     
sequential       -            -          3056       7426.50
1                1            true       3056       7317.50  <-- faster
2                2            false      1528       13358.50  
3                3            false      1018       13476.00  
4                4            false      764        12931.00  
5                5            false      611        12929.00  
6                6            false      509        11545.50  
7                7            false      436        12220.50  
8                8            false      382        13153.50  
test symbolic::codegen::tests::codegen_bvp_diagnostic_tests::diagnose_chunk_granularity_and_fallback ... ok
```
latest result with auto
=== Chunk granularity / fallback diagnostic ===
fields=8, steps=96, nnz=3056, chunks=16
rayon_threads=4, samples=9, jacobian_iters=200
Auto policy derives min_work/job from measured rayon overhead.
min_work/job=399, nnz/chunk=191

mode             jobs         fallback   nnz/job    nnz/chunk  ns/call     
sequential       -            -          3056       -          7334.00
auto             4            true       764        191        7826.50  

Forced sweep (`ParallelFallbackPolicy::Never`) for diagnostics:
requested        actual       fallback   nnz/job    nnz/chunk  ns/call     
1                1            true       3056       191        7621.50  
2                2            false      1528       191        13161.00  
3                3            false      1018       191        15384.00  
4                4            false      764        191        12871.50  
5                5            false      611        191        13180.00  
6                6            false      509        191        11631.00  
7                7            false      436        191        12028.50  
8                8            false      382        191        12618.50  
test symbolic::codegen::tests::codegen_bvp_diagnostic_tests::diagnose_chunk_granularity_and_fallback ... ok
```

Conclusion:

```text
The current code no longer uses a fixed nnz threshold. Auto fallback is derived
from measured Rayon overhead and estimated work per job/chunk. Interpret fresh
release tables through `min_work/job`, `nnz/job`, and `nnz/chunk`.

Latest release result: `min_work/job=399`, `nnz/chunk=191`, and `nnz/job=764`
for the auto plan with four requested jobs. Auto correctly falls back to the
sequential executor because each generated chunk is too small even though the
nominal work per job is above the job-level threshold. This is exactly the
behavior we want: the policy prevents over-fragmented parallel execution without
hardcoding a machine-specific magic number.

The forced sweep proves why fallback is necessary. Sequential evaluation is about
`7334 ns/call`, auto fallback is close at `7826 ns/call`, while forced parallel
execution with 2..8 jobs is much slower, roughly `11631-15384 ns/call`. The
observed break-even is therefore not reached for this fixture. A future parallel
policy should prefer fewer, heavier chunks or direct disjoint-output execution;
it should not force Rayon on chunks around `191` sparse values.
```

### `diagnose_combustion_chunk_ir_amplification`

File: `src/symbolic/codegen/tests/codegen_bvp_diagnostic_tests.rs`

1
```powershell
$env:BVP_CHUNK_IR_STEPS="3000"
cargo test --release diagnose_combustion_chunk_ir_amplification -- --ignored --nocapture
```

=== Combustion BVP chunk IR/source amplification ===
n_steps=1000, unknowns=6000, residuals=6000, sparse_nnz=20988
This does not execute callbacks. It measures generated IR/source size before scheduling/FFI.
matrix   | stage    | strategy | chunks |  outputs |      instr |      amp |  max_blk |     temps |   src_kb
----------------------------------------------------------------------------------------------------------
Shared   | Residual | whole   |      1 |     6000 |      57085 |    1.000 |    57085 |     57085 |   2159.0
Sparse   | Jacobian | whole   |      1 |    20988 |      23034 |    1.000 |    23034 |     23034 |   1315.8
Banded   | Jacobian | whole   |      1 |    20988 |      23034 |    1.000 |    23034 |     23034 |   1315.8
Shared   | Residual | chunk4  |      4 |     6000 |      57231 |    1.003 |    14333 |     57231 |   2072.9
Sparse   | Jacobian | chunk4  |      4 |    20988 |      23168 |    1.006 |     5799 |     23168 |   1267.8
Banded   | Jacobian | chunk4  |      4 |    20988 |      23168 |    1.006 |     5799 |     23168 |   1267.8
Shared   | Residual | chunk8  |      8 |     6000 |      57423 |    1.006 |     7208 |     57423 |   2025.8
Sparse   | Jacobian | chunk8  |      8 |    20988 |      23376 |    1.015 |     2927 |     23376 |   1261.8
Banded   | Jacobian | chunk8  |      8 |    20988 |      23376 |    1.015 |     2927 |     23376 |   1261.8
Shared   | Residual | chunk16 |     16 |     6000 |      57879 |    1.014 |     3648 |     57879 |   2021.8
Sparse   | Jacobian | chunk16 |     16 |    20988 |      23776 |    1.032 |     1497 |     23776 |   1250.0
Banded   | Jacobian | chunk16 |     16 |    20988 |      23776 |    1.032 |     1497 |     23776 |   1250.0

Interpretation: amp > 1 means chunking generated more total straight-line IR than the whole callback. If that happens, runtime parallelism must first pay back duplicated arithmetic/source work plus FFI/scheduler overhead.1


This diagnostic does not solve the BVP and does not compile an AOT artifact.
It asks a lower-level question that became important after the BVP story tables
showed `chunk4` losing to `whole`: before runtime scheduling, FFI, and Newton
linear solves even begin, how much straight-line IR and generated source does
chunking create compared with the whole callback?

The table prints residual, sparse Jacobian, and banded Jacobian rows for
`whole`, `chunk4`, `chunk8`, and `chunk16`. The key column is `amp`, the ratio
of total chunked IR instructions to the corresponding whole-callback IR
instruction count. `amp > 1` means chunking duplicated enough generated work
that runtime parallelism starts behind. In that case a chunked variant can still
win, but only if the useful arithmetic inside each job is large enough to pay
back duplicated IR/source work plus FFI and scheduler overhead.

Current result:

```text
The latest release table at `n_steps=1000` verifies the post-fix structure.

Residual callback:
- chunk4/chunk8/chunk16 stay close to whole, with amplification
  `1.003x`, `1.006x`, and `1.014x`.
- Source size even becomes a little smaller in chunked variants.  Residual
  chunking is therefore not losing because of IR duplication.

Sparse Jacobian callback:
- chunk4/chunk8/chunk16 also stay close to whole, with amplification
  `1.006x`, `1.015x`, and `1.032x`.
- This means sparse AOT chunking is structurally clean at the IR level.  If it
  loses in solver-level stories, the cause is runtime overhead, FFI dispatch,
  Rayon scheduling, cache effects, or too little work per job, not duplicated
  symbolic/IR work.

Banded Jacobian callback:
- Banded now exactly tracks the Sparse rows: `1.006x`, `1.015x`, and
  `1.032x`.
- The earlier `1.650x..2.050x` amplification was the pre-fix result and no
  longer describes the current banded code generation path.
```

Conclusion:

```text
The former structural Banded chunking defect is closed: the generated Banded
Jacobian has essentially the same IR amplification as Sparse at every tested
chunk count. Remaining chunking losses are now runtime economics rather than
duplicated band-window code generation: actual jobs, work per job,
scheduler/FFI overhead, cache behavior, and whether callback evaluation is a
large enough portion of the complete solve.
```

Follow-up implementation note:

```text
The first planner fix changed the BVP bridge banded value stream from
diagonal-major order to row/column locality order.  Native banded assembly does
not require diagonal-major order; it only requires the values slice and
per-value `(diagonal_offset, diagonal_position)` metadata to share the same
order.

After this change, a debug diagnostic at n_steps=1000 showed banded Jacobian IR
amplification matching sparse Jacobian amplification:
- chunk4:  1.003x
- chunk8:  1.007x
- chunk16: 1.016x

Before the fix, the release diagnostics showed about 1.650x for chunk4 and
2.050x for chunk8/chunk16.  This means the main structural banded chunking
defect was not Rayon and not AOT language choice; it was poor value ordering
for codegen locality.
```

### `diagnose_problem_size_crossover`

File: `src/symbolic/codegen/tests/codegen_bvp_diagnostic_tests.rs`

Command:

```powershell
cargo test --release diagnose_problem_size_crossover -- --nocapture
```

This IR-level sweep avoids compiled-fixture build cost and asks the pure
algorithmic question: at what problem size does parallel IR evaluation beat
sequential IR evaluation?

Current result:
=== Problem-size crossover sweep (n_steps=96) ===
samples=7, jacobian_iters=100

fields   unknowns   nnz      seq_ns/call    par_ns/call    par_wins  
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 36.36363636363637 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 59.09090909090909 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 22                │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
╰─────────────────────────────┴───────────────────╯
2        192        573      15878.0        14322.0        YES <--
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 54.54545454545455 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ BC application              │ 43.18181818181818 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 44                │
╰─────────────────────────────┴───────────────────╯
4        384        1528     31464.0        19917.0        YES <--
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 51.515151515151516 │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 46.96969696969697  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0                  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 66                 │
╰─────────────────────────────┴────────────────────╯
6        576        2292     44952.0        25296.0        YES <--
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 44.44444444444444 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 81                │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 53.08641975308642 │
╰─────────────────────────────┴───────────────────╯
8        768        3056     59046.0        28474.0        YES <--
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ total time, sec             │ 120               │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 51.66666666666667 │
├─────────────────────────────┼───────────────────┤
│ BC application              │ 46.66666666666667 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
╰─────────────────────────────┴───────────────────╯
12       1152       4584     89739.0        39590.0        YES <--
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 48.82352941176471 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 50                │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 170               │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
╰─────────────────────────────┴───────────────────╯
16       1536       6112     119312.0       46729.0        YES <--
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 55.97014925373134  │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 42.91044776119403  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0.3731343283582089 │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 268                │
╰─────────────────────────────┴────────────────────╯
24       2304       9168     169624.0       66304.0        YES <--
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ total time, sec             │ 345                │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 41.73913043478261  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0.5797101449275363 │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 57.10144927536231  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
╰─────────────────────────────┴────────────────────╯
32       3072       12224    229893.0       82545.0        YES <--
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 36.33276740237691  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0.6791171477079797 │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 589                │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 62.30899830220714  │
╰─────────────────────────────┴────────────────────╯
48       4608       18336    358228.0       127814.0       YES <--
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ consistency test            │ 0.9142857142857143 │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 875                │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 66.05714285714286  │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 32.68571428571428  │
╰─────────────────────────────┴────────────────────╯
64       6144       24448    486690.0       168491.0       YES <--

First crossover: field_count=2
test symbolic::codegen::tests::codegen_bvp_diagnostic_tests::diagnose_problem_size_crossover ... ok
```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record the first field_count where parallel wins.
```

### `benchmark_stress_multifield_parallel_crossover`

File: `src/symbolic/codegen/tests/codegen_bvp_stress_diagnostic_tests.rs`

Command:

```powershell
cargo test --release benchmark_stress_multifield_parallel_crossover -- --ignored --nocapture
```

This is the larger stress-family crossover search.  It is slower than
`diagnose_problem_size_crossover`, but it is closer to production-size BVPs.

Current result:

```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record crossover sizes for residual and sparse Jacobian separately.
```

### `benchmark_compiled_stress_sparse_parallel_job_counts`

File: `src/symbolic/codegen/tests/codegen_bvp_stress_diagnostic_tests.rs`

Command:

```powershell
cargo test --release benchmark_compiled_stress_sparse_parallel_job_counts -- --ignored --nocapture
```

This is the compiled sparse job-count sweep.  Use it to check whether a compiled
fixture has enough work per job to amortize scheduling overhead.

Current result:

=== Compiled stress sparse parallel job-count sweep ===
fields=8, steps=96
samples=9, jacobian_iters=48
creating discretization equations
╭─────────────────────────────┬───────────────────╮
│ BC application              │ 45.67901234567901 │
├─────────────────────────────┼───────────────────┤
│ bounds and tolerances       │ 0                 │
├─────────────────────────────┼───────────────────┤
│ consistency test            │ 0                 │
├─────────────────────────────┼───────────────────┤
│ flat list creation          │ 0                 │
├─────────────────────────────┼───────────────────┤
│ bc handling                 │ 0                 │
├─────────────────────────────┼───────────────────┤
│ total time, sec             │ 81                │
├─────────────────────────────┼───────────────────┤
│ discretization of equations │ 53.08641975308642 │
╰─────────────────────────────┴───────────────────╯
sequential AOT sparse values: 7527.08 ns/call
parallel AOT sparse values (requested_jobs=1, actual_jobs=1, fallback=true): 9045.83 ns/call
parallel AOT sparse values (requested_jobs=2, actual_jobs=2, fallback=false): 14333.33 ns/call
parallel AOT sparse values (requested_jobs=4, actual_jobs=4, fallback=false): 12629.17 ns/call
test symbolic::codegen::tests::codegen_bvp_stress_diagnostic_tests::benchmark_compiled_stress_sparse_parallel_job_counts ... ok
```text
TODO: paste latest release table here.
```

Conclusion:

```text
TODO: record best requested_jobs and whether it beats sequential.
```

### `benchmark_compiled_xlarge_stress_seq_vs_par`

File: `src/symbolic/codegen/tests/codegen_bvp_stress_diagnostic_tests.rs`

Command:

```powershell
cargo test --release benchmark_compiled_xlarge_stress_seq_vs_par -- --nocapture
```

This is the strongest compiled-fixture check that parallel AOT can work in
principle.  The xlarge fixture is intentionally big enough that each sparse
chunk should contain enough arithmetic to beat Rayon overhead.

Current result:
=
=== XLarge compiled sparse AOT: sequential vs parallel ===
fields=32, steps=256, unknowns=8192, nnz=32704
residual_chunks=32, sparse_chunks=11
samples=7, residual_iters=50, jacobian_iters=20
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ discretization of equations │ 44.11955815464587  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0.3898635477582846 │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 1539               │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 55.16569200779727  │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0.0649772579597141 │
╰─────────────────────────────┴────────────────────╯

config           res_jobs     sparse_jobs  res_ns/call    sparse_ns/call
sequential       -            -            541538         210300        
par-jobs1        1            1            518984         180300          res_speedup=1.04x  sparse_speedup=1.17x  (fallback: res=true, sparse=true)
par-jobs2        2            2            266710         127815          res_speedup=2.03x  sparse_speedup=1.65x  (fallback: res=false, sparse=false)
par-jobs4        4            4            152552         79785           res_speedup=3.55x  sparse_speedup=2.64x  (fallback: res=false, sparse=false)
par-jobs11       11           11           156518         87155           res_speedup=3.46x  sparse_speedup=2.41x  (fallback: res=false, sparse=false)
test symbolic::codegen::tests::codegen_bvp_stress_diagnostic_tests::benchmark_compiled_xlarge_stress_seq_vs_par ... ok
```text
Latest release result is included above.
```

Conclusion:

```text
The xlarge compiled fixture is the positive control for parallel chunking.
With `fields=32`, `steps=256`, `unknowns=8192`, and `nnz=32704`, parallel
execution clearly beats sequential:

- `jobs=2`: residual `2.03x`, sparse Jacobian `1.65x`.
- `jobs=4`: residual `3.55x`, sparse Jacobian `2.64x`.
- `jobs=11`: residual `3.46x`, sparse Jacobian `2.41x`.

On this release run, four jobs are the best measured balance for both
callbacks. Thus the execution path is not merely concurrent but capable of a
substantial speedup when the generated work per job is large enough.
```

## Current Working Interpretation

The existing codegen diagnostics already encode the main hypothesis: parallel
chunking is real, but it only pays when each job receives enough arithmetic.
For small or medium sparse Jacobian callbacks, Rayon scheduling and output merge
overhead can dominate, so parallel execution may be correct yet slower.

The current optimization path should therefore be:

1. Use `diagnose_rayon_overhead_baseline` to know the scheduling floor on the
   current machine.
2. Use `diagnose_chunk_granularity_and_fallback` to verify actual jobs, fallback,
   and work per job.
3. Use `diagnose_problem_size_crossover` and
   `benchmark_stress_multifield_parallel_crossover` to find the size where
   parallel IR becomes worthwhile.
4. Use `benchmark_compiled_stress_sparse_parallel_job_counts` and
   `benchmark_compiled_xlarge_stress_seq_vs_par` to confirm that compiled
   callbacks preserve the expected crossover behavior.
5. Only then read solver-level BVP Damp stories, where Newton iterations,
   Jacobian rebuild policy, and linear solves can hide callback-level wins.

## Results Log

Use this section for short conclusions from release runs.  Keep full console
tables only when they add information that is not already visible in the test
output.

### 2026-05-20

```text
Release diagnostics show that generated BVP callbacks are numerically correct
across Rust, C-gcc, C-tcc, and Zig backends.  Observed residual/Jacobian
differences are at roundoff level, typically 0.0 to ~1e-16.

Pipeline/build economics:
- C-tcc has by far the best time-to-first-output for the current generated BVP
  cases.  On combustion-1000 it reaches callable outputs in ~2.61 s, while
  C-gcc takes ~8.39 s, Rust ~15.51 s, and Zig ~41.93 s.
- Runtime throughput is different from build throughput.  C-gcc is the fastest
  compiled runtime in the measured combustion cases: ~25.8x over lambdify on
  combustion-100 and ~8.7x over lambdify on combustion-1000.  C-tcc is much
  faster to build but slower at callback runtime.
- The practical break-even table confirms the expected tradeoff.  For
  combustion-100, C-tcc pays back after about 67 callback calls, while C-gcc
  pays back after about 251 callback calls.  For larger or repeated solves,
  compiled backends become attractive; for one-shot small problems lambdify
  remains hard to beat end-to-end.

Parallelism economics:
- Rayon no-op join overhead on this machine is ~792 ns for 2-way join and
  ~879 ns for 4-way nested join with 4 workers.
- On the compiled stress fixture with fields=8, steps=96, nnz=3056, sequential
  sparse evaluation is ~7.4 us/call.  Parallel variants with 2..8 jobs are
  slower (~11.5..13.5 us/call).  This is not a correctness bug: the callback is
  too small for Rayon scheduling and merge overhead.
- IR-level problem-size crossover reports the first parallel win already at
  field_count=2 for the tested n_steps=96 sweep.  That means the algorithmic
  parallel path can win, but compiled fixture chunk/job granularity and runtime
  overhead decide whether it wins in practice.
- The xlarge compiled fixture proves real parallelism works.  With fields=32,
  steps=256, unknowns=8192, nnz=32704, par-jobs11 gives ~4.69x residual speedup
  and ~3.36x sparse Jacobian speedup over sequential.

Optimization conclusion:
- Do not force parallel callback execution for medium sparse callbacks.  Auto
  policy should require enough total work and enough work per actual job.
- Keep sequential as the default for small/medium sparse callbacks and switch to
  parallel only above a conservative threshold derived from measured join
  overhead and nnz/output count per job.
- The next implementation target is not "make Rayon run"; it already runs.  The
  target is a better Auto policy and chunk planner: avoid too many tiny jobs,
  prefer fewer heavier jobs, and expose diagnostics for actual_jobs,
  fallback_used, nnz_per_job, and output_count_per_job.

Implementation note:
- `Auto` now carries an explicit per-stage plan instead of a bare yes/no flag.
  The plan records total work, generated chunk count, grouped jobs, work/job,
  work/chunk, and the decision reason (`parallel_candidate`,
  `work_per_chunk_too_small`, `single_chunk_or_job`, etc.).
- Solver-side `BuildIfMissing`/`RebuildAlways` with `AotExecutionPolicy::Auto`
  may now regenerate the prepared bundle with coarse chunking before building
  the AOT artifact, but only when the measured machine/workload plan says that
  parallel execution is worthwhile.  Explicit chunking remains explicit: story
  tests that force `chunk4`, `chunk8`, or `whole` are not silently rewritten.
- Runtime diagnostics now include `aot.auto.*` fields alongside
  `aot.runtime.*`, so solver stories can show both the planned Auto decision and
  the actual linked callback jobs/fallback behavior.
```

### 2026-05-25

```text
The latest combustion-1000 release run shows that the cold-start gap has become
small enough to measure carefully rather than describe qualitatively. Lambdify
reaches callable outputs in about 2.425 s; C-tcc reaches them in about 2.687 s,
only about 0.263 s later. The C-tcc interval is broad because materialization
contains an outlier (`174.7 +/- 236.6 ms`), so its cold interval already
overlaps the Lambdify interval.

Source emission is no longer the main AOT penalty. For C-tcc,
`source_emit_ms` is about 12.1 ms and `source_probe_ms` is zero. The stable
visible costs are module construction at about 169 ms, dominated by residual
lowering, compiler build at about 148 ms, and artifact packaging at about
47 ms. Materialization should be remeasured before treating its noisy mean as
a code bottleneck.

The next low-risk optimization removes work that has no effect on generated
numerics: artifact manifest signatures no longer render and collect the full
residual/Jacobian system as temporary strings. They hash the `Expr` tree
structure directly. This targets `packaging_ms`; because the internal cache
key representation changes, existing AOT cache artifacts are expected to be
rebuilt once rather than reused under an obsolete key.

The following release run confirms that change: on combustion-1000 C-tcc
`packaging_ms` falls from about 47.4 ms to about 21.0 ms. The complete
time-to-outputs result improves to about 2.336 s, versus Lambdify at about
2.140 s, leaving a cold gap of about 196 ms. The leading own-code stage is now
unambiguously `module_ms` at about 188 ms, almost entirely
`residual_lower_ms` at about 188 ms.

Inspection of that stage found another non-numeric overhead: ExprLegacy CSE
lowering requested structural signatures through `TraversalCache`, which also
constructed and merged variable-name vectors for every expression subtree.
Those variable sets are unused by the lowerer. It now uses a signature-only
cache with the identical structural-signature algorithm; the parity test
`signature_only_cache_matches_full_traversal_signature` and generated C-source
parity gate guard this change. The next release run should therefore be judged
primarily by `residual_lower_ms`, `module_ms`, and total C-tcc time.

The same lowerer is used for residual and Jacobian blocks, so the signature
cache correction already applies to `jacobian_lower_ms`. A second small
Jacobian-relevant overhead was also removed: `GeneratedBlock::from_task_plan`
previously allocated a temporary `Vec<&Expr>` before lowering every output.
It now streams the existing planned outputs directly into the lowerer, avoiding
that extra pointer buffer for large sparse/banded Jacobians while preserving
the original solver-order output sequence.

The newest release run validates both lowerer changes. On combustion-1000,
Lambdify reaches outputs in `1922.346 +/- 61.895 ms` and C-tcc in
`1967.020 +/- 52.735 ms`: a mean cold-start gap of only about `44.7 ms`.
Inside C-tcc, `module_ms` is now about `44.7 ms`, `residual_lower_ms` about
`44.6 ms`, and `jacobian_lower_ms` about `20.3 ms`, so the earlier lowering
bottleneck has been removed rather than hidden.

The remaining conspicuous internal asymmetry is C materialization:
`materialize_ms` is about `95 ms` for both C-tcc and C-gcc, while the Rust and
Zig artifact writers take only a few milliseconds. Inspection found that the C
build request spawned `where`/`which` to resolve a bare compiler name on each
cold artifact before spawning that compiler for the real build. This lookup is
not needed for `tcc`/`gcc` found through `PATH`; process spawning already
performs it. C materialization now launches no locator process for bare names,
while explicit compiler paths and environment overrides remain honored. The
next release table should be checked primarily for a fall in C
`materialize_ms` and total C-tcc cold time.
```

### Pipeline stress extension: `combustion-3000`

`bvp_generated_backend_pipeline_comparison_table` now includes a fourth,
larger cold-bootstrap scenario, `combustion-3000`. It is intentionally limited
to this pipeline table: the callback-throughput and compile-preset stories
answer other questions and should not silently multiply an already expensive
build matrix.

For an initial release measurement, run only the new scenario with a small
multi-run sample:

```powershell
$env:BVP_PIPELINE_SCENARIO_FILTER="combustion-3000"
$env:BVP_PIPELINE_RUNS="3"
cargo test --release bvp_generated_backend_pipeline_comparison_table -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_PIPELINE_SCENARIO_FILTER
Remove-Item Env:\BVP_PIPELINE_RUNS
```

If the first table is stable enough, repeat with `BVP_PIPELINE_RUNS="10"` for
the long-run comparison against Lambdify and the compiler/toolchain routes.

Result 6 runs
[BVP backend pipeline compare] scenario=combustion-3000, residuals=18000, vars=18000, nnz=62988, multi-run bootstrap summary
route    | assembly   | variant        | preset      | ok/runs | symbolic_ms mean+/-std [min,max] | callable_prep_ms mean+/-std [min,max] | artifact_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms mean+/-std [min,max] | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         |   5/5   | 53567.073+/-12639.456 [38520.531,76143.996] | 2094.801+/-110.774 [1960.714,2272.534]  | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 7.048+/-0.337  | 55668.922+/-12740.232 [40537.550,78423.333] | ok 5/5
AOT      | ExprLegacy | Rust           | DevFastest  |   5/5   | 53567.073+/-12639.456 [38520.531,76143.996] | 45571.849+/-3103.080 [41643.515,49635.965] | 472.944+/-26.418 | 139.414+/-163.892 | 44944.187+/-3072.423 | 15.303+/-18.977 | 24.180+/-17.521 | 99163.102+/-14849.004 [80178.151,123627.371] | ok 5/5
AOT      | ExprLegacy | C-gcc          | DevFastest  |   5/5   | 53567.073+/-12639.456 [38520.531,76143.996] | 20822.620+/-1397.117 [18528.025,22262.895] | 267.806+/-18.262 | 7.420+/-0.805  | 20349.969+/-1362.248 | 197.424+/-53.772 | 20.763+/-11.948 | 74410.456+/-13554.107 [58515.064,98417.097] | ok 5/5
AOT      | ExprLegacy | C-tcc          | DevFastest  |   5/5   | 53567.073+/-12639.456 [38520.531,76143.996] | 1073.497+/-475.088 [801.918,2022.550]   | 241.528+/-13.668 | 82.936+/-114.749 | 535.767+/-25.106 | 213.266+/-328.395 | 8.813+/-2.419  | 54649.383+/-13062.249 [39354.763,78175.174] | ok 5/5
AOT      | ExprLegacy | Zig            | DevFastest  |   5/5   | 53567.073+/-12639.456 [38520.531,76143.996] | 250075.473+/-19853.316 [223309.374,272172.802] | 472.809+/-36.626 | 13.092+/-5.053 | 249564.480+/-19829.941 | 25.092+/-12.828 | 16.151+/-3.717 | 303658.698+/-30704.579 [261845.032,348328.564] | ok 5/5
AtomView-only planning stages. ExprLegacy rows are expected to be zero here; use the module/source table below for the active legacy module-build cost.
route    | assembly   | variant        | preset      | jac_prepare | lookup | jac_build | chunk_plan | lower | peephole | temp_reuse | module_push
-------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-gcc          | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | C-tcc          | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
AOT      | ExprLegacy | Zig            | DevFastest  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000
route    | assembly   | variant        | preset      | module_ms | module_init | residual_lower | jacobian_lower | source_probe | source_emit | c_header | packaging | artifact_other | source_kb
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000  | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000  | 0.000+/-0.000
AOT      | ExprLegacy | Rust           | DevFastest  | 206.095+/-25.804 | 0.367+/-0.049 | 200.691+/-25.287 | 120.328+/-51.378 | 0.000+/-0.000 | 237.392+/-10.668 | 0.000+/-0.000 | 17.962+/-0.509 | 11.494+/-0.474 | 10457.337+/-0.000
AOT      | ExprLegacy | C-gcc          | DevFastest  | 195.686+/-14.496 | 0.003+/-0.001 | 195.589+/-14.443 | 84.234+/-5.662 | 0.000+/-0.000 | 41.415+/-1.773 | 1.552+/-0.755 | 19.314+/-2.020 | 9.839+/-1.546  | 10548.394+/-0.000
AOT      | ExprLegacy | C-tcc          | DevFastest  | 172.744+/-11.266 | 0.002+/-0.000 | 172.655+/-11.291 | 73.547+/-5.447 | 0.000+/-0.000 | 39.577+/-2.301 | 0.013+/-0.002 | 17.036+/-1.223 | 12.159+/-3.420 | 10548.394+/-0.000
AOT      | ExprLegacy | Zig            | DevFastest  | 178.584+/-12.961 | 0.002+/-0.001 | 178.519+/-12.953 | 79.760+/-2.605 | 0.000+/-0.000 | 266.679+/-19.788 | 0.000+/-0.000 | 16.120+/-1.778 | 11.426+/-3.420 | 11230.216+/-0.000
route    | assembly   | variant        | preset      | residual_diff | jacobian_diff | status
---------------------------------------------------------------------------------------------------------------------
Lambdify | ExprLegacy | Lambdify       | n/a         | 0.000000e0    | 0.000000e0    | ok 5/5
AOT      | ExprLegacy | Rust           | DevFastest  | 0.000000e0    | 0.000000e0    | ok 5/5
AOT      | ExprLegacy | C-gcc          | DevFastest  | 0.000000e0    | 0.000000e0    | ok 5/5
AOT      | ExprLegacy | C-tcc          | DevFastest  | 1.776357e-15  | 8.881784e-16  | ok 5/5
AOT      | ExprLegacy | Zig            | DevFastest  | 3.552714e-15  | 8.881784e-16  | ok 5/5
[BVP backend compare] pipeline comparison finished scenario `combustion-3000`
ok

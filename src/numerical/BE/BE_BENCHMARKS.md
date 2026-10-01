# Backward Euler Benchmarks

Criterion benches live in `benches/`; allocator instrumentation is a separate
target. Keep them separate because a counting global allocator perturbs timing.
No performance ratio is asserted by unit/story tests.

## Targets and Scopes

| Target/group | Scope | Setup included? |
| --- | --- | --- |
| `be_workload_benches` / `be_history_assembly` | Isolated trajectory-matrix assembly; old clone/flatten control versus row-slice and owned-buffer transpose | Ready sample buffer excluded; output matrix construction is timed; owned route's input clone is batch setup |
| `be_workload_benches` / `be_native_solve` `warm-repeated` | Repeated full native BE solve on one initialized instance | Initial configuration excluded; callbacks are reinstalled by the normal solve entry point |
| `be_workload_benches` / `be_native_solve` `native-e2e` | Native configuration plus full solve | Included in each iteration |
| `be_workload_benches` / `be_parameter_continuation` `warm-rebind` | Two symbolic solves on one prepared instance while rebinding rate 1 -> 2 -> 1 | Initial symbolic preparation excluded; parameter binding and both solves included |
| `be_workload_benches` / `be_parameter_continuation` `*-series/targets-N` | N solves across repeated parameter values, using one prepared solver or fresh instances | Equal solve count; setup excluded for reused route and included for fresh route; generated-backend cache may be warm |
| `be_workload_benches` / `be_parameter_continuation` `*-second-segment` | Accepted-state continuation after a shared first segment versus a fresh solver for the second segment and output stitching | Criterion batch setup excludes the common first segment on both routes; timed work includes update+continue versus second-segment setup+solve+stitch |
| `be_history_allocation_audit` | Allocation count and allocated bytes for legacy clone/flatten, row-slice, and owned-buffer transpose | Ready input excluded; not a latency/RSS benchmark |
| `be_symbolic_frontend_benches` / `be_symbolic_frontend` | Dense coupled ExprLegacy versus AtomViewNative preparation and warm residual/Jacobian callback timings | Fixture construction and input cloning excluded from preparation timing; full BE solve is not compared because BE does not expose an AtomView assembly selector |
| `be_symbolic_execution_benches` / `be_symbolic_execution` | Lambdify ExprLegacy, Lambdify AtomViewNative, and dense AtomView AOT/tcc full solves; fresh solver setup and parameter-rebind series | AOT artifact is built/warmed before Criterion timing; fresh solver rows include solver setup but reuse the process artifact cache; not a cold-build comparison |
| `be_aot_build_link_provenance_and_timing_scopes_story` (ignored story) | One-shot cold setup+binding+solve comparison: AtomViewNative Lambdify versus AtomViewNative AOT/tcc on shared diffusion-8 | AOT uses `RebuildAlways` and a unique artifact directory; stage timings/build provenance are reported. Fixed route order, single sample and in-process execution make this diagnostic only, not a performance baseline |
| `be_process_isolated_cold_e2e_matrix_story` (ignored story) | Four child-process samples each for AtomViewNative Lambdify and AOT/tcc on shared diffusion-8, with alternating route order | Child E2E includes workload construction, solver setup, binding and solve; parent reports process wall separately. Each AOT child uses a unique output directory under `target/` (same volume as the lifecycle story), `RebuildAlways`, and `RAYON_NUM_THREADS=1`; parity and build/link provenance are gated |
| `be_workload_benches` / `be_combustion_parameter_continuation` | Three-state nonlinear combustion-like parameter rebind series versus fresh symbolic setup/solve, target counts 1 and 4 | Fixture is AtomViewNative; each route solves the same parameter sequence; a parity precheck runs before timing |

History assembly latency cases cover small shapes through `512x32` and
`256x64`; the separate allocation audit extends to `1024x64`. Both row-slice
and owned-transpose layouts are measured in the latency group. Native
diffusion-chain dimensions are 8, 32 and 64 with 16 steps, plus an
8-state/128-step output-growth case. Both analytic and FD Jacobian paths are
included. A 3-state combustion-like analytic-Jacobian case provides a nonlinear
small-system check. Dense BE scaling is intentional; these sizes are not a
claim that BE should replace a sparse solver for huge ODEs.

The parameter-continuation groups verify trajectory parity before timing,
including a segmented accepted-state reference. The repeated-series comparison
uses the same number of trajectories on both routes. Fresh-instance routes are
not process-isolated cold-compiler measurements; do not call them cold AOT or
use them to claim compilation savings.

The symbolic execution comparison uses the common parameterized diffusion
fixture at `numerical::ivp_workloads::diffusion_chain(8)`. The shared corpus
also provides large configurable diffusion chains, combustion-like kinetics,
Robertson, stiff-scalar and three-body workloads for reuse by BDF and Radau
tests/benches; solver-specific setup belongs in each solver's adapter, not in
duplicate equation definitions.

Solve benches consume `BE::trajectory()` by reference. This avoids timing the
public compatibility method `get_result()`'s full trajectory clone as if it
were solver work. Previously archived solve timings may have included that
consumer-side copy; treat them as a distinct historical scope until a fresh
release baseline is captured. The isolated history-assembly benchmark remains
unchanged because output-matrix construction is exactly what it measures.

## Commands

Cargo builds bench targets with the bench profile; do not add a separate
`--release` flag.

```powershell
cargo bench --no-default-features --bench be_workload_benches -- --noplot
cargo bench --no-default-features --bench be_symbolic_frontend_benches -- --noplot
cargo bench --no-default-features --bench be_symbolic_execution_benches -- --noplot
cargo bench --no-default-features --bench be_history_allocation_audit
```

Criterion output should be redirected to a dated log by the caller. Preserve
the complete stdout/stderr, compiler/toolchain, CPU/OS, commit, dirty state,
profile, command, and whether the run was thermally stable. Do not run the
allocation audit concurrently with Criterion timings.

## Interpretation Rules

- Compare the history assembly group first to verify the local effect of
  removing per-sample `DVector` allocations and the extra flatten pass. Treat
  row-slice versus owned-buffer transpose as a layout implementation comparison.
- Treat it as an isolated assembly microbenchmark, not as end-to-end solve
  speedup. Newton factorization, callbacks, and loop overhead are absent there.
- Use `native-e2e` for instance-construction plus solve and `warm-repeated` for
  repeated full solves. Report absolute times and uncertainty, not just ratios.
- The basic warm-rebind route runs two solves per Criterion iteration, while
  its single fresh-instance control runs one. The `targets-N` repeated-series
  routes have equal solve counts and are the preferred amortization comparison.
- Accepted-state continuation changes the parameter at the segment boundary;
  compare it only with the matching piecewise-rate reference, not a constant-
  parameter full-interval solve.
- `Off` telemetry is used in Criterion to avoid instrumentation cost. Story
  tests separately use Timings for diagnostic stage counts/durations.
- Symbolic execution rows distinguish warm solve, fresh solver setup with a
  process-cached AOT artifact, and warm parameter rebind. Cold build/link is
  measured by the ignored lifecycle story, not folded into these Criterion
  iterations. That story's one-shot matched Lambdify/AOT wall times are useful
  for lifecycle attribution only. Its Rayon environment must be matched against
  the process child (`RAYON_NUM_THREADS=1`) before comparing wall times. The
  process-isolated matrix starts a fresh solver process per sample but does not
  clear OS filesystem/page caches or isolate machine load; retain raw output
  and environment metadata.
- A future Newton-workspace optimization requires a separate before/after
  benchmark with identical problem, callbacks, step grid, convergence settings
  and telemetry. Do not infer its value from the history-assembly group.

## Baseline State

The first complete release baseline is recorded in
`BE_PERFORMANCE_BASELINE.md`, with immutable raw logs in
`test_reports/BE/release/archive/`. The legacy assembly control supplies a local
old-versus-new assembly comparison; it does not reconstruct the old complete BE
solver. Criterion group percentages compare against local persisted state and
must be interpreted together with absolute intervals and run scope.

The symbolic frontend comparison is exploratory. It gates any future public BE
frontend selector; preparation/callback wins alone do not establish a full-solve
benefit. Compare absolute preparation and callback times at equal problem sizes,
and keep ExprLegacy/AtomView parity as a prerequisite.

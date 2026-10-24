# Radau User Guide

This is the guide for the maintained Radau implementation in
`src/numerical/Radau/new`. The archived implementation is kept only as a
reference and is not the default production route.

## 1. Choose The API

There are two public entry points.

`UniversalODESolver::radau(...)` is the small convenience constructor. It uses
the maintained fixed-order Radau route with symbolic Lambdify, ExprLegacy, and
dense storage. It is useful when one facade must switch between several
solvers and no Radau-specific policy is needed.

For frontend, layout, continuation, output, telemetry, execution policy, or
AOT control, use the direct API:

If you deliberately use the universal facade, its production Radau route keeps
telemetry `Off` by default. Opt in with
`UniversalODESolver::with_radau_telemetry_mode(...)`, choosing one of
`RadauTelemetryMode::Counters`, `Timings`, or `Detailed`. Use `Off` to disable
collection explicitly.

```rust
use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauProblem, RadauSolver,
    RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let problem = RadauProblem::new(
    vec![Expr::parse_expression("-k*y")],
    vec!["y".to_owned()],
    "t",
)
.with_parameters(vec!["k".to_owned()])
.with_jacobian(vec![Expr::parse_expression("-k")]);

let config = RadauConfig {
    frontend: RadauFrontend::ExprLegacy,
    matrix_layout: RadauMatrixLayout::Dense,
    telemetry: RadauTelemetryMode::Timings,
    t_bound: 1.0,
    ..RadauConfig::default()
};

let mut solver = RadauSolver::prepare(problem, config)?;
let solution = solver.solve_with_parameters(&[1.0], &[2.0])?;
println!("t={} y={:?}", solution.t, solution.y);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Preparation is fallible and separate from solving. A prepared solver owns the
symbolic callbacks and can be reused for a parameter series.

## 2. Select A Frontend

`RadauFrontend::ExprLegacy` keeps symbolic preparation in the legacy expression
representation. It is a good small-system baseline and a useful independent
correctness reference.

`RadauFrontend::AtomViewNative` converts the model once to AtomView and keeps
the residual/Jacobian preparation and callback path native to that
representation. It is usually the first route to try for large symbolic
systems, but there is no universal winner.

Frontend choice is independent from runtime execution:

- `Lambdify` prepares in-process closures and has no external compiler setup.
- `Aot` generates, compiles, links, and reuses a native artifact.

Both frontends use the same Radau numerical core. Compare them on the same
equations, initial state, tolerances, layout, and output policy.

## 3. Select A Matrix Layout

Choose the layout once during preparation:

```rust
let dense = RadauMatrixLayout::Dense;
let sparse = RadauMatrixLayout::Sparse;
let banded = RadauMatrixLayout::Banded { lower: 1, upper: 1 };
```

Use `Dense` for fully coupled Jacobians and small or medium systems. Use
`Banded` when the Jacobian has a stable narrow bandwidth. Use `Sparse` when a
stable sparse pattern is available and the system is large enough for sparse
storage to amortize its bookkeeping.

There is no silver bullet. A diffusion chain often favors Banded or Sparse;
small combustion or fully coupled kinetics can favor Dense. The layout must
match the actual Jacobian structure. Do not select a structured layout merely
because the residual is cheap.

The backend is not silently converted between Dense, Sparse, and Banded during
scalar callback work. A prepared layout owns its corresponding callback and
linear workspace.

## 4. Jacobian Contract

For symbolic Radau, provide an explicit Jacobian whenever possible. The vector
is row-major: for `n` equations and `n` variables it contains
`df[0]/dy[0] .. df[0]/dy[n-1]`, then the second row, and so on.

```rust
let problem = RadauProblem::new(
    vec![Expr::parse_expression("-k*y")],
    vec!["y".to_owned()],
    "t",
)
.with_parameters(vec!["k".to_owned()])
.with_jacobian(vec![Expr::parse_expression("-k")]);
```

`RadauJacobianSource::Analytic` is the default and is the recommended policy
for stiff production workloads. `FiniteDifference` is useful as a diagnostic
or fallback policy, but it adds residual evaluations and should be compared
against an analytic-Jacobian run. AOT and structured layouts should be tested
with an explicit Jacobian first; this makes the generated contract and pattern
validation unambiguous.

For ExprLegacy Lambdify, an omitted Jacobian with the analytic source selected
is derived from the residual during preparation. The derivative builder keeps
only state-variable dependencies and publishes row-major coordinates, so
Sparse and Banded routes do not construct an intermediate dense symbolic
matrix. Supplying `.with_jacobian(...)` continues to use the caller's explicit
row-major expressions. AOT still requires an explicit Jacobian.

### Native callback API

When the model is already implemented as Rust callbacks, use
`RadauNativeSolver` instead of rebuilding a symbolic expression graph. The
residual callback is mandatory. Supplying a Jacobian selects the analytic
path; passing `None` selects the typed forward-difference fallback. This
boundary currently uses Dense matrices and does not silently convert a native
callback into Sparse or Banded storage.

```rust
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::Radau::{RadauConfig, RadauNativeSolver};

let mut analytic = RadauNativeSolver::prepare(
    RadauConfig { t_bound: 0.5, ..RadauConfig::default() },
    |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
    Some(|_t: f64, _y: &DVector<f64>| DMatrix::from_vec(1, 1, vec![-1.0])),
)?;
let solution = analytic.solve(&[1.0])?;

let mut finite_difference = RadauNativeSolver::prepare(
    RadauConfig { t_bound: 0.5, ..RadauConfig::default() },
    |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
)?;
let fd_solution = finite_difference.solve(&[1.0])?;
# let _ = (solution, fd_solution);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The public configuration reports `RadauExecution::NativeCallbacks` and either
`RadauJacobianSource::Analytic` or `FiniteDifference`. With telemetry enabled,
the latter reports its internal residual probes as
`finite_difference_probes`; those probes are not user-level residual calls.

The native callback boundary is currently Dense-only. Use the symbolic
`RadauProblem` API for Sparse or Banded layouts; it preserves the selected
structured backend without converting a native dense callback at runtime.

## 5. Tolerances And Step Controls

The main numerical controls are:

```rust
let config = RadauConfig {
    rtol: 1e-6,
    atol: 1e-9,
    first_step: None,
    max_step: f64::INFINITY,
    max_steps: 1_000_000,
    max_newton_iterations: 6,
    max_retries: 12,
    ..RadauConfig::default()
};
```

Start with the defaults, then tighten `rtol` and `atol` together with a
problem-scaled validation criterion. Set `max_step` when a known physical
scale or rapidly changing forcing requires it. Set `first_step` only when the
default initial-step estimate is inappropriate. Increasing Newton or retry
budgets can hide a bad Jacobian or poor scaling, so inspect telemetry before
raising them.

## 6. Single Solve And Continuation

For a parameter-free model:

```rust
let solution = solver.solve(&[1.0])?;
```

For a model declared with `with_parameters`, bind values explicitly:

```rust
for k in [0.5, 1.0, 2.0, 4.0] {
    let solution = solver.continue_with_parameters(&[1.0], &[k])?;
    println!("k={k}: y(1)={:.9e}", solution.y[0]);
}
```

Continuation reuses the prepared symbolic model. It does not repeat symbolic
differentiation or frontend preparation merely because parameter values
changed. The first solve pays preparation cost; later solves are the relevant
warm path. For a fair benchmark report the first preparation separately from
the repeated continuation solves.

To reuse the prepared callbacks on a new interval:

```rust
solver.restart(0.0, 2.0)?;
let solution = solver.continue_with_parameters(&[1.0], &[2.0])?;
```

`restart` changes the interval, not the prepared model. Supply a new initial
state explicitly to the following solve.

## 7. Output And Post-Processing

Output retention is a deliberate memory policy:

```rust
use RustedSciThe::numerical::Radau::RadauOutputPolicy;

let final_only = RadauOutputPolicy::FinalOnly;
let dense_output = RadauOutputPolicy::Dense;
let selected_times = RadauOutputPolicy::Sampled(vec![0.25, 0.5, 0.75, 1.0]);
```

`FinalOnly` is the lowest-overhead choice. `Dense` retains continuous-output
segments and enables `solution.sample(t)` and `solution.sample_many(&times)`.
`Sampled` evaluates only requested output times. Do not enable dense output in
a large production run unless post-processing really needs intermediate
states.

## 8. Telemetry

Telemetry is disabled by default and is designed not to charge the normal
path when disabled:

```rust
let config = RadauConfig {
    telemetry: RadauTelemetryMode::Counters,
    ..RadauConfig::default()
};
```

Use `Counters` for calls, accepted/rejected steps, Newton iterations,
factorizations, and linear solves without clock reads. `Timings` collects only
wall-clock durations; `Detailed` collects both counters and durations. The
report separates preparation and runtime scopes; parent and child timings are
diagnostic and must not be added together.

Important fields include residual/Jacobian calls, factorization and solve
counts, callback timing, Newton timing, linear timing, output timing, parameter
rebinds, and AOT build/link/publication timing. Record the frontend, execution
family, layout, output policy, and parameter phase alongside the numbers.

## 9. AOT Lifecycle

AOT is selected explicitly and requires an AOT configuration:

```rust
use RustedSciThe::numerical::Radau::{RadauAotConfig, RadauExecution};

let config = RadauConfig {
    execution: RadauExecution::Aot,
    frontend: RadauFrontend::AtomViewNative,
    matrix_layout: RadauMatrixLayout::Dense,
    aot: Some(RadauAotConfig::build_if_missing_release(
        "target/radau-artifacts",
    )),
    ..RadauConfig::default()
};
```

The supported lifecycle helpers are:

- `build_if_missing_release(path)`: compile only when the artifact is absent;
- `require_prebuilt()`: fail fast if the artifact is not already published;
- `rebuild_always_release(path)`: force a fresh build for cold lifecycle tests.

Use `BuildIfMissing` for normal development, `RequirePrebuilt` for deployment
or process-isolated consumers, and `RebuildAlways` only for cold preparation
measurements. Keep AOT build/link time separate from warm solve time.

## 10. Parallel Policy

The default is safe sequential execution:

```rust
use RustedSciThe::numerical::Radau::RadauExecutionPolicy;

let sequential = RadauExecutionPolicy::Sequential;
let forced = RadauExecutionPolicy::Parallel { min_work: 256 };
let automatic = RadauExecutionPolicy::Auto { min_work: 256 };
```

Use `Parallel` only after measuring the workload. `Auto` is intended to avoid
parallel dispatch when worker overhead cannot amortize. Compare callback-only
and full-solve measurements separately; a callback win can disappear under
linear algebra or controller costs.

## 11. Error Handling And Examples

Preparation and solving return `Result<_, RadauError>`. Match error kinds in
application code rather than parsing display strings. Shape errors, invalid
configuration, unsupported routes, callback failures, Newton failures, step
budget exhaustion, and output-range errors are distinct failure classes.

Runnable examples in this repository:

- `examples/radau_lambdify_single.rs`: one parameterized Lambdify solve;
- `examples/radau_lambdify_continuation.rs`: prepared-model continuation;
- `examples/radau_public_api_guide.rs`: AtomView, counters, and dense output;
- `examples/radau_backends_guide.rs`: frontend/layout comparison;
- `examples/radau_aot_single.rs` and `examples/radau_aot_continuation.rs`:
  explicit AOT lifecycle examples;
- `examples/universal_ode_example.rs`: the compact universal facade.

Validate correctness on the same workload before comparing performance. Keep
separate baselines for preparation, callback-only execution, full solve, and
continuation. A result that is faster on one workload or layout is not a
general claim about Radau.

## 12. Text Task Documents

The task-document route is a compact, reproducible adapter for the maintained
Radau solver. It is intended for examples and batch runs; use the native
`RadauConfig`/`RadauProblem` API when you need full frontend, layout, AOT, or
parallel-executor control.

```text
task
solver: IVP
method: Radau5

equations
arg: t
parameters: k
parameter_values: 2.0
y: -k*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
first_step: 1e-3

postprocessing
output_policy: none
```

The parser dispatches `solver: IVP` plus `method: Radau5` through the native
Radau adapter. `rtol` and `atol` are independent; `tolerance` is a convenient
common value when both tolerances should be equal. `max_step`, `first_step`,
and the iteration/step limits are validated before solver construction.
`Radau3` is not a supported task method: the maintained implementation uses
the fixed Radau IIA order five route.

Task documents currently use the compact symbolic adapter and its supported
defaults. Selecting `ExprLegacy` versus `AtomViewNative`, `Dense` versus
`Sparse`/`Banded`, AOT build policy, telemetry mode, or a custom `Parallel`/
`Auto` worker policy belongs to the typed Radau API. This separation keeps the
text format stable and prevents a task file from silently ignoring an advanced
option.

Continuation reuses the parsed symbolic model when the native contract allows
it:

```text
continuation
parameter: k
values: 1.0, 2.0, 4.0
mode: prepared
restart_each: false
```

`fresh` rebuilds each segment, `warm` reuses prepared callbacks with the
previous accepted state, and `prepared` requests the strongest reusable model
path. A deliberate restart may provide per-segment `y0`, `t0`, and `t_end`.
Keep preparation, callback, full-solve, and continuation timings separate in
reports. `output_policy` accepts `none`, `terminal`, `plotters`, or `gnuplot`;
`plot` remains only a compatibility alias. Comments beginning with `#` are
ignored. The shared grammar and batch-runner conventions are documented in
[`TASK_DOCS_GUIDE_EN.md`](../../command_interpreter/TASK_DOCS_GUIDE_EN.md).

## 13. SciPy Parity And Runtime Evidence

The numerical core is checked against the official SciPy `v1.18.1` Radau
implementation. The reference subset and its provenance are maintained in
`src/numerical/ivp_scipy_reference`; the parity reports record Python, NumPy,
SciPy versions and source hashes. Matched tests cover stiff scalar, Robertson,
combustion-like and HIRES workloads, endpoint and common dense-output samples,
controller traces, backward integration, automatic and explicit initial-step
selection, invalid configuration, underflow and step-budget behavior. There
are also exact-solution/invariant checks and an independent Robertson
cross-check against LSODE2/BDF.

The goal is fidelity of the mathematical method and controller policy, not
bit-for-bit identity of every counter or implementation detail. The Radau
IIA order-five method, transformed collocation solve, modified Newton policy,
embedded estimator, step controller and dense-output polynomial are checked
stage by stage. Remaining observed differences are understood:

- Rust prepares callbacks lazily at solve time; SciPy evaluates the initial
  RHS/Jacobian during solver construction. This is an API lifecycle difference.
- Rust reuses workspace buffers and does not need SciPy's final endpoint RHS
  evaluation, so its residual count can be one lower on matched cases.
- Dense LU implementations differ (Rust in-place real/complex backend versus
  SciPy LAPACK). Near a Newton threshold this can change a Jacobian refresh or
  retry decision, as in the base Robertson case, while the trajectory remains
  within the stated parity envelope. A nearby-tolerance case confirms that the
  branch moves with the numerical boundary.
- On step underflow Rust reports its typed error before a doomed Newton trial;
  SciPy may execute and reject one such trial first. Both terminate without an
  accepted step. Rust's accepted-step budget is an API extension and is
  compared with an explicitly labeled external SciPy driver cap.

These are intentional lifecycle/backend/API differences, not unreviewed
changes to the numerical method. The detailed ledger and compact test tables
are in `src/numerical/ivp_scipy_reference/TODO.md` and
`src/numerical/Radau/RADAU_STORY_TESTS.md`.

### Numerical-Core Timing Snapshot

One release experiment compared a pre-prepared Rust native-callback Radau
solve with SciPy construction plus integration and the same three output
samples. Both used matched equations, analytic Jacobians, tolerances, interval,
step controls, and disabled telemetry. Across the three tested workloads the
reported SciPy/Rust ratios were:

| Workload | Rust solve (median) | SciPy measured scope (median) | Reported ratio |
|---|---:|---:|---:|
| Stiff scalar | 0.0065 ms | 0.4633 ms | 71.28x |
| Robertson | 0.0216 ms | 2.0500 ms | 94.91x |
| Combustion-like | 0.0073 ms | 0.4982 ms | 68.25x |

This is strong evidence that the Rust numerical path is very fast on these
representative small stiff workloads, but it is not a universal speedup claim.
The timing scopes are not identical: Rust excludes its already-completed
callback preparation, while the Python measurement includes solver
construction. The sample is also limited to three workloads on one machine;
other equations, dimensions, layouts, and runtime environments can produce
different results. See the archived report
`test_reports/Radau_SciPy_Parity/release/numerical__Radau__tests__scipy_parity__radau_scipy_release_core_timing_benchmark_story.md`.

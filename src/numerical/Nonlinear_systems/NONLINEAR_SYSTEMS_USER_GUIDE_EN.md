# Nonlinear Systems User Guide

## Recommended Typed Path

For new code, prepare a symbolic system once and keep numeric parameter
bindings separate from that preparation:

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let prepared = PreparedSymbolicNonlinearProblem::from_strings(
    vec!["a*x + y - 3".into(), "x - y".into()],
    SymbolicProblemOptions::new()
        .with_variables(vec!["x".into(), "y".into()])
        .with_equation_parameters(vec!["a".into()]),
)?;

let bound = prepared.bind_values(DVector::from_vec(vec![2.0]))?;
let options = SolveOptions {
    tolerance: 1e-10,
    max_iterations: 64,
    diagnostics: DiagnosticsOptions {
        collect_history: false,
        collect_statistics: true,
        ..DiagnosticsOptions::default()
    },
    ..SolveOptions::default()
};
let result = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
    .solve(&bound, DVector::from_vec(vec![1.0, 1.0]), options)?;
```

`PreparedSymbolicNonlinearProblem` owns the parsed equations, variable order,
symbolic Jacobian preparation, and callable backend. `bind_values` validates a
numeric vector against the declared parameter schema and returns a borrowed
bound view. Rebinding different values does not repeat symbolic preparation;
the same prepared object can be used for continuation-like sweeps and several
initial guesses.

`SymbolicProblemOptions::with_lambdify_backend()` is the default explicit
choice for the in-process symbolic route. Generated AOT uses the generic AOT
lifecycle and must be materialized, built, registered, and linked before an
AOT-only problem can execute. `BuildIfMissing` and `RequirePrebuilt` are
lifecycle policies, not interchangeable solve methods; see the nonlinear
`STORY_TESTS.md` ledger for the tested artifact contract.

### Choosing the symbolic frontend

The dense Lambdify backend exposes two frontends:

```rust
let expr_legacy = SymbolicProblemOptions::new()
    .with_lambdify_frontend(SymbolicLambdifyFrontend::ExprLegacy);
let atom_native = SymbolicProblemOptions::new()
    .with_atom_native_frontend();
```

`ExprLegacy` is the compatibility default. `AtomViewNative` converts the
public `Expr` input once, performs Jacobian differentiation on Atom storage,
and evaluates residuals/Jacobians into caller-owned dense buffers. Generated
dense AOT supports both frontends: the AtomView route lowers residuals and
Jacobians directly from Atom storage through the shared dense AOT ABI, without
converting generated expressions back to Expr.

This solver is intentionally dense. Sparse and banded layouts are not silently
emulated here; use an ODE solver such as LSODE2 when a layout-specific linear
backend is required.

### Parameter Updates And Structural Rebuilds

The order passed to `with_equation_parameters` is the parameter ABI. A call to
`bind_values` (or the compatibility `set_parameter_values`) changes only the
numeric values in that existing schema. It does not reparse equations,
differentiate, lambdify, rebuild an AOT artifact, or change the variable order.
Changing equations, variables, parameter names/order, or generated chunking is
a structural change and requires preparing a new problem.

### Per-Attempt Diagnostics

When statistics are collected, `SolveStatistics::attempts` contains one entry
for each outer nonlinear iteration after the initial state evaluation. The
entries use the same solver-level counter semantics as the aggregate fields:
one residual/Jacobian evaluation means one complete provider request, even if
an AOT provider executes several generated jobs underneath it. Each entry also
reports Jacobian refresh/reuse, factorizations, linear solves, accepted and
rejected trials, and inclusive stage durations:

```rust
for attempt in &result.statistics.attempts {
    println!(
        "iteration={} R/J={}/{} refresh/reuse={}/{} factor/linear={}/{}",
        attempt.iteration,
        attempt.residual_evaluations,
        attempt.jacobian_evaluations,
        attempt.jacobian_refreshes,
        attempt.jacobian_reuses,
        attempt.linear_factorizations,
        attempt.linear_solves,
    );
}
```

The initial residual/Jacobian evaluation belongs to aggregate statistics, not
to an attempt entry. `termination_retries` is explicitly `None` on the
generic engine because it has no separate method-specific termination-retry
boundary. Backend-internal generated job/chunk counts belong to
`SymbolicPreparationReport`, not to solver counters. With statistics disabled,
`attempts` is empty and timing/counter fields are not measurements.

### Detailed Preparation Telemetry

Solver statistics describe numerical attempts. If the cost of building a
prepared symbolic problem also matters, enable the separate opt-in preparation
timeline:

```rust
let prepared = PreparedSymbolicNonlinearProblem::from_strings(
    equations,
    SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_preparation_telemetry(PreparationTelemetryMode::Collect),
)?;

if let Some(report) = prepared.preparation_report().detailed.as_ref() {
    println!(
        "preparation total={:?}, input={:?}, mode={:?}",
        report.total_wall_time, report.input_kind, report.execution_mode,
    );
    for stage in &report.stages {
        println!("{:?}: {:?}", stage.stage, stage.wall_time);
    }
}
```

The aggregate `SymbolicPreparationReport` remains available when detailed
telemetry is disabled. Each detailed stage is an exclusive measurement when
the implementation can isolate it; `None` means that the stage did not apply
or could not be isolated, not that it took zero time. `unattributed_wall_time`
is the remaining preparation time. Binding new parameter values does not
repeat preparation and therefore does not create a new preparation report.

For diagnostics pipelines that must preserve the failure boundary, use the
opt-in detailed constructor. It returns `SymbolicPreparationFailure`, which
contains the original `SolveError` and the completed stage records:

```rust
let attempt = SymbolicNonlinearProblem::from_strings_with_options_detailed(
    equations,
    SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_preparation_telemetry(PreparationTelemetryMode::Collect),
);
if let Err(failure) = attempt {
    eprintln!("{}; telemetry={:?}", failure, failure.telemetry);
}
```

The compatibility constructors remain unchanged and return `SolveError`
directly. Failure telemetry is intended for reporting and debugging; it does
not turn a failed preparation into a usable problem.

### Optional Solver Logging

Solver logging is disabled by default. The setting is scoped to one
`SolverEngine` invocation and does not modify the process-wide logger:

```rust
let quiet = SolveOptions::default();
let verbose = SolveOptions::default().with_logging(EngineLogLevel::Debug);
let quiet_again = verbose.without_logging();
```

The same policy is available on `DiagnosticsOptions` through
`with_logging(...)` and `without_logging()`. Runtime method diagnostics use
this public option; they do not write directly to stdout. The default path
therefore avoids log formatting and emission, while applications that install
the `log` facade can opt into `Info`, `Warn`, or `Debug` records explicitly.

### AOT Artifact Lifecycle

For a cold production-style run, request an output directory and use
`BuildIfMissing`:

```rust
let cold = SymbolicNonlinearProblem::from_strings_with_generated_backend(
    equations.clone(),
    problem_options.clone(),
    SymbolicGeneratedBackendConfig::build_if_missing_release(&output_dir),
)?;
let resolver = cold.updated_resolver.clone();
println!(
    "cold: backend={:?}, action={:?}, manifest_key={:?}, lifecycle_key={:?}, build={:?}",
    cold.selected_backend,
    cold.preparation_report.artifact_action,
    cold.preparation_report.artifact_key,
    cold.preparation_report.artifact_lifecycle_key,
    cold.preparation_report.build_duration,
);
```

After the artifact has been published and its resolver is retained, a strict
warm run uses `RequirePrebuilt` and never compiles:

```rust
let warm = SymbolicNonlinearProblem::from_strings_with_generated_backend(
    equations,
    problem_options,
    SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(resolver),
)?;
assert!(warm.build_result.is_none());
assert_eq!(warm.preparation_report.artifact_action, SymbolicArtifactAction::Reused);
```

`RequirePrebuilt` reports a typed error for a missing, incompatible, or
unlinked artifact; it is not a silent fallback to Lambdify. The resolver is an
in-process snapshot, so a fresh process must discover the persisted compatible
artifact from the configured output location. Parameter rebinding remains a
cheap numerical operation after either lifecycle stage.

`artifact_key` identifies the mathematical/generated problem used by the
resolver. `artifact_lifecycle_key` additionally identifies the known build
profile and `AotCompileConfig`, so a ready marker from a different Rust build
configuration is not reused. When `RequirePrebuilt` receives an externally
supplied resolver whose original build settings are unknown, the lifecycle key
is intentionally unavailable rather than guessed.

## Choosing Runtime Policy And Backend

### Sequential Versus Parallel Lambdify

Select the callback execution policy in the prepared symbolic options:

```rust
let sequential = SymbolicProblemOptions::new()
    .with_variables(variables.clone())
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential);

let parallel = SymbolicProblemOptions::new()
    .with_variables(variables)
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel {
        min_work: 1_500,
    });
```

`Sequential` is the default and is the right first choice for small or sparse
systems. `Parallel { min_work }` is an explicit opt-in: it dispatches only
when the prepared evaluator work reaches the threshold. The threshold is a
runtime dispatch rule, not a convergence or accuracy parameter. Start with
the default policy, then benchmark the actual residual/Jacobian pattern on the
target machine before choosing `min_work`.

The release corpus supports this conservative policy. At dimension `512`,
prepared Sequential was the fastest complete Newton route in all six
large-corpus rows. Parallel was useful for the wider `band-five` callback and
was faster than the legacy compatibility route there, but it remained slower
than prepared Sequential in the complete solve. In the threshold sweep,
`Parallel { min_work: 1 }` was about `9-10%` slower than Sequential on the
tested patterns; activating parallelism exactly at structural `nnz` was slower
again. These results are recorded in `STORY_TESTS.md` Sections 44 and 45.

### Why The New Jacobian Does Not Use A Mutex

The prepared Jacobian already knows the independent nonzero evaluator layout.
The new parallel path assigns disjoint rows/entries to workers and writes
directly into caller-owned `DMatrix` storage. There is no shared accumulator,
per-entry `Mutex`, or intermediate dense matrix. The legacy compatibility path
keeps its old allocated-return and mutex/Rayon dispatch semantics so existing
callers are not broken.

This is both a correctness and performance design: disjoint writes remove the
serialization point, while `jacobian_into` removes a result allocation from
the callback boundary. On the release `band-five` corpus at dimension `512`,
legacy Jacobian evaluation took `1230.7 us`, prepared Sequential `397.15 us`,
and prepared mutex-free Parallel `426.31 us`. Thus Parallel was about `2.89x`
faster than legacy, while Sequential was faster still. Parallel is not
automatically faster than Sequential because worker scheduling can dominate
when each evaluator is cheap. Correctness and deterministic sparse-layout
parity are covered by Sections 41 and 42.

### Lambdify Versus AOT And Break-Even

Use Lambdify when iteration speed, portability, and short startup matter, or
when the problem will be solved only a few times. Use AOT when a toolchain is
available, the artifact can be persisted and reused, and callback throughput
has been demonstrated for the actual workload. The first AOT call includes
materialization, compilation, linking, and publication; compare that cold
cost separately from warm solves.

For a rough estimate, use:

```text
break_even_solves ~=
    (AOT preparation + build - Lambdify preparation)
    / (Lambdify warm solve - AOT warm solve)
```

The denominator must be positive. If AOT is slower when warm, there is no
break-even point for that workload under the measured configuration. The
release large-story result demonstrates why this must be measured: at `512`,
Lambdify warm total was `17.242 ms`, AOT warm total `17.835 ms`; AOT Jacobian
was `2.913 ms` versus `2.296 ms`, while residual and linear stages were close.
At `128`, AOT warm was faster (`0.397 ms` versus `0.516 ms`), but its roughly
`365 ms` build cost implies thousands of repeated solves before amortization.
Therefore the current evidence does not justify a universal AOT speed claim.
It supports AOT as a lifecycle and deployment option whose break-even is
workload-specific. See Sections 53, 64, and 65.

### Architecture Decisions With Evidence

- **Prepared versus bound objects:** symbolic parsing, differentiation, and
  callback construction happen once; parameter rebinding creates a cheap
  solver-facing view. This is why parameter sweeps must reuse the prepared
  object rather than rebuild it.
- **Solver-level versus generated-job counters:** one residual or Jacobian
  counter means one complete provider request. Internal AOT chunks/jobs are
  reported as backend detail, so Lambdify and AOT remain comparable.
- **Separate manifest and lifecycle identities:** the mathematical artifact
  key serves resolver lookup, while profile/compiler configuration participates
  in the on-disk lifecycle key. A `Debug` marker cannot silently satisfy a
  different build configuration.
- **Fail-closed prebuilt policy:** `RequirePrebuilt` reports a typed error for
  missing, stale, incompatible, or unlinked artifacts; it does not silently
  fall back to Lambdify. This keeps deployment reproducible.
- **Whole dense AOT blocks by default:** the `n=512` layout profile found
  `Whole` faster than row chunks 32 and 64. The caller-side adaptation was
  `0.216 ms` of a `0.301 ms` full Jacobian call, so this is a measured dense
  route decision, not a rule for sparse or banded systems.

### Telemetry Cost

Solver statistics and preparation telemetry answer different questions.
`collect_statistics` provides solver-level counts and stage timers;
`collect_history` additionally stores per-iteration snapshots and can cost
more memory. Detailed preparation telemetry is separately opt-in through
`PreparationTelemetryMode::Collect`. The release preparation audit measured
only hundredths of a millisecond difference between disabled and enabled
detailed preparation telemetry at dimensions `40`, `128`, and `512`, with
overlapping run ranges. That audit did not establish a universal warm-solve
percentage, so use disabled history/statistics for pure hot-path benchmarks
unless the diagnostic data is itself the subject of the experiment.

## Bounds And Diagnostics

Use `SolveOptions::bounds` for a box-constrained solve. Bounds belong to the
solver options, while equations and parameter bindings belong to the problem:

```rust
let bounds = Bounds::new(vec![(0.0, 3.0), (0.0, 2.0)])?;
let options = SolveOptions {
    bounds: Some(bounds),
    ..SolveOptions::default()
};
```

With `collect_statistics: true`, `SolveResult::statistics` reports solver-level
residual, Jacobian, linear-solve, accepted/rejected-step counters and
cumulative stage durations. `collect_history` is separate and can add memory
and allocation pressure. Disabled diagnostics should be used for hot-path
timing when history is not part of the question.

For pure numerical problems, implement `NonlinearProblem` and
`JacobianProvider` with caller-owned `DVector`/`DMatrix` values. The optional
`residual_into` and `jacobian_into` hooks let a provider fill solver-owned
buffers and are the appropriate route for allocation-sensitive workloads.
The mathematical method is unchanged; the hooks only change value transport.

## Choosing A Method

Plain Newton is usually the fastest near a well-scaled solution but is more
sensitive to a remote initial guess or a singular Jacobian. Damped Newton is a
good first choice when backtracking or bounds are needed. Levenberg-Marquardt
and trust-region variants are useful for difficult least-squares-like or
poorly scaled systems. No single wall-clock number ranks all methods: compare
convergence, residual quality, rejected trials, and stage timings on the
problem class that matters.

### Backtracking Levenberg-Marquardt

`BacktrackingLevenbergMarquardt` is an explicit peer of the classical,
MINPACK-style, Nielsen, and trust-region methods. It solves the
identity-damped normal equations

`(J^T J + lambda I) delta = -J^T residual`

and tests feasible trial points with `alpha = 1`, halving `alpha` until the
residual norm strictly decreases or `alpha_min` is reached. After an accepted
step, `lambda` is multiplied by `lambda_decrease`; after a failed line search,
it is multiplied by `lambda_increase`. The default multipliers are `0.3` and
`10.0`, respectively. The generic engine still requires the requested residual
tolerance for `Converged`; a successful line-search step is not, by itself, a
convergence claim.

```rust
let method = NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
    BacktrackingLevenbergMarquardtMethod::default(),
);
let result = method.solve(&problem, initial_guess, SolveOptions::default())?;
```

Use this method when identity damping plus a strict feasible backtracking
policy is the intended numerical model. Use `LevenbergMarquardt` when the
classical configurable scaling policy is intended; the two variants are
deliberately separate and neither silently replaces the other.

## Allocation Audit

Run the process-level release audit with:

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

The audit covers dimensions 32, 128, and 512, accepted solves with history
off, history comparisons at the smallest and largest dimensions, reusable
callback outputs, and a separate rejection-heavy control. It reports
allocation/deallocation counts and bytes for the complete solve-result
lifetime. The counting allocator changes process behavior, so its elapsed
times are diagnostic only and must not be compared with ordinary Criterion or
application wall-clock measurements. The audit also does not identify every
individual copy or measure peak resident memory; use it as evidence for a
focused optimization hypothesis, then require correctness tests before a
hot-path change.

The executable example is:

```text
cargo run --example nonlinear_systems_modern_guide
```

The compatibility and lifecycle examples are:

```text
cargo run --example nonlinear_lambdify_legacy_guide
cargo run --example nonlinear_lambdify_prepared_guide
cargo run --example nonlinear_aot_lifecycle_guide
```

The older `nonlinear_systems_guide` remains available as a compatibility
showcase for the legacy LM wrapper.

## Rectangular Least-Squares Problems

The canonical least-squares solver minimizes a residual vector `r(x)` and is
useful for overdetermined or underdetermined models, noisy observations, and
systems whose residuals cannot all be zero at once. It is separate from the
root-finding contract: a root solver searches for `r(x) = 0`, while
least-squares LM minimizes `0.5 * ||r(x)||^2` and reports the final objective.
For an inconsistent fit, a nonzero final objective is expected.

The shared `NonlinearSolver` selector has distinct `Root` and `LeastSquares`
variants. The least-squares route is intentionally **not** a variant of
`NonlinearSolverMethod`: that enum is consumed by the square-system root engine
and returns `SolveResult`, whereas least-squares accepts its own rectangular
problem contract and returns `MinimizationReport`.

### Numerical Residual And Jacobian Callbacks

Implement `LeastSquaresProblem` (or use `ClosureLeastSquaresProblem`) with the
current parameter vector, a residual callback, and its Jacobian. This example
fits a line to three observations; the inconsistent data leave a small
nonzero residual at the optimum:

```rust
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let problem = ClosureLeastSquaresProblem::new(
    DVector::from_vec(vec![0.0, 0.0]), // [intercept, slope]
    |p| DVector::from_vec(vec![p[0] - 1.0, p[0] + p[1] - 2.0, p[0] + 2.0 * p[1] - 2.9]),
    |_| DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]),
);

let method = NonlinearSolver::LeastSquares(
    LeastSquaresLevenbergMarquardt::new().with_tol(1e-10),
);
let (problem, report) = method.try_minimize_least_squares(problem)?;
assert!(report.termination.was_successful());
println!("fit={:?}, objective={:e}", problem.params(), report.objective_function);
```

`try_minimize_least_squares` preserves typed numerical and callback errors.
Convergence and other valid termination outcomes are represented in
`report.termination`; inspect `was_successful()` rather than assuming every
returned report is a successful fit. Runtime telemetry is off by default and
can be enabled with `LeastSquaresTelemetryMode::Counters` or `Detailed` on the
LM method.

### Symbolic Least-Squares Builder

For symbolic residuals, `SymbolicLeastSquaresSolver` prepares the residual and
Jacobian and then uses the same canonical LM core. This builder supports an
initial guess, named unknowns, parameterized equations, opt-in telemetry, and
user-declared positive variables:

```rust
use RustedSciThe::numerical::Nonlinear_systems::prelude::*;

let mut solver = SymbolicLeastSquaresSolver::new()
    .with_equations_str(vec![
        "a - 1".into(),
        "a + b - 2".into(),
        "a + 2*b - 2.9".into(),
    ])
    .with_unknowns(vec!["a".into(), "b".into()])
    .with_initial_guess(vec![0.0, 0.0])
    .with_tolerance(1e-10)
    .with_telemetry(LeastSquaresTelemetryMode::Detailed);

let report = solver.try_solve()?;
if report.termination.was_successful() {
    let coefficients = solver.map_of_solutions.as_ref().expect("successful fit");
    println!("{coefficients:?}; objective={:e}", report.objective_function);
}
```

Use `try_solve`/`try_minimize_least_squares` for typed failures. The symbolic
builder's `set_positive_variables` is an explicit user policy for any model
whose domain requires selected variables to stay strictly positive; the
solver does not infer constraints from an application domain. This protection
is useful for logarithms and other restricted expressions, but is not limited
to chemistry.

### Reusing A Parameterized Symbolic Model

Declare equation parameters separately from unknowns. Each call to
`try_solve_with_params` rebinds numeric values and reuses the prepared
symbolic residual/Jacobian; it does not repeat symbolic preparation. The
wrapper uses its configured initial guess for every call, so this is a fresh
solve per parameter value, not a warm-start continuation from the previous
solution:

```rust
let mut solver = SymbolicLeastSquaresSolver::new()
    .with_equations_str(vec!["a - target".into(), "2*a - 2*target".into()])
    .with_unknowns(vec!["a".into()])
    .with_parameters(vec!["target".into()])
    .with_initial_guess(vec![0.0]);

for target in [1.0, 2.0, 3.0] {
    let report = solver.try_solve_with_params(vec![target])?;
    assert!(report.termination.was_successful());
    println!("target={target}, fit={:?}", solver.map_of_solutions);
}
```

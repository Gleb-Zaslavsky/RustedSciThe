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
    "cold: backend={:?}, action={:?}, key={:?}, build={:?}",
    cold.selected_backend,
    cold.preparation_report.artifact_action,
    cold.preparation_report.artifact_key,
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

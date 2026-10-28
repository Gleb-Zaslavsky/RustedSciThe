# StiffAPI: A Unified Facade for Stiff IVPs

StiffAPI lets an application select a production solver and its settings once,
then prepare and run it without maintaining its own dispatch `match` across
BDF, Radau, Backward Euler and LSODE2. Numerical algorithms, symbolic
preparation, AOT, logging, telemetry and result ownership remain native.

## Select a Route and Solve

`StiffSolverRequest` stores a typed configuration for the selected solver. An
application can keep this request in its solution configuration, prepare it
once, and delegate execution to `PreparedStiffSolver::try_solve()`:

```rust,ignore
use RustedSciThe::numerical::StiffAPI::{
    StiffSolveOutcome, StiffSolverRequest, StiffPreset,
};
use RustedSciThe::numerical::Radau::api::{RadauConfig, RadauProblem};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let request = StiffSolverRequest::Radau {
    problem: RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]),
    config: RadauConfig::default(),
    initial_state: vec![1.0],
    parameters: vec![0.7],
};

let mut solver = request.prepare(Some(&StiffPreset::expr_dense()))?;
let outcome = solver.try_solve()?;
if let StiffSolveOutcome::Radau(solution) = outcome {
    println!("final state: {:?}", solution.y);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The solver dispatch happens inside `try_solve()`. Application code does not
need to call `as_bdf_mut().try_solve()`, `as_radau_mut().solve(...)`, and so on.
The request intentionally uses an enum of native option types rather than an
untyped string map or a flat bag of unrelated `Option` fields. For example,
`BdfSolverOptions`, `RadauProblem`/`RadauConfig`, `BeSolverOptions` and
`Lsode2ProblemConfig` retain solver-specific settings.

## Fresh Runs After Parameter Rebinding

After preparing a route, use `try_solve_with_parameter_values(y0, values)` to
start a fresh solve with a new initial state and parameter values while keeping
the configured `t0` and `t_bound`. Use
`try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)` when the
time interval also changes. Both reuse the prepared symbolic/AOT route and reset
the numerical integration state; neither means solution-seeded parameter
continuation or continuation in time.

```rust,ignore
let outcome = prepared.try_solve_with_parameter_values(
    nalgebra::DVector::from_vec(vec![1.0]),
    nalgebra::DVector::from_vec(vec![1.2]),
)?;
let outcome = prepared.try_solve_with_parameter_values_in_interval(
    0.0,
    nalgebra::DVector::from_vec(vec![0.8]),
    2.0,
    nalgebra::DVector::from_vec(vec![0.9]),
)?;
```

Radau initial values are part of its request (`initial_state` and
`parameters`) because native Radau receives them at every `solve` call. The
other three routes keep their initial state in native configuration. The
number of Radau parameter values must match the declarations in
`RadauProblem::with_parameter_names`.

## Results and Errors

`StiffSolveOutcome` preserves native result ownership:

- Radau returns its native `RadauSolution`, including dense-output sampling and
  the telemetry report.
- LSODE2 returns its native `Lsode2SolveSummary`; the full trajectory remains
  available on the prepared solver.
- BDF and Backward Euler retain their results on the solver and return a
  success marker in the outcome.

After a failure the caller still owns `PreparedStiffSolver`. `StiffSolveError`
contains the selected solver's original error type and exposes it through
`Error::source()`. Inspect `solver_kind()`, native status or a partial result,
then use the matching typed accessor for native restart/recovery. A partial
`RadauSolution` is not promised: native Radau returns that value only on
success.

The facade does not invent a single restart contract. BDF, BE and LSODE2 have
different continuation/restart rules, while Radau accepts initial values on
each solve. Use native methods through `as_bdf_mut()`,
`as_backward_euler_mut()`, `as_lsode2_mut()` or `as_radau_mut()`. This is a
lifecycle escape hatch, not a place to duplicate solver dispatch.

## Presets and Observability

`StiffPreset` changes only the axes it explicitly declares. For example,
`expr_dense()` selects ExprLegacy and Dense without overriding tolerances, step
controls or controllers. Unsupported combinations return `StiffApiError`
instead of silently switching backends. See the
[contract matrix](CONTRACT_AUDIT.md) for route applicability and constraints.

Telemetry remains native and solver-specific. `telemetry_mode()` reports the
selected mode; `telemetry_snapshot()` is available when the solver owns a
snapshot, and Radau telemetry is obtained from `RadauSolution`. StiffAPI does
not merge counters or sum nested timings. Logging is configured through a
preset where the selected native solver supports it.

## Independent Parameter Sweeps

`PreparedStiffSolver::try_solve_parameter_loop(y0, vectors)` applies the same
fixed-initial-state sweep through the solver selected by the request. The
prepared route is reused, but each vector starts a new integration over the
configured interval. The report preserves ordered parameter values, the
time-by-state result and native telemetry per point, or a typed
solver-specific failure. A failed point does not abort the rest of the batch.
This is deliberately not parameter continuation. See
`examples/stiff_api_parameter_loop.rs` and the shared
[`IVP parameter-loop guide`](../IVP_PARAMETER_LOOP_GUIDE_EN.md).

## Postprocessing and Lifecycle

For BDF/BE/LSODE2, `trajectory_view()` provides a borrowed view shaped as
`(time_points, state_variables)`. It can be explicitly converted to a common
`PostprocessDataset` and processed by a `PostprocessPlan`; that conversion
copies arrays. For Radau, pass `RadauSolution` to
`radau_postprocess_dataset()` and provide sampling times explicitly. Dense
output must be requested in the Radau configuration.

Export, plotting, restart, continuation and parameter rebinding are not
implicit and are not forced into an artificial common API. Use the native
accessor for operations whose contracts differ across solver families. The
[contract audit](CONTRACT_AUDIT.md) records detailed route/default/error
differences.

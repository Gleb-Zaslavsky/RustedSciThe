# Backward Euler User Guide

Backward Euler (BE) is an implicit, fixed-step integrator intended for small
and medium systems where robustness is more important than the cost of a dense
Newton solve. The implementation uses a dense `nalgebra` Jacobian and LU
factorization. It is not a sparse large-scale solver; for huge sparse systems,
prefer a solver designed around sparse linear algebra.

## Choose a Callback Route

For a numerical model, configure `BE::try_set_native_initial` or
`BE::try_set_initial` plus `set_native_ode_callbacks`. Supply an analytic dense
Jacobian when available. Passing `None` for the Jacobian selects finite
differences, which costs additional RHS evaluations. The native example shows
the analytic callback route:

```powershell
cargo run --no-default-features --example be_native_callbacks_guide
```

## Step Size and Accuracy

`h: Some(step)` uses a fixed step, clipping only the final step to the requested
bound. Classical BE is first-order accurate; reduce `h` and check convergence
against an independent reference for the problem of interest.

`h: None` is a legacy local step heuristic based on the current RHS/Jacobian
scale and remaining interval. It is not an error estimator, adaptive controller,
or accuracy guarantee. It still holds the chosen step fixed during each Newton
solve. Do not treat it as an automatic accuracy mode.

## Symbolic Lambdify and Parameter Reuse

Use symbolic `Expr` equations when that representation is convenient. Select
`BeSymbolicAssemblyBackend::ExprLegacy` or `AtomViewNative` explicitly through
`BeSolverOptions`; ExprLegacy remains the compatibility default. The best
frontend can depend on model size and expression structure.

Declare symbolic parameters with `try_set_equation_parameters`, then bind values
with `set_parameter_values`. Value-only rebinding reuses the prepared callbacks;
`try_solve` starts again from the original `(t0, y0)`. To continue from the last
accepted `(t, y)`, update values and call `try_continue_to(new_t_bound)` instead.
The symbolic example demonstrates AtomViewNative with Lambdify and accepted-state
continuation:

```powershell
cargo run --no-default-features --example be_symbolic_continuation_guide
```

## Generated AOT

For a compiled symbolic backend, configure the dense AOT route before solving.
The example uses AtomViewNative and C/tcc with a release build policy. The first
run may materialize, compile, and link an artifact; a later run can reuse the
cache. AOT is worthwhile only when repeated or sufficiently expensive callback
work amortizes setup. The example exits successfully with a skip message if
`tcc` is unavailable.

```powershell
cargo run --no-default-features --example be_aot_guide
```

## Errors, Status, and Results

Prefer `BE::try_new_with_options`, `try_set_initial`, `try_solve`, and
`try_continue_to`. They return typed `BeError` values, including invalid
configuration, backend, generated-backend, Newton, step-underflow, and step-limit
failures. `BeStatus` distinguishes `Running`, `Finished`, `StoppedByCondition`,
and `Failed`. Compatibility methods such as `solve` may panic on errors.
`numerical::BE::prelude` re-exports the commonly used BE, NRE, symbolic and
nalgebra types for concise imports.

`get_result()` returns `(times, states)`. Each row of `states` is one time sample
and each column is one state variable. The initial sample is included; a failed
solve retains the accepted prefix rather than exposing a partially computed
step. `trajectory()` provides the same data by reference, avoiding a full
trajectory clone when callers only need to inspect or serialize results.

Stop conditions are checked on accepted samples within the configured
neighborhood. They do not interpolate a crossing or localize an event.

## Telemetry

`BeTelemetryMode::Off` disables instrumentation on the callback/solver hot path;
`Counters` records work counts without clock reads; `Timings` records both
counters and diagnostic elapsed times. Timings can perturb short solves, so use
the same mode when comparing measurements. `statistics_report()` and typed
statistics expose solver and backend work.

## Validation and Limitations

Check dimensions, finite inputs, the fixed-step convergence of the actual model,
and an independent reference. Analytic-Jacobian and finite-difference routes
should be compared where practical. The implementation currently uses dense
Newton matrices; it does not provide sparse Jacobians, adaptive error control,
Jacobian/LU reuse, or event localization. These are separate future extensions,
not implicit properties of Backward Euler.

See [`BE_STORY_TESTS.md`](BE_STORY_TESTS.md),
[`BE_BENCHMARKS.md`](BE_BENCHMARKS.md), and
[`BE_PERFORMANCE_BASELINE.md`](BE_PERFORMANCE_BASELINE.md) for test and
measurement evidence.

## Text Task Documents

Use a task document when a BE experiment should be reproducible or run as part
of a batch. The task route dispatches to the native Backward Euler adapter:

```text
task
solver: IVP
method: BackwardEuler

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
step_size: 1e-2
tolerance: 1e-7
max_iterations: 100

postprocessing
output_policy: none
```

`step_size` selects the fixed BE step; the final step may be clipped to the
requested bound. `tolerance` and `max_iterations` configure the Newton solve.
`output_policy` accepts `none`, `terminal`, `plotters`, or `gnuplot`; `plot` is
only a compatibility alias. Comments beginning with `#` are ignored.

The text adapter is deliberately narrower than the native API. It uses the
dense BE route and does not silently promise sparse matrices, adaptive error
control, Jacobian/LU reuse, or event localization. To choose
`ExprLegacy`/`AtomViewNative`, analytic versus finite-difference Jacobians,
AOT lifecycle, telemetry mode, or a custom continuation/restart policy, use
`BeSolverOptions` and the typed BE API. `None` for the native Jacobian means
finite differences; it is not the same as selecting another symbolic frontend.

Continuation is expressed explicitly:

```text
continuation
parameter: k
values: 1.0, 2.0, 4.0
mode: prepared
restart_each: false
```

`fresh` prepares every segment, `warm` reuses prepared callbacks, and
`prepared` requests the strongest model reuse. `restart_each: true` starts each
segment from its declared initial state; a segment may provide a new `y0`,
`t0`, or `t_end`. Keep fresh/warm/prepared timings separate from the numerical
solve time when reporting continuation performance. The shared grammar and
batch-runner conventions are documented in
[`TASK_DOCS_GUIDE_EN.md`](../../command_interpreter/TASK_DOCS_GUIDE_EN.md).

# Nonlinear Systems and Fitting Guide for Python

The Python package exposes related but distinct problem contracts. Use
`NonlinearSolver` for a square residual system `F(x)=0`; use `FittingSolver`
for symbolic least squares; choose `GavinFittingSolver` when fitting sampled
observations with Gavin's model/weight/bounds semantics; and `VarProSolver`
for variable projection. They share the Rust symbolic engine, not one
interchangeable objective or result type.

## Square nonlinear system

```python
from rustedscithe import Expr, SymbolicSystem, create_nonlinear

x = Expr.variable("x")
system = SymbolicSystem([x * x - Expr.constant(4.0)], ["x"])
solver = create_nonlinear(system, method="newton", telemetry="timings")
solver.prepare()
root = solver.solve([1.0])
print(root.solution_array())  # root near +2 for this initial guess
```

The API includes Newton, damped Newton, LM-family and trust-region methods
where exposed by `capabilities("nonlinear")`. Method-specific options belong
in the corresponding native option mapping; unknown keys are rejected rather
than ignored. `ExprLegacy` and `AtomNative` select symbolic preparation. Dense
AOT is available with `execution="aot"`; its generated-backend lifecycle is
controlled by `aot_policy`, `aot_profile` and `aot_output_dir`. The default
remains Lambdify, and build policies require an explicit output directory.

```python
from pathlib import Path
from rustedscithe import Expr, NonlinearSolver, SymbolicSystem

x = Expr.variable("x")
solver = NonlinearSolver(
    SymbolicSystem([x * x - 2.0], ["x"]),
    execution="aot",
    aot_policy="build_if_missing",
    aot_profile="release",
    aot_output_dir=str(Path("target") / "rst-nonlinear-aot"),
)
root = solver.solve([1.0])
```

The `NonlinearSolver` contract covers all 11 native methods, both ExprLegacy
and AtomNative, sequential/parallel/thresholded-auto evaluation, explicit
idempotent `prepare()`, parameter rebinding, logging, and opt-in counters or
detailed timings. Use `method_options` and `method_flags` for scalar and boolean
controls. `method_vectors={"diag": [...]}` is MINPACK-only and requires
`mode=2`; `method_choices` exposes advanced Nielsen `scaling_method`
(`levenberg`, `marquardt`, `more`) and `reduction_ratio_method` (`nielsen`,
`more`). Named partial box bounds accept `None` on either side. The square
system API does not expose a separate positive-domain declaration or Python
callbacks; these are not silently substituted.

## Least-squares fitting

`FittingSolver` minimizes a vector of symbolic residual equations. Its
parameter schema and optional positive-variable policy belong to the fitting
problem, not to the square-root API. Prepared parameter rebinds are available
through `solve_with_parameters(...)`.

```python
from rustedscithe import Expr, FittingSolver

p = Expr.variable("p")
fit = FittingSolver([p - Expr.constant(3.0)], [0.0], variables=["p"])
result = fit.solve()
print(result.solution_array(), result.termination)
```

## Gavin and VarPro

`GavinFittingSolver` consumes sampled `x_data`, observations, a named sample
parameter, unknown parameters and an initial guess. It is a one-shot fitting
contract, not the prepared `FittingSolver` API. Positive parameter bounds are
user-selected domain constraints; they are not inferred from a chemical model.
Gavin solve telemetry is disabled by default and follows the shared four-mode
contract: `off`, `counters`, `timings` or `detailed` (both channels). Set
`telemetry_mode` on `GavinFittingSolver`, or use
`with_gavin_telemetry_mode(...)` on `UniversalFitting`. The legacy boolean
`telemetry=True` remains shorthand for `detailed`. This is separate from
`preparation_telemetry`, which measures symbolic setup.
`VarProSolver` is also explicitly one-shot and reports its own model/statistics.
Do not assume either supports the parameter-rebind lifecycle of the canonical
fitting solver.

`VarProSolver` exposes the native LM tolerances, iteration and function
evaluation budgets, step bound, diagonal scaling, and telemetry through
`with_tolerances`, `with_max_iterations`, `with_max_function_evaluations`,
`with_step_bound`, `with_scale_diag`, and `with_telemetry`. It is ExprLegacy-only,
sequential, and has no logging or separate symbolic-preparation lifecycle.

`UniversalFitting` is a separate facade over the Rust method selector, not an
alias for any standalone class. Select `"lm"`, `"gavin"`, or `"varpro"`, then
configure the same fluent builder with data, unknowns, initial guesses, and
route-specific equations/basis functions. `build()` consumes it and returns a
common result with `method`, `solution_map()` and `r_squared`; use a standalone
solver when route-specific telemetry/covariance fields are needed. Unsupported
combinations are errors, not ignored settings: VarPro rejects AtomNative and
parallel execution; Gavin rejects `f_tolerance` and `scale_diag`. The shared
`with_telemetry("off"|"counters"|"timings"|"detailed")` selector applies to
the active route, including Gavin. `with_gavin_telemetry(True)` remains a
compatibility shorthand for enabling the detailed snapshot.

```python
from rustedscithe import UniversalFitting

fit = UniversalFitting("lm")
fit.with_data([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
fit.with_arg("x")
fit.with_equation_str("a*x+b")
fit.with_unknowns(["a", "b"])
fit.with_initial_guess([0.0, 0.0])
solution = fit.build()
print(solution.method, solution.solution_map())
```

## Frontends, logging and results

Where supported, select ExprLegacy or AtomNative and sequential/parallel
symbolic evaluation through the wrapper's explicit options. Parallel symbolic
preparation has a workload-dependent break-even point; it does not imply
parallel execution of Python callbacks. Logging is disabled by default and
must be enabled explicitly. Telemetry is a separate opt-in and detached from
the solver's reusable state.

The common `telemetry_counters()` snapshot includes current-state and trial
residual/Jacobian evaluations, trust-region trials, reusable trial points,
linear operations and step acceptance/rejection. `telemetry_timings_ms()` also
reports RHS application and the inclusive trust-region subproblem duration;
these nested durations overlap and are not additive. The same aggregate
statistics mapping is used by standalone LM, VarPro, Gavin and the corresponding
UniversalFitting methods.
When the selected native route retains iteration history,
`telemetry_attempt_counters()` and `telemetry_attempt_timings_ms()` return one
detached row per nonlinear attempt. Attempt durations are nested in aggregate
timings and must not be added to them. Gavin currently has no native attempt
trace, so its attempt-row accessors return empty lists.

Results expose fitted values, termination/objective information and
solver-specific diagnostics. Covariance, R-squared and fitted-output metrics
are only meaningful where the selected fitting contract provides the required
observations and weights. Errors are typed; handle exhaustion separately from
invalid configuration or symbolic preparation errors.

Run `examples/nonlinear_parameter_sweep.py` for prepared nonlinear continuation,
`examples/lm_observation_fit.py` for reusable symbolic least squares,
`examples/gavin_exponential_fit.py` for sampled-data fitting with a positive
parameter, `examples/varpro_exponential_fit.py` for separable fitting, and
`examples/universal_fitting_routes.py` for method selection. The shorter
`examples/nonlinear_and_fitting.py` remains a minimal starting point. See the
native nonlinear and optimization guides for algorithm details.

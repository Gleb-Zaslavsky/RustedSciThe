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
than ignored. `ExprLegacy` and `AtomNative` select symbolic preparation. AOT is
not advertised for this wrapper unless a concrete compiled route is reported
by its capability metadata.

## Least-squares fitting

`FittingSolver` minimizes a vector of symbolic residual equations. Its
parameter schema and optional positive-variable policy belong to the fitting
problem, not to the square-root API. Prepared parameter rebinds are available
through `solve_with_parameters(...)`.

```python
from rustedscithe import Expr, FittingSolver

p = Expr.variable("p")
fit = FittingSolver([p - Expr.constant(3.0)], [0.0], variables=["p"])
fit.prepare()
result = fit.solve()
print(result.solution_array(), result.termination_reason)
```

## Gavin and VarPro

`GavinFittingSolver` consumes sampled `x_data`, observations, a named sample
parameter, unknown parameters and an initial guess. It is a one-shot fitting
contract, not the prepared `FittingSolver` API. Positive parameter bounds are
user-selected domain constraints; they are not inferred from a chemical model.
`VarProSolver` is also explicitly one-shot and reports its own model/statistics.
Do not assume either supports the parameter-rebind lifecycle of the canonical
fitting solver.

## Frontends, logging and results

Where supported, select ExprLegacy or AtomNative and sequential/parallel
symbolic evaluation through the wrapper's explicit options. Parallel symbolic
preparation has a workload-dependent break-even point; it does not imply
parallel execution of Python callbacks. Logging is disabled by default and
must be enabled explicitly. Telemetry is a separate opt-in and detached from
the solver's reusable state.

Results expose fitted values, termination/objective information and
solver-specific diagnostics. Covariance, R-squared and fitted-output metrics
are only meaningful where the selected fitting contract provides the required
observations and weights. Errors are typed; handle exhaustion separately from
invalid configuration or symbolic preparation errors.

Run `examples/nonlinear_and_fitting.py` for a short overview and
`examples/gavin_observation_fit.py` for sampled-data fitting with bounds. See
the native nonlinear and optimization guides for algorithm details.

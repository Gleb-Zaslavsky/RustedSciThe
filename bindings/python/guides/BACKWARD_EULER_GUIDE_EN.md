# Backward Euler Guide for Python

Backward Euler is an implicit first-order time integrator. It is useful as a
simple robust baseline, for fixed-step demonstrations and as a transparent
way to inspect how step size affects accuracy. It should not be mistaken for
the adaptive high-order BDF or Radau solvers.

## Minimal solve

```python
from rustedscithe import BackwardEulerSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BackwardEulerSolver(
    [-y], ["y"], "t", [1.0], 0.0, 1.0, 0.01,
    tolerance=1e-9,
    options=SolverOptions(
        frontend="expr_legacy", execution="lambdify",
        matrix_layout="dense", telemetry="counters",
    ),
)
solution = solver.solve()
```

For `y'=-y`, the exact endpoint is `exp(-1)`. Backward Euler is first order:
halving the step should reduce global error approximately linearly once the
asymptotic regime is reached. The example `backward_euler_basic.py` performs
this check without presenting timing output as a benchmark.

## Route and lifecycle

The wrapper supports Dense symbolic ExprLegacy/AtomNative preparation and
Lambdify/AOT execution. Sparse, Banded and unsupported parallel policies are
rejected explicitly. `solve()` currently performs lazy native preparation;
the wrapper does not claim a separate prepared lifecycle until the core
exposes one. Parameter continuation and restart are available according to
the native adapter contract; verify whether a use case requires preserving
the fixed step or changing the interval.

`h` is the fixed integration step, `tolerance` controls the nonlinear solve,
and `max_iterations` bounds its work. Do not interpret the Newton tolerance
as the global integration error. Use owned NumPy arrays for detached results.

## AOT and diagnostics

AOT shares the native build/cache lifecycle. A compiler is optional for
Lambdify and required only when an AOT route is selected. Record the selected
execution route and artifact provenance. Counter/timing telemetry is opt-in
through `SolverOptions`; typed preparation and exhaustion errors should be
handled instead of parsing their message strings.

See `examples/backward_euler_basic.py` for step refinement and
`examples/backward_euler_routes.py` for a matched ExprLegacy/AtomNative route
comparison.

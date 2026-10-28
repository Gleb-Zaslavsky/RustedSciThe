"""Reuse one prepared nonlinear system for a small parameter sweep.

The equations are ``a*x + y - 3 = 0`` and ``x - y = 0``.  For each value of
``a`` the exact solution is ``x = y = 3 / (a + 1)``.  The example demonstrates
that fixed symbolic preparation can be reused while only numeric parameters
change; the solver's unknown vector contains ``x`` and ``y``, not ``a``.
"""

from rustedscithe import Expr, NonlinearSolver, SymbolicSystem


system = SymbolicSystem(
    [Expr.parse("a*x + y - 3"), Expr.parse("x - y")],
    ["x", "y", "a"],
)
# `a` is declared in the symbolic system so it can appear in the equations,
# then separated as a numeric parameter rather than a Newton unknown.
solver = NonlinearSolver(
    system,
    method="damped_newton",
    frontend="atom_native",
    parameters=["a"],
    telemetry="detailed",
)

# Prepare explicitly so the symbolic setup is outside the repeated solve loop.
solver.prepare()
print("frontend:", solver.frontend(), "backend:", solver.backend())
print("prepared once:", solver.is_prepared())

for a_value in (0.5, 1.0, 2.0, 4.0):
    result = solver.solve_with_parameter_values([1.0, 1.0], [a_value])
    x_value, y_value = result.solution()
    expected = 3.0 / (a_value + 1.0)
    assert abs(x_value - expected) < 1e-7
    assert abs(y_value - expected) < 1e-7
    print(
        f"a={a_value:3.1f}: x={x_value:.8f}, y={y_value:.8f}, "
        f"residual={result.residual_array()}"
    )
    print("  counters:", result.telemetry_counters())
    print("  timings:", result.telemetry_timings_ms())

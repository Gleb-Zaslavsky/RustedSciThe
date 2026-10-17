"""Select the native ODE_api2 nonstiff methods from Python.

These methods use the core's ExprLegacy/Lambdify, Dense, sequential route.
The native fixed-step loop returns its last point after reaching or crossing
the requested bound, so a final sample can be less than one step beyond it.
"""

from rustedscithe import Expr, SolverOptions, create_ivp


y = Expr.variable("y")
options = SolverOptions(
    frontend="expr_legacy",
    execution="lambdify",
    matrix_layout="dense",
    execution_policy="sequential",
    telemetry="off",
)

for method in ("rk45", "dopri", "ab4"):
    solver = create_ivp(
        method,
        [-y],
        ["y"],
        "t",
        [1.0],
        0.0,
        1.0,
        0.01,
        options=options,
    )
    result = solver.solve()
    print(
        f"{result.method}: status={result.status}, "
        f"samples={result.sample_count}, final_t={result.times()[-1]:.6f}, "
        f"final_y={result.states()[-1][0]:.8f}"
    )

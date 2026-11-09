"""Select the native ODE_api2 nonstiff methods from Python.

These methods use the core's ExprLegacy/Lambdify, Dense, sequential route.
The native fixed-step loop returns its last point after reaching or crossing
the requested bound, so a final sample can be less than one step beyond it.
"""

import math

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
    # This API2 family is intentionally separate from the stiff IVP wrappers;
    # it has no symbolic parameter rebinding or sparse/banded layout controls.
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
    final_time = result.times()[-1]
    final_value = result.states()[-1][0]
    # This family is fixed-step; compare against the exact solution at the
    # actual last sample because a native method may cross the requested bound.
    expected = math.exp(-final_time)
    error = abs(final_value - expected)
    if result.status != "finished" or error > 2e-3:
        raise AssertionError(
            f"{method}: status={result.status!r}, exact-solution error={error:.3e}"
        )
    print(
        f"{result.method}: status={result.status}, "
        f"samples={result.sample_count}, final_t={final_time:.6f}, "
        f"final_y={final_value:.8f}, exact_error={error:.3e}"
    )

# DOP853 is an adaptive 8th-order method: `max_step` caps its controller step,
# while rtol/atol determine local error acceptance and the endpoint is clipped.
adaptive = create_ivp(
    "dop853",
    [-y],
    ["y"],
    "t",
    [1.0],
    0.0,
    1.0,
    0.5,
    rtol=1e-11,
    atol=1e-13,
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="detailed",
    ),
)
adaptive_result = adaptive.solve()
adaptive_error = abs(adaptive_result.states()[-1][0] - math.exp(-1.0))
if adaptive_result.times()[-1] != 1.0 or adaptive_error > 2e-10:
    raise AssertionError(
        f"dop853: endpoint={adaptive_result.times()[-1]}, error={adaptive_error:.3e}"
    )
print("dop853:", adaptive_result.diagnostics())

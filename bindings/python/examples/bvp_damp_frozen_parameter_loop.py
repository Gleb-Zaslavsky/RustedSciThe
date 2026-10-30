"""Ordered fixed-parameter sweep with recoverable point failures.

The middle value makes sqrt(alpha) invalid. The loop records that native typed
failure and still solves the final point from the last successful profile.
"""

from rustedscithe import BvpDampFrozenSolver, Expr, SolverOptions


solver = BvpDampFrozenSolver(
    [Expr.parse("alpha^0.5")],
    ["y"],
    {"y": [(0, 0.0)]},
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    [0.0] * 5,
    parameters=["alpha"],
    parameter_values=[1.0],
    policy="Frozen_naive",
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="sparse",
        telemetry="detailed",
    ),
)

report = solver.solve_parameter_loop([[1.0], [-1.0], [4.0]])
for point in report.points:
    if point.status == "converged":
        final_value = point.solution.values()[0][-1]
        print(
            f"point {point.index}: alpha={point.parameters[0]}, "
            f"converged, y(1)={final_value:.6g}, "
            f"iterations={point.telemetry_counters().get('iterations', 0)}"
        )
    else:
        print(
            f"point {point.index}: alpha={point.parameters[0]}, "
            f"{point.status}: {point.error}"
        )

assert [point.status for point in report.points] == [
    "converged",
    "failed",
    "converged",
]
assert report.points[0].solution.values()[0][-1] == 1.0
assert report.points[2].solution.values()[0][-1] == 2.0
assert report.points[1].error.native_kind is not None
assert report.stop_error is None

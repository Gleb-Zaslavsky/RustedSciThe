"""Adaptive Damped BVP parameter sweep with recoverable point failures.

The invalid middle value exercises the typed-error path. The final solve
continues from the last successful profile and its refined mesh.
"""

from rustedscithe import BvpDampSolver, Expr, SolverOptions


solver = BvpDampSolver(
    [Expr.parse("alpha^0.5")],
    ["y"],
    {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.0] * 5,
    parameters=["alpha"],
    parameter_values=[1.0],
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        telemetry="counters",
    ),
    tolerance=1e-9,
    max_refinements=1,
    grid_method="double_points",
    bounds={"y": (-10.0, 10.0)},
)

report = solver.solve_parameter_loop([[1.0], [-1.0], [4.0]])
if report.stop_error is not None:
    raise report.stop_error

for point in report.points:
    if point.solution is not None:
        print(
            f"point {point.index}: alpha={point.parameters[0]}, "
            f"nodes={len(point.solution.mesh())}, "
            f"y(1)={point.solution.values()[0][-1]:.6g}, "
            f"iterations={point.telemetry_counters().get('iterations', 0)}"
        )
    else:
        print(f"point {point.index}: {point.status}: {point.error}")

assert [point.status for point in report.points] == [
    "converged",
    "failed",
    "converged",
]
assert report.points[-1].solution.values()[0][-1] == 2.0

"""Run BVP_sci for ordered fixed model inputs without changing its schema.

The RHS is y' = rate. A boundary callback deliberately rejects the middle
rate, demonstrating that the report keeps the typed failure and still solves
the final point from the last successful profile. ``rate`` is an external
input, not a free Newton unknown.
"""

from rustedscithe import BvpSciSolver, Expr, SolverOptions


def boundary(ya, _yb, runtime_parameters):
    rate = runtime_parameters[0]
    if rate < 0.0:
        raise ValueError("negative rate is outside this model's domain")
    return [ya[0]]


solver = BvpSciSolver(
    [Expr.variable("rate")],
    ["y"],
    [],
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.0] * 5,
    external_parameters=["rate"],
    external_parameter_values=[1.0],
    boundary_callback=boundary,
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="sparse",
        telemetry="detailed",
    ),
    tolerance=1e-6,
)

report = solver.solve_parameter_loop([[1.0], [-1.0], [2.0]])
for index, point in enumerate(report.points):
    if point.succeeded:
        assert point.solution is not None
        print(
            f"point {index}: rate={point.external_parameters[0]}, "
            f"y(1)={point.solution.values()[-1][0]:.6f}, "
            f"Newton iterations={point.solution.iterations}"
        )
    else:
        print(f"point {index}: failed ({point.native_kind}): {point.error_message}")

assert [point.succeeded for point in report.points] == [True, False, True]
assert report.points[2].solution.values()[-1][0] == 2.0
report.raise_if_stopped()

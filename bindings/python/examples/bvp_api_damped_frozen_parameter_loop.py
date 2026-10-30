"""Run Damped and Frozen fixed-input loops through the typed BVP facade.

Damped carries its last successful adaptive mesh/profile; Frozen keeps its
fixed mesh and reuses only the last successful profile. The invalid middle
parameter is retained as a typed point failure while the sweep proceeds.
"""

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def run_route(route: str) -> None:
    mesh = [index / 8 for index in range(9)]
    equation = Expr.parse("1 / rate")  # y(1) = 1/rate for y(0) = 0.
    common = {
        "expressions": [equation],
        "state_variables": ["y"],
        "independent_variable": "x",
        "mesh": mesh,
        "boundary_conditions": [(0, "left", 0.0)],
        "parameters": ["rate"],
        "parameter_values": [1.0],
        "options": SolverOptions(frontend="expr_legacy", telemetry="counters"),
    }
    if route == "damped":
        # Damped's guess has one column per interval, not per node.
        request = BvpSolverRequest.damped(
            **common,
            initial_guess=[[0.0] for _ in range(len(mesh) - 1)],
            bounds={"y": (-10.0, 10.0)},
        )
    else:
        # Frozen uses the reduced Newton iterate, one row per interval.
        request = BvpSolverRequest.frozen(
            **common,
            initial_guess=[[0.0] for _ in range(len(mesh) - 1)],
        )

    report = request.prepare().solve_parameter_loop([[1.0], [0.0], [4.0]])
    for point in report.points:
        if point.solution is None:
            print(f"{route} rate={point.parameters[0]}: {point.status}: {point.error}")
            continue
        final_value = point.solution.values()[0][-1]
        print(
            f"{route} rate={point.parameters[0]}: y(1)={final_value:.6f}, "
            f"iterations={point.telemetry_counters().get('iterations', 0)}"
        )
    if report.stop_error is not None:
        raise report.stop_error


if __name__ == "__main__":
    run_route("damped")
    run_route("frozen")

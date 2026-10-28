"""Reuse one prepared BVP_sci model while sweeping a fixed input parameter."""

import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main() -> None:
    # y' = rate*y, y(0)=1 has the exact solution exp(rate*x).
    y = Expr.variable("y")
    mesh = [index / 40 for index in range(41)]
    request = BvpSolverRequest.sci(
        [Expr.variable("rate") * y],
        ["y"],
        "x",
        mesh,
        [(0, "left", 1.0)],
        [[math.exp(0.5 * x)] for x in mesh],
        options=SolverOptions(frontend="atom_native", telemetry="counters"),
        tolerance=1e-6,
        external_parameters=["rate"],
        external_parameter_values=[0.5],
    )
    solver = request.prepare()

    # The external-input schema and symbolic callbacks are prepared once.
    # Each point changes the parameter and warm-starts from the last success.
    report = solver.solve_parameter_loop([[0.5], [0.75], [1.0]])
    for point in report.points:
        if point.succeeded:
            solution = point.solution
            assert solution is not None
            terminal = solution.values()[-1][0]
            print(
                f"rate={point.external_parameters[0]:.2f}: "
                f"y(1)={terminal:.8f}, "
                f"expected={math.exp(point.external_parameters[0]):.8f}"
            )
        else:
            print(
                f"rate={point.external_parameters[0]:.2f}: "
                f"failed ({point.native_kind}): {point.error_message}"
            )
    report.raise_if_stopped()


if __name__ == "__main__":
    main()

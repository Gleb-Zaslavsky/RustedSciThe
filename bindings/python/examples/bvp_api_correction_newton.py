"""Fixed-grid CorrectionNewton-style Newton solve through the typed BVP facade."""

import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main():
    y = Expr.variable("y")
    steps = 128
    mesh = [index / steps for index in range(steps + 1)]
    guess = [[math.exp(-((index + 1) / steps))] for index in range(steps)]

    request = BvpSolverRequest.correction_newton(
        expressions=[-Expr.variable("rate") * y],
        state_variables=["y"],
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=[(0, "left", 1.0)],
        initial_guess=guess,
        options=SolverOptions(
            frontend="atom_native",
            execution="lambdify",
            matrix_layout="banded",
            telemetry="detailed",
        ),
        tolerance=1e-7,
        external_parameters=["rate"],
        external_parameter_values=[0.5],
    )
    solver = request.prepare()
    solution = solver.solve()
    print(f"route: {solution.route}")
    print(f"y(1), rate=0.5: {solution.values_array()[-1, 0]:.12g}")
    rebound = solver.solve_with_parameter_values([1.0])
    print(f"y(1), rate=1.0: {rebound.values_array()[-1, 0]:.12g}")
    print(f"telemetry: {solver.telemetry_snapshot()}")


if __name__ == "__main__":
    main()

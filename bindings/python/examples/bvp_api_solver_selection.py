"""Choose one symbolic BVP family through the shared native BVP_api facade.

The example solves ``y' = -y`` with ``y(0) = 1``. The common request keeps
application code independent of solver-specific Python classes while leaving
each route's numerical controls explicit at the factory call.
"""

from __future__ import annotations

import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main() -> None:
    y = Expr.variable("y")
    mesh = [index / 20 for index in range(21)]
    guess = [[math.exp(-x)] for x in mesh]

    # This is the application-level choice. Replace "sci" by "damped" or
    # "frozen" and use that route's documented factory arguments when needed.
    request = BvpSolverRequest.sci(
        expressions=[-y],
        state_variables=["y"],
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=[(0, "left", 1.0)],
        initial_guess=guess,
        options=SolverOptions(
            frontend="atom_native",
            execution="lambdify",
            matrix_layout="banded",
            execution_policy="sequential",
            telemetry="detailed",
        ),
        tolerance=1e-5,
        output_policy="mesh",
    )
    solver = request.prepare()
    solution = solver.solve()

    print(f"route: {solution.route}")
    print(f"mesh nodes: {len(solution.mesh())}")
    print(f"y(1): {solution.values()[-1][0]:.8f} (exact {math.exp(-1):.8f})")
    telemetry = solver.telemetry_snapshot()
    print(f"telemetry mode: {telemetry['mode']}")
    print(f"RHS calls: {telemetry['counters'].get('rhs_calls', 0)}")
    if telemetry["timings_ms"]:
        print(f"measured stages (ms): {telemetry['timings_ms']}")


if __name__ == "__main__":
    main()

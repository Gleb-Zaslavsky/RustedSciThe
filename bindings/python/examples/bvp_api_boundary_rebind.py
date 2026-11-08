"""Reuse one prepared BVP_api system while rebinding a numeric boundary value."""

from __future__ import annotations

import math

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main() -> None:
    mesh = [index / 40 for index in range(41)]
    guess = [[math.exp(-x)] for x in mesh]
    request = BvpSolverRequest.sci(
        expressions=[-Expr.variable("y")],
        state_variables=["y"],
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=[(0, "left", 1.0)],
        initial_guess=guess,
        options=SolverOptions(
            frontend="atom_native",
            execution="lambdify",
            matrix_layout="banded",
            telemetry="counters",
        ),
        tolerance=1.0e-6,
    )

    prepared = request.prepare()
    first = prepared.solve()
    rebound = prepared.solve_with_boundary_values([2.0])
    assert abs(first.values_array()[0, 0] - 1.0) < 1.0e-10
    assert abs(rebound.values_array()[0, 0] - 2.0) < 1.0e-10
    print(f"first solve y(0)={first.values_array()[0, 0]:.8f}")
    print(f"rebound solve y(0)={rebound.values_array()[0, 0]:.8f}")
    print(f"same route: {first.route == rebound.route}; route={rebound.route}")


if __name__ == "__main__":
    main()

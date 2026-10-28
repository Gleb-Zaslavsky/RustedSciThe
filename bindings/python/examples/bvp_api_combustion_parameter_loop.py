"""Sweep heat release over a prepared, six-state sparse combustion BVP.

The external `heat_scale` changes at each point. BVP_sci keeps its prepared
symbolic callbacks and warm-starts from the last successful solution profile.
This is a substantial example, not a timing benchmark.
"""

from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main() -> None:
    states = [Expr.variable(name) for name in ("Teta", "q", "C0", "J0", "C1", "J1")]
    teta, q, c0, j0, _c1, j1 = states
    heat_scale = Expr.variable("heat_scale")
    rho_m = (34.2 / 1000.0) * 2.0e6 / (8.314 * 1500.0)
    rate = (
        1.3e5
        * (-5000.0 * 4.184 / (8.314 * (teta * 600.0 + 600.0))).exp()
        * c0
        * (rho_m / 0.342)
    )
    rhs = [
        q / 0.07,
        q * 0.0090168 - heat_scale * (3000.0e3 * 0.034 * (3.0e-4) ** 2 / 600.0) * rate,
        j0 / 2.88e-4,
        j0 * 1.50e-3 + rate * rho_m * (3.0e-4) ** 2,
        j1 / 2.88e-4,
        j1 * 1.50e-3 + rate * rho_m * (3.0e-4) ** 2,
    ]
    nodes = 1_000
    mesh = [index / (nodes - 1) for index in range(nodes)]
    initial_state = [[0.99] * len(states) for _ in mesh]
    request = BvpSolverRequest.sci(
        rhs,
        ["Teta", "q", "C0", "J0", "C1", "J1"],
        "x",
        mesh,
        [
            (0, "left", 400.0 / 600.0),
            (1, "right", 1.0e-10),
            (2, "left", 1.0),
            (3, "right", 1.0e-7),
            (4, "left", 1.0e-3),
            (5, "right", 1.0e-7),
        ],
        initial_state,
        options=SolverOptions(
            frontend="atom_native", matrix_layout="sparse", telemetry="counters"
        ),
        tolerance=1.0e-6,
        max_nodes=nodes + 64,
        max_newton_iterations=20,
        max_mesh_refinements=4,
        external_parameters=["heat_scale"],
        external_parameter_values=[1.0],
    )
    solver = request.prepare()
    heat_values = [[1.0 + step * 0.005] for step in range(10)]
    report = solver.solve_parameter_loop(heat_values)
    for point in report.points:
        if point.succeeded:
            solution = point.solution
            assert solution is not None
            print(
                f"heat_scale={point.external_parameters[0]:.3f}: "
                f"nodes={len(solution.mesh())}, iterations={solution.iterations}, "
                f"residual={solution.residual_norm:.3e}"
            )
        else:
            print(
                f"heat_scale={point.external_parameters[0]:.3f}: "
                f"failed ({point.native_kind}): {point.error_message}"
            )
    report.raise_if_stopped()


if __name__ == "__main__":
    main()

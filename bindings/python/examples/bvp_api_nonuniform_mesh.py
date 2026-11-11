from rustedscithe import BvpSolverRequest, Expr, SolverOptions


def main():
    x = Expr.variable("x")
    mesh = [0.0, 0.07, 0.31, 0.66, 1.0]
    seed = [[0.5 * coordinate * coordinate] for coordinate in mesh]
    seed[2][0] += 0.05

    request = BvpSolverRequest.correction_newton(
        expressions=[x],
        state_variables=["y"],
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=[(0, "left", 0.0)],
        initial_guess=seed,
        initial_guess_layout="node_aligned",
        options=SolverOptions(frontend="atom_native", matrix_layout="banded"),
        tolerance=1e-9,
    )
    solution = request.prepare().solve()
    print("mesh:", solution.mesh())
    print("profile:", solution.values())

    frozen = BvpSolverRequest.frozen(
        expressions=[x],
        state_variables=["y"],
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=[(0, "left", 0.0)],
        initial_guess=[[0.5 * coordinate * coordinate] for coordinate in mesh],
        initial_guess_layout="node_aligned",
        scheme="trapezoid",
        tolerance=1e-10,
        options=SolverOptions(frontend="atom_native", matrix_layout="banded"),
    ).prepare().solve()
    print("Frozen mesh:", frozen.mesh())
    print("Frozen profile:", frozen.values())


if __name__ == "__main__":
    main()

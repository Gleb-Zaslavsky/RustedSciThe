"""Representative statically checked usage from a downstream Python project."""

from numpy import float64
from numpy.typing import NDArray

from rustedscithe import (
    BvpDampSolver,
    BvpSciSolver,
    Expr,
    Lsode2Solver,
    RadauSolution,
    SolverOptions,
    create_bvp,
    create_ivp,
)


def configured_radau_solve() -> tuple[RadauSolution, NDArray[float64]]:
    """Exercise symbolic input, typed solver selection and detached NumPy output."""
    state = Expr.variable("y")
    solver = create_ivp(
        "radau5",
        [-state],
        ["y"],
        "t",
        [1.0],
        0.0,
        1.0,
        0.1,
        options=SolverOptions(
            frontend="atom-native",
            execution="lambdify",
            matrix_layout="dense",
            telemetry="timings",
        ),
        output="dense",
        rtol=1e-6,
        atol=1e-8,
    )
    solution = solver.solve([1.0])
    state_array: NDArray[float64] = solution.final_state_array()
    timing_scopes: dict[str, float] = solution.telemetry_timings_ms()
    assert timing_scopes is not None
    return solution, state_array


def configured_lsoda_solver() -> Lsode2Solver:
    """Check that the LSODA alias retains the concrete LSODE2 solver type."""
    state = Expr.variable("y")
    return create_ivp("lsoda", [-state], ["y"], "t", [1.0], 0.0, 1.0, 0.1)


def configured_lsode2_solver() -> Lsode2Solver:
    """Keep the canonical LSODE2 method literal narrowed in client code."""
    state = Expr.variable("y")
    return create_ivp("lsode2", [-state], ["y"], "t", [1.0], 0.0, 1.0, 0.1)


def configured_bvp_routes() -> tuple[BvpSciSolver, BvpDampSolver]:
    """Keep both native boundary-data schemas visible to static tooling."""
    mesh = [0.0, 0.5, 1.0]
    initial = [0.0, 0.0, 0.0]
    expression = Expr.constant(0.0)
    scipy_like = create_bvp(
        "bvp_sci", [expression], ["y"], [Expr.parse("ya_y - 1")], mesh, initial
    )
    damped = create_bvp(
        "bvp_damp", [expression], ["y"], {"y": [(0, 1.0)]}, mesh, initial
    )
    return scipy_like, damped


def configured_native_bvp_damp() -> BvpDampSolver:
    """Check native BVP_Damp tuple constraints and flattened node-major input."""
    expression = Expr.constant(0.0)
    damped = BvpDampSolver(
        [expression],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        grid_method="easy",
        grid_parameters=[0.5],
        max_refinements=1,
        bounds={"y": (-2.0, 2.0)},
    )
    damped.restart([1.0, 1.0, 1.0])
    return damped


def configured_native_bvp_sci() -> BvpSciSolver:
    """Check the native BVP_sci restart contract uses a flat node-major vector."""
    expression = Expr.constant(0.0)
    mesh = [0.0, 0.5, 1.0]
    solver = BvpSciSolver(
        [expression], ["y"], [Expr.parse("ya_y - 1")], mesh, [1.0, 1.0, 1.0]
    )
    solver.restart(mesh, [1.0, 1.0, 1.0])
    return solver

"""Representative statically checked usage from a downstream Python project."""

from numpy import float64
from numpy.typing import NDArray

from rustedscithe import Expr, RadauSolution, SolverOptions, create_ivp


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

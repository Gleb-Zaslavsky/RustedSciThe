"""Solve a BVP with a free parameter and update its Newton guess.

The equation is y' = p with y(0)=0 and y(1)=1. Here p is an unknown solved
along with y, not a fixed coefficient to rebind; the correct value is p=1.
After the first solve, BvpSciSolver retains its refined mesh and state profile.
"""

from rustedscithe import BvpSciSolver, Expr, SolverOptions


solver = BvpSciSolver(
    [Expr.parse("p")],
    ["y"],
    [Expr.parse("ya_y"), Expr.parse("yb_y - 1")],
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.0] * 5,
    parameters=["p"],
    parameter_guess=[0.7],
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="sparse",
        telemetry="detailed",
    ),
    tolerance=1e-6,
    max_nodes=64,
)

first = solver.solve()
print(f"solved p={first.parameters[0]:.8f}; initial mesh points={len(first.mesh())}")

# The setter changes only the starting estimate of the free parameter block.
# The previous converged y profile and mesh stay in place for the next solve.
solver.set_parameter_guess([1.2])
second = solver.solve()
print(
    f"solved p={second.parameters[0]:.8f}; "
    f"continuation solves={second.telemetry_counters()['continuation_solves']}"
)
assert abs(second.parameters[0] - 1.0) < 1e-5

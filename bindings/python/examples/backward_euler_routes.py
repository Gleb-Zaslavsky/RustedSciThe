"""Check first-order step refinement and symbolic frontends for Backward Euler.

For ``y'=-y`` the exact endpoint is ``exp(-1)``. The same method is run at
two fixed step sizes under both symbolic frontends. Error should generally
decrease with the smaller step; this is a numerical demonstration, not a
statistical performance comparison.
"""

import math

from rustedscithe import BackwardEulerSolver, Expr, SolverOptions


y = Expr.variable("y")
exact = math.exp(-1.0)
for frontend in ("expr_legacy", "atom_native"):
    for step in (0.1, 0.05):
        solver = BackwardEulerSolver(
            [-y], ["y"], "t", [1.0], 0.0, 1.0, step,
            tolerance=1e-10,
            options=SolverOptions(
                frontend=frontend, execution="lambdify",
                matrix_layout="dense", telemetry="counters",
            ),
        )
        solution = solver.solve()
        error = abs(float(solution.states_array()[-1, 0]) - exact)
        print(f"frontend={frontend:11} step={step:.3f} error={error:.3e}")

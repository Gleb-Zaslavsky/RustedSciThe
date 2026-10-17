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
# Hold the equation and interval fixed: halving a fixed step should reduce the
# global error of this first-order method, regardless of symbolic frontend.
errors_by_frontend = {}
for frontend in ("expr_legacy", "atom_native"):
    frontend_errors = {}
    for step in (0.1, 0.05):
        solver = BackwardEulerSolver(
            [-y], ["y"], "t", [1.0], 0.0, 1.0, step,
            # Nonlinear solve tolerance is kept much tighter than the IVP
            # discretization error, so it does not dominate this comparison.
            tolerance=1e-10,
            options=SolverOptions(
                frontend=frontend, execution="lambdify",
                matrix_layout="dense", telemetry="counters",
            ),
        )
        solution = solver.solve()
        error = abs(float(solution.states_array()[-1, 0]) - exact)
        frontend_errors[step] = error
        print(f"frontend={frontend:11} step={step:.3f} error={error:.3e}")
    # This pedagogical check compares discretization error, not wall time; the
    # tighter nonlinear tolerance keeps Newton error below the step-size error.
    if frontend_errors[0.05] >= frontend_errors[0.1]:
        raise AssertionError(f"smaller Backward Euler step did not improve {frontend}")
    errors_by_frontend[frontend] = frontend_errors

for step in (0.1, 0.05):
    if abs(errors_by_frontend["expr_legacy"][step] - errors_by_frontend["atom_native"][step]) > 1e-9:
        raise AssertionError(f"symbolic frontends disagree at fixed step {step}")

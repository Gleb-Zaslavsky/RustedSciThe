"""Matched Radau5 route comparison on a mass-conserving Robertson variant.

The classic Robertson reactions are augmented with a small slow ``y3 -> y1``
recycling channel. This keeps total mass invariant and gives every state a
structural diagonal in the Jacobian, which makes the sparse shifted system's
diagonal pattern explicit. Every row uses the same equations, initial state,
tolerances and analytic Jacobian; only frontend and layout vary. This is a
correctness/configuration demonstration, not a timing benchmark.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


y1, y2, y3 = (Expr.variable(name) for name in ("y1", "y2", "y3"))
recycle_rate = Expr.constant(0.01)
rhs = [
    Expr.constant(-0.04) * y1 + Expr.constant(1.0e4) * y2 * y3
    + recycle_rate * y3,
    Expr.constant(0.04) * y1 - Expr.constant(1.0e4) * y2 * y3
    - Expr.constant(3.0e7) * y2 * y2,
    Expr.constant(3.0e7) * y2 * y2 - recycle_rate * y3,
]
jacobian = [
    Expr.constant(-0.04), Expr.constant(1.0e4) * y3,
    Expr.constant(1.0e4) * y2 + recycle_rate,
    Expr.constant(0.04), Expr.constant(-1.0e4) * y3 - Expr.constant(6.0e7) * y2,
    Expr.constant(-1.0e4) * y2,
    Expr.constant(0.0), Expr.constant(6.0e7) * y2, -recycle_rate,
]

for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend,
            execution="lambdify",
            matrix_layout=layout,
            telemetry="counters",
        )
        if layout == "banded":
            # The Jacobian has one lower and two upper diagonals in this order.
            options.set_banded_width(1, 2)
        print(f"running frontend={frontend} layout={layout}")
        solution = solve_ivp(
            "radau5",
            rhs,
            ["y1", "y2", "y3"],
            "t",
            np.array([1.0, 0.0, 0.0]),
            0.0,
            100.0,
            100.0,
            options=options,
            jacobian=jacobian,
            rtol=1e-7,
            atol=1e-10,
            max_steps=20_000,
        )
        final = solution.final_state_array()
        mass_error = abs(float(final.sum()) - 1.0)
        print(
            f"frontend={frontend:11} layout={layout:6} "
            f"final={final} mass_error={mass_error:.3e} "
            f"accepted={solution.accepted_steps}"
        )

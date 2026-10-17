"""Matched Radau5 route comparison on a mass-conserving Robertson variant.

The classic Robertson reactions are augmented with a small slow ``y3 -> y1``
recycling channel. This keeps total mass invariant and gives every state a
structural diagonal in the Jacobian, which makes the sparse shifted system's
diagonal pattern explicit. Every row uses the same equations, initial state,
tolerances and analytic Jacobian; only frontend and layout vary. This is a
correctness/configuration demonstration, not a timing benchmark.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, render_table, solve_ivp


y1, y2, y3 = (Expr.variable(name) for name in ("y1", "y2", "y3"))
# Each reaction term is written as an explicit rate law. The added recycle
# channel keeps y1+y2+y3 invariant while retaining a stiff, widely separated
# pair of kinetic scales.
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

# The flat Jacobian expression list is row-major, matching the state order
# [y1, y2, y3]. Supplying it avoids finite-difference work in every route.
rows = []
reference_final = None
for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend,
            execution="lambdify",
            matrix_layout=layout,
            telemetry="counters",
        )
        if layout == "banded":
            # The Jacobian's exact lower/upper half-bandwidths in the chosen
            # state ordering are 1 and 2; narrower values would truncate it.
            options.set_banded_width(1, 2)
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
        route = f"frontend={frontend}, layout={layout}"
        if mass_error > 1e-8:
            raise AssertionError(f"{route}: mass conservation error {mass_error:.3e}")
        if reference_final is None:
            reference_final = final.copy()
        elif not np.allclose(final, reference_final, rtol=1e-6, atol=1e-8):
            raise AssertionError(f"{route}: final state disagrees with the first route")
        rows.append(
            {
                "frontend": frontend,
                "layout": layout,
                "final_state": ", ".join(f"{value:.6g}" for value in final),
                "mass_error": f"{mass_error:.3e}",
                "accepted_steps": solution.accepted_steps,
                "rejected_steps": solution.rejected_steps,
            }
        )

print("Radau5 Robertson route correctness matrix (not a performance benchmark)")
print(render_table(rows))

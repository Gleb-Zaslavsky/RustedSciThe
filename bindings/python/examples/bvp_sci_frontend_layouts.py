"""Matched BVP_sci frontend/layout routes on the exact constant solution.

The first-order BVP is ``y'=0`` with ``y(0)=1``; its exact profile is one.
Each row uses the same mesh, initial guess and boundary residual. Sparse and
Banded are included to show route selection, not to imply that structured
storage wins on a tiny problem.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, render_table, solve_bvp


mesh = np.linspace(0.0, 1.0, 12)
# A constant exact solution makes this a route-correctness check, not a hard
# nonlinear workload on which to infer backend performance.
equations = [Expr.constant(0.0)]
boundary = [Expr.parse("ya_y - 1")]
initial_guess = np.full(mesh.size, 0.8)

rows = []
# Keep the equation, mesh and initial guess fixed so only symbolic frontend
# and linear-system layout vary between rows.
for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend, execution="lambdify",
            matrix_layout=layout, telemetry="counters",
        )
        if layout == "banded":
            # This first-order equation has only nearest-neighbor collocation
            # coupling; the declared half-bandwidth is intentionally narrow.
            options.set_banded_width(1, 1)
        result = solve_bvp(
            "bvp_sci", equations, ["y"], boundary, mesh, initial_guess,
            options=options,
            tolerance=1e-7,
            max_nodes=80,
        )
        values = result.values_array()
        max_error = float(np.max(np.abs(values - 1.0)))
        route = f"frontend={frontend}, layout={layout}"
        # A successful status alone is not enough: compare the whole returned
        # profile against the known solution for every route.
        if result.status != "success":
            raise RuntimeError(f"{route}: BVP_sci ended with status {result.status!r}")
        if max_error > 1e-8:
            raise AssertionError(f"{route}: constant-profile error {max_error:.3e}")
        rows.append(
            {
                "frontend": frontend,
                "layout": layout,
                "nodes": result.mesh_array().size,
                "max_profile_error": f"{max_error:.3e}",
                "status": result.status,
            }
        )

print("BVP_sci route correctness matrix (not a performance benchmark)")
print(render_table(rows))

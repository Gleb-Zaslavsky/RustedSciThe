"""Matched BVP_sci frontend/layout routes on the exact constant solution.

The first-order BVP is ``y'=0`` with ``y(0)=1``; its exact profile is one.
Each row uses the same mesh, initial guess and boundary residual. Sparse and
Banded are included to show route selection, not to imply that structured
storage wins on a tiny problem.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


mesh = np.linspace(0.0, 1.0, 12)
equations = [Expr.constant(0.0)]
boundary = [Expr.parse("ya_y - 1")]
initial_guess = np.full(mesh.size, 0.8)

for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend, execution="lambdify",
            matrix_layout=layout, telemetry="counters",
        )
        if layout == "banded":
            options.set_banded_width(1, 1)
        result = solve_bvp(
            "bvp_sci", equations, ["y"], boundary, mesh, initial_guess,
            options=options,
            tolerance=1e-7,
            max_nodes=80,
        )
        values = result.values_array()
        print(
            f"frontend={frontend:11} layout={layout:6} "
            f"nodes={result.mesh_array().size:3} "
            f"max_profile_error={np.max(np.abs(values - 1.0)):.3e}"
        )

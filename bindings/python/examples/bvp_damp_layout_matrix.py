"""Matched BVP_Damp frontend/layout routes on an exact constant profile.

The scalar equation is ``y'=0`` and its one boundary observation is ``y(0)=1``.
BVP_Damp's Python contract takes one total observation per unknown; it is not
the two-endpoint residual API exposed by BVP_sci. Every run starts from the
same deliberately offset guess, then exercises each supported layout and
symbolic frontend. The profile error is a correctness check, not a benchmark.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


mesh = np.linspace(0.0, 1.0, 14)
guess = np.full(mesh.size, 0.6)

for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend, execution="lambdify",
            matrix_layout=layout, telemetry="timings",
        )
        if layout == "banded":
            options.set_banded_width(1, 1)
        solution = solve_bvp(
            "bvp_damp", [Expr.constant(0.0)], ["y"],
            {"y": [(0, 1.0)]}, mesh, guess,
            options=options,
            bounds={"y": (-0.05, 1.1)},
            tolerance=1e-7,
            max_iterations=50,
            max_refinements=2,
            grid_method="double_points",
            logging_mode="off",
            loglevel="error",
        )
        profile = np.asarray(solution.values_array()).reshape(-1)
        max_error = float(np.max(np.abs(profile - 1.0)))
        print(
            f"frontend={frontend:11} layout={layout:6} "
            f"nodes={solution.mesh_array().size:3} "
            f"max_profile_error={max_error:.3e} "
            f"status={solution.status}"
        )

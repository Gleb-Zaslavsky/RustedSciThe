"""Matched BVP_Damp frontend/layout routes on an exact constant profile.

The scalar equation is ``y'=0`` and its one boundary observation is ``y(0)=1``.
BVP_Damp's Python contract takes one total observation per unknown; it is not
the two-endpoint residual API exposed by BVP_sci. Every run starts from the
same deliberately offset guess, then exercises each supported layout and
symbolic frontend. The profile error is a correctness check, not a benchmark.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, render_table, solve_bvp


mesh = np.linspace(0.0, 1.0, 14)
guess = np.full(mesh.size, 0.6)

# BVP_Damp accepts one observation per unknown (here y(0)=1); unlike BVP_sci,
# this is not a general two-endpoint residual vector.
rows = []
for frontend in ("expr_legacy", "atom_native"):
    for layout in ("dense", "sparse", "banded"):
        options = SolverOptions(
            frontend=frontend, execution="lambdify",
            matrix_layout=layout, telemetry="timings",
        )
        if layout == "banded":
            # This scalar nearest-neighbor collocation fixture has one
            # sub- and super-diagonal in its global Jacobian.
            options.set_banded_width(1, 1)
        solution = solve_bvp(
            "bvp_damp", [Expr.constant(0.0)], ["y"],
            {"y": [(0, 1.0)]}, mesh, guess,
            options=options,
            # Bounds constrain Newton trial states, not the returned profile
            # by post-hoc clipping.
            bounds={"y": (-0.05, 1.1)},
            tolerance=1e-7,
            max_iterations=50,
            max_refinements=2,
            # Keep the refinement policy explicit so this route matrix does
            # not accidentally compare different mesh controllers.
            grid_method="double_points",
            logging_mode="off",
            loglevel="error",
        )
        profile = np.asarray(solution.values_array()).reshape(-1)
        max_error = float(np.max(np.abs(profile - 1.0)))
        route = f"frontend={frontend}, layout={layout}"
        if solution.status != "finished":
            raise RuntimeError(f"{route}: BVP_Damp ended with status {solution.status!r}")
        if max_error > 1e-8:
            raise AssertionError(f"{route}: constant-profile error {max_error:.3e}")
        rows.append(
            {
                "frontend": frontend,
                "layout": layout,
                "nodes": solution.mesh_array().size,
                "max_profile_error": f"{max_error:.3e}",
                "status": solution.status,
            }
        )

print("BVP_Damp route correctness matrix (not a performance benchmark)")
print(render_table(rows))

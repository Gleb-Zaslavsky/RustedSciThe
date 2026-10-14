"""BVP_sci symbolic boundary-value solve with a Sparse layout."""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


y = Expr.variable("y")
solution = solve_bvp(
    "bvp_sci",
    [-y],
    ["y"],
    [Expr.parse("ya_y - 1")],
    np.array([0.0, 0.5, 1.0]),
    np.array([1.0, 0.6, 0.37]),
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="sparse",
        telemetry="timings",
    ),
    output="dense",
)

print("mesh shape:", solution.mesh_array().shape)
print("values shape:", solution.values_array().shape)
print("value at x=1:", solution.sample(1.0))

"""Fit a simple constant BVP profile to endpoint observations with BVP_Damp.

BVP_Damp receives observations as ``state -> [(position, value), ...]`` and
uses bounds to constrain the trial state. The equation y'=0 has the constant
solution y=1; the initial mesh values deliberately start away from it.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


solution = solve_bvp(
    "bvp_damp",
    [Expr.parse("0")],
    ["y"],
    {"y": [(0, 1.0)]},
    np.linspace(0.0, 1.0, 5),
    np.full(5, 0.6),
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        telemetry="timings",
    ),
    bounds={"y": (-10.0, 10.0)},
    max_iterations=20,
    loglevel="error",
    logging_mode="off",
)

print("status:", solution.status)
print("mesh nodes:", len(solution.mesh()))
print("endpoint values:", solution.values()[0][0], solution.values()[-1][0])
print("telemetry:", solution.telemetry_timings_ms())

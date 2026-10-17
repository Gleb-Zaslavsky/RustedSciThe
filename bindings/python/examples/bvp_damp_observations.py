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
    # BVP_Damp treats these as state observations on the mesh. A single
    # endpoint observation is sufficient here because y'=0 forces a constant
    # profile; this is not a general recipe for arbitrary BVPs.
    {"y": [(0, 1.0)]},
    np.linspace(0.0, 1.0, 5),
    np.full(5, 0.6),
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        telemetry="timings",
    ),
    # These broad bounds only prevent pathological trial states; they do not
    # encode the solution or clip the returned profile to the observations.
    bounds={"y": (-10.0, 10.0)},
    max_iterations=20,
    loglevel="error",
    logging_mode="off",
)

profile = solution.values()[0]
profile_error = max(abs(value - 1.0) for value in profile)
if solution.status != "finished" or profile_error > 1e-7:
    raise AssertionError(
        f"constant BVP profile check failed: status={solution.status!r}, "
        f"max_error={profile_error:.3e}"
    )
print("status:", solution.status)
print("mesh nodes:", len(solution.mesh()))
# BVP_Damp values are state-major: select the only state first, then its first
# and last mesh samples. Indexing the outer list as a node axis is incorrect.
print("endpoint values:", profile[0], profile[-1])
print(f"maximum constant-profile error: {profile_error:.3e}")
print("telemetry:", solution.telemetry_timings_ms())

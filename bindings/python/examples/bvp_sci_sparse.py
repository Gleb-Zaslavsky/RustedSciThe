"""Solve ``y' = -y`` from a left boundary value with BVP_sci.

The exact profile is ``exp(-x)``. This example asks the symbolic BVP_sci
frontend to prepare a Sparse collocation system and retain dense output. The
reported sample at the right endpoint should be close to ``exp(-1)``; Sparse
is selected to demonstrate the route, not to claim it is fastest for every
problem size or structure.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


y = Expr.variable("y")
solution = solve_bvp(
    "bvp_sci",
    [-y],
    ["y"],
    [Expr.parse("ya_y - 1")],
    # Three nodes are enough to specify a nontrivial initial mesh; refinement
    # is controlled by BVP_sci and the requested tolerance.
    np.array([0.0, 0.5, 1.0]),
    np.array([1.0, 0.6, 0.37]),
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        # Sparse demonstrates the supported structured route. For this tiny
        # scalar problem it is not expected to beat Dense in wall-clock time.
        matrix_layout="sparse",
        telemetry="timings",
    ),
    # Dense output is requested because the example samples at the endpoint
    # through the solver's interpolation API, independent of the mesh.
    output="dense",
)

mesh_values = solution.mesh_array()
profile = solution.values_array()
endpoint = float(solution.sample(1.0)[0])
endpoint_error = abs(endpoint - float(np.exp(-1.0)))
# This loose bound checks interpolation/profile plumbing on the small example
# mesh; it is not a convergence claim for arbitrary BVPs.
if solution.status != "success" or endpoint_error > 1e-4:
    raise AssertionError(
        f"Sparse BVP_sci check failed: status={solution.status!r}, "
        f"endpoint error={endpoint_error:.3e}"
    )
print("mesh shape:", mesh_values.shape)
print("values shape:", profile.shape)
print("value at x=1:", endpoint)
print("exact y(1):", np.exp(-1.0))
print(f"endpoint error: {endpoint_error:.3e}")

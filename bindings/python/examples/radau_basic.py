"""Solve the scalar decay equation with Radau5.

The model is ``y' = -y``, ``y(0) = 1`` and its exact solution is
``exp(-t)``. This is a compact check of the default symbolic IVP route and
dense-output postprocessing. The sampled value at ``t=1`` should be close to
``exp(-1)``; timings are diagnostic output, not a benchmark.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


y = Expr.variable("y")
solution = solve_ivp(
    "radau5",
    [-y],
    ["y"],
    "t",
    np.array([1.0]),
    0.0,
    1.0,
    # This is an upper bound for adaptive steps, not a request for 0.1-sized
    # steps. It keeps the small demo easy to inspect without fixing the mesh.
    0.1,
    # Timings are useful for learning which native stages exist; this example
    # makes no speed claim and leaves counters out of the display.
    options=SolverOptions(telemetry="timings"),
    # The tolerances balance a clear exact-solution check with a short demo.
    rtol=1e-5,
    atol=1e-7,
    # Dense output enables sampling between accepted adaptive steps.
    output="dense",
)

print("final time:", solution.final_time)
final_state = solution.final_state_array()
dense_midpoint = solution.sample(0.5)
final_error = abs(float(final_state[0]) - np.exp(-1.0))
midpoint_error = abs(float(dense_midpoint[0]) - np.exp(-0.5))
# These bounds are looser than roundoff but tighter than the requested solve
# tolerance by a comfortable factor; they guard the lesson's exact solution.
if final_error > 1e-5 or midpoint_error > 1e-5:
    raise AssertionError(
        f"Radau decay errors exceed tolerance: final={final_error:.3e}, "
        f"dense midpoint={midpoint_error:.3e}"
    )
print("final state:", final_state)
print("dense sample at t=0.5:", dense_midpoint)
print("exact y(1):", np.exp(-1.0))
print(f"exact-solution errors: final={final_error:.3e}, midpoint={midpoint_error:.3e}")
print("telemetry:", solution.telemetry_timings_ms())

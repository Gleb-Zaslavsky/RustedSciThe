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
    0.1,
    options=SolverOptions(telemetry="timings"),
    rtol=1e-5,
    atol=1e-7,
    output="dense",
)

print("final time:", solution.final_time)
print("final state:", solution.final_state_array())
print("dense sample at t=0.5:", solution.sample(0.5))
print("exact y(1):", np.exp(-1.0))
print("telemetry:", solution.telemetry_timings_ms())

"""Minimal Radau5 solve with the Rust symbolic frontend and NumPy output."""

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
print("telemetry:", solution.telemetry_timings_ms())

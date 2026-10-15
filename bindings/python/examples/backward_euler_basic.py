"""Solve a scalar decay IVP with fixed-step Backward Euler.

This example uses the common one-shot facade. Backward Euler is a first-order
implicit method; reducing ``step_size`` improves accuracy at additional solve
cost. The reported endpoint is expected to differ slightly from exp(-2).
"""

import math

from rustedscithe import Expr, SolverOptions, solve_ivp


y = Expr.variable("y")
solution = solve_ivp(
    "backward_euler",
    [-y],
    ["y"],
    "t",
    [1.0],
    0.0,
    2.0,
    0.05,
    options=SolverOptions(telemetry="timings"),
)

print("computed endpoint:", solution.states()[-1][0])
print("exact endpoint:", math.exp(-2.0))
print("telemetry:", solution.telemetry_timings_ms())

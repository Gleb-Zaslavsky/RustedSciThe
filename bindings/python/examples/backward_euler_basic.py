"""Solve a scalar decay IVP with fixed-step Backward Euler.

This example uses the common one-shot facade. Backward Euler is a first-order
implicit method; reducing ``step_size`` improves accuracy at additional solve
cost. The reported endpoint is expected to differ slightly from exp(-2).
"""

import math

from rustedscithe import Expr, SolverOptions, solve_ivp


y = Expr.variable("y")
step_size = 0.05
solution = solve_ivp(
    "backward_euler",
    [-y],
    ["y"],
    "t",
    [1.0],
    0.0,
    2.0,
    # This is a fixed step, so the endpoint error reflects first-order method
    # accuracy rather than adaptive error control.
    step_size,
    # Timing collection is opt-in and reports native stages for this solve.
    options=SolverOptions(telemetry="timings"),
)

computed = float(solution.states()[-1][0])
exact = math.exp(-2.0)
# Backward Euler has the closed-form discrete amplification (1+h)^(-N).
# Check it separately from the continuous exact solution to expose method error.
steps = round(2.0 / step_size)
discrete_reference = (1.0 + step_size) ** (-steps)
if abs(computed - discrete_reference) > 1e-9:
    raise AssertionError("computed state does not match the Backward Euler recurrence")
if abs(computed - exact) > 0.01:
    raise AssertionError("fixed-step Backward Euler error is unexpectedly large")
print("computed endpoint:", computed)
print("exact endpoint:", exact)
print(f"discrete-method error: {abs(computed - exact):.3e}")
print("telemetry:", solution.telemetry_timings_ms())

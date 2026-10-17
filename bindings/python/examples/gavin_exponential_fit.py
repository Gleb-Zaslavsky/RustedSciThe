"""Fit exponential decay with Gavin's damped least-squares controller.

The model is ``y = amplitude * exp(-x / tau) + offset``.  The time scale
``tau`` is constrained to be positive by an explicit user policy.  This is a
sampled-observation fit: unlike ``FittingSolver``, the sample array is passed
directly rather than expanded into a residual equation per observation.
"""

import math

from rustedscithe import Expr, GavinFittingSolver


x_data = [index * 0.15 for index in range(40)]
y_data = [2.4 * math.exp(-x_value / 1.7) + 0.35 for x_value in x_data]

solver = GavinFittingSolver(
    equations=[Expr.parse("amplitude*exp(-x/tau) + offset")],
    x_data=x_data,
    observations=y_data,
    sample_parameter="x",
    unknowns=["amplitude", "tau", "offset"],
    initial_guess=[1.8, 1.2, 0.2],
    positive_parameters=["tau"],
    frontend="atom_native",
    execution_policy="sequential",
    update_policy="nielsen",
    max_iterations=500,
    gradient_tolerance=1e-8,
    parameter_step_tolerance=1e-9,
    telemetry=True,
    preparation_telemetry="off",
)
result = solver.solve()  # Gavin is a one-shot builder; create another to refit.

print("parameters:", dict(zip(["amplitude", "tau", "offset"], result.parameters())))
print("termination:", result.termination)
print("R^2:", result.r_squared)
print("solve counters:", result.telemetry_counters())
print("solve timings:", result.telemetry_timings_ms())

assert abs(result.parameters()[0] - 2.4) < 1e-4
assert abs(result.parameters()[1] - 1.7) < 1e-4
assert abs(result.parameters()[2] - 0.35) < 1e-4

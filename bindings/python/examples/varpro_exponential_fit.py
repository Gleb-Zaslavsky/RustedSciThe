"""Use variable projection for a separable exponential model.

For ``y = amplitude * exp(-x/tau) + offset``, ``tau`` changes the basis shape
nonlinearly while amplitude and offset are linear coefficients.  VarPro
optimizes the nonlinear parameter and solves for the linear coefficients
inside each iteration.  Its Python route is ExprLegacy, sequential and
one-shot; use a new builder for a different data set.
"""

import math

from rustedscithe import VarProSolver


x_data = [index * 0.2 for index in range(30)]
y_data = [2.5 * math.exp(-x_value / 1.7) + 0.8 for x_value in x_data]

solver = VarProSolver("x")
solver.with_parameters(["tau"])
solver.with_initial_parameters([1.2])
solver.with_data(x_data, y_data)
solver.with_basis_str("tau", "exp(-x/tau)")
solver.with_invariant_str("1")  # constant offset is another linear term
solver.with_tolerances(1e-9, 1e-9, 1e-10)
solver.with_max_iterations(200)
solver.with_max_function_evaluations(1000)
solver.with_step_bound(10.0)
solver.with_scale_diag(True)
solver.with_telemetry("detailed")

result = solver.solve()
print("fitted coefficients:", result.solution())
print("termination:", result.termination, "successful:", result.successful)
print("R^2:", result.r_squared, "objective:", result.objective_function)
print("function evaluations:", result.function_evaluations)
print("counters:", result.telemetry_counters())
print("timings:", result.telemetry_timings_ms())

assert result.successful
assert abs(result.solution()["tau"] - 1.7) < 0.05
assert abs(result.solution()["c0"] - 2.5) < 0.05
assert abs(result.solution()["c1"] - 0.8) < 0.05

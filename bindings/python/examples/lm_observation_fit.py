"""Fit a straight line to observations with the reusable symbolic LM solver.

Each observation contributes one residual ``a*x_i + b - y_i``.  Sample values
are bound as numeric parameters, so the symbolic residual system is prepared
once and can be reused for another data set with ``solve_with_parameters``.
This is the canonical least-squares equation API, distinct from Gavin's
sampled-model API and from variable projection.
"""

from rustedscithe import Expr, FittingSolver


x_data = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
y_data = [1.2, 2.1, 3.0, 3.9, 4.8, 5.7]  # exact line: y = 1.8*x + 1.2

parameter_names: list[str] = []
equations: list[Expr] = []
parameter_values: list[float] = []
for index, (x_value, y_value) in enumerate(zip(x_data, y_data)):
    x_name = f"x_{index}"
    y_name = f"y_{index}"
    parameter_names.extend((x_name, y_name))
    parameter_values.extend((x_value, y_value))
    equations.append(Expr.parse(f"a*{x_name} + b - {y_name}"))

fit = FittingSolver(
    equations,
    initial_guess=[0.0, 0.0],
    variables=["a", "b"],
    parameters=parameter_names,
    frontend="expr_legacy",
    execution_policy="sequential",
    telemetry="detailed",
    tolerance=1e-10,
    max_iterations=100,
)

# Constructor preparation is reused by every subsequent parameter binding.
first = fit.solve_with_parameters(parameter_values)
print("fit (a, b):", first.solution())
print("termination:", first.termination)
print("objective:", first.objective_function)
print("stage timings:", first.telemetry_timings_ms())

# Reuse the same residual structure for a second observation set.
shifted_values = parameter_values.copy()
for index in range(1, len(shifted_values), 2):
    shifted_values[index] += 2.0
second = fit.solve_with_parameters(shifted_values)
print("after shifting every observation by +2, (a, b):", second.solution())
assert abs(first.solution()[0] - 1.8) < 1e-6
assert abs(first.solution()[1] - 1.2) < 1e-6
assert abs(second.solution()[1] - 3.2) < 1e-6

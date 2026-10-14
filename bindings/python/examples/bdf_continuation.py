"""Prepared BDF continuation: only numeric bindings change between solves."""

import numpy as np

from rustedscithe import BdfSolver, Expr, SolverOptions


y = Expr.variable("y")
alpha = Expr.variable("alpha")
solver = BdfSolver(
    [-alpha * y],
    ["y"],
    "t",
    np.array([1.0]),
    0.0,
    0.5,
    0.1,
    parameters=["alpha"],
    parameter_values=np.array([1.0]),
    options=SolverOptions(frontend="atom_native", telemetry="counters"),
)

solver.prepare()
first = solver.solve()
continued = solver.continue_with_parameters(np.array([2.0]), 1.0)
print("prepared first endpoint:", first.states_array()[-1])
print("continued endpoint:", continued.states_array()[-1])
print("continuation counters:", continued.telemetry_counters())

"""Reuse one prepared BDF model for a two-segment parameter series.

The equation is ``y' = -alpha*y`` with ``y(0)=1``. The first segment uses
``alpha=1`` on ``[0, 0.5]``; the warm continuation changes to ``alpha=2`` on
``[0.5, 1]``. Therefore the exact terminal value is ``exp(-1.5)``. The
symbolic expression is prepared once while only the numeric parameter binding
changes.
"""

import math

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
print("exact continued endpoint:", math.exp(-1.5))
print("continuation counters:", continued.telemetry_counters())

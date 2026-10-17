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
    # AtomNative is selected explicitly so the example shows the native
    # symbolic route. BDF currently uses Dense linear algebra by design.
    options=SolverOptions(frontend="atom_native", telemetry="counters"),
)

# Prepare once: continuation below changes numeric parameter bindings, not the
# symbolic graph or the selected solver route.
solver.prepare()
first = solver.solve()
continued = solver.continue_with_parameters(np.array([2.0]), 1.0)
first_endpoint = float(first.states_array()[-1, 0])
continued_endpoint = float(continued.states_array()[-1, 0])
first_error = abs(first_endpoint - math.exp(-0.5))
continuation_error = abs(continued_endpoint - math.exp(-1.5))
# Check both sides of the rebind boundary: this catches carrying the old
# parameter into segment two as well as an incorrect final integration.
if first_error > 2e-5 or continuation_error > 2e-5:
    raise AssertionError(
        f"BDF continuation errors are too large: first={first_error:.3e}, "
        f"continued={continuation_error:.3e}"
    )
print("prepared first endpoint:", first_endpoint)
print("continued endpoint:", continued_endpoint)
print("exact continued endpoint:", math.exp(-1.5))
print(f"continuation errors: first={first_error:.3e}, continued={continuation_error:.3e}")
print("continuation counters:", continued.telemetry_counters())

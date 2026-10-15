"""Integrate the Lorenz attractor with LSODE2's automatic controller.

The nonlinear three-state system is sensitive to small numerical changes, so
do not compare long-time trajectories pointwise across solvers. This moderate
run checks that the trajectory remains finite and reports the actual
controller/family used. It is adapted from the crate's Lorenz example without
requiring a plotting or animation dependency.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


x, y, z = (Expr.variable(name) for name in ("x", "y", "z"))
rhs = [
    Expr.constant(10.0) * (y - x),
    x * (Expr.constant(28.0) - z) - y,
    x * y - Expr.constant(8.0 / 3.0) * z,
]
solution = solve_ivp(
    "lsode2", rhs, ["x", "y", "z"], "t", [1.0, 1.0, 1.0],
    0.0, 25.0, 0.01,
    controller="auto",
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
    rtol=1e-8,
    atol=1e-10,
)
trajectory = solution.states_array()
assert np.isfinite(trajectory).all()
print("points:", trajectory.shape[0])
print("controller/family:", solution.controller_mode, solution.executed_family)
print("last state:", trajectory[-1])
print("telemetry:", solution.telemetry_timings_ms())

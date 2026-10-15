"""Integrate the crate's planar 12-state gravitational three-body example.

The state order is ``(x, vx, y, vy)`` for each body. Pairwise distances are
defined once as symbolic aliases and recursively substituted into Newtonian
accelerations, illustrating that Python uses the same native Expr tree and
alias semantics as Rust. This short trajectory is for API/correctness
exploration, not a long-term orbital-stability claim.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


names = [
    "x0", "vx0", "y0", "vy0",
    "x1", "vx1", "y1", "vy1",
    "x2", "vx2", "y2", "vy2",
]
variables = {name: Expr.variable(name) for name in names}
aliases = {
    "R01": Expr.parse("((x0-x1)^2 + (y0-y1)^2)^0.5"),
    "R02": Expr.parse("((x0-x2)^2 + (y0-y2)^2)^0.5"),
    "R12": Expr.parse("((x1-x2)^2 + (y1-y2)^2)^0.5"),
    "k": Expr.constant(39.47841760435743),
    "m0": Expr.constant(1.0),
    "m1": Expr.constant(0.5),
    "m2": Expr.constant(0.75),
}

accelerations = [
    Expr.parse("-k*(m1*(x0-x1)/R01^3 + m2*(x0-x2)/R02^3)"),
    Expr.parse("-k*(m1*(y0-y1)/R01^3 + m2*(y0-y2)/R02^3)"),
    Expr.parse("-k*(m0*(x1-x0)/R01^3 + m2*(x1-x2)/R12^3)"),
    Expr.parse("-k*(m0*(y1-y0)/R01^3 + m2*(y1-y2)/R12^3)"),
    Expr.parse("-k*(m0*(x2-x0)/R02^3 + m1*(x2-x1)/R12^3)"),
    Expr.parse("-k*(m0*(y2-y0)/R02^3 + m1*(y2-y1)/R12^3)"),
]
forces = [expr.where_substitute(aliases) for expr in accelerations]
rhs = [
    variables["vx0"], forces[0], variables["vy0"], forces[1],
    variables["vx1"], forces[2], variables["vy1"], forces[3],
    variables["vx2"], forces[4], variables["vy2"], forces[5],
]
initial = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 5.0,
    0.0, -0.4, 33.0, 0.0,
])

solution = solve_ivp(
    "lsode2", rhs, names, "t", initial, 0.0, 0.02, 0.001,
    controller="bdf",
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
    rtol=1e-8,
    atol=1e-10,
)
trajectory = solution.states_array()
assert np.isfinite(trajectory).all()
print("trajectory shape:", trajectory.shape)
print("final body positions:", trajectory[-1, [0, 2, 4, 6, 8, 10]])
print("controller/family:", solution.controller_mode, solution.executed_family)

"""Compare analytic and finite-difference Jacobians in the BDF callback API.

Python callbacks are a convenient route for an existing numeric model. This
route currently uses dense, sequential Lambdify execution; for a symbolic
model, prefer ``BdfSolver`` so preparation and callbacks stay native Rust.
"""

import math

from rustedscithe import BdfCallbackSolver, SolverOptions


def residual(time, state):
    """Return y' - f(t, y) for y' = -y; this model is autonomous."""
    del time
    return [-state[0]]


def jacobian(_time, _state):
    """Return d(residual)/dy in a nested row-major list."""
    return [[-1.0]]


for jacobian_callback in (jacobian, None):
    solver = BdfCallbackSolver(
        1,
        0.0,
        0.5,
        0.1,
        [1.0],
        residual,
        jacobian_callback,
        options=SolverOptions(telemetry="counters"),
    )
    solution = solver.solve()
    print(
        f"jacobian={solver.provenance()['jacobian_source']:16} "
        f"final={solution.states()[-1][0]:.8f} "
        f"exact={math.exp(-0.5):.8f}"
    )
    print("callback counters:", solution.telemetry_counters())

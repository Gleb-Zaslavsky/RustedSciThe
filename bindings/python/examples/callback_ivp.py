"""Compare analytic and finite-difference Jacobians in the BDF callback API.

Python callbacks are a convenient route for an existing numeric model. This
route currently uses dense, sequential Lambdify execution; for a symbolic
model, prefer ``BdfSolver`` so preparation and callbacks stay native Rust.
"""

import math

from rustedscithe import BdfCallbackSolver, SolverOptions


def residual(time, state):
    """Return the native residual vector for the autonomous decay equation.

    The callback contract is ``(time, state) -> residual``; inputs arrive as
    Python floats/lists and the returned sequence must have one finite value
    per state component.
    """
    del time
    return [-state[0]]


def jacobian(_time, _state):
    """Return ``d(residual)/dy`` as a nested row-major matrix.

    Passing ``None`` for this callback below selects the wrapper's native
    finite-difference Jacobian path instead.
    """
    return [[-1.0]]


# Match the same initial condition and interval so only Jacobian acquisition
# differs between the analytic and finite-difference runs.
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
    actual = solution.states()[-1][0]
    exact = math.exp(-0.5)
    if abs(actual - exact) > 1e-5:
        raise AssertionError(
            f"{solver.provenance()['jacobian_source']} route error is too large: "
            f"{actual - exact:.3e}"
        )
    print(
        f"jacobian={solver.provenance()['jacobian_source']:16} "
        f"final={actual:.8f} exact={exact:.8f}"
    )
    print("callback counters:", solution.telemetry_counters())

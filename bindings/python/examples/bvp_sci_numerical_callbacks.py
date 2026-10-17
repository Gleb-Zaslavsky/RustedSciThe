"""Solve y'=-y through BVP_sci's native Dense numerical callback route.

The callback supplies the RHS and exact pointwise state Jacobian. Collocation,
Newton iteration and the Dense linear solve stay in the Rust core.
"""

from __future__ import annotations

import math

import numpy as np

from rustedscithe import BvpSciSolver, Expr, SolverOptions


def main() -> None:
    mesh = np.linspace(0.0, 1.0, 12)
    # Keep the boundary values exact but perturb interior nodes, so Newton
    # actually exercises both the residual and the analytic Jacobian callback.
    initial_guess = np.exp(-mesh) * (1.0 + 0.04 * np.sin(np.pi * mesh))
    solver = BvpSciSolver(
        expressions=[],
        state_variables=["y"],
        boundary_expressions=[Expr.parse("ya_y - 1")],
        mesh=mesh.tolist(),
        initial_state=initial_guess.tolist(),
        options=SolverOptions(
            frontend="numerical",
            execution="numerical",
            # User callbacks currently use the Dense numerical route.
            matrix_layout="dense",
            # Detailed mode includes both event counters and stage timings.
            telemetry="detailed",
        ),
        # The exact Jacobian avoids finite-difference perturbations and makes
        # the analytic-callback contract explicit.
        rhs_callback=lambda _x, state, _parameters: [-state[0]],
        rhs_jacobian_callback=lambda _x, _state, _parameters: [[-1.0]],
        output="dense",
    )
    solution = solver.solve()
    exact = math.exp(-1.0)
    actual = solution.sample(1.0)[0]
    print(f"status={solution.status}; y(1)={actual:.12g}; exact={exact:.12g}")
    print(solution.telemetry_report())
    counters = solution.telemetry_counters()
    if counters["jacobian_calls"] == 0:
        raise AssertionError("the analytic state-Jacobian callback was not exercised")
    if abs(actual - exact) > 1e-6:
        raise AssertionError(f"BVP profile endpoint error is too large: {actual - exact:.3e}")


if __name__ == "__main__":
    main()

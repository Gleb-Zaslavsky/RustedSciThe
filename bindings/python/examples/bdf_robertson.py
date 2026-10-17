"""Solve Robertson kinetics with the dense BDF wrapper.

The model conserves ``y1 + y2 + y3`` and has time scales spanning many orders
of magnitude. BDF is deliberately configured Dense: this wrapper mirrors the
classic dense BDF solver and does not pretend to expose sparse storage. The
example runs the same chemistry through both symbolic frontends.
"""

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


y1, y2, y3 = (Expr.variable(name) for name in ("y1", "y2", "y3"))
rhs = [
    Expr.constant(-0.04) * y1 + Expr.constant(1.0e4) * y2 * y3,
    Expr.constant(0.04) * y1 - Expr.constant(1.0e4) * y2 * y3
    - Expr.constant(3.0e7) * y2 * y2,
    Expr.constant(3.0e7) * y2 * y2,
]

# BDF's public route here is Dense-only, so the frontend is the comparison
# axis; asking for Sparse/Banded would be an unsupported configuration.
reference_final = None
for frontend in ("expr_legacy", "atom_native"):
    solution = solve_ivp(
        "bdf",
        rhs,
        ["y1", "y2", "y3"],
        "t",
        np.array([1.0, 0.0, 0.0]),
        0.0,
        100.0,
        100.0,
        options=SolverOptions(
            frontend=frontend, execution="lambdify",
            matrix_layout="dense", telemetry="timings",
        ),
        rtol=1e-7,
        atol=1e-10,
        # Robertson kinetics is stiff over [0, 100]; permit the adaptive
        # controller enough work to resolve its fast initial transient.
        max_bdf_order=5,
        max_steps=20_000,
    )
    final = solution.states_array()[-1]
    mass_error = abs(float(final.sum()) - 1.0)
    if mass_error > 1e-8:
        raise AssertionError(f"{frontend}: Robertson total mass drifted by {mass_error:.3e}")
    if reference_final is None:
        reference_final = final.copy()
    elif not np.allclose(final, reference_final, rtol=1e-7, atol=1e-10):
        raise AssertionError("BDF symbolic frontends produced different final states")
    print(
        f"frontend={frontend:11} final={final} mass_error={mass_error:.3e} "
        f"telemetry={solution.telemetry_timings_ms()}"
    )

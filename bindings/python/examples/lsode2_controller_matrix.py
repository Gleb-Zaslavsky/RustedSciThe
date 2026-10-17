"""Compare LSODE2's BDF, Adams and automatic controllers on one IVP.

Run from this directory with ``python examples/lsode2_controller_matrix.py``.
The equation and tolerances are identical for every row; only the native
controller policy changes. This is a behavior/configuration demonstration,
not a performance benchmark.
"""

import math

from rustedscithe import Expr, SolverOptions, render_table, solve_ivp


y = Expr.variable("y")
rate = Expr.variable("rate")
initial_value = 1.0
decay_rate = 0.7
end_time = 2.0
expected_final = initial_value * math.exp(-decay_rate * end_time)
rows = []

# All three runs solve the identical scalar decay equation. Only the method
# policy changes: BDF is fixed stiff, Adams is fixed nonstiff, and Auto lets
# LSODE2 switch families using its native controller diagnostics.
for controller in ("bdf", "adams", "auto"):
    solution = solve_ivp(
        "lsode2",
        [-rate * y],
        ["y"],
        "t",
        [initial_value],
        0.0,
        end_time,
        0.1,
        parameters=["rate"],
        parameter_values=[decay_rate],
        controller=controller,
        options=SolverOptions(telemetry="counters"),
        rtol=1e-7,
        atol=1e-9,
    )
    if solution.final_state is None:
        raise RuntimeError(f"controller={controller}: solver returned no final state")
    final = solution.final_state[0]
    abs_error = abs(final - expected_final)
    if not solution.status.startswith("finished") or abs_error > 1e-5:
        raise RuntimeError(
            f"controller={controller}: status={solution.status!r}, "
            f"absolute error={abs_error:.3e}"
        )
    rows.append(
        {
            "requested_controller": controller,
            "active_controller": solution.controller_mode,
            "status": solution.status,
            "final_y": f"{final:.9f}",
            "absolute_error": f"{abs_error:.3e}",
            "residual_calls": solution.telemetry_counters()["residual_calls"],
        }
    )

print("LSODE2 controller correctness matrix (not a performance benchmark)")
print(render_table(rows))

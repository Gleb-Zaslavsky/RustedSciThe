"""Compare LSODE2's BDF, Adams and automatic controllers on one IVP.

Run from this directory with ``python examples/lsode2_controller_matrix.py``.
The equation and tolerances are identical for every row; only the native
controller policy changes. This is a behavior/configuration demonstration,
not a performance benchmark.
"""

from rustedscithe import Expr, SolverOptions, solve_ivp


y = Expr.variable("y")
rate = Expr.variable("rate")
initial_value = 1.0
decay_rate = 0.7
end_time = 2.0

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
    print(
        f"controller={solution.controller_mode:24} "
        f"final={solution.final_state[0]:.9f} "
        f"residual_calls={solution.telemetry_counters()['residual_calls']}"
    )

"""Restart one prepared BDF route for a new initial-value segment.

The equation y'=-rate*y has the closed form y(t)=y0*exp(-rate*(t-t0)).
`restart` deliberately starts a new initial-value segment; it is distinct from
advancing a solver's accepted trajectory in time.
"""

import math

from rustedscithe import Expr, SolverOptions, StiffPreset, StiffSolverRequest


def main() -> None:
    prepared = StiffSolverRequest.bdf(
        [Expr.parse("-rate*y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.5,
        max_step=0.025,
        rtol=1e-9,
        atol=1e-11,
        parameters=["rate"],
        parameter_values=[0.5],
        options=SolverOptions(telemetry="detailed"),
    ).prepare(StiffPreset.solve_1d())

    first = prepared.solve()
    second_start, second_initial = 0.5, 0.7
    second = prepared.restart(second_start, [second_initial], 1.0)
    expected = second_initial * math.exp(-0.5 * (1.0 - second_start))

    print(f"native solver: {prepared.solver}")
    print(f"first segment endpoint: {first.states()[-1][0]:.10g}")
    print(f"restarted endpoint: {second.states()[-1][0]:.10g}")
    print(f"closed-form endpoint: {expected:.10g}")
    print(f"restart status: {prepared.status}")
    print(prepared.telemetry_snapshot())


if __name__ == "__main__":
    main()

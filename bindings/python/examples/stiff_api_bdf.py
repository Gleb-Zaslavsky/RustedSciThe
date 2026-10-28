"""Build and run a parameterized BDF problem through the native StiffAPI.

This small example focuses on the facade boundary: shared route policy goes
through SolverOptions/StiffPreset, while BDF-specific controls stay on its
request factory. For a fixed-step calculation, use the
StiffSolverRequest.backward_euler factory instead of translating `max_step`
into a different numerical meaning.
"""

from rustedscithe import Expr, SolverOptions, StiffPreset, StiffSolverRequest


def main() -> None:
    # A symbolic parameter keeps the equation reusable for later application
    # configurations; its initial numeric binding is explicit and ordered.
    request = StiffSolverRequest.bdf(
        [Expr.parse("-rate*y")],
        state_variables=["y"],
        time_variable="t",
        y0=[1.0],
        t0=0.0,
        t_end=1.0,
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10,
        parameters=["rate"],
        parameter_values=[0.7],
        options=SolverOptions(
            execution="lambdify",
            matrix_layout="dense",
            telemetry="detailed",
        ),
    )

    # The preset selects only common policy axes. It does not erase the BDF
    # tolerance, step controls, parameter binding or method-specific options.
    prepared = request.prepare(StiffPreset.solve_1d())
    solution = prepared.solve()

    print(f"solver: {prepared.solver}")
    print(f"telemetry mode: {prepared.telemetry_mode}")
    print(f"final state: {solution.states()[-1]}")
    print(solution.telemetry_report())


if __name__ == "__main__":
    main()

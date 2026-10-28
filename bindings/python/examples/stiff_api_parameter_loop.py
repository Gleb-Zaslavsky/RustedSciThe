"""Run a fixed-initial-condition parameter loop through the StiffAPI facade."""

from rustedscithe import Expr, SolverOptions, StiffPreset, StiffSolverRequest


request = StiffSolverRequest.bdf(
    [Expr.parse("-rate*y")],
    ["y"],
    "t",
    [1.0],
    0.0,
    1.0,
    0.02,
    options=SolverOptions(telemetry="counters"),
    parameters=["rate"],
    parameter_values=[1.0],
)
prepared = request.prepare(StiffPreset.solve_1d())

for point in prepared.solve_parameter_loop([1.0], [[0.5], [], [2.0]]):
    if not point.succeeded:
        print(
            f"point={point.index}, parameters={point.parameter_values}: "
            f"{type(point.error).__name__}: {point.error}"
        )
        continue
    solution = point.solution
    print(
        f"point={point.index}, parameters={point.parameter_values}, "
        f"final_state={solution.states_array()[-1].tolist()}"
    )
    print(solution.telemetry_report())

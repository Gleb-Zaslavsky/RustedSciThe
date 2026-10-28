"""Run fresh parameterized solves through the common StiffAPI facade.

The first solve keeps the interval from the request. The second supplies a
different interval, initial state and parameter values. Both reuse the
prepared route; neither inherits a previous trajectory implicitly.
"""

import argparse
import math
from typing import Any

from rustedscithe import Expr, SolverOptions, StiffSolverRequest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "solver", choices=("bdf", "backward_euler", "lsode2", "radau"), default="bdf"
    )
    name = parser.parse_args().solver
    equation = [Expr.parse("-rate*y")]
    states = ["y"]
    parameters = ["rate"]
    options = SolverOptions(telemetry="detailed")

    if name == "backward_euler":
        request = StiffSolverRequest.backward_euler(
            equation, states, "t", [1.0], 0.0, 0.1,
            h=0.001, parameters=parameters, parameter_values=[1.0], options=options,
        )
    elif name == "radau":
        request = StiffSolverRequest.radau(
            equation, states, "t", [1.0], 0.0, 0.1, 0.02,
            parameters=parameters, parameter_values=[1.0], options=options,
        )
    elif name == "lsode2":
        request = StiffSolverRequest.lsode2(
            equation, states, "t", [1.0], 0.0, 0.1, 0.02,
            parameters=parameters, parameter_values=[1.0], options=options,
        )
    else:
        request = StiffSolverRequest.bdf(
            equation, states, "t", [1.0], 0.0, 0.1, 0.02,
            parameters=parameters, parameter_values=[1.0], options=options,
        )

    prepared = request.prepare()
    configured = prepared.solve_with_parameter_values([2.0], [1.5])
    configured_value = terminal_value(configured)
    configured_expected = 2.0 * math.exp(-1.5 * 0.1)

    explicit_interval = prepared.solve_with_parameter_values_in_interval(
        0.2, [3.0], 0.4, [0.8]
    )
    explicit_value = terminal_value(explicit_interval)
    explicit_expected = 3.0 * math.exp(-0.8 * 0.2)

    print(f"solver={prepared.solver}")
    print(f"configured interval: {configured_value:.10g} (exact {configured_expected:.10g})")
    print(f"new interval: {explicit_value:.10g} (exact {explicit_expected:.10g})")
    print(explicit_interval.telemetry_report())


def terminal_value(solution: Any) -> float:
    """Read the final state across the solver-specific result types."""
    if hasattr(solution, "states"):
        return float(solution.states()[-1][0])
    return float(solution.final_state[0])


if __name__ == "__main__":
    main()

"""Show time-segment and fresh-solve APIs across stiff solver families.

All routes solve y'=-rate*y. BDF, Backward Euler and LSODE2 can advance from
the last accepted endpoint to a later bound. Radau instead starts a fresh
parameterized solve from the explicitly supplied initial state.
"""

import argparse
import math
from typing import Any

from rustedscithe import Expr, SolverOptions, StiffSolverRequest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("bdf", "backward_euler", "radau", "lsode2"))
    name = parser.parse_args().solver
    equation = [Expr.parse("-rate*y")]
    options = SolverOptions(telemetry="detailed")

    prepared: Any
    first: Any
    result: Any
    if name == "bdf":
        prepared = StiffSolverRequest.bdf(
            equation, ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[0.5], options=options,
        ).prepare()
        first = prepared.solve()
        result = prepared.continue_bdf_with_parameters([1.2], 0.2)
        final = result.states()[-1][0]
        expected = first.states()[-1][0] * math.exp(-1.2 * 0.1)
    elif name == "backward_euler":
        prepared = StiffSolverRequest.backward_euler(
            equation, ["y"], "t", [1.0], 0.0, 0.1,
            h=0.001, parameters=["rate"], parameter_values=[0.5], options=options,
        ).prepare()
        first = prepared.solve()
        result = prepared.continue_backward_euler(0.2, [1.2])
        final = result.states()[-1][0]
        expected = first.states()[-1][0] * math.exp(-1.2 * 0.1)
    elif name == "radau":
        prepared = StiffSolverRequest.radau(
            equation, ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[0.5], output="dense", options=options,
        ).prepare()
        result = prepared.solve_with_parameter_values([0.8], [1.2])
        final = result.sample(0.1)[0]
        expected = 0.8 * math.exp(-1.2 * 0.1)
    else:
        prepared = StiffSolverRequest.lsode2(
            equation, ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[0.5], options=options,
        ).prepare()
        first = prepared.solve()
        result = prepared.continue_lsode2_with_parameters([1.2], 0.2)
        final = result.states()[-1][0]
        expected = first.states()[-1][0] * math.exp(-1.2 * 0.1)

    print(f"solver={prepared.solver} endpoint={final:.10g} expected={expected:.10g}")
    print(f"status={prepared.status}")
    print(result.telemetry_report())


if __name__ == "__main__":
    main()

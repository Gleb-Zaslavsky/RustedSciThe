"""Independent fixed-initial-condition parameter sweeps for first-tier IVPs.

Run, for example, ``python ivp_parameter_loop.py --solver lsode2``. A malformed
point is included deliberately to show that the batch records its typed error
and still solves the following vector.
"""

import argparse

from rustedscithe import (
    BdfSolver,
    BackwardEulerSolver,
    Expr,
    Lsode2Solver,
    RadauSolver,
    SolverOptions,
)


def make_solver(name: str):
    equation = [Expr.parse("-rate*y")]
    common = dict(
        expressions=equation,
        state_variables=["y"],
        time_variable="t",
        t0=0.0,
        t_end=1.0,
        parameters=["rate"],
        parameter_values=[1.0],
        options=SolverOptions(telemetry="counters"),
    )
    if name == "bdf":
        return BdfSolver(y0=[1.0], max_step=0.02, **common)
    if name == "be":
        return BackwardEulerSolver(y0=[1.0], h=0.005, **common)
    if name == "lsode2":
        return Lsode2Solver(y0=[1.0], max_step=0.02, **common)
    if name == "radau":
        return RadauSolver(max_step=0.02, output="dense", **common)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("bdf", "be", "lsode2", "radau"), default="bdf")
    args = parser.parse_args()

    solver = make_solver(args.solver)
    if hasattr(solver, "prepare"):
        solver.prepare()
    points = solver.solve_parameter_loop(
        [1.0],
        [[0.5], [], [2.0]],
    )
    for point in points:
        if point.succeeded:
            solution = point.solution
            if hasattr(solution, "trajectory_arrays"):
                times, states = solution.trajectory_arrays()
            else:
                times, states = solution.times_array(), solution.states_array()
            sample_count = len(times)
            final_state = states[-1].tolist()
            print(
                f"{args.solver} point {point.index}: parameters={point.parameter_values}, "
                f"samples={sample_count}, final_state={final_state}"
            )
            print(solution.telemetry_report())
        else:
            error = point.error
            print(
                f"{args.solver} point {point.index} failed: "
                f"{type(error).__name__}: {error}; "
                f"native_kind={getattr(error, 'native_kind', None)}"
            )


if __name__ == "__main__":
    main()

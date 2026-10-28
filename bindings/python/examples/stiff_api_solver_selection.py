"""Select a stiff solver from application configuration and export its result.

The decay problem y'=-k*y has exact solution exp(-k*t). Shared route and
telemetry settings are selected once; each factory keeps its own numerical
controls (adaptive BDF/Radau/LSODE2 versus fixed-step Backward Euler).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rustedscithe import Expr, SolverOptions, StiffPostprocessPlan, StiffPreset, StiffSolverRequest


def make_request(solver: str):
    expression = [Expr.parse("-rate*y")]
    shared = dict(
        expressions=expression,
        state_variables=["y"],
        time_variable="t",
        y0=[1.0],
        t0=0.0,
        t_end=1.0,
        parameters=["rate"],
        parameter_values=[0.8],
        options=SolverOptions(telemetry="detailed"),
    )
    if solver == "bdf":
        return StiffSolverRequest.bdf(max_step=0.05, rtol=1e-8, atol=1e-10, **shared)
    if solver == "backward_euler":
        return StiffSolverRequest.backward_euler(h=0.002, tolerance=1e-10, **shared)
    if solver == "radau":
        return StiffSolverRequest.radau(
            max_step=0.05,
            rtol=1e-8,
            atol=1e-10,
            output="sampled",
            sample_times=[i / 100 for i in range(101)],
            **shared,
        )
    return StiffSolverRequest.lsode2(max_step=0.05, rtol=1e-8, atol=1e-10, **shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("bdf", "backward_euler", "radau", "lsode2"))
    parser.add_argument("--output-dir", type=Path, default=Path("stiff-output"))
    args = parser.parse_args()

    # This is application-level dispatch: request factories preserve each
    # solver's native settings; the Rust StiffAPI owns preparation and solve.
    prepared = make_request(args.solver).prepare(StiffPreset.solve_1d())
    solution = prepared.solve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = (
        StiffPostprocessPlan()
        .save_csv(str(args.output_dir / f"{args.solver}.csv"))
        .write_report(str(args.output_dir / f"{args.solver}.md"))
    )

    if args.solver == "radau":
        times = [i / 100 for i in range(101)]
        report = solution.postprocess(plan, times, "time", ["y"])
        final_value = solution.sample(1.0)[0]
        telemetry = solution.telemetry_report()
    else:
        report = prepared.postprocess(plan, "time", ["y"])
        _times, states = prepared.trajectory_arrays()
        final_value = float(states[-1, 0])
        telemetry = solution.telemetry_report()

    expected = math.exp(-0.8)
    print(f"solver={prepared.solver} final={final_value:.10g} exact={expected:.10g}")
    print(f"postprocessing: {report}")
    print(telemetry)


if __name__ == "__main__":
    main()

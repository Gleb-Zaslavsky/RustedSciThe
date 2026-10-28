"""Choose a stiff solver for a small configurable reaction-chain model.

The model is A -> B -> sink with k_fast=50 and k_slow=1. Its exact solution
is A(t)=exp(-k_fast*t) and
B(t)=k_fast/(k_fast-k_slow)*(exp(-k_slow*t)-exp(-k_fast*t)). This illustrates
application-level solver selection; it is not a solver benchmark.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rustedscithe import Expr, SolverOptions, StiffPostprocessPlan, StiffPreset, StiffSolverRequest


def build_request(solver_name: str):
    equations = [
        Expr.parse("-k_fast*A"),
        Expr.parse("k_fast*A-k_slow*B"),
    ]
    shared = dict(
        expressions=equations,
        state_variables=["A", "B"],
        time_variable="t",
        y0=[1.0, 0.0],
        t0=0.0,
        t_end=1.0,
        parameters=["k_fast", "k_slow"],
        parameter_values=[50.0, 1.0],
        options=SolverOptions(telemetry="detailed"),
    )
    if solver_name == "bdf":
        return StiffSolverRequest.bdf(max_step=0.05, rtol=1e-8, atol=1e-10, **shared)
    if solver_name == "backward_euler":
        return StiffSolverRequest.backward_euler(h=0.001, tolerance=1e-10, **shared)
    if solver_name == "radau":
        grid = [index / 100 for index in range(101)]
        return StiffSolverRequest.radau(
            max_step=0.05,
            rtol=1e-8,
            atol=1e-10,
            output="sampled",
            sample_times=grid,
            **shared,
        )
    return StiffSolverRequest.lsode2(max_step=0.05, rtol=1e-8, atol=1e-10, **shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("bdf", "backward_euler", "radau", "lsode2"))
    parser.add_argument("--output-dir", type=Path, default=Path("reaction-chain-output"))
    args = parser.parse_args()

    # The physics/application layer chooses the request factory. Common
    # frontend/layout policy is a native preset; numerical controls stay local
    # to each solver's constructor and retain their own meaning.
    prepared = build_request(args.solver).prepare(StiffPreset.expr_dense())
    solution = prepared.solve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = (
        StiffPostprocessPlan()
        .save_csv(str(args.output_dir / f"{args.solver}-trajectory.csv"))
        .write_report(str(args.output_dir / f"{args.solver}-report.md"))
    )

    if args.solver == "radau":
        times = [index / 100 for index in range(101)]
        report = solution.postprocess(plan, times, "time", ["A", "B"])
        final_state = solution.sample(1.0)
    else:
        report = prepared.postprocess(plan, "time", ["A", "B"])
        _times, states = prepared.trajectory_arrays()
        final_state = states[-1].tolist()

    expected_b = 50.0 / 49.0 * (math.exp(-1.0) - math.exp(-50.0))
    print(f"solver={prepared.solver} final_state={final_state}")
    print(f"exact_state=[{math.exp(-50.0):.8g}, {expected_b:.8g}]")
    print(f"postprocessing={report}")
    print(solution.telemetry_report())


if __name__ == "__main__":
    main()

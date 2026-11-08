"""Measure Python StiffAPI boundary stages on a small stiff diffusion IVP.

Python wall timings include native work where noted. Native telemetry stages
are nested diagnostics and are not additive to the wall-clock columns.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np

from rustedscithe import Expr, SolverOptions, StiffSolverRequest

from _support import runtime_provenance


def median_mad(values: list[float]) -> str:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.4f}/{mad:.4f}"


def timed(action):
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1_000.0


def workload(dimension: int):
    variables = [f"y{i}" for i in range(dimension)]
    equations = []
    for i, variable in enumerate(variables):
        terms = [f"-2*{variable}"]
        if i:
            terms.append(variables[i - 1])
        if i + 1 < dimension:
            terms.append(variables[i + 1])
        equations.append(Expr.parse("+".join(terms)))
    initial = np.linspace(0.0, 1.0, dimension, dtype=np.float64).tolist()
    return equations, variables, initial


def run(repeats: int, dimension: int, end_time: float) -> dict[str, list[float]]:
    equations, variables, initial = workload(dimension)
    options = SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="detailed",
    )
    timings: dict[str, list[float]] = {
        name: []
        for name in (
            "request_build_validate_ms",
            "prepare_dispatch_ms",
            "solve_wall_ms",
            "trajectory_numpy_copy_ms",
            "result_extract_ms",
            "native_residual_ms",
            "native_jacobian_ms",
            "native_linear_ms",
        )
    }
    for _ in range(repeats):
        request, elapsed = timed(
            lambda: StiffSolverRequest.bdf(
                equations,
                variables,
                "t",
                initial,
                0.0,
                end_time,
                end_time / 16.0,
                rtol=1.0e-6,
                atol=1.0e-8,
                options=options,
            )
        )
        timings["request_build_validate_ms"].append(elapsed)

        prepared, elapsed = timed(request.prepare)
        timings["prepare_dispatch_ms"].append(elapsed)

        solution, elapsed = timed(prepared.solve)
        timings["solve_wall_ms"].append(elapsed)

        (times, states), elapsed = timed(prepared.trajectory_arrays)
        timings["trajectory_numpy_copy_ms"].append(elapsed)
        if times.shape[0] == 0 or states.shape != (times.shape[0], dimension):
            raise RuntimeError(f"unexpected StiffAPI trajectory shape {states.shape}")
        if not np.isfinite(states).all() or not math.isclose(
            float(times[-1]), end_time, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError("StiffAPI returned an invalid diffusion trajectory")

        extracted, elapsed = timed(
            lambda: (
                solution.states(),
                solution.telemetry_counters(),
                solution.telemetry_timings_ms(),
                solution.diagnostics(),
            )
        )
        timings["result_extract_ms"].append(elapsed)
        result_states, counters, native_timings, diagnostics = extracted
        final_state = np.asarray(result_states[-1], dtype=np.float64)
        if final_state.shape != (dimension,) or not np.isfinite(final_state).all():
            raise RuntimeError("StiffAPI final-state extraction failed")
        if diagnostics.get("solver") != "bdf":
            raise RuntimeError("StiffAPI returned unexpected solver diagnostics")
        if counters.get("residual_calls", 0) <= 0:
            raise RuntimeError("detailed BDF telemetry did not report residual work")
        for target, source in (
            ("native_residual_ms", "residual_ms"),
            ("native_jacobian_ms", "jacobian_ms"),
            ("native_linear_ms", "linear_solve_ms"),
        ):
            if source in native_timings:
                timings[target].append(float(native_timings[source]))
    return timings


def render(timings: dict[str, list[float]], dimension: int, end_time: float, repeats: int) -> str:
    columns = (
        "request_build_validate_ms",
        "prepare_dispatch_ms",
        "solve_wall_ms",
        "native_residual_ms",
        "native_jacobian_ms",
        "native_linear_ms",
        "trajectory_numpy_copy_ms",
        "result_extract_ms",
    )
    row = "| BDF / ExprLegacy / Dense | " + " | ".join(
        median_mad(timings[name]) if timings[name] else "n/a" for name in columns
    ) + " |"
    header = "| route | " + " | ".join(columns) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in columns) + " |"
    return "\n".join(
        (
            "# Python StiffAPI Boundary Benchmark",
            "",
            f"Workload: {dimension}-state diffusion chain, dense BDF / ExprLegacy, `[0, {end_time:g}]`, {repeats} samples.",
            f"Provenance: {runtime_provenance()}",
            "All values are median/MAD milliseconds. Request construction includes Python/native input validation; prepare includes facade dispatch and symbolic/native preparation. `solve_wall_ms` is end-to-end Python wall time. Native stage telemetry is nested and must not be added to wall time. Trajectory copy measures `trajectory_arrays()` NumPy conversion; result extraction measures detached final-state and telemetry access.",
            "",
            header,
            divider,
            row,
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--end-time", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.dimension < 2 or not math.isfinite(args.end_time) or args.end_time <= 0:
        parser.error("repeats must be >= 3, dimension >= 2, and end-time finite/positive")

    timings = run(args.repeats, args.dimension, args.end_time)
    report = render(timings, args.dimension, args.end_time, args.repeats)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Measure Python-boundary costs across LSODE2 controllers and frontends.

All rows solve the same scalar decay IVP. The continuation series advances in
time while rebinding one parameter, matching LSODE2's native lifecycle. This is
wrapper diagnostics, not a core performance benchmark or universal controller
ranking.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import Expr, SolverOptions, create_ivp


def measure(action: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1_000.0


def median_mad(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.4f}/{mad:.4f}"


def median_mad_int(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return f"{median}/{mad}"


def difference(new: dict, old: dict, key: str) -> float | None:
    """Return a cumulative stage delta, preserving unavailable as unavailable."""
    if key not in new or key not in old:
        return None
    delta = float(new[key]) - float(old[key])
    if delta < 0.0:
        raise RuntimeError(f"LSODE2 cumulative timing {key!r} moved backwards")
    return delta


def run_route(frontend: str, controller: str, repeats: int, continuation: int, end_time: float):
    values: dict[str, list[float]] = {
        key: [] for key in (
            "expr_ms", "input_ms", "construct_ms", "prepare_ms", "solve_wall_ms",
            "numpy_output_ms", "continuation_calls_ms", "continuation_series_ms",
            "native_solve_ms", "native_residual_ms", "native_jacobian_ms",
        "native_factor_ms", "native_linear_ms", "evaluation_residual_calls",
        "evaluation_jacobian_calls", "native_steps", "evaluation_linear_solves",
            "continuation_native_solve_ms", "continuation_native_residual_ms",
        )
    }
    max_error = 0.0
    executed_families: set[str] = set()
    evaluation_scopes: set[str] = set()
    for _ in range(repeats):
        expr_start = time.perf_counter()
        y, rate = Expr.variable("y"), Expr.variable("rate")
        rhs = -rate * y
        values["expr_ms"].append((time.perf_counter() - expr_start) * 1_000.0)

        y0 = np.asarray([1.0], dtype=np.float64)
        initial_parameters = np.asarray([0.8], dtype=np.float64)
        input_start = time.perf_counter()
        y0.tolist()
        initial_parameters.tolist()
        values["input_ms"].append((time.perf_counter() - input_start) * 1_000.0)
        options = SolverOptions(
            frontend=frontend, execution="lambdify", matrix_layout="dense",
            execution_policy="sequential", telemetry="detailed",
        )
        solver, construct_ms = measure(lambda: create_ivp(
            "lsode2", [rhs], ["y"], "t", y0, 0.0, end_time, end_time / 8.0,
            parameters=["rate"], parameter_values=initial_parameters,
            options=options, controller=controller, rtol=1e-8, atol=1e-10,
        ))
        values["construct_ms"].append(construct_ms)
        _, prepare_ms = measure(solver.prepare)
        values["prepare_ms"].append(prepare_ms)
        solution, solve_wall_ms = measure(solver.solve)
        values["solve_wall_ms"].append(solve_wall_ms)
        if not solution.status.startswith("finished"):
            raise RuntimeError(f"LSODE2 {controller} failed: {solution.status}")
        expected = math.exp(-0.8 * end_time)
        max_error = max(max_error, abs(float(solution.final_state[0]) - expected))
        executed_families.add(solution.executed_family or "unknown")
        _, output_ms = measure(lambda: (solution.times_array(), solution.states_array()))
        values["numpy_output_ms"].append(output_ms)

        initial_counters = solution.telemetry_counters()
        evaluation_scopes.add(
            getattr(solution, "evaluation_scope", "not_exposed_by_wheel")
        )
        previous_timings = solution.telemetry_timings_ms()
        initial_snapshot = (initial_counters, previous_timings)
        series_start = time.perf_counter()
        continuation_api_ms = 0.0
        continuation_native_solve_ms = 0.0
        continuation_native_residual_ms = 0.0
        solve_stage_available = True
        residual_stage_available = True
        current_end = end_time
        for index in range(continuation):
            current_rate = 0.55 + 0.1 * (index % 4)
            current_end += end_time
            result, call_ms = measure(lambda current_rate=current_rate, current_end=current_end:
                solver.continue_with_parameters([current_rate], current_end)
            )
            continuation_api_ms += call_ms
            if not result.status.startswith("finished"):
                raise RuntimeError(f"LSODE2 continuation failed: {result.status}")
            expected *= math.exp(-current_rate * end_time)
            max_error = max(max_error, abs(float(result.final_state[0]) - expected))
            executed_families.add(result.executed_family or "unknown")

            timings = result.telemetry_timings_ms()
            for target, key in (
                ("continuation_native_solve_ms", "solve_ms"),
                ("continuation_native_residual_ms", "native_residual_ms"),
            ):
                delta = difference(timings, previous_timings, key)
                if delta is None:
                    if target == "continuation_native_solve_ms":
                        solve_stage_available = False
                    else:
                        residual_stage_available = False
                    continue
                if target == "continuation_native_solve_ms":
                    continuation_native_solve_ms += delta
                else:
                    continuation_native_residual_ms += delta
            previous_timings = timings
        values["continuation_calls_ms"].append(continuation_api_ms)
        values["continuation_series_ms"].append((time.perf_counter() - series_start) * 1_000.0)
        if solve_stage_available:
            values["continuation_native_solve_ms"].append(continuation_native_solve_ms)
        if residual_stage_available:
            values["continuation_native_residual_ms"].append(continuation_native_residual_ms)

        # Discrete counters are reported for the first solve only; continuation
        # time stages above are accumulated from per-result snapshot deltas.
        for target, key in (
            ("evaluation_residual_calls", "evaluation_residual_calls"),
            ("evaluation_jacobian_calls", "evaluation_jacobian_calls"),
            ("native_steps", "native_step_attempts"),
            ("evaluation_linear_solves", "evaluation_linear_solves"),
        ):
            if key in initial_snapshot[0]:
                values[target].append(float(initial_snapshot[0][key]))
        for target, key in (
            ("native_solve_ms", "solve_ms"),
            ("native_residual_ms", "native_residual_ms"),
            ("native_jacobian_ms", "native_jacobian_ms"),
            ("native_factor_ms", "linear_factorization_ms"),
            ("native_linear_ms", "native_linear_solve_ms"),
        ):
            if key in initial_snapshot[1]:
                values[target].append(float(initial_snapshot[1][key]))
    return (
        values,
        max_error,
        ",".join(sorted(executed_families)),
        ",".join(sorted(evaluation_scopes)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--continuation", type=int, default=4)
    parser.add_argument("--segment-end", type=float, default=0.2)
    parser.add_argument("--controllers", default="bdf,adams,auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    controllers = [item.strip().lower() for item in args.controllers.split(",") if item.strip()]
    if args.repeats < 3 or args.continuation < 1 or not controllers:
        parser.error("repeats must be >= 3, continuation >= 1, and controllers nonempty")
    if any(value not in {"bdf", "adams", "auto"} for value in controllers):
        parser.error("controllers must contain only bdf, adams or auto")
    if not math.isfinite(args.segment_end) or args.segment_end <= 0:
        parser.error("segment-end must be finite and positive")

    lines = [
        "# Python LSODE2 boundary benchmark", "",
        f"Matched problem: `y'=-rate*y`, initial `y=1`; segment length={args.segment_end:g}, repeats={args.repeats}, prepared continuation segments={args.continuation}.",
        "Rows vary controller and symbolic frontend on the same equation. `auto` may select a different active family; the report lists executed families. Continuation advances time and rebinds `rate`.",
        "Times are median/MAD ms. Python wall time includes native work; nested Rust stages are not additive. LSODE2 timing snapshots are cumulative, so continuation stage times are differenced between results; discrete counters in the table are from the first solve only. Effective evaluation counters are scope-tagged (native faithful inner loop or BDF bridge); older wheels that do not expose scope-tagged keys show `n/a`, never a fabricated zero. Exact piecewise endpoints are checked.", "",
        "| frontend/controller | executed family | evaluation scope | Expr build | NumPy input | construct | prepare wall | first solve wall | Rust solve | residual | Jacobian | factor | linear | evaluation residual calls | evaluation Jacobian calls | step attempts | evaluation linear solves | NumPy output | continuation API | continuation Rust solve | continuation Rust residual | continuation total | max exact error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for controller in controllers:
        for frontend in ("expr_legacy", "atom_native"):
            values, error, executed, evaluation_scope = run_route(
                frontend, controller, args.repeats, args.continuation, args.segment_end
            )
            lines.append("| " + " | ".join((
                f"{frontend}/{controller}", executed, evaluation_scope,
                median_mad(values["expr_ms"]), median_mad(values["input_ms"]),
                median_mad(values["construct_ms"]), median_mad(values["prepare_ms"]),
                median_mad(values["solve_wall_ms"]), median_mad(values["native_solve_ms"]),
                median_mad(values["native_residual_ms"]), median_mad(values["native_jacobian_ms"]),
                median_mad(values["native_factor_ms"]), median_mad(values["native_linear_ms"]),
                median_mad_int(values["evaluation_residual_calls"]), median_mad_int(values["evaluation_jacobian_calls"]),
                median_mad_int(values["native_steps"]), median_mad_int(values["evaluation_linear_solves"]),
                median_mad(values["numpy_output_ms"]), median_mad(values["continuation_calls_ms"]),
                median_mad(values["continuation_native_solve_ms"]),
                median_mad(values["continuation_native_residual_ms"]),
                median_mad(values["continuation_series_ms"]), f"{error:.3e}",
            )) + " |")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

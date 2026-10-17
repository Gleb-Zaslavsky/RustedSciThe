"""Measure Python-boundary cost for a prepared BDF IVP.

The fixture is ``y' = -rate*y`` with piecewise-constant rates across
continuation segments. Each route solves the same mathematical workload;
Python wall time includes native work, while Rust telemetry is nested and not
additive. This is diagnostic wrapper evidence, not a core benchmark.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import BdfCallbackSolver, Expr, SolverOptions, create_ivp

from _support import required_counter, required_timing


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


def run_frontend(frontend: str, repeats: int, continuation: int, end_time: float):
    fields: dict[str, list[float]] = {
        key: []
        for key in (
            "expr_build_ms", "numpy_input_ms", "constructor_ms", "prepare_ms",
            "solve_wall_ms", "numpy_output_ms", "continuation_call_total_ms",
            "continuation_series_ms", "continuation_native_solve_ms",
            "continuation_native_rhs_ms", "native_solve_ms",
            "native_rhs_ms", "native_jacobian_ms", "native_factor_ms",
            "rhs_calls", "jacobian_calls", "linear_solves", "factorizations",
        )
    }
    max_error = 0.0

    for _ in range(repeats):
        expression_start = time.perf_counter()
        rate = Expr.variable("rate")
        state = Expr.variable("y")
        rhs = -rate * state
        expression_ms = (time.perf_counter() - expression_start) * 1_000.0
        fields["expr_build_ms"].append(expression_ms)

        y0_array = np.asarray([1.0], dtype=np.float64)
        parameter_array = np.asarray([0.8], dtype=np.float64)
        input_start = time.perf_counter()
        y0_array.tolist()
        parameter_array.tolist()
        input_ms = (time.perf_counter() - input_start) * 1_000.0
        fields["numpy_input_ms"].append(input_ms)

        options = SolverOptions(
            frontend=frontend,
            execution="lambdify",
            matrix_layout="dense",
            execution_policy="sequential",
            telemetry="detailed",
        )
        solver, constructor_ms = measure(
            lambda: create_ivp(
                "bdf", [rhs], ["y"], "t", y0_array, 0.0, end_time, end_time / 8.0,
                parameters=["rate"], parameter_values=parameter_array, options=options,
                rtol=1e-8, atol=1e-10,
            )
        )
        fields["constructor_ms"].append(constructor_ms)
        _, prepare_ms = measure(solver.prepare)
        fields["prepare_ms"].append(prepare_ms)

        solution, solve_wall_ms = measure(solver.solve)
        fields["solve_wall_ms"].append(solve_wall_ms)
        if solution.status != "finished":
            raise RuntimeError(f"BDF solve failed: {solution.status}")
        expected = math.exp(-0.8 * end_time)
        max_error = max(max_error, abs(float(solution.states_array()[-1, 0]) - expected))

        _, output_ms = measure(lambda: (solution.times_array(), solution.states_array()))
        fields["numpy_output_ms"].append(output_ms)
        counters = solution.telemetry_counters()
        for key, native_key in (
            ("rhs_calls", "residual_calls"),
            ("jacobian_calls", "jacobian_calls"),
            ("linear_solves", "linear_solve_attempts"),
            ("factorizations", "bdf_nlu"),
        ):
            fields[key].append(required_counter(counters, native_key, f"BDF/{frontend}"))
        timings = solution.telemetry_timings_ms()
        for key, candidates in (
            ("native_solve_ms", ("solve_ms",)),
            ("native_rhs_ms", ("residual_ms",)),
            ("native_jacobian_ms", ("jacobian_ms",)),
            ("native_factor_ms", ("linear_factorization_ms",)),
        ):
            value = next((timings[name] for name in candidates if name in timings), None)
            if value is not None:
                fields[key].append(float(value))

        continuation_start = time.perf_counter()
        continuation_call_total = 0.0
        current_rate = 0.8
        current_end = end_time
        continuation_native_solve = 0.0
        continuation_native_rhs = 0.0
        previous_native_solve = required_timing(timings, "solve_ms", f"BDF/{frontend}")
        previous_native_rhs = required_timing(timings, "residual_ms", f"BDF/{frontend}")
        for index in range(continuation):
            current_rate = 0.55 + 0.1 * (index % 4)
            current_end += end_time
            continued_solution, call_ms = measure(
                lambda current_rate=current_rate, current_end=current_end:
                    solver.continue_with_parameters([current_rate], current_end)
            )
            continuation_call_total += call_ms
            if continued_solution.status != "finished":
                raise RuntimeError(
                    f"BDF continuation failed: {continued_solution.status}"
                )
            # BDF continuation integrates the new interval from the previous
            # endpoint, so the exact solution is the product of segment factors.
            expected *= math.exp(-current_rate * end_time)
            continuation_timings = continued_solution.telemetry_timings_ms()
            current_native_solve = required_timing(
                continuation_timings, "solve_ms", f"BDF continuation/{frontend}"
            )
            current_native_rhs = required_timing(
                continuation_timings, "residual_ms", f"BDF continuation/{frontend}"
            )
            solve_delta = current_native_solve - previous_native_solve
            rhs_delta = current_native_rhs - previous_native_rhs
            if solve_delta < 0.0 or rhs_delta < 0.0:
                raise RuntimeError("BDF cumulative continuation telemetry moved backwards")
            continuation_native_solve += solve_delta
            continuation_native_rhs += rhs_delta
            previous_native_solve = current_native_solve
            previous_native_rhs = current_native_rhs
            continued_error = abs(
                float(continued_solution.states_array()[-1, 0]) - expected
            )
            max_error = max(max_error, continued_error)
        fields["continuation_call_total_ms"].append(continuation_call_total)
        fields["continuation_native_solve_ms"].append(continuation_native_solve)
        fields["continuation_native_rhs_ms"].append(continuation_native_rhs)
        fields["continuation_series_ms"].append((time.perf_counter() - continuation_start) * 1_000.0)

    return fields, max_error


def run_python_callbacks(repeats: int, continuation: int, end_time: float):
    """Measure BDF callbacks crossing into Python; not pure dispatch overhead."""

    fields: dict[str, list[float]] = {
        key: []
        for key in (
            "constructor_ms", "prepare_ms", "solve_wall_ms", "numpy_output_ms",
            "continuation_call_total_ms", "continuation_series_ms", "continuation_native_solve_ms",
            "continuation_native_rhs_ms", "native_solve_ms",
            "native_rhs_ms", "native_jacobian_ms", "native_factor_ms", "rhs_calls",
            "jacobian_calls", "linear_solves", "factorizations",
        )
    }
    max_error = 0.0
    for _ in range(repeats):
        rate = [0.8]
        options = SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            execution_policy="sequential", telemetry="detailed",
        )
        y0 = np.asarray([1.0], dtype=np.float64).tolist()
        solver, constructor_ms = measure(lambda: BdfCallbackSolver(
            state_dimension=1,
            t0=0.0,
            t_end=end_time,
            max_step=end_time / 8.0,
            y0=y0,
            residual_callback=lambda _t, y: [-rate[0] * y[0]],
            jacobian_callback=lambda _t, _y: [[-rate[0]]],
            rtol=1e-8,
            atol=1e-10,
            options=options,
        ))
        fields["constructor_ms"].append(constructor_ms)
        _, prepare_ms = measure(solver.prepare)
        fields["prepare_ms"].append(prepare_ms)
        result, solve_ms = measure(solver.solve)
        fields["solve_wall_ms"].append(solve_ms)
        if result.status != "finished":
            raise RuntimeError(f"BDF callback solve failed: {result.status}")
        expected = math.exp(-rate[0] * end_time)
        max_error = max(max_error, abs(float(result.states_array()[-1, 0]) - expected))
        _, output_ms = measure(lambda: (result.times_array(), result.states_array()))
        fields["numpy_output_ms"].append(output_ms)
        counters = result.telemetry_counters()
        timings = result.telemetry_timings_ms()
        for key, native_key in (
            ("rhs_calls", "residual_calls"),
            ("jacobian_calls", "jacobian_calls"),
            ("linear_solves", "linear_solve_attempts"),
            ("factorizations", "bdf_nlu"),
        ):
            fields[key].append(required_counter(counters, native_key, "BDF/python_callbacks"))
        for key, native_key in (
            ("native_solve_ms", "solve_ms"),
            ("native_rhs_ms", "residual_ms"),
            ("native_jacobian_ms", "jacobian_ms"),
            ("native_factor_ms", "linear_factorization_ms"),
        ):
            if native_key in timings:
                fields[key].append(float(timings[native_key]))

        series_start = time.perf_counter()
        continuation_total = 0.0
        continuation_native_solve = 0.0
        continuation_native_rhs = 0.0
        previous_native_solve = required_timing(timings, "solve_ms", "BDF/python_callbacks")
        previous_native_rhs = required_timing(timings, "residual_ms", "BDF/python_callbacks")
        current_end = end_time
        current_value = float(result.states_array()[-1, 0])
        for index in range(continuation):
            rate[0] = 0.55 + 0.1 * (index % 4)
            next_end = current_end + end_time
            next_result, call_ms = measure(
                lambda current_end=current_end, current_value=current_value, next_end=next_end:
                    solver.restart(current_end, [current_value], next_end)
            )
            continuation_total += call_ms
            expected *= math.exp(-rate[0] * end_time)
            current_value = float(next_result.states_array()[-1, 0])
            continuation_timings = next_result.telemetry_timings_ms()
            current_native_solve = required_timing(
                continuation_timings, "solve_ms", "BDF callback restart"
            )
            current_native_rhs = required_timing(
                continuation_timings, "residual_ms", "BDF callback restart"
            )
            solve_delta = current_native_solve - previous_native_solve
            rhs_delta = current_native_rhs - previous_native_rhs
            if solve_delta < 0.0 or rhs_delta < 0.0:
                raise RuntimeError("BDF callback cumulative telemetry moved backwards")
            continuation_native_solve += solve_delta
            continuation_native_rhs += rhs_delta
            previous_native_solve = current_native_solve
            previous_native_rhs = current_native_rhs
            current_end = next_end
            max_error = max(max_error, abs(current_value - expected))
        fields["continuation_call_total_ms"].append(continuation_total)
        fields["continuation_native_solve_ms"].append(continuation_native_solve)
        fields["continuation_native_rhs_ms"].append(continuation_native_rhs)
        fields["continuation_series_ms"].append((time.perf_counter() - series_start) * 1_000.0)
    return fields, max_error


def render(rows: list[dict[str, str]], repeats: int, continuation: int, end_time: float) -> str:
    lines = [
        "# Python IVP boundary benchmark", "",
        f"Matched BDF problem: `y'=-rate*y`, initial `y=1`; segment length={end_time:g}, repeats={repeats}, warm continuation segments={continuation}.",
        "Each symbolic continuation segment rebinds `rate`; callback continuation uses `restart()` with a changed Python closure. Exact endpoints are checked.",
        "Times are median/MAD ms. Python wall times include native work; Rust stage timings are nested and not additive.",
        "Preparation is measured by explicit `prepare()`. The callback row includes Python/GIL crossings inside native residual/Jacobian calls; it does not isolate pure dispatch overhead. Symbolic frontend rows are the matched frontend comparison.",
        "",
        "| frontend | Expr build | NumPy input | construct | prepare wall | first solve wall | Rust solve | RHS stage | Jacobian stage | factor stage | RHS calls | Jacobian calls | linear solves | LU/factor count | NumPy output | continuation calls wall | continuation Rust solve | continuation Rust residual | continuation series | max exact error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {row['frontend']} | {row['expr']} | {row['input']} | {row['construct']} | {row['prepare']} | {row['solve']} | {row['native_solve']} | {row['rhs']} | {row['jac']} | {row['factor']} | {row['rhs_calls']} | {row['jac_calls']} | {row['linear_solves']} | {row['factors']} | {row['output']} | {row['continue_api']} | {row['continue_native_solve']} | {row['continue_native_rhs']} | {row['continue_series']} | {row['error']} |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--continuation", type=int, default=4)
    parser.add_argument("--segment-end", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.continuation < 1 or not math.isfinite(args.segment_end) or args.segment_end <= 0:
        parser.error("repeats must be >= 3, continuation >= 1, and segment-end finite/positive")

    rows = []
    for frontend in ("expr_legacy", "atom_native"):
        fields, error = run_frontend(frontend, args.repeats, args.continuation, args.segment_end)
        rows.append({
            "frontend": frontend,
            "expr": median_mad(fields["expr_build_ms"]),
            "input": median_mad(fields["numpy_input_ms"]),
            "construct": median_mad(fields["constructor_ms"]),
            "prepare": median_mad(fields["prepare_ms"]),
            "solve": median_mad(fields["solve_wall_ms"]),
            "native_solve": median_mad(fields["native_solve_ms"]),
            "rhs": median_mad(fields["native_rhs_ms"]),
            "jac": median_mad(fields["native_jacobian_ms"]),
            "factor": median_mad(fields["native_factor_ms"]),
            "rhs_calls": median_mad_int(fields["rhs_calls"]),
            "jac_calls": median_mad_int(fields["jacobian_calls"]),
            "linear_solves": median_mad_int(fields["linear_solves"]),
            "factors": median_mad_int(fields["factorizations"]),
            "output": median_mad(fields["numpy_output_ms"]),
            "continue_api": median_mad(fields["continuation_call_total_ms"]),
            "continue_native_solve": median_mad(fields["continuation_native_solve_ms"]),
            "continue_native_rhs": median_mad(fields["continuation_native_rhs_ms"]),
            "continue_series": median_mad(fields["continuation_series_ms"]),
            "error": f"{error:.3e}",
        })
    fields, error = run_python_callbacks(args.repeats, args.continuation, args.segment_end)
    rows.append({
        "frontend": "python_callbacks",
        "expr": "n/a",
        "input": "n/a",
        "construct": median_mad(fields["constructor_ms"]),
        "prepare": median_mad(fields["prepare_ms"]),
        "solve": median_mad(fields["solve_wall_ms"]),
        "native_solve": median_mad(fields["native_solve_ms"]),
        "rhs": median_mad(fields["native_rhs_ms"]),
        "jac": median_mad(fields["native_jacobian_ms"]),
        "factor": median_mad(fields["native_factor_ms"]),
        "rhs_calls": median_mad_int(fields["rhs_calls"]),
        "jac_calls": median_mad_int(fields["jacobian_calls"]),
        "linear_solves": median_mad_int(fields["linear_solves"]),
        "factors": median_mad_int(fields["factorizations"]),
        "output": median_mad(fields["numpy_output_ms"]),
        "continue_api": median_mad(fields["continuation_call_total_ms"]),
        "continue_native_solve": median_mad(fields["continuation_native_solve_ms"]),
        "continue_native_rhs": median_mad(fields["continuation_native_rhs_ms"]),
        "continue_series": median_mad(fields["continuation_series_ms"]),
        "error": f"{error:.3e}",
    })
    report = render(rows, args.repeats, args.continuation, args.segment_end)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

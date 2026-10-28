"""Measure Python-boundary cost for symbolic and callback Radau routes.

The matched fixture is ``y' = -rate*y`` with ``y(0)=1``. Radau's prepared
parameter API re-solves the configured interval for each new ``(y0, rate)``;
the repeated rows below are therefore a parameter sweep, not sequential-time
continuation. Python wall time includes native work; native stage times are
nested diagnostic measurements and must not be added to it.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import Expr, RadauCallbackSolver, SolverOptions, create_ivp

from _support import required_counter


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


def _new_fields() -> dict[str, list[float]]:
    names = (
        "expression_ms", "input_ms", "construct_ms", "prepare_ms", "solve_ms",
        "numpy_output_ms", "sweep_api_ms", "sweep_series_ms", "native_residual_ms",
        "native_jacobian_ms", "native_factor_ms", "native_linear_ms",
        "residual_calls", "jacobian_calls", "factorizations", "linear_solves",
    )
    return {name: [] for name in names}


def _record_result(fields: dict[str, list[float]], solution: object) -> None:
    counters = solution.telemetry_counters()
    timings = solution.telemetry_timings_ms()
    for target, key in (
        ("residual_calls", "residual_calls"),
        ("jacobian_calls", "jacobian_calls"),
        ("factorizations", "factorizations"),
        ("linear_solves", "linear_solves"),
    ):
        fields[target].append(required_counter(counters, key, "Radau"))
    for target, key in (
        ("native_residual_ms", "residual_evaluation_ms"),
        ("native_jacobian_ms", "jacobian_assembly_ms"),
        ("native_factor_ms", "factorization_ms"),
        ("native_linear_ms", "linear_ms"),
    ):
        if key in timings:
            fields[target].append(float(timings[key]))


def run_symbolic(frontend: str, repeats: int, sweep: int, end_time: float):
    fields = _new_fields()
    max_error = 0.0
    for _ in range(repeats):
        expression_start = time.perf_counter()
        y, rate = Expr.variable("y"), Expr.variable("rate")
        rhs = -rate * y
        expression_ms = (time.perf_counter() - expression_start) * 1_000.0
        fields["expression_ms"].append(expression_ms)

        y0_array = np.asarray([1.0], dtype=np.float64)
        parameter_array = np.asarray([0.8], dtype=np.float64)
        input_start = time.perf_counter()
        y0 = y0_array.tolist()
        parameters = parameter_array.tolist()
        fields["input_ms"].append((time.perf_counter() - input_start) * 1_000.0)

        options = SolverOptions(
            frontend=frontend, execution="lambdify", matrix_layout="dense",
            execution_policy="sequential", telemetry="detailed",
        )
        solver, constructor_ms = measure(lambda: create_ivp(
            "radau5", [rhs], ["y"], "t", y0, 0.0, end_time, end_time / 8.0,
            parameters=["rate"], parameter_values=parameters, options=options,
            rtol=1e-8, atol=1e-10,
        ))
        fields["construct_ms"].append(constructor_ms)
        _, prepare_ms = measure(solver.prepare)
        fields["prepare_ms"].append(prepare_ms)
        solution, solve_ms = measure(lambda: solver.solve(y0, parameters))
        fields["solve_ms"].append(solve_ms)
        _check_solution(solution, 0.8, end_time)
        expected = math.exp(-0.8 * end_time)
        max_error = max(max_error, abs(float(solution.final_state[0]) - expected))
        _record_result(fields, solution)
        _, output_ms = measure(solution.final_state_array)
        fields["numpy_output_ms"].append(output_ms)

        sweep_start = time.perf_counter()
        sweep_api_ms = 0.0
        for index in range(sweep):
            current_rate = 0.55 + 0.1 * (index % 4)
            result, call_ms = measure(
                lambda current_rate=current_rate: solver.solve_with_parameter_values(
                    y0, [current_rate]
                )
            )
            sweep_api_ms += call_ms
            _check_solution(result, current_rate, end_time)
            max_error = max(
                max_error,
                abs(float(result.final_state[0]) - math.exp(-current_rate * end_time)),
            )
        fields["sweep_api_ms"].append(sweep_api_ms)
        fields["sweep_series_ms"].append((time.perf_counter() - sweep_start) * 1_000.0)
    return fields, max_error


def run_python_callbacks(repeats: int, sweep: int, end_time: float):
    fields = _new_fields()
    max_error = 0.0
    for _ in range(repeats):
        rate = [0.8]
        y0 = np.asarray([1.0], dtype=np.float64).tolist()
        options = SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            execution_policy="sequential", telemetry="detailed",
        )
        solver, constructor_ms = measure(lambda: RadauCallbackSolver(
            state_dimension=1, t0=0.0, t_end=end_time, max_step=end_time / 8.0,
            residual_callback=lambda _t, state: [-rate[0] * state[0]],
            jacobian_callback=lambda _t, _state: [[-rate[0]]],
            rtol=1e-8, atol=1e-10, options=options,
        ))
        fields["construct_ms"].append(constructor_ms)
        # The callback adapter has no symbolic preparation stage; this explicit
        # zero is not reported as a measured preparation duration.
        solution, solve_ms = measure(lambda: solver.solve(y0))
        fields["solve_ms"].append(solve_ms)
        _check_solution(solution, rate[0], end_time)
        max_error = max(
            max_error,
            abs(float(solution.final_state[0]) - math.exp(-rate[0] * end_time)),
        )
        _record_result(fields, solution)
        _, output_ms = measure(solution.final_state_array)
        fields["numpy_output_ms"].append(output_ms)

        sweep_start = time.perf_counter()
        sweep_api_ms = 0.0
        for index in range(sweep):
            rate[0] = 0.55 + 0.1 * (index % 4)
            result, call_ms = measure(lambda: solver.solve(y0))
            sweep_api_ms += call_ms
            _check_solution(result, rate[0], end_time)
            max_error = max(
                max_error,
                abs(float(result.final_state[0]) - math.exp(-rate[0] * end_time)),
            )
        fields["sweep_api_ms"].append(sweep_api_ms)
        fields["sweep_series_ms"].append((time.perf_counter() - sweep_start) * 1_000.0)
    return fields, max_error


def _check_solution(solution: object, rate: float, end_time: float) -> None:
    if not math.isclose(solution.final_time, end_time, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Radau stopped at unexpected time {solution.final_time}")
    error = abs(float(solution.final_state[0]) - math.exp(-rate * end_time))
    if error > 2e-7:
        raise RuntimeError(f"Radau exact-solution gate failed: {error:.3e}")


def _row(label: str, fields: dict[str, list[float]], error: float) -> str:
    value = lambda key: median_mad(fields[key])
    count = lambda key: median_mad_int(fields[key])
    return "| " + " | ".join((
        label, value("expression_ms"), value("input_ms"), value("construct_ms"),
        value("prepare_ms"), value("solve_ms"), value("native_residual_ms"),
        value("native_jacobian_ms"), value("native_factor_ms"), value("native_linear_ms"),
        count("residual_calls"), count("jacobian_calls"), count("factorizations"),
        count("linear_solves"), value("numpy_output_ms"), value("sweep_api_ms"),
        value("sweep_series_ms"), f"{error:.3e}",
    )) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sweep", type=int, default=8)
    parser.add_argument("--segment-end", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.sweep < 1 or not math.isfinite(args.segment_end) or args.segment_end <= 0:
        parser.error("repeats must be >= 3, sweep >= 1, and segment-end finite/positive")

    rows = []
    for frontend in ("expr_legacy", "atom_native"):
        fields, error = run_symbolic(frontend, args.repeats, args.sweep, args.segment_end)
        rows.append(_row(frontend, fields, error))
    fields, error = run_python_callbacks(args.repeats, args.sweep, args.segment_end)
    rows.append(_row("python_callbacks", fields, error))
    report = "\n".join((
        "# Python Radau boundary benchmark", "",
        f"Matched problem: `y'=-rate*y`, `y(0)=1`; interval=[0,{args.segment_end:g}], repeats={args.repeats}, prepared parameter-sweep solves={args.sweep}.",
        "The symbolic Radau route rebinds parameters and supplies a new `y0` for each configured-interval solve. The callback route changes a Python closure and solves from `y0`; these share the mathematical workload but are different lifecycle contracts.",
        "Times are median/MAD ms. Python wall time includes native execution; Rust stages are nested and non-additive. Python callback timings include GIL crossings and callback work, not pure dispatch-only overhead. `n/a` means that stage is not part of that route, not a zero measurement.", "",
        "| route | Expr build | NumPy input | construct | prepare wall | first solve wall | Rust residual | Rust Jacobian | Rust factor | Rust linear | residual calls | Jacobian calls | factors | linear solves | NumPy output | sweep API wall | sweep total wall | max exact error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows, "",
    ))
    print(report, end="\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Measure Python-boundary costs for nonlinear and symbolic fitting APIs.

This diagnostic covers a reusable Newton root solve, reusable symbolic
least-squares fitting with parameter rebinding, and Gavin's one-shot sampled
fit. The latter routes solve different contracts and must not be read as an
apple-to-apple algorithm speed ranking.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import (
    Expr,
    GavinFittingSolver,
    SolverOptions,
    SymbolicSystem,
    create_fit,
    create_nonlinear,
)

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


def median_mad_count(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return f"{median}/{mad}"


def run_newton(frontend: str, repeats: int, warm_solves: int):
    samples = {key: [] for key in (
        "expression_ms", "input_ms", "constructor_ms", "prepare_ms", "solve_wall_ms",
        "warm_series_ms", "numpy_output_ms", "native_total_ms", "residual_ms",
        "jacobian_ms", "linear_solve_ms", "iterations", "residual_calls",
        "jacobian_calls", "linear_solves", "factorizations", "continuation_native_ms",
    )}
    max_error = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        x = Expr.variable("x")
        y = Expr.variable("y")
        system = SymbolicSystem([x * x - 2.0, y - 3.0], ["x", "y"])
        samples["expression_ms"].append((time.perf_counter() - start) * 1_000.0)
        input_start = time.perf_counter()
        initial_guess = np.asarray([1.4, 2.8], dtype=np.float64).tolist()
        samples["input_ms"].append((time.perf_counter() - input_start) * 1_000.0)

        solver, constructor_ms = measure(lambda: create_nonlinear(
            system,
            method="newton",
            frontend=frontend,
            execution="lambdify",
            execution_policy="sequential",
            telemetry="detailed",
            tolerance=1e-11,
            max_iterations=40,
        ))
        samples["constructor_ms"].append(constructor_ms)
        _, prepare_ms = measure(solver.prepare)
        samples["prepare_ms"].append(prepare_ms)

        solution, solve_ms = measure(lambda: solver.solve(initial_guess))
        samples["solve_wall_ms"].append(solve_ms)
        values = solution.solution()
        error = max(abs(values[0] - math.sqrt(2.0)), abs(values[1] - 3.0))
        max_error = max(max_error, error)
        if error > 1e-8:
            raise RuntimeError(f"Newton route {frontend} failed exact root gate: {values}")

        _, output_ms = measure(solution.solution_array)
        samples["numpy_output_ms"].append(output_ms)
        warm_start = time.perf_counter()
        warm_native_total = 0.0
        for index in range(warm_solves):
            candidate_guess = np.asarray(
                [1.25 + 0.01 * (index % 10), 2.8], dtype=np.float64
            ).tolist()
            candidate = solver.solve(candidate_guess)
            candidate_values = candidate.solution()
            warm_native_total += required_timing(
                candidate.telemetry_timings_ms(), "total_ms", f"Newton/{frontend} warm solve"
            )
            max_error = max(
                max_error,
                abs(candidate_values[0] - math.sqrt(2.0)),
                abs(candidate_values[1] - 3.0),
            )
        samples["warm_series_ms"].append((time.perf_counter() - warm_start) * 1_000.0)
        samples["continuation_native_ms"].append(warm_native_total)
        counters = solution.telemetry_counters()
        timings = solution.telemetry_timings_ms()
        for key, native_key in (
            ("iterations", "iterations"),
            ("residual_calls", "residual_evaluations"),
            ("jacobian_calls", "jacobian_evaluations"),
            ("linear_solves", "linear_solves"),
            ("factorizations", "linear_factorizations"),
        ):
            samples[key].append(required_counter(counters, native_key, f"Newton/{frontend}"))
        for key, native_key in (
            ("native_total_ms", "total_ms"),
            ("residual_ms", "residual_ms"),
            ("jacobian_ms", "jacobian_ms"),
            ("linear_solve_ms", "linear_solve_ms"),
        ):
            if native_key in timings:
                samples[key].append(float(timings[native_key]))
    return samples, max_error


def fitting_fixture(sample_count: int):
    names: list[str] = []
    values: list[float] = []
    equations = []
    for index in range(sample_count):
        x_name = f"sample_x_{index}"
        y_name = f"sample_y_{index}"
        names.extend((x_name, y_name))
        x_value = index / max(1, sample_count - 1)
        y_value = 1.5 * x_value + 0.75
        values.extend((x_value, y_value))
        equations.append(Expr.parse(f"slope*{x_name} + intercept - {y_name}"))
    return equations, names, values


def run_fitting(frontend: str, repeats: int, sample_count: int, continuation: int):
    samples = {key: [] for key in (
        "expression_ms", "input_ms", "constructor_ms", "solve_wall_ms",
        "continuation_wall_ms", "numpy_output_ms", "native_total_ms",
        "residual_ms", "jacobian_ms", "linear_solve_ms", "iterations",
        "residual_calls", "jacobian_calls", "linear_solves", "factorizations",
        "continuation_native_ms",
    )}
    max_error = 0.0
    for _ in range(repeats):
        expression_start = time.perf_counter()
        equations, parameter_names, parameter_values = fitting_fixture(sample_count)
        samples["expression_ms"].append((time.perf_counter() - expression_start) * 1_000.0)
        input_start = time.perf_counter()
        parameter_values = np.asarray(parameter_values, dtype=np.float64).tolist()
        samples["input_ms"].append((time.perf_counter() - input_start) * 1_000.0)
        solver, constructor_ms = measure(lambda: create_fit(
            equations,
            np.asarray([0.0, 0.0], dtype=np.float64),
            variables=["slope", "intercept"],
            parameters=parameter_names,
            frontend=frontend,
            execution_policy="sequential",
            telemetry="detailed",
            tolerance=1e-10,
            max_iterations=100,
        ))
        samples["constructor_ms"].append(constructor_ms)
        result, solve_ms = measure(lambda: solver.solve_with_parameters(parameter_values))
        samples["solve_wall_ms"].append(solve_ms)
        fitted = result.solution()
        max_error = max(max_error, abs(fitted[0] - 1.5), abs(fitted[1] - 0.75))
        if max_error > 1e-7:
            raise RuntimeError(f"Fitting route {frontend} failed exact linear-fit gate: {fitted}")
        _, output_ms = measure(result.solution_array)
        samples["numpy_output_ms"].append(output_ms)

        continuation_start = time.perf_counter()
        continuation_native_total = 0.0
        for index in range(continuation):
            shifted = parameter_values.copy()
            for position in range(1, len(shifted), 2):
                shifted[position] += 0.05 * (index + 1)
            continued = solver.solve_with_parameters(shifted)
            continuation_native_total += required_timing(
                continued.telemetry_timings_ms(), "total_ms", f"Fitting/{frontend} continuation"
            )
            expected_intercept = 0.75 + 0.05 * (index + 1)
            if abs(continued.solution()[1] - expected_intercept) > 1e-7:
                raise RuntimeError("prepared fitting continuation returned the wrong intercept")
        samples["continuation_wall_ms"].append((time.perf_counter() - continuation_start) * 1_000.0)
        samples["continuation_native_ms"].append(continuation_native_total)

        counters = result.telemetry_counters()
        timings = result.telemetry_timings_ms()
        for key, native_key in (
            ("iterations", "iterations"),
            ("residual_calls", "residual_evaluations"),
            ("jacobian_calls", "jacobian_evaluations"),
            ("linear_solves", "linear_solves"),
            ("factorizations", "linear_factorizations"),
        ):
            samples[key].append(required_counter(counters, native_key, f"Fitting/{frontend}"))
        for key, native_key in (
            ("native_total_ms", "total_ms"),
            ("residual_ms", "residual_ms"),
            ("jacobian_ms", "jacobian_ms"),
            ("linear_solve_ms", "linear_solve_ms"),
        ):
            if native_key in timings:
                samples[key].append(float(timings[native_key]))
    return samples, max_error


def run_gavin(repeats: int, sample_count: int):
    samples = {key: [] for key in (
        "expression_ms", "input_ms", "constructor_ms", "solve_wall_ms",
        "numpy_output_ms", "native_total_ms", "objective_ms", "linear_solve_ms",
        "iterations", "objective_calls", "jacobian_calls", "linear_solves",
        "continuation_native_ms",
    )}
    max_error = 0.0
    for _ in range(repeats):
        expression_start = time.perf_counter()
        model = Expr.parse("amplitude*exp(-x/tau) + offset")
        samples["expression_ms"].append((time.perf_counter() - expression_start) * 1_000.0)
        input_start = time.perf_counter()
        x_data = np.linspace(0.0, 4.0, sample_count, dtype=np.float64).tolist()
        observations = [2.0 * math.exp(-x / 1.25) + 0.4 for x in x_data]
        samples["input_ms"].append((time.perf_counter() - input_start) * 1_000.0)
        solver, constructor_ms = measure(lambda: GavinFittingSolver(
            equations=[model], x_data=x_data, observations=observations,
            sample_parameter="x", unknowns=["amplitude", "tau", "offset"],
            initial_guess=[1.7, 1.0, 0.2], positive_parameters=["tau"],
            frontend="expr_legacy", execution_policy="sequential",
            telemetry_mode="detailed", max_iterations=300,
            gradient_tolerance=1e-8, parameter_step_tolerance=1e-9,
        ))
        samples["constructor_ms"].append(constructor_ms)
        result, solve_ms = measure(solver.solve)
        samples["solve_wall_ms"].append(solve_ms)
        parameters = result.parameters()
        error = max(abs(parameters[0] - 2.0), abs(parameters[1] - 1.25), abs(parameters[2] - 0.4))
        max_error = max(max_error, error)
        if error > 1e-4:
            raise RuntimeError(f"Gavin failed exact synthetic fit gate: {parameters}")
        # Gavin currently returns fitted parameters as an owned Python list;
        # this measures the caller's explicit NumPy materialization cost.
        _, output_ms = measure(lambda: np.asarray(result.parameters(), dtype=np.float64))
        samples["numpy_output_ms"].append(output_ms)
        counters = result.telemetry_counters()
        timings = result.telemetry_timings_ms()
        for key, native_key in (
            ("iterations", "iterations"),
            ("objective_calls", "objective_evaluations"),
            ("jacobian_calls", "analytic_jacobian_evaluations"),
            ("linear_solves", "linear_solves"),
        ):
            samples[key].append(required_counter(counters, native_key, "Gavin"))
        for key, native_key in (
            ("native_total_ms", "total_ms"),
            ("objective_ms", "objective_ms"),
            ("linear_solve_ms", "linear_solve_ms"),
        ):
            if native_key in timings:
                samples[key].append(float(timings[native_key]))
    return samples, max_error


def render(rows: list[dict[str, str]], repeats: int, sample_count: int, continuation: int) -> str:
    lines = [
        "# Python nonlinear and fitting boundary benchmark", "",
        f"Repeats={repeats}; fitting samples={sample_count}; prepared-fit rebinds={continuation}.",
        "Times are median/MAD ms. Wall time includes Rust work; native telemetry is nested and not additive.",
        "Newton and FittingSolver use separate exact fixtures. Gavin is a sampled-observation algorithm, so its row is descriptive and not an apple-to-apple speed comparison with equation least squares.",
        "FittingSolver constructor includes native preparation; NonlinearSolver has explicit prepare(). Gavin's builder also combines model setup/preparation.",
        "",
        "| route | expression build | input conversion | constructor | explicit prepare | solve wall | native total | residual/objective | Jacobian | linear solve | iterations | residual/objective calls | Jacobian calls | linear solves | factorizations | NumPy output | warm/rebind wall | warm/rebind native | max exact error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['route']} | {row['expression']} | {row['input']} | {row['constructor']} | {row['prepare']} | {row['solve']} | {row['native']} | {row['residual']} | {row['jacobian']} | {row['linear']} | {row['iterations']} | {row['residual_calls']} | {row['jac_calls']} | {row['linear_solves']} | {row['factorizations']} | {row['output']} | {row['series']} | {row['series_native']} | {row['error']} |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def row(route: str, samples: dict[str, list[float]], error: float, explicit_prepare: bool, series_key: str):
    return {
        "route": route,
        "expression": median_mad(samples["expression_ms"]),
        "input": median_mad(samples["input_ms"]),
        "constructor": median_mad(samples["constructor_ms"]),
        "prepare": median_mad(samples.get("prepare_ms", [])) if explicit_prepare else "included in constructor",
        "solve": median_mad(samples["solve_wall_ms"]),
        "native": median_mad(samples["native_total_ms"]),
        "residual": median_mad(samples.get("residual_ms", samples.get("objective_ms", []))),
        "jacobian": median_mad(samples.get("jacobian_ms", samples.get("analytic_jacobian_ms", []))),
        "linear": median_mad(samples.get("linear_solve_ms", [])),
        "iterations": median_mad_count(samples["iterations"]),
        "residual_calls": median_mad_count(samples.get("residual_calls", samples.get("objective_calls", []))),
        "jac_calls": median_mad_count(samples.get("jacobian_calls", [])),
        "linear_solves": median_mad_count(samples["linear_solves"]),
        "factorizations": median_mad_count(samples.get("factorizations", [])),
        "output": median_mad(samples["numpy_output_ms"]),
        "series": median_mad(samples.get(series_key, [])),
        "series_native": median_mad(samples.get("continuation_native_ms", [])),
        "error": f"{error:.3e}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--warm-solves", type=int, default=4)
    parser.add_argument("--continuation", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.samples < 4 or args.warm_solves < 1 or args.continuation < 1:
        parser.error("repeats >= 3, samples >= 4, and both series lengths >= 1 are required")

    rows = []
    for frontend in ("expr_legacy", "atom_native"):
        samples, error = run_newton(frontend, args.repeats, args.warm_solves)
        rows.append(row(f"NonlinearSolver/Newton/{frontend}", samples, error, True, "warm_series_ms"))
        samples, error = run_fitting(frontend, args.repeats, args.samples, args.continuation)
        rows.append(row(f"FittingSolver/{frontend}", samples, error, False, "continuation_wall_ms"))
    samples, error = run_gavin(args.repeats, args.samples)
    rows.append(row("GavinFittingSolver/ExprLegacy", samples, error, False, "continuation_wall_ms"))

    report = render(rows, args.repeats, args.samples, args.continuation)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

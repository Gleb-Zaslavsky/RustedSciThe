"""Compare nonlinear Python-boundary preparation with cold AOT builds.

The matched square systems are ``x_i**2 - p_i = 0`` with positive ``p_i``.
Every route uses the same Dense Newton problem and exact positive-root gate.
Cold AOT uses a fresh artifact directory and ``rebuild_always`` for every
sample; its preparation wall time includes symbolic lowering, native compile,
and link. This is wrapper-level diagnostic evidence, not a core benchmark.
"""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from rustedscithe import Expr, NonlinearSolver, SymbolicSystem

from _support import (
    required_counter,
    required_timing,
    runtime_provenance,
    wall_minus_native_ms,
)


def median_mad(values: list[float]) -> str:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.3f}/{mad:.3f}"


def fixture(dimension: int) -> tuple[SymbolicSystem, list[str], list[float]]:
    unknown_names = [f"x{index}" for index in range(dimension)]
    parameter_names = [f"p{index}" for index in range(dimension)]
    unknowns = [Expr.variable(name) for name in unknown_names]
    parameters = [Expr.variable(name) for name in parameter_names]
    equations = [x * x - p for x, p in zip(unknowns, parameters)]
    system = SymbolicSystem(
        equations,
        [*unknown_names, *parameter_names],
    )
    parameter_values = [1.5 + 0.1 * index for index in range(dimension)]
    return system, parameter_names, parameter_values


def run_sample(
    dimension: int, frontend: str, execution: str, warm_solves: int
) -> dict[str, float]:
    system, parameter_names, parameter_values = fixture(dimension)
    output_directory: tempfile.TemporaryDirectory[str] | None = None
    if execution == "aot":
        output_directory = tempfile.TemporaryDirectory(
            prefix="rustedscithe-nonlinear-cold-aot-",
            ignore_cleanup_errors=True,
        )

    try:
        constructor_start = time.perf_counter()
        if execution == "aot":
            assert output_directory is not None
            solver = NonlinearSolver(
                system,
                method="newton",
                frontend=frontend,
                execution="aot",
                aot_policy="rebuild_always",
                aot_profile="debug",
                aot_output_dir=output_directory.name,
                telemetry="detailed",
                parameters=parameter_names,
                parameter_values=parameter_values,
                tolerance=1e-10,
                max_iterations=80,
            )
        else:
            solver = NonlinearSolver(
                system,
                method="newton",
                frontend=frontend,
                execution="lambdify",
                telemetry="detailed",
                parameters=parameter_names,
                parameter_values=parameter_values,
                tolerance=1e-10,
                max_iterations=80,
            )
        constructor_ms = (time.perf_counter() - constructor_start) * 1_000.0

        prepare_start = time.perf_counter()
        solver.prepare()
        prepare_ms = (time.perf_counter() - prepare_start) * 1_000.0
        solve_start = time.perf_counter()
        result = solver.solve([1.0] * dimension)
        solve_wall_ms = (time.perf_counter() - solve_start) * 1_000.0
        if result.residual_norm > 1e-8:
            raise RuntimeError(
                f"{execution}/{frontend}/n={dimension} residual norm "
                f"{result.residual_norm:.3e}: {result.termination}"
            )

        values = result.solution_array().tolist()
        expected = [math.sqrt(value) for value in parameter_values]
        max_error = max(abs(actual - target) for actual, target in zip(values, expected))
        if max_error > 1e-7:
            raise RuntimeError(
                f"{execution}/{frontend}/n={dimension} exact-root error {max_error:.3e}"
            )

        counters = result.telemetry_counters()
        timings = result.telemetry_timings_ms()
        residual_evaluations = required_counter(
            counters, "residual_evaluations", f"nonlinear/{execution}/{frontend}"
        )
        jacobian_evaluations = required_counter(
            counters, "jacobian_evaluations", f"nonlinear/{execution}/{frontend}"
        )
        linear_solves = required_counter(
            counters, "linear_solves", f"nonlinear/{execution}/{frontend}"
        )
        initial_native_solve_ms = required_timing(
            timings, "total_ms", f"nonlinear/{execution}/{frontend}"
        )
        native_series_solve_ms = initial_native_solve_ms
        warm_native_solve_ms = 0.0
        guess = result.solution()
        warm_series_start = time.perf_counter()
        for index in range(warm_solves):
            current_parameters = [
                value * (1.0 + 0.01 * (index + 1)) for value in parameter_values
            ]
            result = solver.solve(guess, parameters=current_parameters)
            if result.residual_norm > 1e-8:
                raise RuntimeError(
                    f"{execution}/{frontend}/n={dimension} warm solve {index} "
                    f"residual {result.residual_norm:.3e}"
                )
            guess = result.solution()
            expected = [math.sqrt(value) for value in current_parameters]
            max_error = max(
                max_error,
                max(abs(actual - target) for actual, target in zip(guess, expected)),
            )
            warm_counters = result.telemetry_counters()
            warm_timings = result.telemetry_timings_ms()
            residual_evaluations += required_counter(
                warm_counters,
                "residual_evaluations",
                f"nonlinear warm solve/{execution}/{frontend}",
            )
            jacobian_evaluations += required_counter(
                warm_counters,
                "jacobian_evaluations",
                f"nonlinear warm solve/{execution}/{frontend}",
            )
            linear_solves += required_counter(
                warm_counters,
                "linear_solves",
                f"nonlinear warm solve/{execution}/{frontend}",
            )
            current_native_solve_ms = required_timing(
                warm_timings,
                "total_ms",
                f"nonlinear warm solve/{execution}/{frontend}",
            )
            warm_native_solve_ms += current_native_solve_ms
            native_series_solve_ms += current_native_solve_ms
        warm_series_ms = (time.perf_counter() - warm_series_start) * 1_000.0
        return {
            "constructor_ms": constructor_ms,
            "prepare_ms": prepare_ms,
            "solve_wall_ms": solve_wall_ms,
            "native_solve_ms": native_series_solve_ms,
            "initial_wall_minus_native_ms": wall_minus_native_ms(
                solve_wall_ms, initial_native_solve_ms
            ),
            "warm_series_ms": warm_series_ms,
            "warm_wall_minus_native_ms": wall_minus_native_ms(
                warm_series_ms, warm_native_solve_ms
            ),
            "warm_call_ms": warm_series_ms / warm_solves,
            "full_lifecycle_ms": constructor_ms + prepare_ms + solve_wall_ms + warm_series_ms,
            "residual_evaluations": float(residual_evaluations),
            "jacobian_evaluations": float(jacobian_evaluations),
            "linear_solves": float(linear_solves),
            "max_error": max_error,
        }
    finally:
        if output_directory is not None:
            output_directory.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="4,16,32")
    parser.add_argument("--frontends", default="expr_legacy,atom_native")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-solves", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dimensions = [int(item) for item in args.dimensions.split(",")]
    frontends = [item.strip() for item in args.frontends.split(",")]
    if (
        args.repeats < 1
        or args.warm_solves < 1
        or not dimensions
        or any(value < 1 for value in dimensions)
    ):
        parser.error("repeats, warm-solves and dimensions must be positive")
    if any(frontend not in {"expr_legacy", "atom_native"} for frontend in frontends):
        parser.error("frontends must contain expr_legacy and/or atom_native")
    if not (shutil.which("tcc") or shutil.which("gcc")):
        parser.error("cold AOT requires tcc or gcc on PATH")

    samples: dict[tuple[int, str, str], list[dict[str, float]]] = {}
    for dimension in dimensions:
        for frontend in frontends:
            for execution in ("lambdify", "aot"):
                key = (dimension, frontend, execution)
                samples[key] = [
                    run_sample(dimension, frontend, execution, args.warm_solves)
                    for _ in range(args.repeats)
                ]

    lines = [
        "# Nonlinear Python Boundary: Lambdify vs Cold AOT",
        "",
        f"Runtime: {runtime_provenance()}.",
        "Matched dense systems: `x_i^2 - p_i = 0`, with exact positive roots.",
        "Times are median/MAD in ms. `prepare_ms` includes a fresh AOT lowering,",
        "compile and link for AOT rows (`RebuildAlways`, fresh artifact directory).",
        "Each prepared model then performs a warm parameter-rebind solve series.",
        "Lambdify preparation is symbolic preparation only. Solve telemetry is",
        "nested inside Python wall time and must not be added to it.",
        "Counters and exact-root checks are included to make route mismatches visible.",
        "This measures the installed wrapper path, not the Rust core alone.",
        f"Warm-series length: {args.warm_solves}; full lifecycle includes constructor, preparation, first solve and all warm calls.",
        "Rust solve time sums native stages for the initial and warm solves; it remains nested in Python wall time.",
        "`wall - Rust` values are signed attribution residuals, not pure Python time; they include PyO3 conversion/dispatch and native work outside solver timers.",
        "Break-even is a median-based estimate: extra AOT preparation divided by per-call warm-solve savings; `not reached` means AOT warm calls were not faster.",
        "",
        "| n | frontend | execution | constructor ms | prepare ms | first solve ms | first wall - Rust ms | warm series ms | warm wall - Rust ms | avg warm call ms | full lifecycle ms | estimated AOT break-even rebinds | Rust solve ms | residual evals | Jacobian evals | linear solves | max root error |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    paired: dict[tuple[int, str], dict[str, list[dict[str, float]]]] = {}
    for (dimension, frontend, execution), runs in samples.items():
        paired.setdefault((dimension, frontend), {})[execution] = runs

    break_even: dict[tuple[int, str], str] = {}
    for route_pair, routes in paired.items():
        lambdify = routes["lambdify"]
        aot = routes["aot"]
        compile_overhead = max(
            0.0,
            statistics.median(sample["prepare_ms"] for sample in aot)
            - statistics.median(sample["prepare_ms"] for sample in lambdify),
        )
        per_call_saving = (
            statistics.median(sample["warm_call_ms"] for sample in lambdify)
            - statistics.median(sample["warm_call_ms"] for sample in aot)
        )
        break_even[route_pair] = (
            str(math.ceil(compile_overhead / per_call_saving))
            if per_call_saving > 0.0
            else "not reached"
        )
    for (dimension, frontend, execution), runs in samples.items():
        def timing(key: str) -> str:
            return median_mad([sample[key] for sample in runs])

        def count(key: str) -> str:
            values = [sample[key] for sample in runs]
            return "n/a" if any(not math.isfinite(value) for value in values) else str(int(statistics.median(values)))

        max_error = max(sample["max_error"] for sample in runs)
        lines.append(
            f"| {dimension} | {frontend} | {execution} | {timing('constructor_ms')} | "
            f"{timing('prepare_ms')} | {timing('solve_wall_ms')} | "
            f"{timing('initial_wall_minus_native_ms')} | {timing('warm_series_ms')} | "
            f"{timing('warm_wall_minus_native_ms')} | {timing('warm_call_ms')} | "
            f"{timing('full_lifecycle_ms')} | "
            f"{break_even[(dimension, frontend)] if execution == 'aot' else 'n/a'} | "
            f"{timing('native_solve_ms')} | "
            f"{count('residual_evaluations')} | "
            f"{count('jacobian_evaluations')} | {count('linear_solves')} | {max_error:.3e} |"
        )
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

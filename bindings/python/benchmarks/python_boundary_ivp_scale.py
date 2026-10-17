"""Measure Python-boundary BDF costs as a diagonal IVP grows in dimension.

The matched system is ``y_i' = -rate_i*y_i`` with ``y_i(0)=1``. The exact
solution makes every row correctness-gated. ExprLegacy and AtomNative are the
symbolic route comparison; the analytical Python callback route is reported
separately because each residual/Jacobian call crosses the Python/GIL boundary.
This is a wrapper benchmark, not a Rust-core performance ranking.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

from rustedscithe import BdfCallbackSolver, BdfSolver, Expr, SolverOptions

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


def run_sample(dimension: int, route: str, end_time: float) -> dict[str, float]:
    rate_array = np.asarray(
        [0.4 + 0.01 * index for index in range(dimension)], dtype=np.float64
    )
    y0_array = np.ones(dimension, dtype=np.float64)
    input_start = time.perf_counter()
    rates = rate_array.tolist()
    y0 = y0_array.tolist()
    input_ms = (time.perf_counter() - input_start) * 1_000.0
    if route == "atom_native":
        symbolic_frontend: Literal["expr_legacy", "atom_native"] = "atom_native"
    else:
        symbolic_frontend = "expr_legacy"
    options = SolverOptions(
        frontend=symbolic_frontend,
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="detailed",
    )

    expression_ms = 0.0
    if route != "python_callback":
        expression_start = time.perf_counter()
        state_names = [f"y{index}" for index in range(dimension)]
        parameter_names = [f"rate{index}" for index in range(dimension)]
        symbolic_states = [Expr.variable(name) for name in state_names]
        parameter_exprs = [Expr.variable(name) for name in parameter_names]
        equations = [
            -rate * state for rate, state in zip(parameter_exprs, symbolic_states)
        ]
        expression_ms = (time.perf_counter() - expression_start) * 1_000.0
        solver_factory: Callable[[], BdfSolver | BdfCallbackSolver] = lambda: BdfSolver(
            equations,
            state_names,
            "t",
            y0,
            0.0,
            end_time,
            end_time / 8.0,
            rtol=1e-8,
            atol=1e-10,
            options=options,
            parameters=parameter_names,
            parameter_values=rates,
        )
    else:
        residual = lambda _t, values: [
            -rate * value for rate, value in zip(rates, values)
        ]

        def jacobian(_t: float, _values: Sequence[float]) -> list[list[float]]:
            return [
                [-rates[row] if row == column else 0.0 for column in range(dimension)]
                for row in range(dimension)
            ]

        solver_factory = lambda: BdfCallbackSolver(
            state_dimension=dimension,
            t0=0.0,
            t_end=end_time,
            max_step=end_time / 8.0,
            y0=y0,
            residual_callback=residual,
            jacobian_callback=jacobian,
            rtol=1e-8,
            atol=1e-10,
            options=options,
        )

    construction_start = time.perf_counter()
    solver = solver_factory()
    construction_ms = (time.perf_counter() - construction_start) * 1_000.0

    prepare_start = time.perf_counter()
    solver.prepare()
    prepare_ms = (time.perf_counter() - prepare_start) * 1_000.0
    solve_start = time.perf_counter()
    result = solver.solve()
    solve_wall_ms = (time.perf_counter() - solve_start) * 1_000.0
    if result.status != "finished":
        raise RuntimeError(f"BDF/{route}/n={dimension} failed: {result.status}")

    trajectory = result.states_array()
    expected = np.exp(-np.asarray(rates) * end_time)
    max_error = float(np.max(np.abs(trajectory[-1, :] - expected)))
    if max_error > 1e-7:
        raise RuntimeError(f"BDF/{route}/n={dimension} exact error {max_error:.3e}")
    output_start = time.perf_counter()
    result.times_array()
    result.states_array()
    output_ms = (time.perf_counter() - output_start) * 1_000.0

    counters = result.telemetry_counters()
    timings = result.telemetry_timings_ms()
    native_solve_ms = required_timing(timings, "solve_ms", f"BDF/{route}")
    return {
        "expression_ms": expression_ms,
        "input_ms": input_ms,
        "construction_ms": construction_ms,
        "prepare_ms": prepare_ms,
        "solve_wall_ms": solve_wall_ms,
        "native_solve_ms": native_solve_ms,
        "wall_minus_native_ms": wall_minus_native_ms(solve_wall_ms, native_solve_ms),
        "residual_ms": required_timing(timings, "residual_ms", f"BDF/{route}"),
        "jacobian_ms": required_timing(timings, "jacobian_ms", f"BDF/{route}"),
        "factor_ms": required_timing(
            timings, "linear_factorization_ms", f"BDF/{route}"
        ),
        "output_ms": output_ms,
        "residual_calls": float(
            required_counter(counters, "residual_calls", f"BDF/{route}")
        ),
        "jacobian_calls": float(
            required_counter(counters, "jacobian_calls", f"BDF/{route}")
        ),
        "linear_solves": float(
            required_counter(counters, "linear_solve_attempts", f"BDF/{route}")
        ),
        "factorizations": float(required_counter(counters, "bdf_nlu", f"BDF/{route}")),
        "max_error": max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="1,8,32,128")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",")]
    if args.repeats < 3 or any(value < 1 for value in dimensions):
        parser.error("repeats must be >= 3 and dimensions must be positive")
    if not math.isfinite(args.end_time) or args.end_time <= 0.0:
        parser.error("end-time must be finite and positive")

    rows = []
    for dimension in dimensions:
        for route in ("expr_legacy", "atom_native", "python_callback"):
            runs = [run_sample(dimension, route, args.end_time) for _ in range(args.repeats)]
            rows.append((dimension, route, runs))

    lines = [
        "# Scaled Python BDF Boundary Workloads",
        "",
        f"Runtime: {runtime_provenance()}.",
        f"Matched diagonal decay IVP, `n={','.join(map(str, dimensions))}`, end time={args.end_time:g}, repeats={args.repeats}.",
        "The exact endpoint is `exp(-rate_i*t_end)` for each component. Times are median/MAD ms.",
        "Rust stage timings and counters are nested in Python solve wall time; do not add them together.",
        "`wall - Rust solve` is a signed attribution residual, not pure Python time; it can include PyO3 conversion/dispatch and native work outside solver timers.",
        "The Python callback route crosses the GIL during numerical evaluation and is not a matched symbolic frontend claim.",
        "",
        "| n | route | Expr build | NumPy input | constructor | prepare wall | solve wall | Rust solve | wall - Rust solve | residual | Jacobian | factorization | residual calls | Jacobian calls | linear solves | LU count | NumPy output | max exact error |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    timing_keys = (
        "expression_ms", "input_ms", "construction_ms", "prepare_ms", "solve_wall_ms",
        "native_solve_ms", "wall_minus_native_ms", "residual_ms", "jacobian_ms", "factor_ms", "output_ms",
    )
    count_keys = ("residual_calls", "jacobian_calls", "linear_solves", "factorizations")
    for dimension, route, runs in rows:
        values = {key: median_mad([sample[key] for sample in runs]) for key in timing_keys}
        counts = {
            key: str(int(statistics.median(sample[key] for sample in runs)))
            for key in count_keys
        }
        max_error = max(sample["max_error"] for sample in runs)
        lines.append(
            f"| {dimension} | {route} | {values['expression_ms']} | {values['input_ms']} | "
            f"{values['construction_ms']} | {values['prepare_ms']} | {values['solve_wall_ms']} | "
            f"{values['native_solve_ms']} | {values['wall_minus_native_ms']} | "
            f"{values['residual_ms']} | {values['jacobian_ms']} | "
            f"{values['factor_ms']} | {counts['residual_calls']} | {counts['jacobian_calls']} | "
            f"{counts['linear_solves']} | {counts['factorizations']} | {values['output_ms']} | {max_error:.3e} |"
        )
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

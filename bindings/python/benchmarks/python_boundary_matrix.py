"""Measure Python/native boundary costs on matched BVP_sci workloads.

This is wrapper-level diagnostic evidence, not a replacement for Rust release
benchmarks. Python wall-clock includes native work; nested native timings are
reported alongside it and must not be added to it. The numerical route crosses
the Python callback/GIL boundary, unlike symbolic Lambdify and AOT routes.
"""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import BvpSciSolver, Expr, SolverOptions

from _support import required_counter


def median_mad(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.3f}/{mad:.3f}"


def median_mad_count(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return f"{median}/{mad}"


def timed_ms(action: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1_000.0


def telemetry_value(snapshot: dict[str, float], *names: str) -> float | None:
    for name in names:
        value = snapshot.get(name)
        if value is not None:
            return float(value)
    return None


def make_solver(
    route: str,
    nodes: int,
    aot_dir: Path | None,
    compiler: str | None,
) -> tuple[BvpSciSolver, float, float, float]:
    expr_start = time.perf_counter()
    rate = Expr.variable("rate")
    boundary = [Expr.parse("ya_y"), Expr.parse("yb_y - 1")]
    expression_ms = (time.perf_counter() - expr_start) * 1_000.0

    mesh_array = np.linspace(0.0, 1.0, nodes, dtype=np.float64)
    guess_array = mesh_array.copy()
    marshal_start = time.perf_counter()
    mesh = mesh_array.tolist()
    guess = guess_array.tolist()
    marshal_ms = (time.perf_counter() - marshal_start) * 1_000.0

    if route == "numerical_callback":
        frontend, execution = "numerical", "numerical"
    else:
        frontend = "expr_legacy" if route.endswith("expr_legacy") else "atom_native"
        execution = "aot" if route.startswith("aot_") else "lambdify"

    options_kwargs: dict[str, str] = {
        "frontend": frontend,
        "execution": execution,
        "matrix_layout": "dense",
        "telemetry": "detailed",
    }
    if execution == "aot":
        if aot_dir is None or compiler is None:
            raise RuntimeError("AOT route requires a compiler and an artifact directory")
        options_kwargs.update(
            aot_policy="rebuild_always",
            aot_toolchain=compiler,
            aot_profile="debug",
            aot_output_dir=str(aot_dir),
        )
    options = SolverOptions(**options_kwargs)

    kwargs: dict[str, object] = {}
    if route == "numerical_callback":
        kwargs.update(
            rhs_callback=lambda _x, _state, parameters: [parameters[0]],
            rhs_jacobian_callback=lambda _x, _state, _parameters: [[0.0]],
            rhs_parameter_jacobian_callback=lambda _x, _state, _parameters: [[1.0]],
        )

    constructor_start = time.perf_counter()
    solver = BvpSciSolver(
        [] if route == "numerical_callback" else [rate],
        ["y"],
        boundary,
        mesh,
        guess,
        parameters=["rate"],
        parameter_values=[0.35],
        independent_variable="x",
        options=options,
        tolerance=1e-6,
        max_nodes=max(nodes * 2, nodes + 16),
        max_mesh_refinements=8,
        output="dense",
        **kwargs,
    )
    constructor_ms = (time.perf_counter() - constructor_start) * 1_000.0
    return solver, expression_ms, marshal_ms, constructor_ms


def validate_solution(solution: object, nodes: int) -> float:
    if solution.status_code != 0:
        raise RuntimeError(f"BVP solve failed: {solution.status}")
    if not math.isclose(solution.parameters[0], 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise RuntimeError(f"unknown parameter mismatch: {solution.parameters}")
    points = np.linspace(0.0, 1.0, min(nodes, 17))
    error = max(abs(solution.sample(float(x))[0] - x) for x in points)
    if error > 2e-5:
        raise RuntimeError(f"linear exact-solution error too large: {error:.3e}")
    return error


def measure_route(
    route: str, nodes: int, repeats: int, continuation: int, compiler: str | None
) -> dict[str, object]:
    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "expression_build_ms",
            "input_materialize_ms",
            "constructor_ms",
            "cold_python_solve_ms",
            "native_preparation_ms",
            "native_full_solve_ms",
            "native_residual_ms",
            "native_jacobian_ms",
            "native_rhs_callback_ms",
            "native_factorization_ms",
            "rhs_calls",
            "jacobian_calls",
            "linear_solves",
            "factorizations",
            "numpy_output_ms",
            "continuation_series_ms",
            "continuation_rebind_total_ms",
            "continuation_solve_total_ms",
            "aot_build_ms",
            "aot_link_ms",
        )
    }
    max_error = 0.0

    for repeat in range(repeats):
        # On Windows, a loaded AOT DLL can remain locked until process exit.
        # Cleanup is best-effort so that this OS lifecycle detail does not turn
        # a completed measurement into a benchmark failure.
        with tempfile.TemporaryDirectory(
            prefix="rustedscithe-boundary-aot-", ignore_cleanup_errors=True
        ) as temp:
            aot_dir = Path(temp) if route.startswith("aot_") else None
            solver, expression_ms, marshal_ms, constructor_ms = make_solver(
                route, nodes, aot_dir, compiler
            )
            samples["expression_build_ms"].append(expression_ms)
            samples["input_materialize_ms"].append(marshal_ms)
            samples["constructor_ms"].append(constructor_ms)

            solution, cold_ms = timed_ms(solver.solve)
            samples["cold_python_solve_ms"].append(cold_ms)
            max_error = max(max_error, validate_solution(solution, nodes))

            native = solution.telemetry_timings_ms()
            counters = solution.telemetry_counters()
            for name in ("rhs_calls", "jacobian_calls", "linear_solves", "factorizations"):
                samples[name].append(required_counter(counters, name, route))
            for output_key, aliases in {
                "native_preparation_ms": ("reported_preparation_ms", "preparation_ms"),
                "native_full_solve_ms": ("full_solve_ms",),
                "native_residual_ms": ("residual_evaluation_ms", "residual_ms"),
                "native_jacobian_ms": ("jacobian_evaluation_ms", "jacobian_ms"),
                "native_rhs_callback_ms": ("rhs_callback_ms",),
                "native_factorization_ms": ("factorization_ms", "linear_factorization_ms"),
            }.items():
                value = telemetry_value(native, *aliases)
                if value is not None:
                    samples[output_key].append(value)

            if route.startswith("aot_"):
                aot_timings = solution.telemetry_aot_timings_ms()
                for key, candidates in {
                    "aot_build_ms": ("cold.aot_build.ms", "aot_build_ms"),
                    "aot_link_ms": ("cold.aot_link.ms", "aot_link_ms"),
                }.items():
                    value = telemetry_value(aot_timings, *candidates)
                    if value is not None:
                        samples[key].append(value)

            _, output_ms = timed_ms(lambda: (solution.mesh_array(), solution.values_array()))
            samples["numpy_output_ms"].append(output_ms)

            continuation_start = time.perf_counter()
            rebind_total_ms = 0.0
            continuation_solve_total_ms = 0.0
            for index in range(continuation):
                # The problem is unchanged; varying this parameter value varies
                # the initial estimate, exercising the prepared rebind path.
                _, rebind_ms = timed_ms(
                    lambda index=index: solver.set_parameters(
                        [0.2 + (index % 5) * 0.2]
                    )
                )
                continued, continued_solve_ms = timed_ms(solver.solve)
                rebind_total_ms += rebind_ms
                continuation_solve_total_ms += continued_solve_ms
                max_error = max(max_error, validate_solution(continued, nodes))
            samples["continuation_rebind_total_ms"].append(rebind_total_ms)
            samples["continuation_solve_total_ms"].append(continuation_solve_total_ms)
            samples["continuation_series_ms"].append(
                (time.perf_counter() - continuation_start) * 1_000.0
            )

    return {
        "route": route,
        "nodes": nodes,
        "expression_build": median_mad(samples["expression_build_ms"]),
        "input_materialize": median_mad(samples["input_materialize_ms"]),
        "constructor": median_mad(samples["constructor_ms"]),
        "cold_solve": median_mad(samples["cold_python_solve_ms"]),
        "native_prepare": median_mad(samples["native_preparation_ms"]),
        "native_solve": median_mad(samples["native_full_solve_ms"]),
        "native_residual": median_mad(samples["native_residual_ms"]),
        "native_jacobian": median_mad(samples["native_jacobian_ms"]),
        "python_rhs": median_mad(samples["native_rhs_callback_ms"]),
        "native_factor": median_mad(samples["native_factorization_ms"]),
        "rhs_calls": median_mad_count(samples["rhs_calls"]),
        "jacobian_calls": median_mad_count(samples["jacobian_calls"]),
        "linear_solves": median_mad_count(samples["linear_solves"]),
        "factorizations": median_mad_count(samples["factorizations"]),
        "numpy_output": median_mad(samples["numpy_output_ms"]),
        "continuation_rebind": median_mad(samples["continuation_rebind_total_ms"]),
        "continuation_solve": median_mad(samples["continuation_solve_total_ms"]),
        "continuation_series": median_mad(samples["continuation_series_ms"]),
        "aot_build": median_mad(samples["aot_build_ms"]),
        "aot_link": median_mad(samples["aot_link_ms"]),
        "max_exact_error": f"{max_error:.2e}",
    }


def render(rows: list[dict[str, object]], repeats: int, continuation: int) -> str:
    lines = [
        "# Python binding boundary benchmark",
        "",
        "Matched BVP: `y'=rate`, `y(0)=0`, `y(1)=1`; exact solution `y=x`, `rate=1`.",
        f"Repeats={repeats}; each continuation series has {continuation} warm rebinds.",
        "Values are median/MAD milliseconds. `constructor` excludes Expr creation and",
        "Python list materialization; native preparation is reported separately where",
        "the solver exposes it. Constructor and solve wall times include their native",
        "work; native stages (including AOT build) are nested, not additive. Numerical",
        "callbacks cross Python/GIL per call.",
        "AOT rows compile in fresh artifact directories; build/link are cold stages.",
        "This is wrapper evidence, not a core speed claim or a statistical baseline.",
        "",
        "| route | nodes | Expr build | input lists | constructor | cold Python solve | native prep | native solve | residual | Jacobian | native factor | Python RHS | RHS calls | Jacobian calls | factorizations | linear solves | NumPy output | rebinds | warm solves | warm series | AOT build | AOT link | max error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {nodes} | {expression_build} | {input_materialize} | "
            "{constructor} | {cold_solve} | {native_prepare} | {native_solve} | "
            "{native_residual} | {native_jacobian} | {native_factor} | {python_rhs} | {rhs_calls} | "
            "{jacobian_calls} | {factorizations} | {linear_solves} | "
            "{numpy_output} | {continuation_rebind} | {continuation_solve} | "
            "{continuation_series} | {aot_build} | {aot_link} | "
            "{max_exact_error} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def parse_csv_ints(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 4 for item in result):
        raise argparse.ArgumentTypeError("each workload size must be at least 4")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=parse_csv_ints, default=parse_csv_ints("32,64"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--continuation", type=int, default=4)
    parser.add_argument("--include-aot", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.continuation < 1:
        parser.error("--repeats must be >= 3 and --continuation must be >= 1")

    routes = ["lambdify_expr_legacy", "lambdify_atom_native", "numerical_callback"]
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if args.include_aot:
        if compiler is None:
            parser.error("--include-aot requires tcc or gcc on PATH")
        routes.extend(("aot_expr_legacy", "aot_atom_native"))
    compiler_name = (
        "tcc" if compiler and Path(compiler).stem.lower() == "tcc" else "gcc"
    )

    rows = [
        measure_route(route, nodes, args.repeats, args.continuation, compiler_name)
        for nodes in args.nodes
        for route in routes
    ]
    report = render(rows, args.repeats, args.continuation)
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Measure Python-facing BVP construction and solve boundaries on one task.

This is diagnostic wrapper evidence, not a Rust-core performance benchmark.
The numerical callback row includes Python callback/GIL crossings; the
ExprLegacy row measures native Lambdify callbacks. Compare the reported Python
wall time with the nested Rust telemetry instead of treating them as additive.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np

from rustedscithe import BvpSciSolver, Expr, SolverOptions


def median_mad(values: list[float]) -> str:
    if not values:
        return "n/a"
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.4f}/{mad:.4f}"


def measure_route(route: str, repeats: int, nodes: int) -> dict[str, object]:
    mesh = np.linspace(0.0, 1.0, nodes)
    guess = np.exp(-mesh)
    constructor_ms: list[float] = []
    wall_solve_ms: list[float] = []
    native: dict[str, list[float]] = {}

    for _ in range(repeats):
        options = SolverOptions(
            frontend="expr_legacy" if route == "expr_legacy" else "numerical",
            execution="lambdify" if route == "expr_legacy" else "numerical",
            matrix_layout="dense",
            telemetry="detailed",
        )
        start = time.perf_counter()
        if route == "expr_legacy":
            solver = BvpSciSolver(
                [-Expr.variable("y")], ["y"], [Expr.parse("ya_y - 1")],
                mesh.tolist(), guess.tolist(), options=options, output="dense",
            )
        else:
            solver = BvpSciSolver(
                [], ["y"], [Expr.parse("ya_y - 1")],
                mesh.tolist(), guess.tolist(), options=options, output="dense",
                rhs_callback=lambda _x, state, _p: [-state[0]],
                rhs_jacobian_callback=lambda _x, _state, _p: [[-1.0]],
            )
        constructor_ms.append((time.perf_counter() - start) * 1_000.0)

        start = time.perf_counter()
        solution = solver.solve()
        wall_solve_ms.append((time.perf_counter() - start) * 1_000.0)
        if solution.status_code != 0:
            raise RuntimeError(f"{route} solve returned status={solution.status}")
        if not math.isclose(solution.sample(1.0)[0], math.exp(-1.0), rel_tol=5e-3):
            raise RuntimeError(f"{route} solution failed the exact-solution gate")
        for key, value in solution.telemetry_timings_ms().items():
            native.setdefault(key, []).append(value)

    return {
        "route": route,
        "constructor": median_mad(constructor_ms),
        "python_solve": median_mad(wall_solve_ms),
        "native_full_solve": median_mad(native.get("full_solve_ms", [])),
        "native_rhs_callback": median_mad(native.get("rhs_callback_ms", [])),
        "native_residual": median_mad(native.get("residual_evaluation_ms", [])),
    }


def render(rows: list[dict[str, object]], repeats: int, nodes: int) -> str:
    lines = [
        "# Python BVP boundary diagnostic",
        "",
        f"Matched problem: `y'=-y`, `y(0)=1`; nodes={nodes}, repeats={repeats}.",
        "Times are median/MAD in ms. Constructor includes route preparation;",
        "Python solve wall time contains native work and is not additive with stages.",
        "The numerical callback route crosses Python/GIL per callback; this is not a",
        "matched frontend speed claim and is not a substitute for core release benches.",
        "",
        "| route | constructor_ms | python_solve_wall_ms | rust_full_solve_ms | rust_rhs_callback_ms | rust_residual_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {constructor} | {python_solve} | {native_full_solve} | "
            "{native_rhs_callback} | {native_residual} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.nodes < 4:
        parser.error("--repeats must be >= 3 and --nodes must be >= 4")
    rows = [measure_route(route, args.repeats, args.nodes) for route in ("expr_legacy", "numerical")]
    report = render(rows, args.repeats, args.nodes)
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

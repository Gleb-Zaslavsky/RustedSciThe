"""Measure only Python-boundary routes for a prepared symbolic system.

Each route evaluates the same native residual/Jacobian provider. This script
does not compare symbolic frontends or claim to benchmark the Rust core.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import Expr, SymbolicFrontend, SymbolicSystem


def median_mad(samples: list[float]) -> str:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return f"{median:.4f}/{mad:.4f}"


def measure(action: Callable[[], object], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        action()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples


def run(dimension: int, samples: int, repeats: int, frontend_name: str, strided: bool) -> str:
    variables = [Expr.variable(f"x{i}") for i in range(dimension)]
    residuals = [
        variable**2 + 0.25 * variables[(index + 1) % dimension]
        for index, variable in enumerate(variables)
    ]
    system = SymbolicSystem(residuals, [f"x{i}" for i in range(dimension)])
    frontend = {
        "expr_legacy": SymbolicFrontend.ExprLegacy,
        "atom_native": SymbolicFrontend.AtomNative,
    }[frontend_name]
    prepared = system.lambdify(frontend=frontend)

    rng = np.random.default_rng(20261025)
    points = rng.uniform(0.25, 1.25, size=(samples, dimension)).astype(np.float64)
    if strided:
        padded = np.empty((samples, dimension * 2), dtype=np.float64)
        padded[:, ::2] = points
        padded[:, 1::2] = -99.0
        points = padded[:, ::2]

    expected_residuals = prepared.evaluate_batch(points)
    expected_jacobians = prepared.evaluate_jacobian_batch(points)
    residual_out = np.empty((samples, dimension), dtype=np.float64)
    jacobian_out = np.empty((samples, dimension, dimension), dtype=np.float64)

    def residual_point_loop() -> np.ndarray:
        return np.asarray([prepared.evaluate(row.tolist()) for row in points])

    def jacobian_point_loop() -> np.ndarray:
        return np.asarray([prepared.evaluate_jacobian(row.tolist()) for row in points])

    routes: list[tuple[str, Callable[[], object], np.ndarray]] = [
        ("residual point loop + Python lists", residual_point_loop, expected_residuals),
        ("residual NumPy batch return", lambda: prepared.evaluate_batch(points), expected_residuals),
        (
            "residual NumPy batch into reusable out",
            lambda: prepared.evaluate_batch_into(points, residual_out),
            expected_residuals,
        ),
        ("Jacobian point loop + Python lists", jacobian_point_loop, expected_jacobians),
        (
            "Jacobian NumPy batch return",
            lambda: prepared.evaluate_jacobian_batch(points),
            expected_jacobians,
        ),
        (
            "Jacobian NumPy batch into reusable out",
            lambda: prepared.evaluate_jacobian_batch_into(points, jacobian_out),
            expected_jacobians,
        ),
    ]

    rows = []
    for label, action, expected in routes:
        for _ in range(2):
            result = action()
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        results = measure(action, repeats)
        rows.append(f"| {label} | {median_mad(results)} |")

    layout = "strided" if strided else "C-contiguous"
    return "\n".join(
        [
            "# Python SymbolicSystem Boundary Benchmark",
            "",
            f"Python={platform.python_version()}, NumPy={np.__version__}, frontend={frontend_name}, ",
            f"variables={dimension}, residuals={dimension}, samples={samples}, input={layout}, repeats={repeats}.",
            "Times are median/MAD milliseconds per complete operation. Every row uses the same prepared Rust residual/Jacobian provider and passes a numerical parity gate. The point-loop rows include Python iteration, list conversion and repeated native calls; batch rows cross the Python/Rust boundary once per batch. This is wrapper-boundary evidence, not a Rust-core frontend or solver benchmark.",
            "",
            "| route | median/MAD ms per batch |",
            "| --- | ---: |",
            *rows,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--frontend", choices=("expr_legacy", "atom_native"), default="expr_legacy")
    parser.add_argument("--strided", action="store_true", help="use a non-contiguous NumPy input view")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dimension < 1 or args.samples < 1 or args.repeats < 3:
        parser.error("dimension/samples must be positive and repeats must be >= 3")

    report = run(args.dimension, args.samples, args.repeats, args.frontend, args.strided)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

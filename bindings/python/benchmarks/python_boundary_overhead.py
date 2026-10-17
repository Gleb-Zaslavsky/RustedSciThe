"""Measure small Python/native and NumPy materialization boundary costs.

The native row is a minimal PyO3 round trip around constant-expression
evaluation, not a pure FFI instruction cost. NumPy rows isolate explicit
array/list materialization from solver preparation and numerical work.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rustedscithe import Expr


def median_mad(values: list[float]) -> str:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.6f}/{mad:.6f}"


def time_per_call(action: Callable[[], object], batch: int, repeats: int) -> list[float]:
    """Measure repeated calls in batches to reduce timer granularity noise."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(batch):
            action()
        samples.append((time.perf_counter_ns() - start) / batch / 1_000.0)
    return samples


def run(sizes: list[int], repeats: int, batch: int) -> str:
    expression = Expr.constant(1.0)
    native_value = expression.evaluate([])
    if native_value != 1.0:
        raise RuntimeError(f"native constant-expression gate failed: {native_value}")

    noop_samples = time_per_call(lambda: None, batch, repeats)
    native_samples = time_per_call(lambda: expression.evaluate([]), batch, repeats)
    lines = [
        "# Python Boundary Overhead Microbenchmark",
        "",
        f"Python={platform.python_version()}, NumPy={np.__version__}, repeats={repeats}, batch={batch}.",
        "Times are median/MAD microseconds per operation. The native row is a minimal PyO3 round-trip proxy (Rust constant evaluation plus argument/result conversion), not a pure FFI cost. NumPy rows measure explicit materialization only; they exclude solver work.",
        "",
        "| operation | elements | median/MAD us |",
        "| --- | ---: | ---: |",
        f"| Python no-op call baseline | 0 | {median_mad(noop_samples)} |",
        f"| PyO3 constant evaluate round-trip | 0 | {median_mad(native_samples)} |",
    ]

    for size in sizes:
        array = np.linspace(0.0, 1.0, size, dtype=np.float64)
        python_values = array.tolist()
        if len(python_values) != size or not np.allclose(python_values, array):
            raise RuntimeError(f"NumPy input materialization gate failed at size={size}")
        restored = np.asarray(python_values, dtype=np.float64)
        if restored.shape != array.shape or not np.array_equal(restored, array):
            raise RuntimeError(f"NumPy output materialization gate failed at size={size}")

        input_samples = time_per_call(array.tolist, batch, repeats)
        output_samples = time_per_call(
            lambda: np.asarray(python_values, dtype=np.float64), batch, repeats
        )
        lines.append(f"| NumPy input `.tolist()` | {size} | {median_mad(input_samples)} |")
        lines.append(f"| Python sequence to NumPy float64 | {size} | {median_mad(output_samples)} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1,8,64,256,1024")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--batch", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        sizes = [int(token.strip()) for token in args.sizes.split(",") if token.strip()]
    except ValueError as error:
        parser.error(f"sizes must be comma-separated positive integers: {error}")
    if not sizes or any(size < 1 for size in sizes):
        parser.error("sizes must contain positive integers")
    if args.repeats < 3 or args.batch < 1:
        parser.error("repeats must be >= 3 and batch must be positive")

    report = run(sizes, args.repeats, args.batch)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

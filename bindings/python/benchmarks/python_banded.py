"""Measure the Python boundary for the compact faithful banded API.

This is not a dense-versus-banded Rust benchmark. The coefficient matrix is
created directly in packed band storage, and each row reports the Python
boundary cost of assembly, factorization, warm solves, multiple RHS and
explicit refinement separately.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from rustedscithe import BandedMatrix, LapackStyleBandedLuFaithful


def median_mad(values: list[float]) -> str:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return f"{median:.4f}/{mad:.4f}"


def compact_fixture(n: int, kl: int, ku: int) -> np.ndarray:
    values = np.zeros((kl + ku + 1, n), dtype=np.float64)
    values[ku, :] = 4.0
    for row in range(kl + ku + 1):
        offset = row - ku
        if offset < 0:
            values[row, -offset:] = 0.05
        elif offset > 0:
            values[row, : n - offset] = 0.05
    return values


def measure(n: int, kl: int, ku: int, repeats: int) -> dict[str, object]:
    assembly: list[float] = []
    factor: list[float] = []
    warm_solve: list[float] = []
    multiple_rhs: list[float] = []
    refinement: list[float] = []
    storage_bytes = 0

    compact = compact_fixture(n, kl, ku)
    rhs = np.ones(n, dtype=np.float64)
    rhs_multiple = np.ones((n, 4), dtype=np.float64)
    for _ in range(repeats):
        start = time.perf_counter()
        matrix = BandedMatrix(n, kl, ku, compact)
        assembly.append((time.perf_counter() - start) * 1_000.0)
        storage_bytes = matrix.storage_bytes

        lu = LapackStyleBandedLuFaithful(n, kl, ku)
        start = time.perf_counter()
        lu.factor(matrix)
        factor.append((time.perf_counter() - start) * 1_000.0)

        start = time.perf_counter()
        lu.solve(rhs)
        warm_solve.append((time.perf_counter() - start) * 1_000.0)

        start = time.perf_counter()
        lu.solve_multiple(rhs_multiple)
        multiple_rhs.append((time.perf_counter() - start) * 1_000.0)

        start = time.perf_counter()
        report = lu.solve_with_refinement(matrix, rhs, max_iters=2, tolerance=1e-12)
        refinement.append((time.perf_counter() - start) * 1_000.0)
        if not report.converged:
            raise RuntimeError(
                f"banded refinement failed for n={n}, kl={kl}, ku={ku}: "
                f"relative_residual={report.relative_residual_inf}"
            )

    return {
        "n": n,
        "kl": kl,
        "ku": ku,
        "storage_bytes": storage_bytes,
        "assembly_ms": median_mad(assembly),
        "factor_ms": median_mad(factor),
        "warm_solve_ms": median_mad(warm_solve),
        "multiple_rhs_ms": median_mad(multiple_rhs),
        "refinement_ms": median_mad(refinement),
    }


def render(rows: list[dict[str, object]], repeats: int) -> str:
    lines = [
        "# Python compact faithful banded boundary benchmark",
        "",
        f"Repeats: `{repeats}`. Times are median/MAD in milliseconds.",
        "The coefficient matrix remains `(kl + ku + 1, n)` throughout.",
        "This measures Python/NumPy boundary overhead and native calls; it is",
        "not a replacement for Rust-core performance evidence.",
        "",
        "| n | kl | ku | storage_bytes | assembly_ms | factor_ms | warm_solve_ms | multiple_rhs_ms | refinement_ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {n} | {kl} | {ku} | {storage_bytes} | {assembly_ms} | "
            "{factor_ms} | {warm_solve_ms} | {multiple_rhs_ms} | {refinement_ms} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="32,128,512")
    parser.add_argument("--bandwidths", default="1,2")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be >= 3")

    dimensions = [int(value) for value in args.dimensions.split(",") if value]
    bandwidths = [int(value) for value in args.bandwidths.split(",") if value]
    if not dimensions or any(value <= 0 for value in dimensions):
        parser.error("--dimensions must contain positive integers")
    if not bandwidths or any(value <= 0 for value in bandwidths):
        parser.error("--bandwidths must contain positive integers")

    rows = [
        measure(n, bandwidth, bandwidth, args.repeats)
        for n in dimensions
        for bandwidth in bandwidths
        if bandwidth < n
    ]
    report = render(rows, args.repeats)
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

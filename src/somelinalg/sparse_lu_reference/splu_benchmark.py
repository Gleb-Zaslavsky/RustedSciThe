"""Compact SciPy 1.18.1 SuperLU benchmark for shared sparse fixtures."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from statistics import median
from time import perf_counter

# Set this before importing NumPy/SciPy so the same thread budget can be used
# when comparing their OpenBLAS route with the sequential SuperLU control.
_blas_threads = os.environ.get("SOMELINALG_SCIPY_BLAS_THREADS")
if _blas_threads:
    os.environ["OPENBLAS_NUM_THREADS"] = _blas_threads

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def benchmark_fixture(
    input_path: Path,
    repeats: int = 9,
    solution_output: Path | None = None,
) -> dict[str, str]:
    lines = input_path.read_text(encoding="ascii").splitlines()
    header = list(map(int, lines[0].split()))
    exact_solution: np.ndarray | None
    if len(header) == 5:
        dimension, mesh_nodes, state_dimension, border_dimension, entry_count = header
        rhs_line, entries_start = 1, 3
        exact_solution = np.fromstring(lines[2], sep=" ", dtype=float)
    elif len(header) == 3:
        dimension, mesh_nodes, entry_count = header
        state_dimension = border_dimension = 0
        rhs_line, entries_start = 1, 2
        exact_solution = None
    else:
        raise ValueError("fixture header must have either 3 legacy or 5 shared fields")

    rhs = np.fromstring(lines[rhs_line], sep=" ", dtype=float)
    if rhs.size != dimension or (exact_solution is not None and exact_solution.size != dimension):
        raise ValueError("fixture vector dimensions are inconsistent")
    if len(lines) != entry_count + entries_start:
        raise ValueError("fixture triplet count is inconsistent")

    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    values = np.empty(entry_count, dtype=float)
    for index, line in enumerate(lines[entries_start:]):
        row, column, value = line.split()
        rows[index] = int(row)
        columns[index] = int(column)
        values[index] = float(value)

    assembly_ms: list[float] = []
    factor_ms: list[float] = []
    solve_ms: list[float] = []
    residual_inf = float("nan")
    solution_max_diff = float("nan")
    matrix_nnz = numerical_nonzeros = factor_l_nnz = factor_u_nnz = 0
    for _ in range(max(1, repeats)):
        started = perf_counter()
        matrix = coo_matrix(
            (values, (rows, columns)), shape=(dimension, dimension)
        ).tocsc()
        matrix.sum_duplicates()
        assembly_ms.append((perf_counter() - started) * 1e3)
        matrix_nnz = int(matrix.nnz)
        numerical_nonzeros = int(np.count_nonzero(matrix.data))

        started = perf_counter()
        factor = splu(matrix)
        factor_ms.append((perf_counter() - started) * 1e3)
        factor_l_nnz, factor_u_nnz = int(factor.L.nnz), int(factor.U.nnz)

        started = perf_counter()
        solution = factor.solve(rhs)
        solve_ms.append((perf_counter() - started) * 1e3)
        residual_inf = float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))
        if exact_solution is not None:
            solution_max_diff = float(np.max(np.abs(solution - exact_solution)))

    assembly_median = median(assembly_ms)
    factor_median = median(factor_ms)
    solve_median = median(solve_ms)
    scipy_config = scipy.__config__.CONFIG
    blas_config = scipy_config.get("Build Dependencies", {}).get("blas", {})
    blas_name = blas_config.get("name", "unknown")
    blas_version = blas_config.get("version", "unknown")
    thread_policy = _blas_threads or os.environ.get("OPENBLAS_NUM_THREADS", "runtime-default")
    # Keep legacy aliases consumed by the BVP_sci same-matrix benchmark.
    fields = {
        "status": "ok",
        "solver": "scipy-splu",
        "scipy_version": scipy.__version__,
        "blas": f"{blas_name}-{blas_version}",
        "blas_threads": str(thread_policy),
        "superlu_options": "ColPerm=COLAMD,DiagPivotThresh=1.0,default panel/relax",
        "dimension": str(dimension),
        "mesh_nodes": str(mesh_nodes),
        "state_dimension": str(state_dimension),
        "border_dimension": str(border_dimension),
        "input_triplets": str(entry_count),
        "csc_nnz": str(matrix_nnz),
        "csc_stored_entries": str(matrix_nnz),
        "csc_numerical_nonzeros": str(numerical_nonzeros),
        "factor_L_nnz": str(factor_l_nnz),
        "factor_U_nnz": str(factor_u_nnz),
        "assembly_median_ms": f"{assembly_median:.9f}",
        "assembly_median_us": f"{assembly_median * 1e3:.3f}",
        "symbolic_plus_numeric_factor_median_ms": f"{factor_median:.9f}",
        "factor_median_ms": f"{factor_median:.9f}",
        "solve_median_ms": f"{solve_median:.9f}",
        "residual_inf": f"{residual_inf:.17e}",
        "relative_residual_inf": f"{residual_inf / max(1.0, float(np.linalg.norm(rhs, ord=np.inf))):.17e}",
        "solution_inf": f"{float(np.linalg.norm(solution, ord=np.inf)):.17e}",
    }
    if exact_solution is not None:
        fields["known_solution_max_diff"] = f"{solution_max_diff:.17e}"
    if solution_output is not None:
        solution_output.parent.mkdir(parents=True, exist_ok=True)
        solution_bytes = " ".join(f"{value:.17e}" for value in solution).encode("ascii")
        solution_output.write_bytes(solution_bytes)
        fields["solution_sha256"] = hashlib.sha256(solution_bytes).hexdigest()
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--solution-output", type=Path)
    args = parser.parse_args()
    fields = benchmark_fixture(args.input, args.repeats, args.solution_output)
    print("\t".join(f"{key}={value}" for key, value in fields.items()))


if __name__ == "__main__":
    main()

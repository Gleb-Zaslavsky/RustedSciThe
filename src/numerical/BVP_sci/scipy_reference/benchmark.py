"""Release timing rows for matched SciPy 1.18.1 BVP workloads.

Only ``solve_bvp`` is timed. The vendored module is loaded without the parity
telemetry wrapper so Python wall-clock measurements do not include trace or
counter instrumentation overhead.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import numpy as np
import scipy


HERE = Path(__file__).resolve().parent


def load_solve_bvp():
    spec = importlib.util.spec_from_file_location(
        "rustedscithe_scipy_bvp_benchmark", HERE / "bvp.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load vendored SciPy _bvp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_bvp


def problem(case: str):
    if case == "exponential-analytic":
        def fun(_, y):
            return np.vstack((y[1], y[0]))

        def fun_jac(x, y):
            values = np.zeros((2, 2, x.size))
            values[0, 1, :] = 1.0
            values[1, 0, :] = 1.0
            return values

        def bc(ya, yb):
            return np.array([ya[0] - 1.0, yb[0]])

        def bc_jac(ya, yb):
            return (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [1.0, 0.0]]),
            )

        return {
            "fun": fun,
            "bc": bc,
            "x": np.linspace(0.0, 1.0, 5),
            "y": np.zeros((2, 5)),
            "kwargs": {"tol": 1e-3, "max_nodes": 100, "fun_jac": fun_jac,
                       "bc_jac": bc_jac},
        }

    if case == "big-parameterized-analytic":
        systems = 30
        dimension = systems * 2

        def fun(_, y, p):
            values = np.zeros_like(y)
            values[::2] = y[1::2]
            values[1::4] = -(p[0] ** 2) * y[::4]
            values[3::4] = -(p[1] ** 2) * y[2::4]
            return values

        def fun_jac(x, y, p):
            points = x.size
            df_dy = np.zeros((dimension, dimension, points))
            df_dy[np.arange(0, dimension, 2), np.arange(1, dimension, 2)] = 1.0
            df_dy[np.arange(1, dimension, 4), np.arange(0, dimension, 4)] = -(p[0] ** 2)
            df_dy[np.arange(3, dimension, 4), np.arange(2, dimension, 4)] = -(p[1] ** 2)
            df_dp = np.zeros((dimension, 2, points))
            df_dp[np.arange(1, dimension, 4), 0] = (
                -2.0 * p[0] * y[np.arange(0, dimension, 4)]
            )
            df_dp[np.arange(3, dimension, 4), 1] = (
                -2.0 * p[1] * y[np.arange(2, dimension, 4)]
            )
            return df_dy, df_dp

        def bc(ya, yb, p):
            return np.hstack((ya[::2], yb[::2], ya[1] - p[0], ya[3] - p[1]))

        def bc_jac(ya, yb, p):
            dya = np.zeros((dimension + 2, dimension))
            dyb = np.zeros((dimension + 2, dimension))
            dya[np.arange(systems), np.arange(0, dimension, 2)] = 1.0
            dyb[np.arange(systems, 2 * systems), np.arange(0, dimension, 2)] = 1.0
            dp = np.zeros((dimension + 2, 2))
            dya[dimension, 1] = 1.0
            dya[dimension + 1, 3] = 1.0
            dp[dimension, 0] = -1.0
            dp[dimension + 1, 1] = -1.0
            return dya, dyb, dp

        return {
            "fun": fun,
            "bc": bc,
            "x": np.linspace(0.0, np.pi, 5),
            "y": np.ones((dimension, 5)),
            "p": np.array([0.5, 0.5]),
            "kwargs": {"fun_jac": fun_jac, "bc_jac": bc_jac},
        }

    raise ValueError(f"unsupported matched benchmark case: {case}")


def solve_once(solve_bvp, fixture):
    return solve_bvp(
        fixture["fun"],
        fixture["bc"],
        fixture["x"],
        fixture["y"].copy(),
        p=fixture["p"].copy() if "p" in fixture else None,
        **fixture["kwargs"],
    )


def benchmark(case: str, warmups: int, repeats: int) -> str:
    solve_bvp = load_solve_bvp()
    fixture = problem(case)
    for _ in range(warmups):
        result = solve_once(solve_bvp, fixture)
        if not result.success:
            raise RuntimeError(f"warmup failed for {case}: status={result.status}")

    samples_ms = []
    observed = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = solve_once(solve_bvp, fixture)
        samples_ms.append((time.perf_counter() - started) * 1e3)
        observed.append((int(result.status), int(result.x.size), int(result.niter)))
        if not result.success:
            raise RuntimeError(f"measured solve failed for {case}: status={result.status}")
    if len(set(observed)) != 1:
        raise RuntimeError(f"unstable SciPy outcomes for {case}: {observed}")

    median_ms = statistics.median(samples_ms)
    minimum_ms = min(samples_ms)
    maximum_ms = max(samples_ms)
    status, nodes, iterations = observed[0]
    return "\t".join(
        (
            case,
            scipy.__version__,
            np.__version__,
            str(status),
            str(nodes),
            str(iterations),
            f"{median_ms:.9f}",
            f"{minimum_ms:.9f}",
            f"{maximum_ms:.9f}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("exponential-analytic", "big-parameterized-analytic"),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    print(benchmark(args.case, args.warmups, args.repeats))


if __name__ == "__main__":
    main()

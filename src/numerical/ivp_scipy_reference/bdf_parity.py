"""Matched BDF workloads for the Rust/SciPy numerical-core parity story."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import statistics
import sys
from time import perf_counter_ns

import numpy as np
import scipy

from .bdf import BDF
from .telemetry import BdfTelemetry

UPSTREAM_SCIPY_TAG = "v1.18.1"
EXPECTED_NUMPY_VERSION = "2.5.4"
EXPECTED_SCIPY_VERSION = "1.18.1"


def source_hashes():
    from pathlib import Path

    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("bdf.py", "base.py", "common.py", "telemetry.py")
    }


def _cases():
    def stiff_scalar(_t, y):
        return -35.0 * y

    def robertson(_t, y):
        return np.array([
            -0.04 * y[0] + 1.0e4 * y[1] * y[2],
            0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] ** 2,
            3.0e7 * y[1] ** 2,
        ])

    def robertson_jac(_t, y):
        return np.array([
            [-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
            [0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
            [0.0, 6.0e7 * y[1], 0.0],
        ])

    def hires(_t, y):
        y1, y2, y3, y4, y5, y6, y7, y8 = y
        return np.array([
            -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007,
            1.71 * y1 - 8.75 * y2,
            -10.03 * y3 + 0.43 * y4 + 0.035 * y5,
            8.32 * y2 + 1.71 * y3 - 1.12 * y4,
            -1.745 * y5 + 0.43 * y6 + 0.43 * y7,
            -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7,
            280.0 * y6 * y8 - 1.81 * y7,
            -280.0 * y6 * y8 + 1.81 * y7,
        ])

    dense_matrix = np.array([
        [-30.0, 3.0, 2.0, 1.0],
        [4.0, -20.0, 1.0, 2.0],
        [1.0, 2.0, -10.0, 1.0],
        [2.0, 1.0, 3.0, -5.0],
    ])

    def dense_coupled(_t, y):
        return dense_matrix @ y

    def mixed_scale(t, y):
        return np.array([
            -80.0 * y[0] + np.sin(t),
            -0.2 * y[1] + 1.0e-7 * np.cos(t),
        ])

    def combustion_like(_t, y):
        a, b, temperature = y
        rate = 1.0e7 * np.exp(-5.0e4 / (8.314 * temperature)) * a * a
        return np.array([
            -rate,
            0.5 * rate,
            500.0 * rate - 0.5 * (temperature - 300.0),
        ])

    return [
        {
            "name": "stiff-scalar",
            "fun": stiff_scalar,
            # Match Rust's constant-J lifecycle rather than a callable J.
            "jac": np.array([[-35.0]]),
            "y0": np.array([1.0]),
            "t0": 0.0,
            "t_bound": 0.4,
            "first_step": 0.005,
            "max_step": 0.05,
            "rtol": 1.0e-8,
            "atol": 1.0e-11,
        },
        {
            "name": "robertson",
            "fun": robertson,
            "jac": robertson_jac,
            "y0": np.array([1.0, 0.0, 0.0]),
            "t0": 0.0,
            "t_bound": 0.002,
            "first_step": 2.0e-5,
            "max_step": 5.0e-4,
            "rtol": 1.0e-7,
            "atol": 1.0e-10,
        },
        {
            "name": "stiff-scalar-fd",
            "fun": stiff_scalar,
            "jac": None,
            "y0": np.array([1.0]),
            "t0": 0.0,
            "t_bound": 0.4,
            "first_step": 0.005,
            "max_step": 0.05,
            "rtol": 1.0e-8,
            "atol": 1.0e-11,
        },
        {
            "name": "constant-jacobian-newton-retry",
            "fun": lambda _t, y: -1000.0 * y,
            # Deliberately stale constant J forces the retry/step-reduction branch.
            "jac": np.array([[0.0]]),
            "y0": np.array([1.0]),
            "t0": 0.0,
            "t_bound": 0.1,
            "first_step": 0.1,
            "max_step": 0.1,
            "rtol": 1.0e-6,
            "atol": 1.0e-10,
        },
        {
            "name": "backward-scalar",
            "fun": lambda _t, y: -3.0 * y,
            "jac": np.array([[-3.0]]),
            "y0": np.array([np.exp(-0.6)]),
            "t0": 0.2,
            "t_bound": 0.0,
            "first_step": 0.0025,
            "max_step": 0.02,
            "rtol": 1.0e-8,
            "atol": 1.0e-11,
        },
        {
            "name": "auto-first-step",
            "fun": lambda _t, y: -8.0 * y,
            "jac": np.array([[-8.0]]),
            "y0": np.array([1.0]),
            "t0": 0.0,
            "t_bound": 0.5,
            "first_step": None,
            "max_step": 0.05,
            "rtol": 1.0e-8,
            "atol": 1.0e-11,
        },
        {
            "name": "hires-short",
            "fun": hires,
            "jac": None,
            "y0": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]),
            "t0": 0.0,
            "t_bound": 0.5,
            "first_step": None,
            "max_step": 0.05,
            "rtol": 1.0e-7,
            "atol": 1.0e-10,
        },
        {
            "name": "nonautonomous-mixed-scale",
            "fun": mixed_scale,
            "jac": np.diag([-80.0, -0.2]),
            "y0": np.array([1.0e3, 1.0e-6]),
            "t0": 0.0,
            "t_bound": 0.5,
            "first_step": 1.0e-3,
            "max_step": 0.02,
            "rtol": 1.0e-8,
            "atol": 1.0e-12,
        },
        {
            "name": "dense-coupled-4",
            "fun": dense_coupled,
            "jac": dense_matrix,
            "y0": np.array([1.0, -0.5, 0.25, 2.0]),
            "t0": 0.0,
            "t_bound": 0.3,
            "first_step": None,
            "max_step": 0.025,
            "rtol": 1.0e-8,
            "atol": 1.0e-11,
        },
        {
            "name": "combustion-like",
            "fun": combustion_like,
            "jac": None,
            "y0": np.array([1.0, 0.0, 300.0]),
            "t0": 0.0,
            "t_bound": 0.01,
            "first_step": None,
            "max_step": 0.001,
            "rtol": 1.0e-7,
            "atol": 1.0e-10,
        },
    ]


def run_case(case):
    telemetry = BdfTelemetry()
    solver = BDF(
        case["fun"], case["t0"], case["y0"], case["t_bound"],
        first_step=case["first_step"], max_step=case["max_step"],
        rtol=case["rtol"], atol=case["atol"], jac=case["jac"],
        telemetry=telemetry,
    )
    dense_segments = []
    step_lu_calls = []
    while solver.status == "running":
        previous_nlu = int(solver.nlu)
        solver.step()
        step_lu_calls.append(
            (float(solver.t_old), float(solver.t), int(solver.nlu) - previous_nlu)
        )
        if solver.status != "failed":
            dense_segments.append(solver.dense_output())

    fractions = (0.25, 0.5, 0.75)
    sample_times = [
        case["t0"] + fraction * (case["t_bound"] - case["t0"])
        for fraction in fractions
    ]
    samples = []
    if solver.status == "finished":
        for sample_time in sample_times:
            segment = next(
                item for item in dense_segments
                if min(item.t_old, item.t) <= sample_time <= max(item.t_old, item.t)
            )
            samples.append(np.asarray(segment(sample_time), dtype=float))
    snapshot = telemetry.snapshot(solver)
    counters = snapshot["counters"]
    return {
        "name": case["name"],
        "t": float(solver.t),
        "t_bound": float(solver.t_bound),
        "t_old": float(solver.t_old),
        "h_abs": float(solver.h_abs),
        "order": int(solver.order),
        "direction": float(solver.direction),
        "y": np.asarray(solver.y, dtype=float),
        "samples": samples,
        "status": str(solver.status),
        "nfev": int(solver.nfev),
        "njev": int(solver.njev),
        "nlu": int(solver.nlu),
        "accepted": int(counters.get("accepted_steps", 0)),
        "rejected": int(counters.get("rejected_steps", 0)),
        "attempts": int(counters.get("step_attempts", 0)),
        "newton_iterations": int(counters.get("newton_iterations", 0)),
        "newton_solves": int(counters.get("newton_solves", 0)),
        "linear_solve_calls": int(counters.get("linear_solve_calls", 0)),
        "factorization_calls": int(counters.get("factorization_calls", 0)),
        "jacobian_refreshes": int(counters.get("jacobian_refreshes", 0)),
        "order_changes": int(counters.get("order_changes", 0)),
        "rhs_callback_calls": int(counters.get("rhs_callback_calls", 0)),
        "fd_baseline_rhs_calls": int(counters.get("fd_baseline_rhs_calls", 0)),
        "fd_rhs_callback_calls": int(counters.get("fd_rhs_callback_calls", 0)),
        "fd_state_probes": int(counters.get("fd_state_probes", 0)),
        "rtol": float(case["rtol"]),
        "atol": float(case["atol"]),
        "trace": telemetry.controller_trace,
        "step_lu_calls": step_lu_calls,
    }


def run_case_wall_clock(case):
    """Time one SciPy solver construction, integration, and trajectory capture."""
    solver = BDF(
        case["fun"], case["t0"], case["y0"], case["t_bound"],
        first_step=case["first_step"], max_step=case["max_step"],
        rtol=case["rtol"], atol=case["atol"], jac=case["jac"],
        telemetry=None,
    )
    trajectory = [np.asarray(solver.y, dtype=float).copy()]
    while solver.status == "running":
        solver.step()
        if solver.status != "failed":
            trajectory.append(np.asarray(solver.y, dtype=float).copy())
    if solver.status != "finished":
        raise RuntimeError(f"SciPy BDF failed for {case['name']}: {solver.status}")
    return np.asarray(trajectory), solver


def benchmark_case(case, warmups: int, repeats: int):
    """Measure with telemetry off; collect counters in a separate diagnostic run."""
    for _ in range(warmups):
        run_case_wall_clock(case)
    elapsed_ns = []
    final_state = None
    for _ in range(repeats):
        started = perf_counter_ns()
        trajectory, solver = run_case_wall_clock(case)
        elapsed_ns.append(perf_counter_ns() - started)
        final_state = trajectory[-1]

    diagnostic = run_case(case)
    ordered = sorted(elapsed_ns)
    return {
        "name": case["name"],
        "median_ms": statistics.median(elapsed_ns) / 1.0e6,
        "p10_ms": ordered[max(0, int((len(ordered) - 1) * 0.1))] / 1.0e6,
        "p90_ms": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.9))] / 1.0e6,
        "samples": repeats,
        "accepted": diagnostic["accepted"],
        "rejected": diagnostic["rejected"],
        "nfev": diagnostic["nfev"],
        "njev": diagnostic["njev"],
        "nlu": diagnostic["nlu"],
        "rtol": float(case["rtol"]),
        "atol": float(case["atol"]),
        "final_state": ",".join(f"{value:.17g}" for value in final_state),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 3:
        raise ValueError("benchmark requires warmups >= 0 and repeats >= 3")
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(f"expected NumPy {EXPECTED_NUMPY_VERSION}, got {np.__version__}")
    if scipy.__version__ != EXPECTED_SCIPY_VERSION:
        raise RuntimeError(f"expected SciPy {EXPECTED_SCIPY_VERSION}, got {scipy.__version__}")

    if args.benchmark:
        benchmark_names = {
            "stiff-scalar", "robertson", "stiff-scalar-fd",
            "backward-scalar", "auto-first-step",
        }
        cases = [
            benchmark_case(case, args.warmups, args.repeats)
            for case in _cases()
            if case["name"] in benchmark_names
        ]
        if args.format == "tsv":
            writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
            writer.writerow([
                "# scipy_tag", UPSTREAM_SCIPY_TAG,
                "python", sys.version.split()[0], "numpy", np.__version__,
                "scipy", scipy.__version__, "platform", platform.platform(),
                "timed_scope", "constructor+integration+accepted-state-collection",
                "telemetry", "off",
                "source_sha256_bdf", source_hashes()["bdf.py"],
            ])
            writer.writerow([
                "workload", "median_ms", "p10_ms", "p90_ms", "samples",
                "accepted", "rejected", "nfev", "njev", "nlu", "rtol", "atol",
                "final_state",
            ])
            for case in cases:
                writer.writerow([
                    case["name"], f"{case['median_ms']:.9g}",
                    f"{case['p10_ms']:.9g}", f"{case['p90_ms']:.9g}",
                    case["samples"], case["accepted"], case["rejected"],
                    case["nfev"], case["njev"], case["nlu"],
                    f"{case['rtol']:.17g}", f"{case['atol']:.17g}", case["final_state"],
                ])
            return
        import json
        print(json.dumps({"benchmark": cases}, separators=(",", ":")))
        return

    cases = [run_case(case) for case in _cases()]
    hashes = source_hashes()
    if args.format == "tsv":
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "# scipy_tag", UPSTREAM_SCIPY_TAG,
            "python", sys.version.split()[0], "numpy", np.__version__,
            "scipy", scipy.__version__, "platform", platform.platform(),
            "bdf_sha256_instrumented", hashes["bdf.py"],
            "base_sha256", hashes["base.py"], "common_sha256", hashes["common.py"],
            "telemetry_sha256", hashes["telemetry.py"],
        ])
        writer.writerow([
            "workload", "t", "t_bound", "t_old", "h_abs", "order", "direction",
            "y", "samples", "status", "nfev", "njev", "nlu",
            "accepted", "rejected", "attempts", "newton_iterations",
            "newton_solves", "linear_solve_calls", "factorization_calls",
            "jacobian_refreshes", "order_changes", "rhs_callback_calls",
            "fd_baseline_rhs_calls", "fd_rhs_callback_calls",
            "fd_state_probes", "rtol", "atol",
            "trace_entries", "step_lu_calls",
        ])
        for case in cases:
            writer.writerow([
                case["name"], f"{case['t']:.17g}",
                f"{case['t_bound']:.17g}",
                f"{case['t_old']:.17g}", f"{case['h_abs']:.17g}",
                case["order"], f"{case['direction']:.17g}",
                ",".join(f"{value:.17g}" for value in case["y"]),
                ";".join(",".join(f"{value:.17g}" for value in sample)
                         for sample in case["samples"]),
                case["status"], case["nfev"], case["njev"], case["nlu"],
                case["accepted"], case["rejected"], case["attempts"],
                case["newton_iterations"], case["newton_solves"],
                case["linear_solve_calls"], case["factorization_calls"],
                case["jacobian_refreshes"],
                case["order_changes"],
                case["rhs_callback_calls"], case["fd_baseline_rhs_calls"],
                case["fd_rhs_callback_calls"],
                case["fd_state_probes"],
                f"{case['rtol']:.17g}", f"{case['atol']:.17g}",
                len(case["trace"]),
                ";".join(
                    f"{t_old:.17g}:{t_new:.17g}:{count}"
                    for t_old, t_new, count in case["step_lu_calls"]
                ),
            ])
        return

    import json

    serializable = [{
        **case,
        "y": case["y"].tolist(),
        "samples": [sample.tolist() for sample in case["samples"]],
    } for case in cases]
    print(json.dumps({
        "environment": {
            "scipy_tag": UPSTREAM_SCIPY_TAG,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "source_sha256": hashes,
        },
        "cases": serializable,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()

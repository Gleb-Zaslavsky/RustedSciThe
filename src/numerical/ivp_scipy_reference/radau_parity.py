"""Matched, deterministic fixtures for the Rust/SciPy Radau parity story."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
import scipy

from .radau import Radau
from .telemetry import RadauTelemetry

UPSTREAM_SCIPY_TAG = "v1.18.1"
EXPECTED_NUMPY_VERSION = "2.5.4"
EXPECTED_SCIPY_VERSION = "1.18.1"
_DEFAULT_FIRST_STEP = object()


def source_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("radau.py", "bdf.py", "base.py", "common.py", "telemetry.py")
    }


def _cases():
    # Keep equations, initial states, and parameters aligned with
    # numerical::ivp_workloads and Radau/tests/story_support.rs.
    def stiff_scalar(t, y):
        return np.array([-1000.0 * (y[0] - np.cos(t)) - np.sin(t)])

    def stiff_scalar_jac(_t, _y):
        return np.array([[-1000.0]])

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

    # k, E, R, T0, Q/(c*rho), k_loss, cooling
    p = (1.0e7, 5.0e4, 8.314, 300.0, 5.0e2, 0.0, 0.5)

    def combustion(_t, y):
        A, B, T = y
        k, E, gas, T0, heat, loss, cooling = p
        rate = k * np.exp(-E / (gas * T)) * A * A
        return np.array([-rate, 0.5 * rate - loss * B,
                         heat * rate - cooling * (T - T0)])

    def combustion_jac(_t, y):
        A, _B, T = y
        k, E, gas, _T0, heat, loss, cooling = p
        q = k * np.exp(-E / (gas * T))
        dr_da = 2.0 * q * A
        dr_dt = q * A * A * E / (gas * T * T)
        return np.array([
            [-dr_da, 0.0, -dr_dt],
            [0.5 * dr_da, -loss, 0.5 * dr_dt],
            [heat * dr_da, 0.0, heat * dr_dt - cooling],
        ])

    return [
        ("stiff-scalar", stiff_scalar, stiff_scalar_jac, [1.0], 0.01),
        ("robertson", robertson, robertson_jac, [1.0, 0.0, 0.0], 0.002),
        ("combustion-like", combustion, combustion_jac, [1.0, 0.0, 300.0], 0.002),
    ]


def run_case(
    name, fun, jac, y0, t_bound, telemetry_enabled=True, *, t0=0.0,
    first_step=_DEFAULT_FIRST_STEP, max_step=None, accepted_step_budget=None,
    rtol=1.0e-7, atol=1.0e-10,
):
    span = t_bound - t0
    if first_step is _DEFAULT_FIRST_STEP:
        first_step = 0.1 * abs(span)
    if max_step is None:
        max_step = 0.25 * abs(span)
    sample_times = t0 + np.asarray([0.25, 0.5, 0.75]) * span
    telemetry = RadauTelemetry(enabled=telemetry_enabled)
    started = perf_counter()
    solver = Radau(
        fun, t0, np.asarray(y0, dtype=float), t_bound,
        first_step=first_step,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        jac=jac,
        telemetry=telemetry if telemetry_enabled else None,
    )
    dense_segments = []
    driver_steps = 0
    while solver.status == "running" and (
        accepted_step_budget is None or driver_steps < accepted_step_budget
    ):
        solver.step()
        driver_steps += 1
        if solver.status != "failed":
            dense_segments.append(solver.dense_output())
    samples = []
    if solver.status == "finished":
        for sample_time in sample_times:
            segment = next(
                item for item in dense_segments
                if min(item.t_old, item.t) <= sample_time <= max(item.t_old, item.t)
            )
            samples.append(np.asarray(segment(sample_time), dtype=float))
    total_ms = (perf_counter() - started) * 1_000.0
    status = (
        "runner_step_budget_exhausted"
        if solver.status == "running" and accepted_step_budget is not None
        else solver.status
    )
    return {
        "workload": name,
        "t0": float(t0),
        "t": float(solver.t),
        "y": [float(v) for v in solver.y],
        "samples": [[float(v) for v in sample] for sample in samples],
        "status": status,
        "scipy_nfev": solver.nfev,
        "scipy_njev": solver.njev,
        "scipy_nlu": solver.nlu,
        "driver_steps": driver_steps,
        "total_solve_ms": total_ms,
        "counters": telemetry.counters if telemetry_enabled else {},
        "timings_ms": telemetry.timings_ms if telemetry_enabled else {},
        "controller_trace": telemetry.controller_trace if telemetry_enabled else [],
    }


def run_all():
    return [run_case(*case) for case in _cases()]


def _hires_case():
    def rhs(_t, y):
        return np.array([
            -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] - 7.0e-4,
            1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
            8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
            -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4]
            - 0.43 * y[5] + 0.69 * y[6],
            280.0 * y[5] * y[7] - 1.81 * y[6],
            -1.81 * y[6] + 1.81 * y[7],
        ])

    def jac(_t, y):
        return np.array([
            [-1.71, 0.43, 8.32, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.71, -8.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -10.03, 0.43, 0.035, 0.0, 0.0, 0.0],
            [0.0, 8.32, 1.71, -1.12, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.745, 0.43, 0.43, 0.0],
            [0.0, 0.0, 0.0, 0.69, 1.71, -280.0 * y[7] - 0.43,
             0.69, -280.0 * y[5]],
            [0.0, 0.0, 0.0, 0.0, 0.0, 280.0 * y[7], -1.81, 280.0 * y[5]],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.81, 1.81],
        ])

    return rhs, jac, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]


def run_extended_cases():
    stiff_scalar = _cases()[0]
    _, scalar_rhs, scalar_jac, _, _ = stiff_scalar
    _, robertson_rhs, robertson_jac, robertson_y0, robertson_bound = _cases()[1]
    hires_rhs, hires_jac, hires_y0 = _hires_case()
    backward = run_case(
        "backward-auto-step", scalar_rhs, scalar_jac, [float(np.cos(0.01))],
        0.0, t0=0.01, first_step=None, max_step=0.0025,
    )
    hires = run_case(
        "hires-auto-step", hires_rhs, hires_jac, hires_y0, 1.0,
        first_step=None, max_step=0.25,
    )
    robertson_nearby = run_case(
        "robertson-nearby-tolerance", robertson_rhs, robertson_jac,
        robertson_y0, robertson_bound, first_step=0.0002, max_step=0.0005,
        rtol=8.0e-8,
    )

    try:
        Radau(
            lambda _t, y: -y, 0.0, np.array([1.0]), 0.1,
            first_step=0.2, jac=lambda _t, _y: np.array([[-1.0]]),
        )
    except ValueError as error:
        invalid_first_step = {
            "workload": "oversize-first-step",
            "status": "configuration_error",
            "message": str(error),
        }
    else:
        raise AssertionError("SciPy accepted first_step larger than the interval")

    tiny_t0 = 1.0e16
    underflow_telemetry = RadauTelemetry()
    underflow = Radau(
        lambda _t, y: -y, tiny_t0, np.array([1.0]), tiny_t0 + 2.0,
        first_step=1.0, jac=lambda _t, _y: np.array([[-1.0]]),
        telemetry=underflow_telemetry,
    )
    underflow_message = underflow.step()
    underflow_case = {
        "workload": "step-underflow",
        "status": underflow.status,
        "message": underflow_message,
        "t": float(underflow.t),
        "accepted_steps": underflow_telemetry.counters.get("accepted_steps", 0),
        "rejected_steps": underflow_telemetry.counters.get("rejected_steps", 0),
        "step_attempts": underflow_telemetry.counters.get("step_attempts", 0),
    }

    budget = run_case(
        "accepted-step-budget", lambda _t, y: -y,
        lambda _t, _y: np.array([[-1.0]]), [1.0], 0.01,
        first_step=0.005, max_step=0.005, accepted_step_budget=1,
    )
    return [backward, hires, robertson_nearby, invalid_first_step, underflow_case, budget]


def _quantile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def run_benchmark_case(case, repeats, warmups):
    name, fun, jac, y0, t_bound = case
    for _ in range(warmups):
        run_case(name, fun, jac, y0, t_bound, telemetry_enabled=False)
    samples = [
        run_case(name, fun, jac, y0, t_bound, telemetry_enabled=False)["total_solve_ms"]
        for _ in range(repeats)
    ]
    return {
        "workload": name,
        "warmups": warmups,
        "repeats": repeats,
        "wall_min_ms": min(samples),
        "wall_median_ms": _quantile(samples, 0.5),
        "wall_p90_ms": _quantile(samples, 0.9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--benchmark-warmups", type=int, default=3)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"Expected NumPy {EXPECTED_NUMPY_VERSION}, found {np.__version__}"
        )
    if scipy.__version__ != EXPECTED_SCIPY_VERSION:
        raise RuntimeError(
            f"Expected SciPy {EXPECTED_SCIPY_VERSION}, found {scipy.__version__}"
        )
    if args.benchmark_repeats:
        if args.benchmark_repeats < 1 or args.benchmark_warmups < 0:
            raise ValueError("benchmark repeats must be positive and warmups nonnegative")
        cases = [
            run_benchmark_case(case, args.benchmark_repeats, args.benchmark_warmups)
            for case in _cases()
        ]
        if args.format == "tsv":
            writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
            writer.writerow([
                "# upstream_scipy_tag", UPSTREAM_SCIPY_TAG,
                "# python", sys.version.split()[0], "numpy", np.__version__,
                "scipy", scipy.__version__, "platform", platform.platform(),
                "scope", "Radau construction + integration + sampled output; telemetry off",
            ])
            writer.writerow([
                "workload", "warmups", "repeats", "wall_min_ms",
                "wall_median_ms", "wall_p90_ms",
            ])
            for case in cases:
                writer.writerow([
                    case["workload"], case["warmups"], case["repeats"],
                    case["wall_min_ms"], case["wall_median_ms"], case["wall_p90_ms"],
                ])
            return
        print(json.dumps({"environment": {
            "upstream_scipy_tag": UPSTREAM_SCIPY_TAG,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
        }, "cases": cases}, separators=(",", ":")))
        return
    if args.extended:
        cases = run_extended_cases()
        if args.format == "tsv":
            writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
            writer.writerow([
                "scenario", "status", "message", "t0", "t", "y", "samples",
                "initial_h_abs", "accepted_steps", "rejected_steps", "step_attempts",
                "nfev", "njev", "nlu", "driver_steps",
            ])
            for case in cases:
                trace = case.get("controller_trace", [])
                counters = case.get("counters", {})
                writer.writerow([
                    case["workload"], case["status"], case.get("message", ""),
                    case.get("t0", ""), case.get("t", ""),
                    ",".join(map(str, case.get("y", []))),
                    ";".join(",".join(map(str, sample)) for sample in case.get("samples", [])),
                    trace[0][0] if trace else "",
                    case.get("accepted_steps", counters.get("accepted_steps", 0)),
                    case.get("rejected_steps", counters.get("rejected_steps", 0)),
                    case.get("step_attempts", counters.get("step_attempts", 0)),
                    case.get("scipy_nfev", 0), case.get("scipy_njev", 0),
                    case.get("scipy_nlu", 0), case.get("driver_steps", 0),
                ])
            return
        print(json.dumps({"cases": cases}, separators=(",", ":")))
        return
    cases = run_all()
    if args.format == "tsv":
        writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
        hashes = source_hashes()
        writer.writerow([
            "# upstream_scipy_tag", UPSTREAM_SCIPY_TAG,
            "# python", sys.version.split()[0], "numpy", np.__version__,
            "scipy", scipy.__version__, "platform", platform.platform(),
            "radau_sha256_instrumented", hashes["radau.py"],
            "bdf_sha256_instrumented", hashes["bdf.py"],
            "base_sha256", hashes["base.py"],
            "common_sha256", hashes["common.py"], "telemetry_sha256", hashes["telemetry.py"],
        ])
        writer.writerow([
            "workload", "t", "y", "samples", "status", "nfev", "njev", "nlu",
            "rhs_calls", "rhs_state_evaluations", "fd_rhs_calls", "fd_state_probes",
            "jacobian_calls", "jacobian_refreshes", "lu_calls", "factor_pair_attempts",
            "linear_solve_calls", "step_attempts", "newton_iterations", "accepted_steps",
            "rejected_steps", "error_estimates", "controller_trace", "total_solve_ms",
        ])
        for case in cases:
            counters = case["counters"]
            writer.writerow([
                case["workload"], case["t"], ",".join(map(str, case["y"])),
                ";".join(",".join(map(str, sample)) for sample in case["samples"]),
                case["status"], case["scipy_nfev"], case["scipy_njev"], case["scipy_nlu"],
                counters.get("rhs_callback_calls", 0),
                counters.get("rhs_state_evaluations", 0),
                counters.get("fd_rhs_callback_calls", 0),
                counters.get("fd_state_probes", 0),
                counters.get("jacobian_callback_calls", 0),
                counters.get("jacobian_refreshes", 0),
                counters.get("lu_factorization_calls", 0),
                counters.get("factorization_pair_attempts", 0),
                counters.get("linear_solve_calls", 0),
                counters.get("step_attempts", 0),
                counters.get("newton_iterations", 0),
                counters.get("accepted_steps", 0), counters.get("rejected_steps", 0),
                counters.get("error_estimates", 0),
                ";".join(
                    f"{trace[0]:.17g}:{trace[1]:.17g}:{trace[2]:.17g}:"
                    f"{trace[3]}:{int(trace[4])}:{int(trace[5])}"
                    for trace in case["controller_trace"]
                ),
                case["total_solve_ms"],
            ])
        return
    payload = {
        "environment": {
            "upstream_scipy_tag": UPSTREAM_SCIPY_TAG,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "vendored_source_sha256": source_hashes(),
        },
        "cases": cases,
    }
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()

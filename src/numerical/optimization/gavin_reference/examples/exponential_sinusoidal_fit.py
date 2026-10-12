"""Run challenging deterministic fits through Gavin's Python reference."""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import levenberg_marquardt as reference


def run_case(name: str, t: np.ndarray, noise: np.ndarray) -> dict[str, str]:
    expected = np.array([[2.0], [0.5], [1.0], [2.0]])
    initial = np.array([[1.9], [0.55], [0.9], [1.9]])
    observations = reference.lm_func(t, expected) + noise
    telemetry = reference.ReferenceTelemetry()
    started = perf_counter()

    # Keep the reference's own verbose console report out of the parity table.
    with redirect_stdout(StringIO()):
        fitted, reduced_chi_square, *_rest, history = reference.lm(
            initial, t, observations, telemetry=telemetry
        )
    wall_ms = (perf_counter() - started) * 1e3

    assert np.all(np.isfinite(fitted))
    assert np.isfinite(reduced_chi_square).all()
    assert telemetry.objective_evaluations == reference.func_calls
    assert telemetry.iterations == len(history)
    if not np.any(noise):
        np.testing.assert_allclose(fitted, expected, rtol=2e-2, atol=2e-2)
    return {
        "case": name,
        "samples": str(len(t)),
        "iterations": str(telemetry.iterations),
        "accepted/rejected": f"{telemetry.accepted_steps}/{telemetry.rejected_steps}",
        "objective_calls": str(telemetry.objective_evaluations),
        "fd_jacobians": str(telemetry.finite_difference_jacobians),
        "fd_columns": str(telemetry.finite_difference_columns),
        "broyden_updates": str(telemetry.broyden_updates),
        "linear_solves": str(telemetry.linear_solves),
        "objective_ms": f"{telemetry.objective_seconds * 1e3:.3f}",
        "linearize_ms": f"{telemetry.linearization_seconds * 1e3:.3f}",
        "linear_solve_ms": f"{telemetry.linear_solve_seconds * 1e3:.3f}",
        "finite_difference_ms": f"{telemetry.finite_difference_seconds * 1e3:.3f}",
        "covariance_ms": f"{telemetry.covariance_seconds * 1e3:.3f}",
        "total_ms": f"{wall_ms:.3f}",
        "termination": telemetry.termination,
        "reduced_chi_square": f"{float(reduced_chi_square[0, 0]):.12e}",
        "parameters": np.array2string(fitted.ravel(), precision=8),
    }


def run_controller_case(
    name: str,
    update_type: int,
    initial_values: list[float],
    t: np.ndarray,
    include_history: bool = True,
) -> dict[str, str]:
    expected = np.array([[2.0], [0.5], [1.0], [2.0]])
    initial = np.array(initial_values, dtype=float).reshape((-1, 1))
    observations = reference.lm_func(t, expected)
    telemetry = reference.ReferenceTelemetry()
    started = perf_counter()
    with redirect_stdout(StringIO()):
        fitted, reduced_chi_square, *_rest, history = reference.lm(
            initial,
            t,
            observations,
            telemetry=telemetry,
            update_type=update_type,
        )
    wall_ms = (perf_counter() - started) * 1e3
    history_text = (
        "; ".join(
            ",".join(f"{float(value):.12g}" for value in row[1:]) for row in history
        )
        if include_history
        else "; ".join(
            f"{index + 1}:" + ",".join(f"{float(value):.12g}" for value in history[index, 1:])
            for index in (0, 3, 99, 199, 299, 383, 384)
        )
    )
    return {
        "case": name,
        "policy": {1: "Marquardt", 2: "Quadratic", 3: "Nielsen"}[update_type],
        "iterations": str(telemetry.iterations),
        "accepted/rejected": f"{telemetry.accepted_steps}/{telemetry.rejected_steps}",
        "objective_calls": str(telemetry.objective_evaluations),
        "fd_jacobians": str(telemetry.finite_difference_jacobians),
        "fd_columns": str(telemetry.finite_difference_columns),
        "broyden_updates": str(telemetry.broyden_updates),
        "linear_solves": str(telemetry.linear_solves),
        "termination": telemetry.termination,
        "reduced_chi_square": f"{float(reduced_chi_square[0, 0]):.12e}",
        "parameters": np.array2string(fitted.ravel(), precision=10),
        "history (x2,p...)": history_text,
        "total_ms": f"{wall_ms:.3f}",
    }


def main() -> None:
    rows = []

    # The reference implementation expects one-dimensional samples, despite
    # its original docstring describing an m-by-1 column.
    broad_t = np.linspace(0.0, 20.0, 201)
    no_noise = np.zeros_like(broad_t)
    rows.append(run_case("broad_exact", broad_t, no_noise))

    deterministic_noise = 0.001 * np.sin(1.7 * broad_t) + 0.0005 * np.cos(
        2.3 * broad_t
    )
    rows.append(run_case("broad_noisy", broad_t, deterministic_noise))

    unit_interval = np.linspace(0.0, 1.0, 81)
    nonuniform_t = 20.0 * unit_interval**1.25
    rows.append(run_case("nonuniform_exact", nonuniform_t, np.zeros_like(nonuniform_t)))

    controller_t = np.linspace(0.0, 20.0, 201)
    controller_fixtures = [
        ("baseline", 1, [1.9, 0.55, 0.9, 1.9]),
        ("baseline", 2, [1.9, 0.55, 0.9, 1.9]),
        ("baseline", 3, [1.9, 0.55, 0.9, 1.9]),
        ("rejection_refresh", 1, [1.94, 0.55, 1.0, 1.38]),
        ("rejection_quadratic", 2, [0.63, 0.74, 0.98, 3.06]),
        ("rejection_nielsen", 3, [0.76, 0.77, 0.94, 1.95]),
    ]
    rows.extend(
        run_controller_case(f"controller_{name}", update_type, initial, controller_t)
        for name, update_type, initial in controller_fixtures
    )
    rows.append(
        run_controller_case(
            "controller_active_bound",
            1,
            [0.01, 0.55, 0.9, 1.9],
            controller_t,
            include_history=False,
        )
    )

    regular_rows = rows[:3]
    controller_rows = rows[3:]

    def render_table(table_rows: list[dict[str, str]]) -> list[str]:
        columns = list(table_rows[0])
        table = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        table.extend(
            "| " + " | ".join(row[column] for column in columns) + " |"
            for row in table_rows
        )
        return table

    report = [
        "# Gavin Python reference telemetry",
        "",
        "Stage timings overlap: objective evaluation occurs inside linearization and finite differences.",
        "Controller histories serialize each iteration as reduced chi-square followed by parameters.",
        "",
        "## Fit fixtures",
        "",
    ]
    report.extend(render_table(regular_rows))
    report.extend(["", "## Controller histories", ""])
    report.extend(render_table(controller_rows))
    text = "\n".join(report) + "\n"
    print(text, end="")
    output = os.environ.get("GAVIN_REFERENCE_REPORT")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

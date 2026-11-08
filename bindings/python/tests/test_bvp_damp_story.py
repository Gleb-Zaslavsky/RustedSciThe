"""Slow, explicit BVP_Damp stories.

This module is intentionally separate from ``test_contract.py``.  The
contract suite should stay fast; this file mirrors the native combustion
fixture and writes one compact report for a deliberate diagnostic run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import pytest

from rustedscithe import BvpDampSolver, Expr, SolverOptions


def _number(value: float) -> str:
    """Render a finite Python number as a parser-friendly Rust expression."""

    return format(value, ".17g")


def _combustion_fixture():
    """Return the verified native BVP_Damp combustion equations and data."""

    q_heat = 3000.0 * 1.0e3 * 0.034
    delta_t = 600.0
    t_scale = 600.0
    length = 3.0e-4
    molecular_mass = 34.2 / 1000.0
    peclet_q = 0.0090168
    density_diffusion = 2.88e-4
    peclet_d = 1.50e-3
    density_m = molecular_mass * 2.0e6 / (8.314 * 1500.0)
    q_m = length**2 / t_scale
    q_s = length**2

    teta = "Teta"
    q = "q"
    c0 = "C0"
    j0 = "J0"
    c1 = "C1"
    j1 = "J1"
    rate = (
        f"{_number(1.3e5)} * exp(-({_number(5000.0 * 4.184)}) / "
        f"({_number(8.314)} * ({teta} * {_number(t_scale)} + "
        f"{_number(delta_t)}))) * {c0} * ({_number(density_m)} / 0.342)"
    )
    expressions = [
        Expr.parse(f"{q} / 0.07"),
        Expr.parse(
            f"{q} * {_number(peclet_q)} - {_number(q_heat)} * ({rate}) * "
            f"{_number(q_m)}"
        ),
        Expr.parse(f"{j0} / {_number(density_diffusion)}"),
        Expr.parse(
            f"{j0} * {_number(peclet_d)} + ({rate}) * {_number(density_m)} * "
            f"{_number(q_s)}"
        ),
        Expr.parse(f"{j1} / {_number(density_diffusion)}"),
        Expr.parse(
            f"{j1} * {_number(peclet_d)} - ({rate}) * {_number(density_m)} * "
            f"{_number(q_s)}"
        ),
    ]
    states = [teta, q, c0, j0, c1, j1]
    nodes = 1000
    mesh = [index / (nodes - 1) for index in range(nodes)]
    initial_state = [0.99] * (len(mesh) * len(states))
    boundary_conditions = {
        teta: [(0, (1000.0 - delta_t) / t_scale)],
        q: [(1, 1.0e-10)],
        c0: [(0, 1.0)],
        j0: [(1, 1.0e-7)],
        c1: [(0, 1.0e-3)],
        j1: [(1, 1.0e-7)],
    }
    bounds = {
        teta: (0.0, 10.0),
        q: (-1.0e20, 1.0e20),
        c0: (0.0, 1.5),
        j0: (-1.0e2, 1.0e2),
        c1: (0.0, 1.5),
        j1: (-1.0e2, 1.0e2),
    }
    return expressions, states, boundary_conditions, mesh, initial_state, bounds


def _report_path() -> Path:
    root = Path(
        os.environ.get(
            "RST_PYTHON_REPORT_DIR", "test_reports/python_bvp_damp_story"
        )
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"bvp_damp_combustion_{timestamp}.md"


@pytest.mark.slow
def test_bvp_damp_combustion_story_writes_compact_matrix(capsys):
    """Run the native combustion fixture through both Lambdify frontends."""

    (
        expressions,
        states,
        boundary_conditions,
        mesh,
        initial_state,
        bounds,
    ) = _combustion_fixture()
    rows = []
    for frontend in ("expr_legacy", "atom_native"):
        solver = BvpDampSolver(
            expressions,
            states,
            boundary_conditions,
            mesh,
            initial_state,
            options=SolverOptions(
                frontend=frontend,
                matrix_layout="sparse",
                execution="lambdify",
                telemetry="detailed",
            ),
            scheme="forward",
            tolerance=1.0e-5,
            max_iterations=100,
            max_jacobian_refreshes=6,
            max_damping_iterations=6,
            damping_factor=0.5,
            bounds=bounds,
        )
        solution = solver.solve()
        values = solution.values()
        flat_values = [value for row in values for value in row]
        counters = solution.telemetry_counters()
        timings = solution.telemetry_timings_ms()
        assert solution.status == "finished"
        assert len(values) == len(states)
        assert all(len(row) == len(solution.mesh()) for row in values)
        assert all(value == value and abs(value) != float("inf") for value in flat_values)
        assert counters.get("iterations", 0) > 0
        rows.append(
            {
                "frontend": frontend,
                "layout": solver.matrix_layout,
                "nodes": len(solution.mesh()),
                "iterations": counters.get("iterations", 0),
                "residual_calls": counters.get("residual_calls", 0),
                "jacobian_requests": counters.get("jacobian_requests", 0),
                "factorizations": counters.get("factorizations", 0),
                "total_ms": timings.get("total_ms", float("nan")),
                "symbolic_ms": timings.get("symbolic_operations_ms", float("nan")),
                "residual_ms": timings.get("residual_ms", float("nan")),
                "jacobian_ms": timings.get("jacobian_ms", float("nan")),
                "factorization_ms": timings.get("factorization_ms", float("nan")),
                "max_abs": max(abs(value) for value in flat_values),
            }
        )

    # Native legacy diagnostics must not turn this compact story into a noisy
    # console log.  The useful data is written below as one Markdown table.
    capsys.readouterr()
    report = _report_path()
    lines = [
        "# BVP_Damp Python combustion story",
        "",
        "Native verified combustion fixture; initial 1000-node Sparse mesh; Lambdify only.",
        "This is an explicit slow story, separate from the fast Python contract suite.",
        "",
        "| frontend | layout | nodes | iterations | residual_calls | jacobian_requests | factorizations | total_ms | symbolic_ms | residual_ms | jacobian_ms | factorization_ms | max_abs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {frontend} | {layout} | {nodes} | {iterations} | {residual_calls} | "
            "{jacobian_requests} | {factorizations} | {total_ms:.6f} | "
            "{symbolic_ms:.6f} | {residual_ms:.6f} | {jacobian_ms:.6f} | "
            "{factorization_ms:.6f} | {max_abs:.6e} |".format(**row)
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

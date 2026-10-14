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
    length = 0.3e-4
    molecular_mass = 34.2 / 1000.0
    conductivity = 0.07
    heat_capacity = 0.35 * 4.184 * 1000.0
    pressure = 2.0e6
    t0 = 800.0
    t_m = 1500.0
    gas_constant = 8.314
    mass_flux = 0.077 * (pressure / 1.0e5) ** 0.748 / 100.0
    density_0 = molecular_mass * pressure / (gas_constant * t0)
    diffusivity = conductivity / (heat_capacity * density_0)
    peclet_q = length * heat_capacity * mass_flux / conductivity
    density_diffusion = diffusivity * density_0 * (t_m / t0) ** 0.5
    peclet_d = mass_flux * length / density_diffusion
    density_m = molecular_mass * pressure / (gas_constant * t_m)
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
        f"({_number(gas_constant)} * ({teta} * {_number(t_scale)} + "
        f"{_number(delta_t)}))) * {c0} * ({_number(density_m)} / "
        f"{_number(molecular_mass)})"
    )
    expressions = [
        Expr.parse(f"{q} / {_number(conductivity)}"),
        Expr.parse(
            f"{q} * {_number(peclet_q)} - {_number(q_heat)} * ({rate}) * "
            f"{_number(q_m)}"
        ),
        Expr.parse(f"{j0} / {_number(density_diffusion)}"),
        Expr.parse(
            f"{j0} * {_number(peclet_d)} - ({rate}) * {_number(density_m)} * "
            f"{_number(q_s)}"
        ),
        Expr.parse(f"{j1} / {_number(density_diffusion)}"),
        Expr.parse(
            f"{j1} * {_number(peclet_d)} - ({rate}) * {_number(density_m)} * "
            f"{_number(q_s)}"
        ),
    ]
    states = [teta, q, c0, j0, c1, j1]
    mesh = [index / 19.0 for index in range(20)]
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
        teta: (-100.0, 100.0),
        q: (-1.0e20, 1.0e20),
        c0: (-1.0, 1.5),
        j0: (-1.0e20, 1.0e20),
        c1: (-1.0, 1.5),
        j1: (-1.0e20, 1.0e20),
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
                matrix_layout="dense",
                execution="lambdify",
                telemetry="timings",
            ),
            scheme="forward",
            tolerance=1.0e-5,
            max_iterations=100,
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
        "Native verified combustion fixture; initial 20-node mesh; Lambdify only.",
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

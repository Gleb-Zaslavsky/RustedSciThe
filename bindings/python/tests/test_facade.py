"""Smoke tests for the thin Python convenience facade."""

import math

import numpy as np

import pytest
from rustedscithe import facade

from rustedscithe import (
    Expr,
    Lsode2Solver,
    SolverOptions,
    SymbolicSystem,
    create_bvp,
    create_fit,
    create_ivp,
    create_nonlinear,
    render_table,
    run_task_batch,
    run_task_batch_report,
    solve_bvp,
    solve_ivp,
    task,
)


def test_public_facade_helpers_document_lifecycle_and_failure_contracts():
    public_helpers = (
        facade.create_ivp,
        facade.solve_ivp,
        facade.create_bvp,
        facade.solve_bvp,
        facade.create_nonlinear,
        facade.solve_nonlinear,
        facade.create_fit,
        facade.fit,
        facade.task,
        facade.capabilities,
        facade.render_table,
        facade.write_table,
        facade.run_task_batch,
        facade.run_task_batch_report,
    )
    for helper in public_helpers:
        assert helper.__doc__ is not None
        assert len(helper.__doc__.strip()) >= 120, helper.__name__
        assert "Returns" in helper.__doc__ or "Raises" in helper.__doc__

    assert "controller" in facade.create_ivp.__doc__
    assert "one-shot" in facade.solve_ivp.__doc__
    assert "boundary-data" in facade.create_bvp.__doc__
    assert "TaskDocumentError" in facade.task.__doc__


def test_facade_normalizes_ivp_creation_without_hiding_native_lifecycle():
    y = Expr.variable("y")
    solver = create_ivp(
        "radau",
        [-y],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.1,
        rtol=1e-5,
        atol=1e-7,
    )
    assert not solver.is_prepared()
    solver.prepare()
    result = solver.solve([1.0], None)
    assert solver.is_prepared()
    assert result.final_time == pytest.approx(0.2)


def test_facade_solve_ivp_uses_native_bdf_result_and_task_parser():
    y = Expr.variable("y")
    result = solve_ivp(
        "bdf",
        [-y],
        ["y"],
        "t",
        np.array([1.0]),
        0.0,
        0.2,
        0.1,
        rtol=1e-5,
        atol=1e-7,
    )
    assert result.status == "finished"
    assert result.times_array().shape == (len(result.times()),)
    assert result.states_array().shape == (len(result.times()), 1)
    assert result.times_array().dtype == np.float64
    assert result.states_array().dtype == np.float64
    document = task(
        """task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
    )
    assert "solver" in document.preview().lower()


@pytest.mark.parametrize(
    ("controller", "expected_mode"),
    [
        ("bdf", "bdf_only"),
        ("adams", "adams_only"),
        ("auto", "automatic_adams_bdf"),
    ],
)
def test_facade_lsode2_selects_controller_separately_from_solver_family(
    controller, expected_mode
):
    y = Expr.variable("y")
    solver = create_ivp(
        "lsode2",
        [-y],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.1,
        0.05,
        controller=controller,
    )

    assert isinstance(solver, Lsode2Solver)
    result = solver.solve()
    assert result.status.startswith("finished")
    assert result.controller_mode == expected_mode

    with pytest.raises(ValueError, match="only supported by the LSODE2"):
        create_ivp(
            "bdf", [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            controller="adams",
        )


def test_facade_lsoda_alias_selects_automatic_controller_by_default():
    y = Expr.variable("y")
    solver = create_ivp("lsoda", [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05)
    assert isinstance(solver, Lsode2Solver)
    assert solver.solve().controller_mode == "automatic_adams_bdf"

    # The explicit controller remains authoritative when callers need to pin
    # the LSODE2 policy while using the familiar LSODA family alias.
    pinned = create_ivp(
        "lsoda", [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, controller="adams"
    )
    assert pinned.solve().controller_mode == "adams_only"


def test_facade_batch_and_table_keep_later_tasks_after_failure(tmp_path):
    valid = """task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
    rows = run_task_batch([valid, "task\nmethod: BDF\n"])
    assert [row["status"] for row in rows] == ["ok", "error"]
    assert rows[1]["error_type"] == "TaskDocumentError"
    table = render_table(rows, columns=["status", "error_type"])
    assert "TaskDocumentError" in table
    assert "<inline:1>" == rows[0]["source"]
    output = tmp_path / "summary.md"
    output.write_text(table, encoding="utf-8")
    assert "status" in output.read_text(encoding="utf-8")

    report_path = tmp_path / "batch-report.md"
    report = run_task_batch_report([valid, "task\nmethod: BDF\n"], path=report_path)
    assert "universal_nonstiff" not in report
    assert "TaskDocumentError" in report
    assert "matrix_layout" in report
    assert report_path.read_text(encoding="utf-8") == report


def test_facade_dispatches_bvp_families_without_losing_native_options():
    y = Expr.variable("y")
    sci = create_bvp(
        "scipy",
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0, 0.6, 0.37]),
        options=SolverOptions(frontend="expr_legacy", matrix_layout="dense"),
        tolerance=1e-4,
    )
    assert sci.frontend == "expr_legacy"
    sci_solution = sci.solve()
    assert sci_solution.status_code == 0
    assert sci_solution.mesh_array().shape == (len(sci_solution.mesh()),)
    assert sci_solution.values_array().shape[0] == len(sci_solution.mesh())

    damp = create_bvp(
        "damped",
        [-y],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        options=SolverOptions(frontend="expr_legacy", matrix_layout="dense"),
        bounds={"y": (-10.0, 10.0)},
        max_iterations=12,
    )
    assert damp.provenance()["strategy"] == "damped"
    assert damp.solve().status == "finished"

    direct = solve_bvp(
        "bvp_sci",
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        [0.0, 1.0],
        [1.0, math.exp(-1.0)],
        options=SolverOptions(frontend="expr_legacy", matrix_layout="dense"),
    )
    assert direct.status_code == 0


def test_facade_returns_preparable_nonlinear_and_fitting_solvers():
    system = SymbolicSystem([Expr.parse("x^2 - 4")], ["x"])
    nonlinear = create_nonlinear(system)
    assert not nonlinear.is_prepared()
    nonlinear.prepare()
    assert nonlinear.is_prepared()
    assert nonlinear.solve([1.0]).solution()[0] == pytest.approx(2.0, rel=1e-6)

    fitting = create_fit([Expr.parse("x - 2")], np.array([0.0]))
    assert fitting.solve().solution()[0] == pytest.approx(2.0, rel=1e-6)

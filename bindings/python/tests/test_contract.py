"""Black-box smoke tests for the documented Python contract."""

import ast
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import rustedscithe.facade as python_facade

from rustedscithe import (
    AotArtifactError,
    BvpError,
    BvpDampSolution,
    BvpDampFrozenSolver,
    BvpDampNaiveSolver,
    BvpDampSolver,
    BvpSciSolution,
    BvpSciSolver,
    BvpTaskResult,
    BdfCallbackSolver,
    BdfSolver,
    BackwardEulerSolution,
    BackwardEulerSolver,
    CallbackError,
    ExhaustionError,
    Expr,
    ExprMatrix,
    ExprVector,
    FittingSolution,
    InvalidConfigurationError,
    InvalidExpressionError,
    IvpTaskResult,
    FittingSolver,
    GavinFittingSolver,
    GavinFittingSolution,
    Lsode2Solver,
    Lsode2Solution,
    LeastSquaresBSpline,
    NonlinearSolution,
    NonlinearSolver,
    PreparationError,
    PanicError,
    PoisonedSolverError,
    RadauCallbackSolver,
    RadauSolver,
    RadauSolution,
    SolverError,
    SolverOptions,
    SymbolicSystem,
    SymbolicFrontend,
    UnsupportedSymbolicOperationError,
    TaskDocument,
    TaskDocumentError,
    TelemetryError,
    TelemetryMode,
    UnsupportedOptionError,
    UniversalNonStiffSolver,
    VarProSolver,
    VarProSolution,
    UniversalFitting,
    UniversalFittingSolution,
    capabilities,
    animate_solution_2d,
    animate_solution_3d,
    animate_trajectory_2d,
    animate_trajectory_3d,
    fit_quality,
    indexed_symbols,
    lowess_smooth,
    least_squares_bspline,
    savgol_filter,
    spline_resample,
    symbols,
)


def test_animation_api_rejects_invalid_inputs_before_opening_a_window():
    valid_2d = np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    valid_3d = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    times = np.asarray([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match=r"shape \(samples, 2\)"):
        animate_trajectory_2d(valid_3d, times)
    with pytest.raises(ValueError, match=r"shape \(samples, 3\)"):
        animate_trajectory_3d(valid_2d, times)
    with pytest.raises(ValueError, match="timestamps"):
        animate_trajectory_2d(valid_2d, np.asarray([0.0, 0.0]))
    with pytest.raises(ValueError, match="speed"):
        animate_trajectory_2d(valid_2d, times, speed=0.0)
    with pytest.raises(ValueError, match="coordinate"):
        animate_trajectory_3d(
            np.asarray([[0.0, 0.0, np.inf], [1.0, 2.0, 3.0]]), times
        )
    with pytest.raises(ValueError, match="too large for playback"):
        animate_trajectory_2d(valid_2d, np.asarray([0.0, 2.0e9]))
    with pytest.raises(ValueError, match="coordinate range"):
        animate_trajectory_2d(
            np.asarray(
                [
                    [np.finfo(np.float32).max, 0.0],
                    [-np.finfo(np.float32).max, 0.0],
                ]
            ),
            times,
        )


def test_animation_api_rejects_worker_thread_before_opening_a_window():
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    times = np.asarray([0.0, 1.0], dtype=np.float64)

    # Bevy's desktop event loop is main-thread-bound on supported platforms.
    # This contract test must fail before any renderer/window is initialized.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(animate_trajectory_2d, points, times)
        with pytest.raises(RuntimeError, match="Python's main thread"):
            future.result()


def test_animation_facade_runs_one_isolated_child_and_preserves_array_layout(
    monkeypatch, capsys
):
    def fake_run(arguments, **kwargs):
        assert arguments[1:3] == ["-m", "rustedscithe._animation_worker"]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "check": False,
        }
        input_path = Path(arguments[arguments.index("--input") + 1])
        with np.load(input_path, allow_pickle=False) as trajectory:
            np.testing.assert_array_equal(
                trajectory["points"], np.asarray([[1.0, 2.0], [3.0, 4.0]])
            )
            np.testing.assert_array_equal(trajectory["times"], [0.0, 0.5])
        return subprocess.CompletedProcess(arguments, 0, "animation closed\n", "")

    monkeypatch.setattr(python_facade.subprocess, "run", fake_run)
    animate_trajectory_2d([[1.0, 2.0], [3.0, 4.0]], [0.0, 0.5])
    assert capsys.readouterr().out == "animation closed\n"


def test_animation_facade_restores_typed_child_input_errors(monkeypatch):
    def fake_run(arguments, **kwargs):
        error = {"type": "ValueError", "message": "invalid animation speed"}
        stderr = "RUSTEDSCITHE_ANIMATION_ERROR:" + json.dumps(error)
        return subprocess.CompletedProcess(arguments, 2, "", stderr)

    monkeypatch.setattr(python_facade.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="invalid animation speed"):
        animate_trajectory_2d([[0.0, 0.0]], [0.0])


def test_solution_animation_adapter_only_reads_selected_solution_columns():
    class DetachedSolution:
        def states_array(self):
            return np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def times_array(self):
            return np.asarray([0.0, 1.0])

    solution = DetachedSolution()
    with pytest.raises(ValueError, match="speed"):
        animate_solution_2d(solution, (0, 2), ("x", "z"), speed=0.0)
    with pytest.raises(ValueError, match="speed"):
        animate_solution_3d(solution, (0, 1, 2), speed=0.0)
    with pytest.raises(ValueError, match="outside the solution dimension"):
        animate_solution_2d(solution, (0, 3), speed=0.0)


def test_task_document_parse_preview_check_and_file_roundtrip(tmp_path):
    ivp_source = """
task
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
    document = TaskDocument.parse(ivp_source)
    assert document.kind == "ivp"
    assert document.source == ivp_source
    assert "[Task preview] IVP" in document.preview()
    assert "[Task check]" in document.check()
    assert "BDF" in document.preview()
    data = document.to_dict()
    assert data["dict_schema_version"] == 1
    assert data["kind"] == "ivp"
    assert data["solver"]["selection"]["method"] == "bdf"
    assert data["problem"]["equations"] == ["-1 * y"]
    assert data["problem"]["initial_conditions"] == {
        "t0": 0.0,
        "t_end": 0.1,
        "y0": [1.0],
    }
    assert data["continuation"] is None
    assert python_facade.task_dict(ivp_source) == data

    bvp_path = tmp_path / "sample_bvp.txt"
    bvp_path.write_text(
        """task
solver: BVP
strategy: Damped

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0
""",
        encoding="utf-8",
    )
    bvp = TaskDocument.from_file(str(bvp_path))
    assert bvp.kind == "bvp"
    assert "[Task preview] BVP" in bvp.preview()
    assert "[Task check]" in bvp.check()
    bvp_data = bvp.to_dict()
    assert bvp_data["solver"]["selection"]["solver"] == "bvp_damp"
    assert bvp_data["problem"]["boundary_conditions"]["y"] == [
        {"derivative_order": 0, "value": 1.0}
    ]
    assert python_facade.task_dict(bvp_path) == bvp_data


def test_task_document_reports_typed_parse_and_io_errors(tmp_path):
    with pytest.raises(TaskDocumentError, match="missing_solver|solver"):
        TaskDocument.parse("task\nmethod: BDF\n")
    with pytest.raises(TaskDocumentError, match="io:"):
        TaskDocument.from_file(str(tmp_path / "missing-task.txt"))


def test_task_document_preserves_source_positions_on_semantic_diagnostics():
    with pytest.raises(TaskDocumentError) as captured:
        TaskDocument.parse(
            """task
solver: IVP
method: BDF

equations
arg: t
y: -misspelled_gain*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
        )
    error = captured.value
    assert error.category == "configuration"
    assert error.line is not None
    assert error.column is not None
    assert error.line > 1
    assert error.column > 1


def test_task_document_runs_ivp_and_returns_detached_typed_result():
    document = TaskDocument.parse(
        """task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
    )
    result = document.run()
    assert isinstance(result, IvpTaskResult)
    assert result.times()
    assert result.times()[-1] == pytest.approx(0.1)
    assert result.states()[-1][0] == pytest.approx(math.exp(-0.1), rel=2e-3)
    assert result.status_code in {"finished", "unknown"}
    assert result.diagnostics()["solver"] == "ivp_task"
    assert result.diagnostics()["sample_count"] == str(len(result.times()))
    assert not result.has_continuation


def test_task_document_run_keeps_ivp_and_bvp_routes_explicit():
    document = TaskDocument.parse(
        """task
solver: BVP
strategy: Damped

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0
"""
    )
    with pytest.raises(TaskDocumentError, match="unsupported_route"):
        document.run()


def test_task_document_runs_bvp_sci_with_detached_mesh_and_telemetry(tmp_path):
    document = TaskDocument.parse(
        """task
solver: BVP_sci
frontend: ExprLegacy
execution: Lambdify
method: Dense

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0

solver_options
tolerance: 1e-2
max_iterations: 20
telemetry: detailed
"""
    )

    result = document.run_bvp()
    assert isinstance(result, BvpTaskResult)
    assert result.solver == "bvp_sci"
    assert result.frontend == "expr_legacy"
    assert result.execution == "lambdify"
    assert result.matrix_layout == "dense"
    assert result.status == "finished"
    assert result.status_code == "ok"
    assert result.mesh()[0] == pytest.approx(0.0)
    assert result.mesh()[-1] == pytest.approx(1.0)
    assert len(result.values()) == 1
    assert len(result.values()[0]) == len(result.mesh())
    assert result.continuation_info() == (1, 1, 0, 0, 0)
    assert result.telemetry_counters()["linear_solves"] >= 1
    assert result.telemetry_timings_ms()["full_solve_ms"] >= 0.0
    assert result.telemetry_timings_ms()["preparation_ms"] >= 0.0
    assert result.diagnostics()["mesh_nodes"] == str(len(result.mesh()))
    assert "BVP task telemetry" in result.telemetry_report()

    report = tmp_path / "bvp_telemetry.md"
    result.write_telemetry_report(str(report))
    assert "solver=bvp_sci" in report.read_text(encoding="utf-8")


def test_bvp_execution_failures_use_the_dedicated_typed_error():
    document = TaskDocument.parse(
        """task
solver: BVP_sci

equations
arg: x
unknowns: y
rhs: log(y)

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0

solver_options
max_newton_iterations: 1
"""
    )
    with pytest.raises(BvpError, match="callback") as captured:
        document.run_bvp()
    assert captured.value.stage == "task_run"
    assert captured.value.solver == "bvp_task"
    assert captured.value.route == "native"


def test_bvp_capabilities_expose_frontend_execution_layout_and_continuation_contract():
    damp = capabilities("bvp_damp")
    assert damp.frontends == ["expr_legacy", "atom_native"]
    assert damp.executions == ["lambdify", "aot"]
    assert damp.layouts == ["dense", "sparse", "banded"]
    assert damp.methods == ["damped", "naive", "frozen"]
    assert damp.supports_continuation

    scipy_like = capabilities("bvp_sci")
    assert scipy_like.frontends == ["expr_legacy", "atom_native"]
    assert scipy_like.layouts == ["dense", "sparse", "banded"]
    assert scipy_like.methods == ["scipy_like"]
    assert scipy_like.supports_continuation


def test_nonlinear_solver_exposes_prepared_symbolic_methods_and_telemetry():
    system = SymbolicSystem([Expr.parse("x^2 - 4")], ["x"])
    solver = NonlinearSolver(
        system,
        method="newton",
        frontend="atom_native",
        telemetry="detailed",
        tolerance=1e-10,
    )
    assert solver.frontend() == "atom-native"
    assert not solver.is_prepared()
    solver.prepare()
    assert solver.is_prepared()
    result = solver.solve([1.0])
    assert result.solution()[0] == pytest.approx(2.0, rel=1e-6)
    assert result.residual_norm < 1e-8
    assert result.telemetry_counters()["jacobian_evaluations"] >= 1
    assert "total_ms" in result.telemetry_timings_ms()


def test_nonlinear_solver_accepts_method_specific_options_and_rejects_typos():
    system = SymbolicSystem([Expr.parse("x^2 - 4")], ["x"])
    solver = NonlinearSolver(
        system,
        method="levenberg_marquardt",
        method_options={"lambda_init": 1e-2, "max_lambda": 1e4},
        method_flags={"diag_scaling": False},
    )
    result = solver.solve([1.0])
    assert result.solution()[0] == pytest.approx(2.0, rel=1e-6)

    with pytest.raises(UnsupportedOptionError, match="does not support option"):
        NonlinearSolver(
            system,
            method="levenberg_marquardt",
            method_options={"lamdba_init": 1e-2},
        )


def test_nonlinear_exposes_vector_and_enum_method_controls_without_silent_defaults():
    system = SymbolicSystem([Expr.parse("x - 2")], ["x"])
    minpack = NonlinearSolver(
        system,
        method="levenberg_marquardt_minpack",
        method_options={"mode": 2},
        method_vectors={"diag": [1.5]},
    )
    assert minpack.solve([0.5]).solution()[0] == pytest.approx(2.0, abs=1e-5)

    advanced = NonlinearSolver(
        system,
        method="nielsen_levenberg_marquardt_advanced",
        method_choices={
            "scaling_method": "more",
            "reduction_ratio_method": "more",
        },
    )
    assert advanced.solve([0.5]).solution()[0] == pytest.approx(2.0, abs=1e-5)

    with pytest.raises(InvalidConfigurationError, match="requires method_vectors"):
        NonlinearSolver(system, method="lm_minpack", method_options={"mode": 2})
    with pytest.raises(InvalidConfigurationError, match="scaling_method"):
        NonlinearSolver(
            system,
            method="nielsen_lm_advanced",
            method_choices={"scaling_method": "automatic"},
        )


_NONLINEAR_METHOD_CASES = [
    ("newton", {}, {}),
    ("damped_newton", {"max_line_search_steps": 12, "sufficient_decrease": 1e-4, "shrink_factor": 0.5}, {}),
    ("damped_newton_advanced", {"max_damping_iterations": 12, "sufficient_decrease": 1e-4, "shrink_factor": 0.5}, {}),
    ("levenberg_marquardt", {"lambda_init": 1e-2, "increase_factor": 8.0, "decrease_factor": 2.0, "min_lambda": 1e-12, "max_lambda": 1e8}, {"diag_scaling": True}),
    ("backtracking_levenberg_marquardt", {"lambda_init": 1e-2, "lambda_decrease": 0.2, "lambda_increase": 8.0, "alpha_min": 1e-6, "max_lambda": 1e8}, {}),
    ("levenberg_marquardt_minpack", {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "maxfev": 100, "mode": 1, "factor": 10.0}, {}),
    ("nielsen_levenberg_marquardt", {"tau": 1e-3, "nu_init": 2.0, "factor_up": 2.0, "factor_down": 0.5, "rho_threshold": 1e-4, "max_rejections": 8}, {"use_column_scaling": True}),
    ("nielsen_levenberg_marquardt_advanced", {"tau": 1e-3, "nu_init": 2.0, "factor_up": 2.0, "factor_down": 0.5, "rho_threshold": 1e-4, "max_rejections": 8}, {}),
    ("trust_region", {"delta_init": 1.0, "delta_max": 100.0, "eta": 0.1, "shrink_factor": 0.25, "expand_factor": 2.0}, {}),
    ("powell_dogleg", {"delta_init": 1.0, "delta_max": 100.0, "eta": 0.1, "shrink_factor": 0.25, "expand_factor": 2.0}, {"use_double_dogleg": True, "use_column_scaling": True}),
    ("trust_region_lm", {"step_bound": 10.0, "lambda_init": 0.0, "acceptance_threshold": 1e-4}, {"scale_diag": True}),
]


@pytest.mark.parametrize("method,method_options,method_flags", _NONLINEAR_METHOD_CASES)
@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
def test_nonlinear_full_method_frontend_and_options_contract(method, method_options, method_flags, frontend):
    """Every native nonlinear method is exercised through both symbolic frontends."""
    system = SymbolicSystem([Expr.parse("x - 2")], ["x"])
    solver = NonlinearSolver(
        system,
        method=method,
        frontend=frontend,
        execution_policy="parallel",
        min_work=1,
        telemetry="detailed",
        method_options=method_options,
        method_flags=method_flags,
        tolerance=1e-9,
        max_iterations=80,
        bounds={"x": (0.0, 3.0)},
    )
    assert solver.frontend() in {"expr-legacy", "atom-native"}
    result = solver.solve([0.5])
    assert result.solution()[0] == pytest.approx(2.0, abs=2e-5), (method, frontend, result.diagnostics())
    assert result.telemetry_counters()["residual_evaluations"] > 0


def test_nonlinear_parallel_parameter_rebind_and_named_partial_bounds_contract():
    system = SymbolicSystem([Expr.parse("rate*x - 2")], ["rate", "x"])
    solver = NonlinearSolver(
        system,
        method="damped_newton_advanced",
        frontend="atom_native",
        execution_policy="auto",
        min_work=2,
        parameters=["rate"],
        parameter_values=[1.0],
        bounds={"x": (0.0, None)},
        telemetry="counters",
    )
    for rate in [1.0, 2.0, 0.5, 4.0]:
        result = solver.solve([1.0], [rate])
        assert result.solution()[0] == pytest.approx(2.0 / rate, rel=1e-7)


def test_nonlinear_rejects_unknown_named_bounds_and_telemetry_modes():
    system = SymbolicSystem([Expr.parse("x - 1")], ["x"])
    with pytest.raises(InvalidConfigurationError, match="unknown variable"):
        NonlinearSolver(system, bounds={"missing": (0.0, 1.0)})
    with pytest.raises(InvalidConfigurationError, match="telemetry"):
        NonlinearSolver(system, telemetry="typo")


@pytest.mark.parametrize("method,frontend", [("lm", "expr_legacy"), ("lm", "atom_native"), ("gavin", "expr_legacy"), ("gavin", "atom_native"), ("varpro", "expr_legacy")])
def test_universal_fitting_facade_dispatches_native_routes(method, frontend):
    x = [index * 0.2 for index in range(12)]
    y = (
        [2.0 * math.exp(-value / 1.5) for value in x]
        if method == "varpro"
        else [1.7 * value + 0.4 for value in x]
    )
    solver = UniversalFitting(method)
    solver.with_data(x, y)
    solver.with_arg("x")
    solver.with_unknowns(["a", "b"] if method != "varpro" else ["tau"])
    solver.with_initial_guess([1.0, 0.0] if method != "varpro" else [1.0])
    solver.with_frontend(frontend)
    if method == "gavin":
        solver.with_gavin_telemetry(True)
    else:
        solver.with_telemetry("detailed")
    if method == "varpro":
        solver.with_basis_str("tau", "exp(-x/tau)")
        solver.with_equations_str(["1"])
    else:
        solver.with_equations_str(["a*x+b"])
    result = solver.build()
    assert isinstance(result, UniversalFittingSolution)
    assert result.method == method
    assert result.solution_map()
    assert solver.is_built()
    with pytest.raises(SolverError, match="one-shot"):
        solver.build()


def test_universal_fitting_preserves_method_specific_options_and_rejects_unsupported_routes():
    x = [index * 0.25 for index in range(12)]
    gavin = UniversalFitting("gavin")
    gavin.with_data(x, [2.0 * value + 1.0 for value in x])
    gavin.with_arg("x")
    gavin.with_equations_str(["a*x+b"])
    gavin.with_unknowns(["a", "b"])
    gavin.with_initial_guess([1.9, 0.9])
    gavin.with_gavin_solver_controls(1000, 1e-3, 1e-3, 1e-1)
    gavin.with_gavin_update_policy("marquardt")
    assert gavin.build().solution_map()["a"] == pytest.approx(2.0, rel=1e-5)

    varpro = UniversalFitting("varpro")
    varpro.with_frontend("atom_native")
    with pytest.raises(UnsupportedOptionError, match="symbolic_frontend"):
        varpro.build()

    unsupported_gavin = UniversalFitting("gavin")
    unsupported_gavin.with_f_tolerance(1e-8)
    with pytest.raises(UnsupportedOptionError, match="f_tolerance"):
        unsupported_gavin.build()

    with pytest.raises(InvalidConfigurationError, match="with_gavin_telemetry"):
        UniversalFitting("gavin").with_telemetry("counters")

    enabled_gavin = UniversalFitting("gavin")
    enabled_gavin.with_gavin_telemetry(True)
    enabled_gavin.with_data(x, [2.0 * value + 1.0 for value in x])
    enabled_gavin.with_arg("x")
    enabled_gavin.with_equations_str(["a*x+b"])
    enabled_gavin.with_unknowns(["a", "b"])
    enabled_gavin.with_initial_guess([1.9, 0.9])
    assert enabled_gavin.build().solution_map()["a"] == pytest.approx(2.0, rel=1e-5)


def test_universal_fitting_polynomial_shortcut_matches_native_lm_route():
    x = [index * 0.25 for index in range(10)]
    y = [2.0 * value * value + 3.0 * value + 1.0 for value in x]
    solver = UniversalFitting()
    solver.with_data(x, y)
    solver.with_polynomial(2, "x")
    solver.with_initial_guess([0.0, 0.0, 0.0])
    result = solver.build()
    assert result.method == "lm"
    assert result.solution_map()["c2"] == pytest.approx(2.0, rel=1e-7)


def test_standalone_fitting_long_rebind_series_and_one_shot_failure_contracts():
    solver = FittingSolver(
        [Expr.parse("a*x-y")], [1.0], variables=["a"], parameters=["x", "y"],
        frontend="expr_legacy", telemetry="counters",
    )
    for target in [1.0, 2.0, 0.5, 4.0, 3.0, 1.5]:
        result = solver.solve_with_parameters([1.0, target])
        assert result.solution()[0] == pytest.approx(target, rel=1e-7)
        assert result.telemetry_counters()["residual_evaluations"] > 0

    with pytest.raises(InvalidConfigurationError, match="positive"):
        FittingSolver(
            [Expr.parse("a-1")], [1.0], variables=["a"], positive_variables=["missing"]
        )

    invalid_gavin = GavinFittingSolver(
        [Expr.parse("a*x")], [0.0, 1.0], [1.0, 2.0], "x", ["a"], [1.0],
        weights=[1.0],
    )
    with pytest.raises(InvalidConfigurationError):
        invalid_gavin.solve()
    with pytest.raises(TypeError):
        GavinFittingSolver(
            [Expr.parse("a*x")], [0.0, 1.0], [1.0, 2.0], "x", ["a"], [1.0],
            telemetry="counters",
        )

    missing_basis = VarProSolver("x")
    missing_basis.with_parameters(["tau"])
    missing_basis.with_initial_parameters([1.0])
    missing_basis.with_data([0.0, 1.0], [1.0, 0.5])
    with pytest.raises(InvalidConfigurationError):
        missing_basis.solve()
    with pytest.raises(SolverError, match="one-shot"):
        missing_basis.solve()


def test_gavin_and_varpro_long_series_keep_route_local_statistics():
    x = [index * 0.2 for index in range(10)]
    for _ in range(5):
        gavin = GavinFittingSolver(
            [Expr.parse("a*x+b")], x, [1.7 * value + 0.4 for value in x],
            "x", ["a", "b"], [1.65, 0.35], telemetry=True,
        )
        gavin_result = gavin.solve()
        assert gavin_result.telemetry_counters()["objective_evaluations"] > 0

        varpro = VarProSolver("x")
        varpro.with_parameters(["tau"])
        varpro.with_initial_parameters([1.4])
        varpro.with_data(x, [2.0 * math.exp(-value / 1.5) for value in x])
        varpro.with_basis_str("tau", "exp(-x/tau)")
        varpro.with_telemetry("counters")
        varpro_result = varpro.solve()
        assert varpro_result.function_evaluations > 0
        assert varpro_result.telemetry_counters()["iterations"] >= 0


def test_fitting_solver_exposes_expr_atom_and_parameter_rebind():
    equations = [Expr.parse("a*x - y"), Expr.parse("a*z - w")]
    solver = FittingSolver(
        equations,
        [1.0],
        variables=["a"],
        parameters=["x", "y", "z", "w"],
        frontend="atom_native",
        execution_policy="parallel",
        min_work=1,
        telemetry="detailed",
    )
    result = solver.solve_with_parameters([2.0, 4.0, 3.0, 6.0])
    assert result.solution()[0] == pytest.approx(2.0, rel=1e-6)
    assert result.telemetry_counters()["jacobian_evaluations"] >= 1
    assert result.objective_function >= 0.0
    assert result.termination
    diagnostics = result.diagnostics()
    assert diagnostics["termination"] == result.termination
    assert int(diagnostics["iterations"]) >= 0
    assert int(diagnostics["residual_evaluations"]) >= 1


def test_gavin_fitting_exposes_statistics_and_native_telemetry():
    x = [index * 0.2 for index in range(10)]
    y = [1.7 * value + 0.4 for value in x]
    solver = GavinFittingSolver(
        [Expr.parse("a*x+b")],
        x,
        y,
        "x",
        ["a", "b"],
        [1.2, 0.2],
        frontend="atom_native",
        telemetry=True,
        preparation_telemetry="timings",
    )
    result = solver.solve()
    assert result.parameters()[0] == pytest.approx(1.7, rel=1e-6)
    assert result.parameters()[1] == pytest.approx(0.4, rel=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)
    assert result.parameter_correlation()
    assert result.telemetry_counters()["objective_evaluations"] >= 1
    assert result.telemetry_timings_ms()["total_ms"] >= 0.0
    assert solver.is_solved()
    with pytest.raises(SolverError, match="one-shot"):
        solver.solve()


def test_gavin_telemetry_is_binary_and_disabled_by_default():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [2.0 * value + 1.0 for value in x]
    disabled = GavinFittingSolver(
        [Expr.parse("a*x+b")], x, y, "x", ["a", "b"], [1.5, 0.5]
    ).solve()
    assert disabled.telemetry_counters() == {}
    assert disabled.telemetry_timings_ms() == {}

    enabled = GavinFittingSolver(
        [Expr.parse("a*x+b")], x, y, "x", ["a", "b"], [1.5, 0.5], telemetry=True
    ).solve()
    assert enabled.telemetry_counters()["objective_evaluations"] > 0
    assert "total_ms" in enabled.telemetry_timings_ms()


def test_varpro_exposes_r_squared_and_solution_map():
    x = [index * 0.25 for index in range(9)]
    y = [2.0 * math.exp(-value / 1.5) for value in x]
    solver = VarProSolver("x")
    solver.with_parameters(["tau"])
    solver.with_initial_parameters([1.0])
    solver.with_data(x, y)
    solver.with_basis("tau", Expr.parse("exp(-x/tau)"))
    solver.with_tolerances(1e-8, 1e-8, 1e-10)
    solver.with_max_iterations(200)
    solver.with_max_function_evaluations(500)
    solver.with_step_bound(10.0)
    solver.with_scale_diag(True)
    solver.with_telemetry("detailed")
    result = solver.solve()
    assert result.successful
    assert result.solution()["tau"] == pytest.approx(1.5, rel=2e-2)
    assert result.solution()["c0"] == pytest.approx(2.0, rel=2e-2)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.best_fit()
    assert result.objective_function >= 0.0
    assert result.function_evaluations > 0
    assert result.telemetry_counters()["iterations"] >= 0
    assert "total_ms" in result.telemetry_timings_ms()


def test_varpro_rejects_invalid_native_lm_settings_with_typed_errors():
    solver = VarProSolver("x")
    with pytest.raises(InvalidConfigurationError):
        solver.with_max_iterations(0)
    with pytest.raises(InvalidConfigurationError, match="telemetry"):
        solver.with_telemetry("verbose-ish")


def test_varpro_preserves_native_lm_exhaustion_error():
    x = [index * 0.1 for index in range(30)]
    solver = VarProSolver("x")
    solver.with_parameters(["tau"])
    solver.with_initial_parameters([8.0])
    solver.with_data(x, [2.0 * math.exp(-value / 0.5) for value in x])
    solver.with_basis_str("tau", "exp(-x/tau)")
    solver.with_tolerances(1e-16, 1e-16, 1e-16)
    solver.with_max_iterations(1)
    with pytest.raises(ExhaustionError) as captured:
        solver.solve()
    assert captured.value.native_kind


def test_nonlinear_capabilities_do_not_advertise_unwired_aot():
    nonlinear = capabilities("nonlinear_system")
    assert nonlinear.frontends == ["expr_legacy", "atom_native"]
    assert nonlinear.executions == ["lambdify"]
    assert nonlinear.methods == [
        "newton",
        "damped_newton",
        "damped_newton_advanced",
        "levenberg_marquardt",
        "backtracking_levenberg_marquardt",
        "levenberg_marquardt_minpack",
        "nielsen_levenberg_marquardt",
        "nielsen_levenberg_marquardt_advanced",
        "trust_region",
        "powell_dogleg",
        "trust_region_lm",
    ]
    fitting = capabilities("fitting")
    assert fitting.methods == ["levenberg_marquardt"]


def test_task_document_runs_prepared_ivp_continuation_with_segment_timings():
    document = TaskDocument.parse(
        """task
solver: IVP
method: BDF

equations
arg: t
parameters: rate
parameter_values: 1.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
rtol: 1e-5
atol: 1e-7
max_step: 0.02

continuation
parameter: rate
values: 0.5, 1.0, 2.0
mode: prepared
restart_each: false
"""
    )
    result = document.run()
    assert result.has_continuation
    info = result.continuation_info()
    assert info[0] == "rate"
    assert info[2] == "prepared"
    assert info[4:] == (1, 2)
    segments = result.continuation_segments()
    assert len(segments) == 3
    assert all(segment[3] >= 0.0 and segment[4] >= 0.0 for segment in segments)


def test_task_document_exposes_explicit_ivp_options_without_inventing_defaults():
    document = TaskDocument.parse(
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

solver_options
rtol: 1e-6
atol: 1e-9
max_step: 0.02
telemetry: timings
"""
    )
    options = document.options()
    assert options["rtol"] == "1e-6"
    assert options["atol"] == "1e-9"
    assert options["max_step"] == "0.02"
    assert options["telemetry"] == "Timings"
    assert "first_step" not in options
    structured = document.to_dict()["solver"]["options"]
    assert structured["rtol"] == pytest.approx(1e-6)
    assert structured["atol"] == pytest.approx(1e-9)
    assert structured["max_step"] == pytest.approx(0.02)
    assert structured["telemetry"] == "Timings"


def test_task_document_preserves_where_substitution_and_continuation_metadata():
    document = TaskDocument.parse(
        """task
schema_version: 1
solver: IVP
method: BDF

equations
arg: t
unknowns: y1, y2, y3
rhs: -forward + backward, forward - backward - decay, decay
parameters: k1, k2, k3
parameter_values: 0.04, 1.0e4, 3.0e7

where
forward: k1*y1
backward: k2*y2*y3
decay: k3*y2*y2

initial_conditions
t0: 0.0
t_end: 1.0
y0: [1.0, 0.0, 0.0]

continuation
parameters: k1, k3
k1_values: 0.02, 0.04
k3_values: 1.5e7, 3.0e7
mode: fresh
restart_policy: restart_with_state
monotonic: allow
y0_values: [1.0, 0.0, 0.0]
"""
    )
    assert document.kind == "ivp"
    assert document.route()["continuation"] == "true"
    assert document.options() == {}
    assert "0.04 * y1" in document.preview()
    continuation = document.to_dict()["continuation"]
    assert continuation["parameters"] == ["k1", "k3"]
    assert continuation["value_grid"] == [
        [0.02, 15000000.0],
        [0.02, 30000000.0],
        [0.04, 15000000.0],
        [0.04, 30000000.0],
    ]
    assert continuation["mode"] == "fresh"
    assert continuation["restart_policy"] == "restart_with_state"
    assert continuation["symbolic_equations"]


def test_task_document_exposes_explicit_bvp_options():
    document = TaskDocument.parse(
        """task
solver: BVP_sci
frontend: AtomNative
execution: Lambdify
method: Sparse

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0

solver_options
tolerance: 1e-4
max_nodes: 64
telemetry: counters
"""
    )
    options = document.options()
    assert options["tolerance"] == "0.0001"
    assert options["max_nodes"] == "64"
    assert options["telemetry"] == "Counters"
    assert document.route()["frontend"] == "atom_native"
    assert document.route()["matrix_layout"] == "sparse"


def test_task_document_exposes_normalized_dispatch_route_before_execution():
    stiff = TaskDocument.parse(
        """task
solver: IVP
method: Radau5

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
    )
    assert stiff.route() == {
        "family": "ivp",
        "method": "radau5",
        "dispatch": "native_stiff",
        "frontend": "default",
        "execution": "default",
        "matrix_layout": "default",
        "continuation": "false",
    }

    nonstiff = TaskDocument.parse(
        """task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"""
    )
    assert nonstiff.route()["method"] == "rk45"
    assert nonstiff.route()["dispatch"] == "universal_nonstiff"


@pytest.mark.parametrize("method", ["rk45", "dopri", "ab4"])
def test_ode_api2_nonstiff_methods_return_detached_trajectory(method):
    solver = UniversalNonStiffSolver(
        method, [Expr.parse("1")], ["y"], "t", [0.0], 0.0, 0.1, 0.01,
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            execution_policy="sequential", telemetry="off",
        ),
    )
    assert solver.method == method
    assert solver.provenance()["solver"] == "ode_api2_nonstiff"
    result = solver.solve()
    assert result.method == method
    assert result.status == "finished"
    # ODE_api2's fixed-step loop stops after crossing the requested bound.
    assert 0.1 <= result.times()[-1] < 0.11 + 1e-12
    assert result.states()[-1][0] == pytest.approx(result.times()[-1], abs=1e-10)
    assert result.states_array().shape == (result.sample_count, 1)
    assert result.diagnostics()["telemetry"] == "not_provided_by_native_route"


def test_ode_api2_nonstiff_facade_and_stop_condition():
    result = python_facade.solve_ivp(
        "rk45", [Expr.parse("1")], ["y"], "t", np.array([0.0]),
        0.0, 1.0, 0.01, stop_condition={"y": 0.05}, neighborhood_check=0.005,
    )
    assert result.status == "stopped_by_condition"
    assert result.states()[-1][0] == pytest.approx(0.05, abs=1e-12)
    assert capabilities("nonstiff").methods == ["rk45", "dopri", "ab4"]
    assert capabilities("nonstiff").supports_continuation is False
    with pytest.raises(UnsupportedOptionError, match="does not expose symbolic parameter"):
        python_facade.create_ivp(
            "rk45", [Expr.parse("rate")], ["y"], "t", [0.0], 0.0, 0.1, 0.01,
            parameters=["rate"], parameter_values=[1.0],
        )


def test_ode_api2_nonstiff_rejects_unsupported_options_and_invalid_inputs():
    base = ([Expr.parse("1")], ["y"], "t", [0.0], 0.0, 0.1, 0.01)
    with pytest.raises(UnsupportedOptionError, match="expected rk45, dopri or ab4"):
        UniversalNonStiffSolver("rk4", *base)
    with pytest.raises(UnsupportedOptionError, match="ExprLegacy, Lambdify, Dense and Sequential"):
        UniversalNonStiffSolver(
            "rk45", *base, options=SolverOptions(frontend="atom_native", telemetry="off")
        )
    with pytest.raises(UnsupportedOptionError, match="does not provide native telemetry"):
        UniversalNonStiffSolver("rk45", *base, options=SolverOptions(telemetry="timings"))
    with pytest.raises(InvalidConfigurationError, match="same non-zero length"):
        UniversalNonStiffSolver("rk45", [Expr.parse("1")], ["y", "z"], "t", [0.0], 0.0, 0.1, 0.01)
    with pytest.raises(InvalidConfigurationError, match="forward-only"):
        UniversalNonStiffSolver("rk45", *base[:4], 1.0, 0.0, 0.01)
    with pytest.raises(InvalidConfigurationError, match="undeclared state"):
        UniversalNonStiffSolver("rk45", *base, stop_condition={"z": 1.0})


def test_bvp_task_route_preserves_frontend_execution_and_layout():
    document = TaskDocument.parse(
        """task
solver: BVP_sci
frontend: AtomViewNative
execution: Lambdify
method: Sparse

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 4

initial_guess
y: 0.0
"""
    )
    route = document.route()
    assert route["family"] == "bvp"
    assert route["solver"] == "bvp_sci"
    assert route["frontend"] == "atom_native"
    assert route["execution"] == "lambdify"
    assert route["matrix_layout"] == "sparse"


@pytest.mark.parametrize("method", ["BackwardEuler", "Radau5", "LSODE2"])
def test_task_document_runs_all_stiff_ivp_families(method):
    solver_options = {
        "BackwardEuler": "step_size: 1e-3\ntolerance: 1e-8\nmax_iterations: 100",
        "Radau5": "rtol: 1e-6\natol: 1e-8\nmax_step: 0.05",
        "LSODE2": (
            "rtol: 1e-6\natol: 1e-8\nmax_step: 0.05\n"
            "lsode2_symbolic_execution: LambdifyExpr\n"
            "lsode2_linear_structure: dense\n"
            "lsode2_linear_solver_policy: auto"
        ),
    }[method]
    document = TaskDocument.parse(
        f"""task
solver: IVP
method: {method}

equations
arg: t
y: -10.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
{solver_options}
"""
    )
    result = document.run()
    assert result.times()
    assert result.states()
    assert result.status_code in {"finished", "unknown"}


def test_expr_workflow_matches_readme():
    x = Expr.variable("x")
    y = Expr.variable("y")
    expression = x.pow(Expr.constant(2.0)) + y

    assert expression.variables() == ["x", "y"]
    assert expression.evaluate([2.0, 3.0]) == pytest.approx(7.0)
    assert expression.diff("x").evaluate([2.0]) == pytest.approx(4.0)
    assert expression.substitute("y", 5.0).evaluate([2.0]) == pytest.approx(9.0)


def test_exposed_symbolic_operations_are_native_and_composable():
    x = Expr.variable("x")
    replacement = Expr.constant(2.0)

    assert (x ** replacement).evaluate([3.0]) == pytest.approx(9.0)
    assert x.substitute_expr("x", replacement).evaluate([]) == pytest.approx(2.0)
    assert x.exp().evaluate([0.0]) == pytest.approx(1.0)
    assert x.ln().evaluate([1.0]) == pytest.approx(0.0)
    assert x.sin().evaluate([0.0]) == pytest.approx(0.0)
    assert x.cos().evaluate([0.0]) == pytest.approx(1.0)
    assert x.tan().evaluate([0.0]) == pytest.approx(0.0)
    assert x.asin().evaluate([0.0]) == pytest.approx(0.0)
    assert x.acos().evaluate([1.0]) == pytest.approx(0.0)
    assert x.atan().evaluate([0.0]) == pytest.approx(0.0)
    assert x.acot().evaluate([1.0]) == pytest.approx(0.7853981633974483)
    assert x.arcsin().evaluate([0.0]) == pytest.approx(0.0)
    assert x.arccos().evaluate([1.0]) == pytest.approx(0.0)
    assert x.arctg().evaluate([0.0]) == pytest.approx(0.0)
    assert x.arcctg().evaluate([1.0]) == pytest.approx(0.7853981633974483)
    assert Expr.parse("(x + 0) * 1").simplify().evaluate([3.0]) == pytest.approx(3.0)


def test_scalar_expression_numeric_operators_and_reflected_forms():
    x = Expr.variable("x")
    expression = 2 * x + 1 / x - (x - 3) ** 2

    assert expression.evaluate([2.0]) == pytest.approx(3.5)
    assert (x + 1).evaluate([2.0]) == pytest.approx(3.0)
    assert (1 - x).evaluate([2.0]) == pytest.approx(-1.0)
    assert (x * 0.5).evaluate([2.0]) == pytest.approx(1.0)
    with pytest.raises(InvalidExpressionError, match="finite"):
        _ = x + float("inf")
    with pytest.raises(TypeError, match="boolean"):
        _ = x + True


def test_symbol_batches_renaming_pretty_print_and_scalar_derivatives():
    x, y = Expr.symbols(["x", "y"])
    x0, x1, x2 = Expr.indexed("x", 3)
    assert [str(item) for item in symbols(["a", "b"])] == ["a", "b"]
    assert [str(item) for item in indexed_symbols("u", 2)] == ["u0", "u1"]
    expression = x**2 + y * x0 + x1 - x2

    assert [item.variables() for item in (x, y, x0, x1, x2)] == [
        ["x"], ["y"], ["x0"], ["x1"], ["x2"]
    ]
    assert expression.variables() == ["x", "x0", "x1", "x2", "y"]
    assert expression.contains_variable("x")
    assert not expression.contains_variable("missing")
    renamed = expression.rename_variable("x", "z")
    assert renamed.variables() == ["x0", "x1", "x2", "y", "z"]
    assert expression.variables() == ["x", "x0", "x1", "x2", "y"]
    assert "z" in renamed.pretty_print()

    polynomial = Expr.parse("x^3 + y*x")
    assert polynomial.diff_n("x", 2).evaluate([2.0]) == pytest.approx(12.0)
    gradient = polynomial.gradient()
    assert gradient[0].evaluate([2.0, 4.0]) == pytest.approx(16.0)
    assert gradient[1].evaluate([2.0]) == pytest.approx(2.0)
    taylor = Expr.parse("x^2 + y").taylor_series("x", 1.0, 2)
    assert taylor.evaluate([1.25, 3.0]) == pytest.approx(4.5625)
    assert Expr.variable("x").log10().evaluate([10.0]) == pytest.approx(1.0)
    assert Expr.variable("x").cot().evaluate([1.0]) == pytest.approx(1.0 / np.tan(1.0))

    with pytest.raises(InvalidExpressionError, match="more than once"):
        Expr.symbols(["x", "x"])
    with pytest.raises(InvalidExpressionError, match="must not exceed"):
        polynomial.taylor_series("x", 0.0, 33)


def test_symbolic_indefinite_integration_and_typed_unsupported_form():
    x = Expr.variable("x")
    primitive = (x**3 + 2 * x).integrate("x")
    assert primitive.diff("x").simplify().evaluate([1.75]) == pytest.approx(
        (x**3 + 2 * x).evaluate([1.75])
    )
    assert Expr.parse("1/x").integrate("x").variables() == ["x"]

    with pytest.raises(UnsupportedSymbolicOperationError) as error:
        Expr.parse("exp(x^2)").integrate("x")
    assert error.value.stage == "symbolic_integration"
    assert error.value.category == "unsupported_symbolic_operation"
    assert error.value.operation == "integrate"


def test_supported_symbolic_primitives_differentiate_back_to_integrands():
    cases = [
        ("x^3 + 2*x", [-0.7, 0.3, 1.4]),
        ("exp(x)", [-0.7, 0.3, 1.4]),
        ("sin(x)", [-0.7, 0.3, 1.4]),
        ("cos(x)", [-0.7, 0.3, 1.4]),
        ("1/x", [0.25, 0.7, 2.0]),
    ]

    for source, samples in cases:
        integrand = Expr.parse(source)
        derivative = integrand.integrate("x").diff("x").simplify()
        for value in samples:
            assert derivative.evaluate([value]) == pytest.approx(
                integrand.evaluate([value]), rel=1e-11, abs=1e-12
            ), f"d/dx(integrate({source!r})) disagrees at x={value}"


def test_integral_of_reciprocal_documents_positive_real_branch():
    reciprocal = Expr.parse("1/x")
    primitive = reciprocal.integrate("x")

    # The native primitive is ln(x), whose real-valued domain is x > 0.
    # Its symbolic derivative agrees with 1/x on that connected domain.
    derivative = primitive.diff("x").simplify()
    for value in [0.25, 0.7, 2.0]:
        assert derivative.evaluate([value]) == pytest.approx(1.0 / value)

    # Do not silently reinterpret ln(x) as ln(abs(x)); crossing x=0 also
    # crosses a singularity, so the API leaves branch/domain handling explicit.
    with pytest.raises(InvalidExpressionError, match="non-finite"):
        primitive.evaluate([-1.0])


def test_simpson_integration_is_explicit_and_validates_panic_inputs():
    x = Expr.variable("x")
    quadratic = x**2
    assert quadratic.integrate_simpson(0.0, 1.0, 100) == pytest.approx(1.0 / 3.0)
    assert quadratic.integrate_simpson(1.0, 0.0, 100) == pytest.approx(-1.0 / 3.0)

    for subdivisions in [0, 3]:
        with pytest.raises(InvalidConfigurationError, match="positive even integer"):
            quadratic.integrate_simpson(0.0, 1.0, subdivisions)
    with pytest.raises(InvalidConfigurationError, match="bounds must be finite"):
        quadratic.integrate_simpson(0.0, float("inf"), 100)
    with pytest.raises(InvalidConfigurationError, match="at most one free variable"):
        (x + Expr.variable("y")).integrate_simpson(0.0, 1.0, 100)
    with pytest.raises(InvalidExpressionError, match="non-finite result"):
        (1.0 / x).integrate_simpson(-1.0, 1.0, 100)


def test_gauss_legendre_integration_is_unweighted_and_validated():
    x = Expr.variable("x")
    quartic = x**4
    assert quartic.integrate_gauss_legendre(-1.0, 1.0, 3) == pytest.approx(
        2.0 / 5.0, abs=1e-13
    )
    assert quartic.integrate_gauss_legendre(1.0, -1.0, 3) == pytest.approx(
        -2.0 / 5.0, abs=1e-13
    )

    with pytest.raises(InvalidConfigurationError, match="degree must be between 1"):
        quartic.integrate_gauss_legendre(0.0, 1.0, 0)
    with pytest.raises(InvalidConfigurationError, match="bounds must be finite"):
        quartic.integrate_gauss_legendre(0.0, float("inf"), 4)
    with pytest.raises(InvalidConfigurationError, match="at most one free variable"):
        (x + Expr.variable("y")).integrate_gauss_legendre(0.0, 1.0, 4)
    with pytest.raises(InvalidExpressionError, match="non-finite result"):
        Expr.parse("exp(1000*x)").integrate_gauss_legendre(0.0, 1.0, 8)


def test_weighted_gaussian_quadrature_uses_documented_weights_and_checks_inputs():
    one = Expr.constant(1.0)
    x = Expr.variable("x")
    assert one.integrate_gauss_hermite(8) == pytest.approx(math.sqrt(math.pi))
    assert (x**3).integrate_gauss_laguerre(4, alpha=0.0) == pytest.approx(6.0)
    assert (x**2).integrate_gauss_laguerre(4, alpha=1.0) == pytest.approx(6.0)

    with pytest.raises(InvalidConfigurationError, match="degree must be between 1"):
        one.integrate_gauss_hermite(0)
    with pytest.raises(InvalidConfigurationError, match="degree must be between 1"):
        one.integrate_gauss_laguerre(4097)
    with pytest.raises(InvalidConfigurationError, match="alpha must be finite and greater than -1"):
        one.integrate_gauss_laguerre(4, alpha=-1.0)
    with pytest.raises(InvalidConfigurationError, match="alpha must be finite and greater than -1"):
        one.integrate_gauss_laguerre(4, alpha=float("nan"))
    with pytest.raises(InvalidConfigurationError, match="at most one free variable"):
        (x + Expr.variable("y")).integrate_gauss_hermite(8)
    with pytest.raises(InvalidExpressionError, match="non-finite result"):
        Expr.parse("exp(1000*x)").integrate_gauss_hermite(8)
    with pytest.raises(InvalidExpressionError, match="non-finite result"):
        Expr.parse("exp(1000*x)").integrate_gauss_laguerre(8, alpha=0.0)


def test_symbolic_simplification_contract_does_not_preserve_domain_singularities():
    x = Expr.variable("x")
    original = x / x
    simplified = original.simplify()

    # Keep this behavior explicit: simplification does not carry domain assumptions.
    assert str(simplified) == "1"
    assert simplified.evaluate([]) == pytest.approx(1.0)
    with pytest.raises(InvalidExpressionError, match="non-finite"):
        original.evaluate([0.0])


def test_symbolic_vectors_and_matrices_validate_shapes_and_return_owned_numpy():
    x, y = Expr.symbols(["x", "y"])
    vector = ExprVector([x + y, x * y])
    assert len(vector) == 2
    assert vector.variables() == ["x", "y"]
    np.testing.assert_allclose(vector.evaluate([2.0, 3.0]), [5.0, 6.0])
    assert vector.dot(ExprVector([x, y])).evaluate([2.0, 3.0]) == pytest.approx(28.0)
    assert vector.diff("x").evaluate([3.0]).tolist() == pytest.approx([1.0, 3.0])
    with pytest.raises(IndexError):
        _ = vector[2]
    with pytest.raises(InvalidConfigurationError, match="lengths differ"):
        vector.dot(ExprVector([x]))

    matrix = ExprMatrix([[x, y], [x * y, x**2]])
    assert matrix.shape() == (2, 2)
    assert matrix.variables() == ["x", "y"]
    np.testing.assert_allclose(matrix.evaluate([2.0, 3.0]), [[2.0, 3.0], [6.0, 4.0]])
    np.testing.assert_allclose(
        matrix.transpose().evaluate([2.0, 3.0]), [[2.0, 6.0], [3.0, 4.0]]
    )
    np.testing.assert_allclose(
        matrix.substitute_expr("y", x).evaluate([2.0]), [[2.0, 2.0], [4.0, 4.0]]
    )
    np.testing.assert_allclose(
        matrix.matvec(ExprVector([x, y])).evaluate([2.0, 3.0]), [13.0, 24.0]
    )
    right = ExprMatrix([[Expr.constant(1.0), y], [x, Expr.constant(2.0)]])
    np.testing.assert_allclose(
        matrix.matmul(right).evaluate([2.0, 3.0]), [[8.0, 12.0], [14.0, 26.0]]
    )
    np.testing.assert_allclose(
        matrix.hadamard(right).evaluate([2.0, 3.0]), [[2.0, 9.0], [12.0, 8.0]]
    )
    assert matrix.trace().evaluate([2.0]) == pytest.approx(6.0)
    assert matrix.determinant().evaluate([2.0, 3.0]) == pytest.approx(-10.0)
    with pytest.raises(IndexError):
        matrix.get(2, 0)
    with pytest.raises(InvalidConfigurationError, match="same length"):
        ExprMatrix([[x], [x, y]])
    with pytest.raises(InvalidConfigurationError, match="size 1 through 3"):
        ExprMatrix.identity(4).determinant()
    with pytest.raises(InvalidConfigurationError, match="matrix-product dimensions"):
        matrix.matmul(ExprMatrix.identity(3))
    with pytest.raises(InvalidConfigurationError, match="Hadamard-product shapes"):
        matrix.hadamard(ExprMatrix.identity(3))
    assert ExprMatrix.zeros(0, 3).evaluate([]).shape == (0, 3)
    assert ExprMatrix.zeros(2, 0).evaluate([]).shape == (2, 0)
    assert ExprMatrix.zeros(0, 2).matmul(ExprMatrix.zeros(2, 3)).evaluate([]).shape == (0, 3)
    assert ExprMatrix.zeros(0, 3).hadamard(ExprMatrix.zeros(0, 3)).evaluate([]).shape == (0, 3)


def test_lambdified_expression_evaluates_numpy_batches_with_explicit_axes():
    x, y = Expr.symbols(["x", "y"])
    compiled = (x + 2 * y).lambdify(["y", "x"], diagnostics=True)
    samples = np.array([[3.0, 2.0], [5.0, 4.0]])

    assert compiled.evaluate_batch(samples) == pytest.approx(np.array([8.0, 14.0]))
    strided_samples = np.array([[3.0, -1.0, 2.0], [5.0, -1.0, 4.0]])[:, ::2]
    assert compiled.evaluate_batch(strided_samples) == pytest.approx(np.array([8.0, 14.0]))
    assert compiled.preparation_ms >= 0.0
    assert compiled.diagnostics_enabled
    assert compiled.diagnostics()["evaluations"] == 4
    with pytest.raises(InvalidExpressionError, match="columns"):
        compiled.evaluate_batch(np.ones((2, 1)))


def test_where_substitution_resolves_nested_aliases_independent_of_mapping_order():
    x = Expr.variable("x")
    y = Expr.variable("y")
    rate = Expr.variable("rate")
    scaled = Expr.variable("scaled")

    expanded = (scaled + y).where_substitute(
        {"scaled": rate * x, "rate": Expr.constant(2.0)}
    )
    assert expanded.evaluate([3.0, 4.0]) == pytest.approx(10.0)

    system = SymbolicSystem([scaled + y, rate * x], ["scaled", "rate", "x", "y"])
    expanded_system = system.where_substitute(
        {"scaled": rate * x, "rate": Expr.constant(2.0)}
    )
    assert expanded_system.variables() == ["x", "y"]
    assert expanded_system.evaluate([3.0, 4.0]) == pytest.approx([10.0, 6.0])

    # The same dependency graph is applied to every residual in one prepared pass.
    many_rows = SymbolicSystem([scaled + y, scaled * x, scaled - rate])
    rebound = many_rows.substitute_many(
        [("scaled", rate * x), ("rate", Expr.constant(2.0))]
    )
    assert rebound.variables() == ["x", "y"]
    assert rebound.evaluate([3.0, 4.0]) == pytest.approx([10.0, 18.0, 4.0])


def test_where_substitution_rejects_cycles_and_duplicate_list_names():
    x = Expr.variable("x")
    a = Expr.variable("a")
    b = Expr.variable("b")

    with pytest.raises(InvalidExpressionError, match="cyclic"):
        x.where_substitute({"a": b, "b": a})
    with pytest.raises(InvalidExpressionError, match="more than once"):
        x.substitute_many([("a", Expr.constant(1.0)), ("a", Expr.constant(2.0))])


def test_lambdify_compiles_once_and_keeps_explicit_input_order():
    x = Expr.variable("x")
    y = Expr.variable("y")
    compiled = (x + y).lambdify(["y", "x"])

    assert compiled.variables() == ["y", "x"]
    assert compiled.evaluate([3.0, 2.0]) == pytest.approx(5.0)
    assert "x" in compiled.expression
    assert compiled.preparation_ms is None
    assert not compiled.diagnostics_enabled
    assert compiled.diagnostics()["evaluations"] == 0

    with pytest.raises(InvalidExpressionError):
        (x + y).lambdify(["x"])
    with pytest.raises(InvalidExpressionError):
        (x + y).lambdify(["x", "x", "y"])


def test_expr_legacy_and_atom_native_standalone_callables_match_with_diagnostics():
    x, y = Expr.symbols(["x", "y"])
    expression = x.sin() * y + x**2
    legacy = expression.lambdify(["y", "x"], frontend=SymbolicFrontend.ExprLegacy)
    atom = expression.lambdify(
        ["y", "x"], frontend=SymbolicFrontend.AtomNative, diagnostics=True
    )

    point = [0.25, 1.5]
    assert atom.frontend == "atom_native"
    assert legacy.frontend == "expr_legacy"
    assert atom.evaluate(point) == pytest.approx(legacy.evaluate(point))
    assert atom.preparation_ms is not None and atom.preparation_ms >= 0.0
    assert atom.diagnostics_enabled
    assert atom.diagnostics()["evaluations"] == 1
    assert atom.diagnostics()["callback_time_ms"] >= 0.0

    system = SymbolicSystem([expression, expression.diff("x")], ["x", "y"])
    prepared = system.lambdify(frontend=SymbolicFrontend.AtomNative, diagnostics=True)
    residual = prepared.evaluate([0.25, 1.5])
    jacobian = prepared.evaluate_jacobian([0.25, 1.5])
    assert prepared.frontend == "atom_native"
    assert prepared.diagnostics_enabled
    assert prepared.preparation_ms is not None and prepared.preparation_ms >= 0.0
    assert len(residual) == 2 and len(jacobian) == 2
    assert prepared.diagnostics()["evaluations"] == 6


def test_symbolic_system_exposes_residual_jacobian_and_reusable_vector_callable():
    x = Expr.variable("x")
    y = Expr.variable("y")
    system = SymbolicSystem([x.pow(Expr.constant(2.0)) + y, x * y], ["x", "y"])

    assert system.variables() == ["x", "y"]
    assert system.evaluate([2.0, 3.0]) == pytest.approx([7.0, 6.0])
    jacobian = system.jacobian()
    assert jacobian[0][0].evaluate([2.0]) == pytest.approx(4.0)
    assert jacobian[0][1].evaluate([]) == pytest.approx(1.0)
    assert jacobian[1][0].evaluate([3.0]) == pytest.approx(3.0)
    assert jacobian[1][1].evaluate([2.0]) == pytest.approx(2.0)

    compiled = system.lambdify()
    assert compiled.variables() == ["x", "y"]
    assert compiled.evaluate([2.0, 3.0]) == pytest.approx([7.0, 6.0])
    compiled_jacobian = compiled.evaluate_jacobian([2.0, 3.0])
    assert compiled_jacobian[0] == pytest.approx([4.0, 1.0])
    assert compiled_jacobian[1] == pytest.approx([3.0, 2.0])


def test_detached_solution_data_is_not_mutable_through_returned_lists():
    y = Expr.variable("y")
    solver = BdfSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        options=SolverOptions(telemetry="off"),
    )
    result = solver.solve()

    times = result.times()
    states = result.states()
    times[0] = 99.0
    states[0][0] = 99.0

    assert result.times()[0] == pytest.approx(0.0)
    assert result.states()[0][0] == pytest.approx(1.0)
    assert result.telemetry_mode == "off"
    assert result.telemetry_counters() == {}
    assert result.telemetry_timings_ms() == {}


def test_bdf_symbolic_dense_lambdify_vertical_slice_returns_detached_solution():
    y = Expr.variable("y")
    solver = BdfSolver(
        [-y],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.5,
        0.1,
        rtol=1e-5,
        atol=1e-7,
        options=SolverOptions(telemetry="counters"),
    )

    assert not solver.is_prepared()
    solver.prepare()
    assert solver.is_prepared()
    result = solver.solve()
    assert result.status == "finished"
    assert result.times()[0] == pytest.approx(0.0)
    assert result.times()[-1] == pytest.approx(0.5)
    assert result.states()[-1][0] == pytest.approx(math.exp(-0.5), rel=2e-3)
    assert "residual_calls=" in solver.telemetry_report()
    assert result.telemetry_counters()["accepted_steps"] > 0
    assert result.telemetry_timings_ms() == {}
    assert result.telemetry_mode == "counters"
    assert "telemetry=counters" in result.telemetry_report()
    assert result.diagnostics()["solver"] == "bdf"
    assert result.diagnostics()["status"] == "finished"
    assert result.diagnostics()["state_dimension"] == "1"
    assert solver.provenance() == {
        "solver": "bdf",
        "frontend": "expr_legacy",
        "execution": "lambdify",
        "matrix_layout": "dense",
        "execution_policy": "sequential",
        "telemetry": "counters",
        "first_step": "auto",
        "max_bdf_order": "5",
        "max_steps": "1000000",
    }

    restarted = solver.restart(0.5, [1.0], 1.0)
    assert restarted.status == "finished"
    assert restarted.times()[0] == pytest.approx(0.5)
    assert restarted.times()[-1] == pytest.approx(1.0)
    assert restarted.states()[-1][0] == pytest.approx(math.exp(-0.5), rel=2e-3)


def test_bdf_exposes_native_step_controls_and_validates_them():
    y = Expr.variable("y")
    solver = BdfSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        first_step=0.01,
        max_bdf_order=2,
        max_steps=100,
        options=SolverOptions(telemetry="counters"),
    )
    result = solver.solve()
    assert result.status == "finished"
    assert solver.provenance()["first_step"] == "0.01"
    assert solver.provenance()["max_bdf_order"] == "2"
    assert solver.provenance()["max_steps"] == "100"

    with pytest.raises(InvalidConfigurationError, match="first_step"):
        BdfSolver(
            [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            first_step=0.0,
        )
    with pytest.raises(InvalidConfigurationError, match="max_bdf_order"):
        BdfSolver(
            [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            max_bdf_order=6,
        )
    with pytest.raises(InvalidConfigurationError, match="max_steps"):
        BdfSolver(
            [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            max_steps=0,
        )


def test_bdf_rejects_undeclared_rhs_symbols_before_preparation():
    y = Expr.variable("y")
    unknown = Expr.variable("unknown")
    with pytest.raises(InvalidConfigurationError, match="undeclared symbol"):
        BdfSolver([unknown - y], ["y"], "t", [1.0], 0.0, 1.0, 0.1)

    with pytest.raises(InvalidExpressionError):
        SymbolicSystem([])


def test_radau_exposes_native_step_budgets_and_validates_them():
    y = Expr.variable("y")
    solver = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.2, 0.1,
        first_step=0.05,
        max_steps=100,
        max_newton_iterations=4,
        max_retries=8,
    )
    solution = solver.solve([1.0])
    assert solution.final_time == pytest.approx(0.2)
    assert solver.provenance()["first_step"] == "0.05"
    assert solver.provenance()["max_steps"] == "100"
    assert solver.provenance()["max_newton_iterations"] == "4"
    assert solver.provenance()["max_retries"] == "8"

    with pytest.raises(InvalidConfigurationError, match="first_step"):
        RadauSolver(
            [-y], ["y"], "t", 0.0, 0.2, 0.1,
            first_step=0.0,
        )
    with pytest.raises(InvalidConfigurationError, match="first_step"):
        RadauSolver(
            [-y], ["y"], "t", 0.0, 0.2, 0.1,
            first_step=0.3,
        )
    with pytest.raises(InvalidConfigurationError, match="max_steps"):
        RadauSolver(
            [-y], ["y"], "t", 0.0, 0.2, 0.1,
            max_steps=0,
        )


def test_radau_callback_solver_supports_analytic_and_fd_jacobians():
    def residual(time, state):
        return [-state[0] + 0.0 * time]

    def jacobian(_time, _state):
        return [[-1.0]]

    analytic = RadauCallbackSolver(
        1, 0.0, 0.2, 0.1, residual,
        jacobian_callback=jacobian,
        options=SolverOptions(telemetry="counters"),
        output="dense",
    )
    analytic_solution = analytic.solve([1.0])
    assert analytic_solution.final_time == pytest.approx(0.2)
    assert analytic_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=3e-3)
    assert analytic.provenance()["jacobian_source"] == "analytic"
    assert analytic_solution.sample(0.1)[0] == pytest.approx(math.exp(-0.1), rel=3e-3)

    finite_difference = RadauCallbackSolver(
        1, 0.0, 0.2, 0.1, residual,
        options=SolverOptions(telemetry="counters"),
    )
    fd_solution = finite_difference.solve([1.0])
    assert fd_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=3e-3)
    assert finite_difference.provenance()["jacobian_source"] == "finite_difference"

    with pytest.raises(CallbackError, match="residual_callback"):
        RadauCallbackSolver(
            1, 0.0, 0.2, 0.1,
            lambda _time, _state: [math.nan],
        ).solve([1.0])

    def raising_residual(_time, _state):
        raise LookupError("radau callback cause")

    with pytest.raises(CallbackError, match="radau callback cause") as callback_error:
        RadauCallbackSolver(1, 0.0, 0.2, 0.1, raising_residual).solve([1.0])
    assert isinstance(callback_error.value.__cause__, LookupError)
    assert str(callback_error.value.__cause__) == "radau callback cause"
    assert callback_error.value.stage == "solve"
    assert callback_error.value.solver == "radau5"
    assert callback_error.value.route == "python_callback"

    with pytest.raises(UnsupportedOptionError, match="matrix_layout"):
        RadauCallbackSolver(
            1, 0.0, 0.2, 0.1, residual,
            options=SolverOptions(matrix_layout="sparse"),
        )

    radau_capabilities = capabilities("radau")
    assert radau_capabilities.supports_analytical_callbacks
    assert radau_capabilities.analytical_callback_layouts == ["dense"]


def test_stiff_ivp_step_budget_is_exposed_as_typed_exhaustion():
    y = Expr.variable("y")
    with pytest.raises(ExhaustionError) as bdf_exhaustion:
        BdfSolver(
            [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            max_steps=1,
        ).solve()
    assert bdf_exhaustion.value.native_kind == "MaxStepsExceeded"
    assert bdf_exhaustion.value.solver == "bdf"

    with pytest.raises(ExhaustionError) as radau_exhaustion:
        RadauSolver(
            [-y], ["y"], "t", 0.0, 0.2, 0.1,
            max_steps=1,
        ).solve([1.0])
    assert radau_exhaustion.value.native_kind == "StepBudget"


def test_bdf_callback_solver_supports_analytic_and_fd_jacobians():
    residual = lambda _t, state: [-state[0]]
    analytic_jacobian = lambda _t, _state: [[-1.0]]

    for jacobian, expected_source in (
        (analytic_jacobian, "analytic"),
        (None, "finite_difference"),
    ):
        solver = BdfCallbackSolver(
            1,
            0.0,
            0.5,
            0.1,
            [1.0],
            residual,
            jacobian,
            options=SolverOptions(telemetry="counters"),
        )
        solution = solver.solve()

        assert solution.status == "finished"
        assert solution.states()[-1][0] == pytest.approx(math.exp(-0.5), rel=5e-3)
        assert solver.provenance()["jacobian_source"] == expected_source
        assert solver.provenance()["matrix_layout"] == "dense"

    configured = BdfCallbackSolver(
        1,
        0.0,
        0.5,
        0.1,
        [1.0],
        residual,
        analytic_jacobian,
        first_step=0.01,
        max_bdf_order=3,
        max_steps=100,
    )
    assert configured.provenance()["first_step"] == "0.01"
    assert configured.provenance()["max_bdf_order"] == "3"
    assert configured.provenance()["max_steps"] == "100"


def test_bdf_callback_solver_reports_typed_callback_failures():
    def broken_residual(_t, _state):
        raise RuntimeError("callback boom")

    solver = BdfCallbackSolver(1, 0.0, 0.2, 0.1, [1.0], broken_residual)
    with pytest.raises(CallbackError, match="callback boom") as caught:
        solver.solve()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "callback boom"

    with pytest.raises(CallbackError, match="must return 1 finite values") as malformed:
        BdfCallbackSolver(
            1, 0.0, 0.2, 0.1, [1.0], lambda _t, _state: [0.0, 1.0]
        ).solve()
    assert malformed.value.__cause__ is None

    with pytest.raises(UnsupportedOptionError, match="does not use a symbolic frontend"):
        BdfCallbackSolver(
            1,
            0.0,
            0.2,
            0.1,
            [1.0],
            lambda _t, state: [-state[0]],
            options=SolverOptions(frontend="atom_native"),
        )


def test_bdf_continuation_before_first_solve_is_typed_configuration_error():
    y = Expr.variable("y")
    a = Expr.variable("a")
    solver = BdfSolver(
        [-a * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
        options=SolverOptions(telemetry="counters"),
        parameters=["a"],
        parameter_values=[1.0],
    )

    with pytest.raises(InvalidConfigurationError):
        solver.continue_with_parameters([2.0], 1.0)


def test_bdf_atom_native_parameter_continuation_reuses_prepared_callbacks():
    y = Expr.variable("y")
    a = Expr.variable("a")
    solver = BdfSolver(
        [-a * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
        rtol=1e-5,
        atol=1e-7,
        options=SolverOptions(frontend="atom_native", telemetry="counters"),
        parameters=["a"],
        parameter_values=[1.0],
    )

    solver.prepare()
    first = solver.solve()
    assert first.status == "finished"
    continued = solver.continue_with_parameters([2.0], 1.0)
    assert continued.status == "finished"
    assert continued.times()[-1] == pytest.approx(1.0)
    assert continued.states()[-1][0] == pytest.approx(math.exp(-1.5), rel=4e-3)


def test_bdf_restart_keeps_prepared_callbacks_and_uses_new_initial_state():
    y = Expr.variable("y")
    solver = BdfSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
        options=SolverOptions(telemetry="counters"),
    )

    solver.prepare()
    first = solver.solve()
    assert first.status == "finished"
    assert solver.is_prepared()

    restarted = solver.restart(0.5, [2.0], 1.0)
    assert restarted.status == "finished"
    assert solver.is_prepared()
    assert restarted.times()[0] == pytest.approx(0.5)
    assert restarted.times()[-1] == pytest.approx(1.0)
    assert restarted.states()[0][0] == pytest.approx(2.0)
    assert restarted.states()[-1][0] == pytest.approx(2.0 * math.exp(-0.5), rel=4e-3)


def test_radau_symbolic_dense_lambdify_supports_frontend_output_and_continuation():
    y = Expr.variable("y")
    a = Expr.variable("a")
    solver = RadauSolver(
        [-a * y],
        ["y"],
        "t",
        0.0,
        1.0,
        0.25,
        rtol=1e-5,
        atol=1e-7,
        options=SolverOptions(frontend="atom_native", telemetry="detailed"),
        parameters=["a"],
        jacobian=[-a],
        output="dense",
    )

    assert not solver.is_prepared()
    solution = solver.solve([1.0], [1.0])
    assert solver.is_prepared()
    assert solution.final_time == pytest.approx(1.0)
    assert solution.final_state[0] == pytest.approx(math.exp(-1.0), rel=3e-3)
    assert solution.sample(0.5)[0] == pytest.approx(math.exp(-0.5), rel=3e-3)
    assert solution.telemetry_mode == "detailed"
    assert solution.telemetry_counters()["accepted_steps"] > 0
    assert "preparation_ms" in solution.telemetry_timings_ms()
    assert "counters:" in solution.telemetry_report()
    assert solution.diagnostics()["solver"] == "radau5"
    assert float(solution.diagnostics()["final_time"]) == pytest.approx(solution.final_time)
    assert solver.provenance() == {
        "solver": "radau5",
        "frontend": "atom_native",
        "execution": "lambdify",
        "matrix_layout": "dense",
        "execution_policy": "sequential",
        "telemetry": "detailed",
        "first_step": "auto",
        "max_steps": "1000000",
        "max_newton_iterations": "6",
        "max_retries": "12",
    }
    continued = solver.continue_with_parameters([1.0], [2.0])
    assert continued.final_state[0] == pytest.approx(math.exp(-2.0), rel=4e-3)


def test_radau_telemetry_modes_have_disjoint_collection_contracts():
    y = Expr.variable("y")
    expected = {
        "off": (False, False),
        "counters": (True, False),
        "timings": (False, True),
        "detailed": (True, True),
    }
    for mode, (has_counters, has_timings) in expected.items():
        solution = RadauSolver(
            [-y],
            ["y"],
            "t",
            0.0,
            0.1,
            0.025,
            options=SolverOptions(telemetry=mode),
        ).solve([1.0])
        assert solution.telemetry_mode == mode
        assert bool(solution.telemetry_counters()) is has_counters
        assert bool(solution.telemetry_timings_ms()) is has_timings


def test_radau_restart_keeps_prepared_callbacks_for_a_new_interval():
    y = Expr.variable("y")
    solver = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.5, 0.1,
        rtol=1e-5,
        atol=1e-7,
        options=SolverOptions(telemetry="counters"),
    )

    first = solver.solve([1.0])
    assert first.final_time == pytest.approx(0.5)
    assert solver.is_prepared()

    solver.restart(0.5, 1.0)
    restarted = solver.solve([2.0])
    assert solver.is_prepared()
    assert restarted.final_time == pytest.approx(1.0)
    assert restarted.final_state[0] == pytest.approx(2.0 * math.exp(-0.5), rel=4e-3)
    assert restarted.telemetry_counters()["finite_difference_probes"] > 0


def test_radau_accepts_dense_sparse_and_banded_layouts():
    y = Expr.variable("y")
    # Structured layouts require a layout-aware Jacobian.  Supplying the
    # analytic derivative keeps this gate focused on layout selection rather
    # than accidentally testing the dense-only finite-difference fallback.
    jacobian = [Expr.constant(-1.0)]
    solver = RadauSolver(
        [-y], ["y"], "t", 0.0, 1.0, 0.1,
        options=SolverOptions(execution="aot"),
    )
    assert solver.provenance()["execution"] == "aot"

    sparse = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.1, 0.05,
        options=SolverOptions(matrix_layout="sparse", telemetry="counters"),
        jacobian=jacobian,
    )
    sparse_solution = sparse.solve([1.0])
    assert sparse_solution.final_time == pytest.approx(0.1)
    assert sparse.provenance()["matrix_layout"] == "sparse"

    banded_options = SolverOptions(matrix_layout="banded", telemetry="counters")
    banded_options.set_banded_width(1, 1)
    banded = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.1, 0.05,
        options=banded_options,
        jacobian=jacobian,
    )
    banded_solution = banded.solve([1.0])
    assert banded_solution.final_time == pytest.approx(0.1)
    assert banded.provenance()["matrix_layout"] == "banded"

    with pytest.raises(UnsupportedOptionError, match="layout-aware Jacobian"):
        RadauSolver(
            [-y], ["y"], "t", 0.0, 0.1, 0.05,
            options=SolverOptions(matrix_layout="sparse"),
        ).solve([1.0])

    with pytest.raises(InvalidConfigurationError, match="set_banded_width"):
        RadauSolver(
            [-y], ["y"], "t", 0.0, 1.0, 0.1,
            options=SolverOptions(matrix_layout="banded"),
        )


def test_dense_aot_routes_build_and_solve_for_radau_bdf_and_backward_euler(tmp_path):
    compiler_path = shutil.which("tcc") or shutil.which("gcc")
    if compiler_path is None:
        pytest.skip("no supported C compiler is available for the AOT route gate")
    toolchain = "tcc" if compiler_path.lower().endswith(("tcc", "tcc.exe")) else "gcc"
    y = Expr.variable("y")

    def options(name):
        return SolverOptions(
            execution="aot",
            aot_policy="build_if_missing",
            aot_toolchain=toolchain,
            aot_profile="debug",
            aot_output_dir=str(tmp_path / name),
            telemetry="counters",
        )

    bdf = BdfSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
        options=options("bdf"),
    )
    bdf.prepare()
    bdf_solution = bdf.solve()
    assert bdf_solution.times()[-1] == pytest.approx(0.1)
    assert bdf_solution.telemetry_counters()["aot_build_attempts"] >= 1
    assert bdf_solution.aot_artifact_keys()
    assert bdf.provenance()["aot_policy"] == "build_if_missing"
    assert bdf.provenance()["aot_toolchain"] == toolchain

    yr = Expr.variable("yr")
    radau = RadauSolver(
        [-yr], ["yr"], "t", 0.0, 0.1, 0.05,
        options=options("radau"),
    )
    radau.prepare()
    radau_solution = radau.solve([1.0])
    assert radau_solution.final_time == pytest.approx(0.1)
    assert radau_solution.telemetry_counters()["aot_build_attempts"] >= 1
    assert radau_solution.aot_artifact_keys()
    assert radau.provenance()["aot_profile"] == "debug"

    ye = Expr.variable("ye")
    backward_euler = BackwardEulerSolver(
        [-ye], ["ye"], "t", [1.0], 0.0, 0.1, h=0.05,
        options=options("backward-euler"),
    )
    be_solution = backward_euler.solve()
    assert be_solution.times()[-1] == pytest.approx(0.1)
    assert be_solution.telemetry_counters()["aot_build_attempts"] >= 1
    assert be_solution.aot_artifact_keys()
    assert backward_euler.provenance()["execution"] == "aot"


@pytest.mark.parametrize("solver_name", ["bdf", "radau", "backward_euler"])
def test_dense_aot_process_isolated_handoff_for_all_ivp_adapters(tmp_path, solver_name):
    compiler = shutil.which("gcc") or shutil.which("tcc")
    if compiler is None:
        pytest.skip("no supported C compiler is available for the AOT handoff gate")

    toolchain = "gcc" if compiler.lower().endswith(("gcc", "gcc.exe")) else "tcc"
    root = tmp_path / solver_name
    root.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "RST_AOT_HANDOFF_ROOT": str(root),
            "RST_AOT_TOOLCHAIN": toolchain,
            "RST_AOT_SOLVER": solver_name,
        }
    )
    child = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path
        from rustedscithe import BdfSolver, BackwardEulerSolver, Expr, RadauSolver, SolverOptions

        root = Path(os.environ["RST_AOT_HANDOFF_ROOT"])
        solver_name = os.environ["RST_AOT_SOLVER"]
        variable = Expr.variable("y_" + solver_name)

        def make_options(policy):
            options = SolverOptions(
                execution="aot",
                aot_policy=policy,
                aot_toolchain=os.environ["RST_AOT_TOOLCHAIN"],
                aot_profile="debug",
                aot_output_dir=str(root),
                telemetry="counters",
            )
            options.set_aot_handoff_path(str(root / "registry.txt"))
            return options

        options = make_options(POLICY)
        if solver_name == "bdf":
            solver = BdfSolver(
                [-variable], ["y_" + solver_name], "t", [1.0], 0.0, 0.1, 0.05,
                options=options,
            )
            solver.prepare()
            solution = solver.solve()
            final_time = solution.times()[-1]
        elif solver_name == "radau":
            solver = RadauSolver(
                [-variable], ["y_" + solver_name], "t", 0.0, 0.1, 0.05,
                options=options,
            )
            solver.prepare()
            solution = solver.solve([1.0])
            final_time = solution.final_time
        else:
            solver = BackwardEulerSolver(
                [-variable], ["y_" + solver_name], "t", [1.0], 0.0, 0.1,
                h=0.05, options=options,
            )
            solution = solver.solve()
            final_time = solution.times()[-1]

        print(json.dumps({
            "provenance": solver.provenance(),
            "handoff_exists": (root / "registry.txt").is_file(),
            "final_time": final_time,
            "aot_build_attempts": solution.telemetry_counters()["aot_build_attempts"],
            "aot_reconnects": solution.telemetry_counters()["aot_reconnects"],
            "aot_artifact_keys": solution.aot_artifact_keys(),
        }))
        """
    )

    def run_process(policy):
        result = subprocess.run(
            [sys.executable, "-c", child.replace("POLICY", repr(policy))],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"AOT {solver_name}/{policy} child process failed with exit={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    producer = run_process("build_if_missing")
    assert producer["handoff_exists"]
    assert producer["final_time"] == pytest.approx(0.1)
    assert producer["aot_build_attempts"] >= 1
    assert producer["aot_artifact_keys"]

    consumer = run_process("require_prebuilt")
    assert consumer["provenance"]["aot_policy"] == "require_prebuilt"
    assert consumer["provenance"]["aot_handoff_path"].endswith("registry.txt")
    assert consumer["final_time"] == pytest.approx(0.1)
    assert consumer["aot_build_attempts"] == 0
    assert consumer["aot_reconnects"] >= 1
    assert consumer["aot_artifact_keys"] == producer["aot_artifact_keys"]


def test_backward_euler_dense_lambdify_validates_solves_and_continues():
    y = Expr.variable("y")
    solver = BackwardEulerSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.5,
        h=0.1,
        tolerance=1e-8,
        options=SolverOptions(telemetry="counters"),
    )

    solver.validate()
    solution = solver.solve()
    assert solution.status == "finished"
    assert solution.times()[0] == pytest.approx(0.0)
    assert solution.times()[-1] == pytest.approx(0.5)
    assert solution.states()[-1][0] == pytest.approx((1.0 / 1.1) ** 5, rel=2e-3)
    assert solution.telemetry_counters()["accepted_steps"] > 0
    assert "counters:" in solution.telemetry_report()
    assert solution.telemetry_mode == "counters"
    assert solution.diagnostics()["solver"] == "backward_euler"
    assert solution.diagnostics()["sample_count"] == str(len(solution.times()))
    assert solver.provenance() == {
        "solver": "backward_euler",
        "frontend": "expr_legacy",
        "execution": "lambdify",
        "matrix_layout": "dense",
        "execution_policy": "sequential",
        "telemetry": "counters",
    }

    continued = solver.continue_to(1.0)
    assert continued.status == "finished"
    assert continued.times()[-1] == pytest.approx(1.0)


def test_backward_euler_restart_keeps_prepared_callbacks_and_new_state():
    y = Expr.variable("y")
    solver = BackwardEulerSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.5,
        h=0.1,
        tolerance=1e-8,
        options=SolverOptions(telemetry="counters"),
    )

    solver.validate()
    first = solver.solve()
    assert first.status == "finished"

    restarted = solver.restart(0.5, [2.0], 1.0)
    assert restarted.status == "finished"
    assert restarted.times()[0] == pytest.approx(0.5)
    assert restarted.times()[-1] == pytest.approx(1.0)
    assert restarted.states()[0][0] == pytest.approx(2.0)
    assert restarted.states()[-1][0] == pytest.approx(2.0 / (1.1 ** 5), rel=2e-3)


def test_backward_euler_supports_atom_native_and_dense_aot():
    y = Expr.variable("y")
    solver = BackwardEulerSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, h=0.1,
        options=SolverOptions(frontend="atom_native", telemetry="timings"),
    )
    assert solver.telemetry_mode == "timings"
    assert solver.solve().status == "finished"

    aot = BackwardEulerSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1,
        options=SolverOptions(execution="aot"),
    )
    assert aot.provenance()["execution"] == "aot"


def test_telemetry_off_is_reported_as_unmeasured_for_radau_and_backward_euler():
    y = Expr.variable("y")

    radau = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.2, 0.1,
        options=SolverOptions(telemetry="off"),
    )
    radau_solution = radau.solve([1.0])
    assert radau_solution.telemetry_mode == "off"
    assert radau_solution.telemetry_counters() == {}
    assert radau_solution.telemetry_timings_ms() == {}

    backward_euler = BackwardEulerSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.2,
        h=0.1,
        options=SolverOptions(telemetry="off"),
    )
    be_solution = backward_euler.solve()
    assert be_solution.telemetry_mode == "off"
    assert be_solution.telemetry_counters() == {}
    assert be_solution.telemetry_timings_ms() == {}


def test_telemetry_collection_modes_have_consistent_map_semantics():
    y = Expr.variable("y")

    def solve_family(mode):
        options = SolverOptions(telemetry=mode)
        return (
            BdfSolver([-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=options).solve(),
            RadauSolver([-y], ["y"], "t", 0.0, 0.1, 0.05, options=options).solve([1.0]),
            BackwardEulerSolver(
                [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=options
            ).solve(),
            Lsode2Solver(
                [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=options
            ).solve(),
            BvpSciSolver(
                [Expr.constant(1.0)],
                ["y"],
                [Expr.parse("ya_y")],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.0],
                options=options,
                tolerance=1e-3,
            ).solve(),
        )

    for mode in ("off", "counters", "timings"):
        for index, solution in enumerate(solve_family(mode)):
            assert solution.telemetry_mode == mode
            counters = solution.telemetry_counters()
            timings = solution.telemetry_timings_ms()
            if mode == "off":
                assert counters == {}
                assert timings == {}
            elif mode == "counters":
                assert counters
                assert timings == {}
            elif mode == "timings" and index != 4:
                assert counters == {}
                assert timings
            elif mode == "detailed" and index != 4:
                assert counters, f"expected counters for mode={mode!r}, solver_index={index}"
                assert timings, f"expected timings for mode={mode!r}, solver_index={index}"
            else:
                # BVP_Sci retains its combined Timings behavior pending its
                # own telemetry migration; the other IVP solvers are disjoint.
                assert counters
                assert timings

    detailed = SolverOptions(telemetry="detailed")
    detailed_solutions = (
        BdfSolver([-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=detailed).solve(),
        RadauSolver([-y], ["y"], "t", 0.0, 0.1, 0.05, options=detailed).solve([1.0]),
        BackwardEulerSolver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=detailed
        ).solve(),
        Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05, options=detailed
        ).solve(),
    )
    for solution in detailed_solutions:
        assert solution.telemetry_mode == "detailed"
        assert solution.telemetry_counters()
        assert solution.telemetry_timings_ms()


def test_ivp_adapters_reject_nonfinite_and_invalid_numeric_configuration_early():
    y = Expr.variable("y")

    with pytest.raises(InvalidConfigurationError, match="finite"):
        BdfSolver([-y], ["y"], "t", [math.nan], 0.0, 1.0, 0.1)
    with pytest.raises(InvalidConfigurationError, match="finite"):
        BdfSolver([-y], ["y"], "t", [1.0], 0.0, 1.0, 0.1, rtol=math.inf)

    with pytest.raises(InvalidConfigurationError, match="finite"):
        RadauSolver([-y], ["y"], "t", math.nan, 1.0, 0.1)
    with pytest.raises(InvalidConfigurationError, match="greater than zero"):
        RadauSolver([-y], ["y"], "t", 0.0, 1.0, 0.0)

    with pytest.raises(InvalidConfigurationError, match="finite"):
        BackwardEulerSolver(
            [-y], ["y"], "t", [1.0], 0.0, 1.0,
            h=0.1,
            tolerance=math.inf,
        )
    with pytest.raises(InvalidConfigurationError, match="greater than zero"):
        BackwardEulerSolver(
            [-y], ["y"], "t", [1.0], 0.0, 1.0,
            h=0.0,
        )


def test_invalid_expression_is_typed():
    with pytest.raises(InvalidExpressionError):
        Expr.parse("x +")


def test_solver_contract_separates_frontend_route_and_layout():
    options = SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
    )
    options.validate_for("bdf")
    assert capabilities("bdf").frontends == ["expr_legacy", "atom_native"]
    assert capabilities("bdf").layouts == ["dense"]
    assert capabilities("bdf").executions == ["lambdify", "aot"]
    assert capabilities("bdf").policies == ["sequential", "parallel", "auto"]
    assert capabilities("bdf").supports_analytical_callbacks
    assert capabilities("bdf").analytical_callback_layouts == ["dense"]

    with pytest.raises(UnsupportedOptionError):
        SolverOptions(matrix_layout="sparse").validate_for("backward_euler")


@pytest.mark.parametrize("policy", ["sequential", "parallel", "auto"])
def test_bdf_and_radau_expose_native_execution_policies(policy):
    y_bdf = Expr.variable("y_bdf_" + policy)
    bdf = BdfSolver(
        [-y_bdf], ["y_bdf_" + policy], "t", [1.0], 0.0, 0.1, 0.05,
        options=SolverOptions(execution_policy=policy, telemetry="counters"),
    )
    bdf_solution = bdf.solve()
    assert bdf_solution.status == "finished"
    assert bdf.provenance()["execution_policy"] == policy

    y_radau = Expr.variable("y_radau_" + policy)
    radau = RadauSolver(
        [-y_radau], ["y_radau_" + policy], "t", 0.0, 0.1, 0.05,
        options=SolverOptions(execution_policy=policy, telemetry="counters"),
    )
    radau_solution = radau.solve([1.0])
    assert radau_solution.final_time == pytest.approx(0.1)
    assert radau.provenance()["execution_policy"] == policy


def test_lsode2_dense_lambdify_lifecycle_and_restart():
    y = Expr.variable("y")
    solver = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
        rtol=1e-5,
        atol=1e-7,
        options=SolverOptions(
            frontend="expr_legacy",
            execution="lambdify",
            matrix_layout="dense",
            execution_policy="sequential",
            telemetry="counters",
        ),
    )

    assert not solver.is_prepared
    solver.prepare()
    assert solver.is_prepared
    solution = solver.solve()
    assert solution.status.startswith("finished")
    assert solution.method == "bdf"
    assert solution.controller_mode == "bdf_only"
    assert solution.active_family == "bdf"
    assert solution.executed_family == "bdf"
    assert solution.switch_uses_fallback is False
    assert solution.final_time == pytest.approx(0.5)
    assert solution.final_state[0] == pytest.approx(math.exp(-0.5), rel=4e-3)
    assert solution.telemetry_mode == "counters"
    assert solution.telemetry_counters()["solve_calls"] > 0
    assert "aot_build_attempts" in solution.telemetry_counters()
    assert "aot_link_attempts" in solution.telemetry_counters()
    assert solution.diagnostics()["solver"] == "lsode2"
    assert solution.diagnostics()["controller_mode"] == "bdf_only"
    assert solution.diagnostics()["active_family"] == "bdf"
    assert "evaluation_scope=" in solver.telemetry_report()
    assert "controller=bdf_only" in solver.telemetry_report()
    assert solver.provenance() == {
        "solver": "lsode2",
        "method": "bdf",
        "controller_mode": "bdf_only",
        "active_family": "bdf",
        "frontend": "expr_legacy",
        "execution": "lambdify",
        "matrix_layout": "dense",
        "execution_policy": "sequential",
        "telemetry": "counters",
        "aot_policy": "not_applicable",
        "aot_toolchain": "not_applicable",
        "aot_profile": "not_applicable",
        "aot_output_dir": "default",
        "aot_handoff_path": "default",
    }

    restarted = solver.restart(0.5, [2.0], 1.0)
    assert restarted.final_time == pytest.approx(1.0)
    assert restarted.final_state[0] == pytest.approx(2.0 * math.exp(-0.5), rel=4e-3)


def test_lsode2_frontend_layout_policy_and_controller_contract():
    y = Expr.variable("y")
    for layout in ("sparse", "banded"):
        kwargs = {"band_lower": 1, "band_upper": 1} if layout == "banded" else {}
        solver = Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            options=SolverOptions(
                frontend="atom_native",
                matrix_layout=layout,
                execution_policy="auto",
                telemetry="off",
            ),
            **kwargs,
        )
        assert solver.provenance()["frontend"] == "atom_native"
        assert solver.provenance()["matrix_layout"] == layout
        assert solver.provenance()["execution_policy"] == "auto"
        assert "counters: disabled" in solver.telemetry_report()
        solution = solver.solve()
        assert solution.status.startswith("finished")
        assert solution.final_time == pytest.approx(0.1)

    assert capabilities("lsode2").analytical_callback_layouts == [
        "dense", "sparse", "banded"
    ]
    assert capabilities("lsode2").methods == ["bdf", "adams", "auto", "lsoda"]

    for requested, expected_controller, expected_method in (
        ("adams", "adams_only", "adams"),
        ("auto", "automatic_adams_bdf", "auto"),
        ("lsoda", "automatic_adams_bdf", "auto"),
    ):
        controller_solver = Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            method=requested,
            options=SolverOptions(telemetry="counters"),
        )
        controller_solution = controller_solver.solve()
        assert controller_solution.status.startswith("finished")
        assert controller_solution.method == expected_method
        assert controller_solution.controller_mode == expected_controller
        assert controller_solution.executed_family in ("adams", "bdf")

    with pytest.raises(UnsupportedOptionError, match="'bdf'.*'adams'.*'auto'"):
        Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            method="not-a-method",
        )

    aot = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
        options=SolverOptions(
            execution="aot",
            aot_policy="require_prebuilt",
            aot_toolchain="tcc",
            aot_profile="debug",
            aot_output_dir="target/python-ivp-aot",
        ),
    )
    assert aot.provenance()["execution"] == "aot"
    assert aot.provenance()["aot_policy"] == "require_prebuilt"
    assert aot.provenance()["aot_toolchain"] == "tcc"
    assert aot.provenance()["aot_profile"] == "debug"
    assert aot.provenance()["aot_output_dir"] == "target/python-ivp-aot"
    assert capabilities("lsode2").executions == ["lambdify", "aot"]

    with pytest.raises(UnsupportedOptionError, match="require execution='aot'"):
        Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            options=SolverOptions(aot_toolchain="tcc"),
        )
    with pytest.raises(UnsupportedOptionError, match="execution='lambdify'"):
        Lsode2Solver(
            [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            options=SolverOptions(execution="aot", matrix_layout="sparse"),
            residual_callback=lambda _t, state: [-state[0]],
            jacobian_callback=lambda _t, _state: [[-1.0]],
        )


def test_lsode2_require_prebuilt_reports_missing_artifact_type():
    y = Expr.variable("y")
    solver = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
        options=SolverOptions(
            execution="aot",
            aot_policy="require_prebuilt",
            aot_output_dir="target/python-ivp-aot-test-missing",
        ),
    )

    with pytest.raises(AotArtifactError):
        solver.prepare()


def test_lsode2_aot_handoff_path_is_explicit_and_preserved_in_provenance():
    y = Expr.variable("y")
    options = SolverOptions(
        execution="aot",
        aot_policy="require_prebuilt",
        aot_output_dir="target/python-ivp-aot-consumer",
    )
    options.set_aot_handoff_path("target/python-ivp-aot/registry.txt")
    solver = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1, options=options
    )

    assert solver.provenance()["aot_handoff_path"] == "target/python-ivp-aot/registry.txt"


def test_lsode2_aot_process_isolated_producer_consumer_handoff(tmp_path):
    compiler = shutil.which("gcc") or shutil.which("tcc")
    if compiler is None:
        pytest.skip("no supported C compiler is available for the AOT handoff gate")

    toolchain = "gcc" if compiler.lower().endswith("gcc.exe") or compiler.lower().endswith("gcc") else "tcc"
    root = tmp_path / "aot-handoff"
    root.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "RST_AOT_HANDOFF_ROOT": str(root),
            "RST_AOT_TOOLCHAIN": toolchain,
        }
    )
    common = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path
        from rustedscithe import Expr, Lsode2Solver, SolverOptions

        root = Path(os.environ["RST_AOT_HANDOFF_ROOT"])
        y = Expr.variable("y")

        def make_options(policy):
            options = SolverOptions(
                execution="aot",
                aot_policy=policy,
                aot_toolchain=os.environ["RST_AOT_TOOLCHAIN"],
                aot_profile="debug",
                aot_output_dir=str(root),
                telemetry="counters",
            )
            options.set_aot_handoff_path(str(root / "registry.txt"))
            return options

        solver = Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            options=make_options(POLICY),
        )
        solver.prepare()
        solution = solver.solve()
        RESULT = {
            "provenance": solver.provenance(),
            "handoff_exists": (root / "registry.txt").is_file(),
            "final_time": solution.final_time,
            "aot_build_attempts": solution.telemetry_counters()["aot_build_attempts"],
            "aot_reconnects": solution.telemetry_counters()["aot_reconnects"],
            "aot_artifact_keys": solution.aot_artifact_keys(),
        }
        print(json.dumps(RESULT))
        """
    )
    def run_process(policy):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                common.replace("POLICY", repr(policy)),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"AOT {policy} child process failed with exit={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return result

    producer = run_process("build_if_missing")
    producer_result = json.loads(producer.stdout.strip().splitlines()[-1])
    assert producer_result["handoff_exists"]
    assert producer_result["final_time"] == pytest.approx(0.1)
    assert producer_result["aot_build_attempts"] >= 1
    assert producer_result["aot_artifact_keys"]

    consumer = run_process("require_prebuilt")
    consumer_result = json.loads(consumer.stdout.strip().splitlines()[-1])
    assert consumer_result["provenance"]["aot_policy"] == "require_prebuilt"
    assert consumer_result["provenance"]["aot_handoff_path"].endswith("registry.txt")
    assert consumer_result["final_time"] == pytest.approx(0.1)
    assert consumer_result["aot_build_attempts"] == 0
    assert consumer_result["aot_reconnects"] >= 1
    assert consumer_result["aot_artifact_keys"] == producer_result["aot_artifact_keys"]


def test_lsode2_parameter_continuation_starts_at_last_successful_state():
    y = Expr.variable("y")
    a = Expr.variable("a")
    solver = Lsode2Solver(
        [-a * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
        parameters=["a"],
        parameter_values=[1.0],
        options=SolverOptions(telemetry="counters"),
    )

    with pytest.raises(InvalidConfigurationError, match="successful solve"):
        solver.continue_with_parameters([2.0], 0.5)

    first = solver.solve()
    continued = solver.continue_with_parameters([2.0], 1.0)
    assert first.final_time == pytest.approx(0.5)
    assert continued.final_time == pytest.approx(1.0)
    # The second segment starts at y(0.5), then evolves for another 0.5
    # units with the rebound coefficient a=2.
    assert continued.final_state[0] == pytest.approx(math.exp(-1.5), rel=8e-3)

    fresh = Lsode2Solver(
        [-a * y], ["y"], "t", [1.0], 0.0, 1.0, 0.1,
        parameters=["a"],
        parameter_values=[2.0],
    ).solve()
    assert continued.final_state[0] != pytest.approx(fresh.final_state[0], rel=1e-2)


def test_lsode2_sparse_python_analytical_callbacks_and_typed_callback_failure():
    def residual(_t, state):
        return [-state[0]]

    def jacobian(_t, _state):
        return [[-1.0]]

    solver = Lsode2Solver(
        [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        options=SolverOptions(
            matrix_layout="sparse",
            execution_policy="sequential",
            telemetry="counters",
        ),
        residual_callback=residual,
        jacobian_callback=jacobian,
    )
    solution = solver.solve()
    assert solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=5e-3)
    assert solver.provenance()["matrix_layout"] == "sparse"

    dense = Lsode2Solver(
        [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        options=SolverOptions(matrix_layout="dense"),
        residual_callback=residual,
        jacobian_callback=jacobian,
    )
    dense_solution = dense.solve()
    assert dense_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=5e-3)
    assert dense.provenance()["matrix_layout"] == "dense"

    banded = Lsode2Solver(
        [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        options=SolverOptions(matrix_layout="banded"),
        band_lower=0,
        band_upper=0,
        residual_callback=residual,
        jacobian_callback=jacobian,
    )
    banded_solution = banded.solve()
    assert banded_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=5e-3)
    assert banded.provenance()["matrix_layout"] == "banded"

    def failing_residual(_t, _state):
        raise RuntimeError("callback fixture failure")

    failing = Lsode2Solver(
        [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
        options=SolverOptions(matrix_layout="sparse"),
        residual_callback=failing_residual,
        jacobian_callback=jacobian,
    )
    with pytest.raises(CallbackError, match="callback fixture failure") as callback_error:
        failing.solve()
    assert isinstance(callback_error.value.__cause__, RuntimeError)
    assert str(callback_error.value.__cause__) == "callback fixture failure"

    with pytest.raises(InvalidConfigurationError, match="provided together"):
        Lsode2Solver(
            [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            residual_callback=residual,
        )
    with pytest.raises(UnsupportedOptionError, match="'bdf'.*'adams'.*'auto'"):
        Lsode2Solver(
            [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            method="unsupported",
        )


def test_ivp_solution_telemetry_report_writes_compact_file_without_noise(tmp_path):
    y = Expr.variable("y")
    solver = BdfSolver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
        options=SolverOptions(telemetry="counters"),
    )
    solution = solver.solve()
    report_path = tmp_path / "bdf_telemetry.md"
    solution.write_telemetry_report(str(report_path))
    report = report_path.read_text(encoding="utf-8")
    assert report == solution.telemetry_report()
    assert "telemetry=counters" in report
    assert "Compiling" not in report

    with pytest.raises(TelemetryError, match="path must not be empty"):
        solution.write_telemetry_report("")


@pytest.mark.parametrize("telemetry", ["timings", "detailed"])
def test_bvp_sci_direct_lambdify_builder_supports_mesh_output_and_restart(tmp_path, telemetry):
    y = Expr.variable("y")
    ya_y = Expr.parse("ya_y - 1")
    solver = BvpSciSolver(
        [-y],
        ["y"],
        [ya_y],
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        options=SolverOptions(frontend="expr_legacy", matrix_layout="dense", telemetry=telemetry),
        tolerance=1e-4,
        output="dense",
    )
    assert solver.frontend == "expr_legacy"
    assert solver.matrix_layout == "dense"
    solution = solver.solve()
    assert solution.status_code == 0
    assert solution.mesh()[0] == pytest.approx(0.0)
    assert solution.values()[0][0] == pytest.approx(1.0, rel=2e-3)
    assert solution.sample(1.0)[0] == pytest.approx(math.exp(-1.0), rel=3e-3)
    timings = solution.telemetry_timings_ms()
    if telemetry == "timings":
        assert solution.telemetry_counters() == {}
    else:
        assert solution.telemetry_counters()["linear_solves"] >= 1
    assert timings["preparation_ms"] >= 0.0
    assert timings["collocation_ms"] >= 0.0
    assert timings["linear_solve_ms"] >= 0.0
    assert timings["full_solve_ms"] >= 0.0
    report_path = tmp_path / "bvp_sci.md"
    solution.write_telemetry_report(str(report_path))
    assert "BVP_sci" in report_path.read_text(encoding="utf-8")

    solver.restart([0.0, 1.0], [1.0, math.exp(-1.0)])
    restarted = solver.solve()
    assert restarted.status_code == 0
    assert restarted.mesh()[0] == pytest.approx(0.0)
    assert restarted.mesh()[-1] == pytest.approx(1.0)


def test_bvp_sci_direct_aot_preserves_typed_lifecycle_failure(tmp_path):
    y = Expr.variable("y")
    ya_y = Expr.parse("ya_y - 1")
    options = SolverOptions(
        execution="aot",
        aot_policy="require_prebuilt",
        aot_output_dir=str(tmp_path / "missing-aot"),
    )
    with pytest.raises((AotArtifactError, PreparationError)) as failure:
        BvpSciSolver(
            [-y],
            ["y"],
            [ya_y],
            [0.0, 1.0],
            [1.0, math.exp(-1.0)],
            options=options,
        )
    assert failure.value.stage == "construct"
    assert failure.value.solver == "bvp_sci"
    assert failure.value.native_kind in {"AotPreparation", "AotHandoff"}


def test_bvp_sci_direct_atom_native_and_boundary_fd_routes_are_available():
    y = Expr.variable("y")
    boundary = [Expr.parse("ya_y - 1")]
    for frontend, boundary_jacobian in (
        ("atom_native", "symbolic"),
        ("expr_legacy", "finite_difference"),
    ):
        solver = BvpSciSolver(
            [-y],
            ["y"],
            boundary,
            [0.0, 1.0],
            [1.0, math.exp(-1.0)],
            options=SolverOptions(frontend=frontend, telemetry="counters"),
            boundary_jacobian=boundary_jacobian,
        )
        solution = solver.solve()
        assert solution.status_code == 0
        assert solution.telemetry_counters()["linear_solves"] >= 1


def test_bvp_sci_direct_singular_term_is_configured_and_reported():
    """The public adapter forwards SciPy's optional S*y/(x-a) contract."""
    y = Expr.variable("y")
    solver = BvpSciSolver(
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        singular_term_a=0.0,
        singular_term_matrix=[0.0],
        options=SolverOptions(telemetry="counters"),
    )

    solution = solver.solve()
    assert solution.status_code == 0
    assert solution.values()[0][0] == pytest.approx(1.0, rel=2e-3)
    assert solution.telemetry_counters()["singular_term_applications"] > 0


def test_bvp_sci_direct_singular_term_validation_is_typed():
    y = Expr.variable("y")
    common = ([(-y)], ["y"], [Expr.parse("ya_y - 1")], [0.0, 1.0], [1.0, math.exp(-1.0)])

    with pytest.raises(InvalidConfigurationError, match="provided together"):
        BvpSciSolver(*common, singular_term_a=0.0)

    with pytest.raises(InvalidConfigurationError, match="left mesh endpoint"):
        BvpSciSolver(
            *common,
            singular_term_a=1.0,
            singular_term_matrix=[0.0],
        )

    with pytest.raises(SolverError, match="invalid shape"):
        BvpSciSolver(
            *common,
            singular_term_a=0.0,
            singular_term_matrix=[0.0, 0.0],
        )


def test_bvp_sci_direct_sparse_factorization_is_explicit_and_fallback_safe():
    y = Expr.variable("y")
    options = SolverOptions(matrix_layout="sparse", telemetry="counters")
    options.set_sparse_factorization("superlu")
    solver = BvpSciSolver(
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        options=options,
    )
    assert solver.sparse_factorization == "superlu"
    solution = solver.solve()
    assert solution.status_code == 0
    # With the default binding build the bridge is absent and the core falls
    # back to StandardFaer; with a SuperLU-enabled build this counter is zero.
    assert solution.telemetry_counters().get("superlu_fallbacks", 0) >= 0

    with pytest.raises(UnsupportedOptionError, match="matrix_layout='sparse'"):
        dense_options = SolverOptions(matrix_layout="dense")
        dense_options.set_sparse_factorization("superlu")
        BvpSciSolver(
            [-y], ["y"], [Expr.parse("ya_y - 1")],
            [0.0, 1.0], [1.0, math.exp(-1.0)], options=dense_options,
        )


def test_bvp_sci_direct_python_boundary_callbacks_support_analytic_and_fd_jacobians():
    y = Expr.variable("y")

    def boundary(ya, _yb, _parameters):
        return [ya[0] - 1.0]

    def boundary_jacobian(_ya, _yb, _parameters):
        return ([[1.0]], [[0.0]], [[]])

    analytic = BvpSciSolver(
        [-y], ["y"], [], [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
        boundary_callback=boundary,
        boundary_jacobian_callback=boundary_jacobian,
        options=SolverOptions(telemetry="counters"),
    )
    analytic_solution = analytic.solve()
    assert analytic_solution.status_code == 0
    analytic_counters = analytic_solution.telemetry_counters()
    assert analytic_counters["boundary_jacobian_calls"] > 0

    finite_difference = BvpSciSolver(
        [-y], ["y"], [], [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
        boundary_callback=boundary,
        boundary_jacobian="finite_difference",
        options=SolverOptions(telemetry="counters"),
    )
    fd_solution = finite_difference.solve()
    assert fd_solution.status_code == 0
    fd_counters = fd_solution.telemetry_counters()
    assert fd_counters["finite_difference_probes"] > 0


def test_bvp_sci_direct_python_boundary_callback_failures_are_typed():
    y = Expr.variable("y")

    def failing_boundary(_ya, _yb, _parameters):
        raise RuntimeError("boundary fixture failure")

    solver = BvpSciSolver(
        [-y], ["y"], [], [0.0, 1.0], [1.0, math.exp(-1.0)],
        boundary_callback=failing_boundary,
    )
    with pytest.raises(CallbackError, match="boundary fixture failure") as callback_error:
        solver.solve()
    assert isinstance(callback_error.value.__cause__, RuntimeError)
    assert str(callback_error.value.__cause__) == "boundary fixture failure"

    with pytest.raises(InvalidConfigurationError, match="must be empty"):
        BvpSciSolver(
            [-y], ["y"], [Expr.parse("ya_y - 1")], [0.0, 1.0],
            [1.0, math.exp(-1.0)], boundary_callback=lambda *_: [0.0],
        )


def test_bvp_damp_direct_dense_builder_returns_detached_solution_and_telemetry(capsys):
    solver = BvpDampSolver(
        [Expr.parse("0")],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        options=SolverOptions(
            frontend="expr_legacy",
            execution="lambdify",
            matrix_layout="dense",
            execution_policy="sequential",
            telemetry="detailed",
        ),
        bounds={"y": (-10.0, 10.0)},
    )
    assert solver.frontend == "expr_legacy"
    assert solver.execution == "lambdify"
    assert solver.matrix_layout == "dense"

    solution = solver.solve()
    assert solution.status == "finished"
    assert solution.telemetry_mode == "detailed"
    assert solution.mesh()[0] == pytest.approx(0.0)
    assert solution.mesh()[-1] == pytest.approx(1.0)
    assert len(solution.mesh()) >= 3
    assert solution.values()[0][0] == pytest.approx(1.0, rel=2e-3)
    assert solution.values()[-1][0] == pytest.approx(1.0, rel=2e-3)
    counters = solution.telemetry_counters()
    timings = solution.telemetry_timings_ms()
    assert counters["iterations"] >= 1
    assert counters["linear_solves"] >= 1
    assert timings["total_ms"] >= 0.0
    assert timings["symbolic_operations_ms"] >= 0.0
    assert timings["linear_system_ms"] >= 0.0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("mode", "has_counters", "has_timings"),
    [
        ("off", False, False),
        ("counters", True, False),
        ("timings", False, True),
        ("detailed", True, True),
    ],
)
def test_bvp_damp_telemetry_modes_are_disjoint(mode, has_counters, has_timings):
    solution = BvpDampSolver(
        [Expr.parse("0")],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        options=SolverOptions(telemetry=mode),
        bounds={"y": (-10.0, 10.0)},
    ).solve()
    assert solution.telemetry_mode == mode
    assert bool(solution.telemetry_counters()) is has_counters
    assert bool(solution.telemetry_timings_ms()) is has_timings

    detached_mesh = solution.mesh()
    detached_values = solution.values()
    detached_mesh[0] = 99.0
    detached_values[0][0] = 99.0
    assert solution.mesh()[0] == pytest.approx(0.0)
    assert solution.values()[0][0] == pytest.approx(1.0, rel=2e-3)


@pytest.mark.parametrize("strategy", ["naive", "frozen"])
@pytest.mark.parametrize(
    ("mode", "has_counters", "has_timings"),
    [
        ("off", False, False),
        ("counters", True, False),
        ("timings", False, True),
        ("detailed", True, True),
    ],
)
def test_bvp_damp_fixed_mesh_telemetry_modes_are_disjoint(
    strategy, mode, has_counters, has_timings
):
    solver_type = BvpDampNaiveSolver if strategy == "naive" else BvpDampFrozenSolver
    kwargs = {"policy": "Frozen_naive"} if strategy == "frozen" else {}
    solution = solver_type(
        [Expr.parse("rate")],
        ["y"],
        {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        parameters=["rate"],
        parameter_values=[1.0],
        options=SolverOptions(telemetry=mode),
        **kwargs,
    ).solve()
    assert solution.telemetry_mode == mode
    assert bool(solution.telemetry_counters()) is has_counters
    assert bool(solution.telemetry_timings_ms()) is has_timings


def test_bvp_damp_direct_validation_rejects_invalid_boundary_and_scheme():
    y = Expr.variable("y")
    common = ([-y], ["y"], {"y": [(0, 1.0)]}, [0.0, 1.0], [1.0, math.exp(-1.0)])
    with pytest.raises(UnsupportedOptionError, match="scheme"):
        BvpDampSolver(*common, scheme="collocation")
    with pytest.raises(InvalidConfigurationError, match="position 0 or 1"):
        BvpDampSolver(*([-y], ["y"], {"y": [(2, 1.0)]}, [0.0, 1.0], [1.0, math.exp(-1.0)]))


def test_bvp_damp_naive_prepared_parameter_rebind_uses_native_route(capsys):
    solver = BvpDampNaiveSolver(
        [Expr.parse("alpha")],
        ["y"],
        {"y": [(0, 0.0)]},
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0] * 5,
        parameters=["alpha"],
        parameter_values=[1.0],
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            telemetry="detailed",
        ),
    )
    assert solver.provenance()["strategy"] == "Naive"
    assert solver.frontend == "expr_legacy"
    assert solver.matrix_layout == "dense"

    first = solver.solve()
    assert first.status == "finished"
    assert first.diagnostics()["solver"] == "bvp_damp_naive"
    assert first.values_array().shape == (1, 5)
    assert first.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert first.telemetry_counters()["iterations"] >= 1
    assert first.telemetry_timings_ms()["total_ms"] >= 0.0

    solver.set_parameters([2.0])
    warm = solver.solve_prepared()
    assert warm.parameters == pytest.approx([2.0])
    assert warm.values()[0][-1] == pytest.approx(2.0, abs=1e-8)
    assert capsys.readouterr().out == ""


def test_bvp_damp_naive_rejects_nonuniform_mesh_and_unconsumed_bandwidth():
    common = (
        [Expr.parse("0")], ["y"], {"y": [(0, 1.0)]},
        [0.0, 0.2, 1.0], [1.0, 1.0, 1.0],
    )
    with pytest.raises(UnsupportedOptionError, match="uniform mesh"):
        BvpDampNaiveSolver(*common)

    with pytest.raises(InvalidConfigurationError, match="same symbol names"):
        BvpDampNaiveSolver(
            [Expr.parse("y")], ["y"], {"y": [(0, 1.0)]},
            [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], parameters=["y"],
        )

    with pytest.raises(UnsupportedOptionError, match="does not consume explicit band widths"):
        banded_options = SolverOptions(matrix_layout="banded")
        banded_options.set_banded_width(1, 1)
        BvpDampNaiveSolver(
            [Expr.parse("0")], ["y"], {"y": [(0, 1.0)]},
            [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], options=banded_options,
        )


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["dense", "sparse", "banded"])
def test_bvp_damp_naive_lambdify_frontend_layout_contract(frontend, layout):
    solver = BvpDampNaiveSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["rate"], parameter_values=[1.0],
        options=SolverOptions(
            frontend=frontend, execution="lambdify", matrix_layout=layout,
            telemetry="counters",
        ),
    )
    solution = solver.solve()
    assert solver.frontend == frontend
    assert solver.matrix_layout == layout
    assert solution.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert solution.telemetry_counters()["linear_solves"] >= 1


def test_bvp_damp_naive_exhaustion_is_typed():
    solver = BvpDampNaiveSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["rate"], parameter_values=[1.0], max_iterations=1,
    )
    with pytest.raises(ExhaustionError) as failure:
        solver.solve()
    assert failure.value.solver == "bvp_damp_naive"
    assert failure.value.category == "exhaustion"


@pytest.mark.parametrize(
    ("policy", "policy_parameters"),
    [
        ("Frozen_naive", None),
        ("every_m", [2.0]),
        ("at_high_morm", [1.0]),
        ("at_low_speed", [0.5]),
        ("complex", [2.0, 1.0, 0.5]),
    ],
)
def test_bvp_damp_frozen_native_policy_contract(policy, policy_parameters):
    solver = BvpDampFrozenSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["rate"], parameter_values=[1.0],
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            telemetry="counters",
        ),
        policy=policy, policy_parameters=policy_parameters,
    )
    assert solver.provenance()["solver"] == "bvp_damp_frozen"
    assert solver.provenance()["strategy"] == "Frozen"
    assert solver.provenance()["frozen_policy"] == policy
    solution = solver.solve()
    assert solution.diagnostics()["solver"] == "bvp_damp_frozen"
    assert solution.status == "finished"
    assert solution.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert solution.telemetry_counters()["jacobian_recalculations"] >= 1


def test_bvp_damp_frozen_prepared_parameter_rebind():
    solver = BvpDampFrozenSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
        parameters=["rate"], parameter_values=[1.0],
        options=SolverOptions(frontend="atom_native", execution="lambdify",
                              matrix_layout="sparse", telemetry="timings"),
        policy="every_m", policy_parameters=[2.0],
    )
    solver.prepare()
    first = solver.solve_prepared()
    solver.set_parameters([2.0])
    second = solver.solve_prepared()
    assert first.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert second.values()[0][-1] == pytest.approx(2.0, abs=1e-8)
    assert second.parameters == pytest.approx([2.0])
    assert second.telemetry_timings_ms()["total_ms"] >= 0.0


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["dense", "sparse", "banded"])
def test_bvp_damp_frozen_frontend_layout_contract(frontend, layout):
    solver = BvpDampFrozenSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["rate"], parameter_values=[1.0],
        options=SolverOptions(
            frontend=frontend, execution="lambdify", matrix_layout=layout,
            telemetry="counters",
        ),
        policy="Frozen_naive",
    )
    result = solver.solve()
    assert solver.frontend == frontend
    assert solver.matrix_layout == layout
    assert result.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert result.telemetry_counters()["linear_solves"] >= 1


@pytest.mark.parametrize(
    ("policy", "parameters", "error", "message"),
    [
        ("unknown", None, UnsupportedOptionError, "unsupported Frozen policy"),
        ("Frozen_naive", [1.0], InvalidConfigurationError, "does not accept"),
        ("every_m", None, InvalidConfigurationError, "requires 1"),
        ("every_m", [1.5], InvalidConfigurationError, "positive integer"),
        ("at_high_morm", [0.0], InvalidConfigurationError, "positive norm"),
        ("at_low_speed", [1.1], InvalidConfigurationError, "no greater than 1"),
        ("complex", [1.5, 1.0, 0.5], InvalidConfigurationError, "positive integer"),
    ],
)
def test_bvp_damp_frozen_policy_validation(policy, parameters, error, message):
    with pytest.raises(error, match=message):
        BvpDampFrozenSolver(
            [Expr.parse("0")], ["y"], {"y": [(0, 1.0)]},
            [0.0, 0.5, 1.0], [1.0, 1.0, 1.0],
            policy=policy, policy_parameters=parameters,
        )


def test_bvp_damp_prepared_parameter_continuation_and_restart_contract():
    solver = BvpDampSolver(
        [Expr.parse("alpha * y - z"), Expr.parse("-z")],
        ["y", "z"],
        {"y": [(0, 0.0)], "z": [(0, 1.0)]},
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.25] * 10,
        parameters=["alpha"],
        parameter_values=[2.0],
        options=SolverOptions(
            frontend="expr_legacy",
            execution="lambdify",
            matrix_layout="dense",
            telemetry="counters",
        ),
        bounds={"y": (-10.0, 10.0), "z": (-10.0, 10.0)},
        max_iterations=12,
    )
    assert solver.parameter_names == ["alpha"]
    assert solver.parameter_values == pytest.approx([2.0])

    solver.prepare()
    first = solver.solve_prepared()
    first_counters = first.telemetry_counters()
    assert first.status == "finished"

    solver.set_parameters([5.0])
    assert solver.parameter_values == pytest.approx([5.0])
    second = solver.solve_prepared()
    second_counters = second.telemetry_counters()
    assert second.status == "finished"
    assert second_counters["factorizations"] >= first_counters["factorizations"]
    assert second.parameters == pytest.approx([5.0])

    # A prepared iterate is a warm restart and does not require rebuilding
    # the symbolic callback graph.
    solver.set_prepared_iterate([0.25] * 10)
    warm = solver.solve_prepared()
    assert warm.status == "finished"

    # A restart replaces only the numeric initial guess and keeps the prepared
    # symbolic/runtime plan reusable.
    solver.restart([0.25] * 10)
    assert solver.solve_prepared().status == "finished"


def test_bvp_damp_prepared_lifecycle_and_parameter_failures_are_typed():
    solver = BvpDampSolver(
        [Expr.parse("0")],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        parameters=["alpha"],
        parameter_values=[0.0],
        bounds={"y": (-10.0, 10.0)},
    )
    with pytest.raises(SolverError, match="prepared plan"):
        solver.solve_prepared()
    with pytest.raises(InvalidConfigurationError, match="must contain 1 values"):
        solver.set_parameters([])
    with pytest.raises(InvalidConfigurationError, match="finite"):
        solver.set_parameters([math.nan])
    with pytest.raises(InvalidConfigurationError, match="initial_state"):
        solver.restart([1.0, 1.0])


def test_bvp_damp_solver_specific_options_and_compact_exports(tmp_path):
    solver = BvpDampSolver(
        [Expr.parse("0")],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        options=SolverOptions(telemetry="counters"),
        max_jacobian_refreshes=2,
        max_damping_iterations=3,
        damping_factor=0.25,
        logging_mode="detailed",
        logging_max_events=8,
        bounds={"y": (-10.0, 10.0)},
    )
    solution = solver.solve()
    assert solver.provenance()["strategy"] == "damped"
    assert solver.provenance()["prepared"] == "true"
    assert solution.diagnostics()["solver"] == "bvp_damp"
    csv_path = tmp_path / "solution.csv"
    txt_path = tmp_path / "solution.txt"
    solution.write_solution_csv(str(csv_path))
    solution.write_solution_txt(str(txt_path))
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] in {"mesh,y0", "node_index,y0"}
    assert txt_path.read_text(encoding="utf-8").startswith("BVP_Damp solution\n")

    common = (
        [Expr.parse("0")],
        ["y"],
        {"y": [(0, 1.0)]},
        [0.0, 1.0],
        [1.0, 1.0],
    )
    with pytest.raises(UnsupportedOptionError, match="strategy='damped'"):
        BvpDampSolver(*common, strategy="frozen", bounds={"y": (-10.0, 10.0)})
    with pytest.raises(InvalidConfigurationError, match="grid_method"):
        BvpDampSolver(
            *common,
            max_refinements=1,
            bounds={"y": (-10.0, 10.0)},
        )


def test_failure_and_telemetry_matrix_keeps_categories_and_context():
    """Exercise representative failure categories without relying on stdout."""
    y = Expr.variable("y")

    with pytest.raises(CallbackError, match="callback boom") as captured:
        BdfCallbackSolver(
            1,
            0.0,
            0.2,
            0.1,
            [1.0],
            lambda _t, _state: (_ for _ in ()).throw(RuntimeError("callback boom")),
            options=SolverOptions(telemetry="counters"),
        ).solve()
    assert captured.value.stage == "solve"
    assert captured.value.solver == "bdf"
    assert captured.value.route == "callback"

    with pytest.raises(InvalidConfigurationError, match="greater than zero") as captured:
        BdfSolver(
            [-y],
            ["y"],
            "t",
            [1.0],
            0.0,
            0.2,
            0.0,
        )
    assert isinstance(captured.value, InvalidConfigurationError)

    with pytest.raises(ExhaustionError) as captured:
        BdfSolver(
            [-y],
            ["y"],
            "t",
            [1.0],
            0.0,
            1.0,
            0.01,
            max_steps=1,
            options=SolverOptions(telemetry="counters"),
        ).solve()
    assert captured.value.category == "exhaustion"
    assert captured.value.stage == "solve"

    for mode in ("off", "counters", "timings"):
        solver = BackwardEulerSolver(
            [-y],
            ["y"],
            "t",
            [1.0],
            0.0,
            0.1,
            0.05,
            options=SolverOptions(telemetry=mode),
        )
        solution = solver.solve()
        assert solution.telemetry_mode == mode
        assert "telemetry=" in solution.telemetry_report()
        if mode == "off":
            assert solution.telemetry_counters() == {}
            assert solution.telemetry_timings_ms() == {}
        elif mode == "counters":
            assert solution.telemetry_counters()
            assert solution.telemetry_timings_ms() == {}
        elif mode == "timings":
            assert solution.telemetry_counters() == {}
            assert solution.telemetry_timings_ms()
        else:
            assert solution.telemetry_counters()
            assert solution.telemetry_timings_ms()


def test_bdf_telemetry_modes_have_disjoint_collection_contracts():
    y = Expr.variable("y")
    expected = {
        "off": (False, False),
        "counters": (True, False),
        "timings": (False, True),
        "detailed": (True, True),
    }
    for mode, (has_counters, has_timings) in expected.items():
        solution = BdfSolver(
            [-y],
            ["y"],
            "t",
            [1.0],
            0.0,
            0.1,
            0.05,
            options=SolverOptions(telemetry=mode),
        ).solve()
        assert solution.telemetry_mode == mode
        assert bool(solution.telemetry_counters()) is has_counters
        assert bool(solution.telemetry_timings_ms()) is has_timings
    assert str(TelemetryMode.Detailed) == "Detailed"


def test_public_panic_error_is_exported_for_guarded_native_failures():
    """The typed class is part of the public exception taxonomy."""
    assert issubclass(PanicError, Exception)
    assert issubclass(PanicError, __import__("rustedscithe").RustedSciTheError)
    assert issubclass(PoisonedSolverError, SolverError)
    assert issubclass(PoisonedSolverError, __import__("rustedscithe").RustedSciTheError)


def test_core_ivp_pyo3_types_and_lifecycle_methods_have_python_docstrings():
    """Rust API comments must survive PyO3 export and be useful via help()."""
    documented = (
        (SolverOptions, ("frontend", "execution", "telemetry")),
        (BdfSolver, ("prepare", "solve", "restart", "continue_with_parameters")),
        (BackwardEulerSolver, ("validate", "solve", "continue_to", "restart")),
        (RadauSolver, ("prepare", "solve", "continue_with_parameters", "restart")),
        (Lsode2Solver, ("prepare", "solve", "continue_with_parameters", "restart")),
    )
    for exposed_type, expected_terms in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in expected_terms:
            method = getattr(exposed_type, method_name, None)
            assert method is not None, f"{exposed_type.__name__}.{method_name} is not exposed"
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} lacks a Python-visible contract"
            )


def test_internal_native_helpers_are_not_exposed_as_python_methods():
    """Panic-guard bypass helpers must remain private to the Rust wrapper."""
    internal_methods = {
        BdfSolver: ("solve_inner",),
        BvpDampSolver: ("solve_inner", "detached_solution"),
        NonlinearSolver: ("prepare_inner", "solve_inner"),
        RadauSolver: ("solve_inner",),
        RadauCallbackSolver: ("solve_checked",),
    }
    for exposed_type, names in internal_methods.items():
        for name in names:
            assert not hasattr(exposed_type, name), (
                f"internal helper {exposed_type.__name__}.{name} leaked through PyO3"
            )


def test_bvp_pyo3_types_document_shapes_lifecycle_and_telemetry():
    documented = (
        (
            BvpSciSolution,
            ("values", "interval_residuals", "telemetry_counters", "telemetry_timings_ms", "sample"),
        ),
        (
            BvpSciSolver,
            ("set_parameters", "restart", "solve", "mesh", "parameters"),
        ),
        (
            BvpDampSolution,
            ("mesh", "values", "telemetry_counters", "telemetry_timings_ms", "write_solution_csv"),
        ),
        (
            BvpDampSolver,
            ("solve", "prepare", "solve_prepared", "set_parameters", "restart"),
        ),
    )
    for exposed_type, method_names in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in method_names:
            method = getattr(exposed_type, method_name)
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} needs a useful Python docstring"
            )


def test_callback_ivp_and_nonlinear_pyo3_docs_cover_public_contracts():
    documented = (
        (
            BdfCallbackSolver,
            ("prepare", "is_prepared", "solve", "restart", "telemetry_counters", "telemetry_report", "provenance"),
        ),
        (
            RadauCallbackSolver,
            ("solve", "continue_with_initial_state", "restart", "provenance"),
        ),
        (
            NonlinearSolution,
            ("solution", "residual", "telemetry_counters", "telemetry_timings_ms", "diagnostics"),
        ),
        (
            NonlinearSolver,
            ("variables", "backend", "frontend", "is_prepared", "prepare", "prepare_report", "solve"),
        ),
    )
    for exposed_type, method_names in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in method_names:
            method = getattr(exposed_type, method_name)
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} needs a useful Python docstring"
            )


def test_fitting_and_varpro_pyo3_docs_cover_lifecycle_results_and_telemetry():
    documented = (
        (FittingSolution, ("solution", "report", "diagnostics", "telemetry_counters", "telemetry_timings_ms")),
        (FittingSolver, ("variables", "solve", "solve_with_parameters")),
        (
            GavinFittingSolution,
            (
                "parameters",
                "parameter_standard_errors",
                "fitted_value_standard_errors",
                "parameter_correlation",
                "preparation_report",
                "telemetry_counters",
                "telemetry_timings_ms",
                "diagnostics",
            ),
        ),
        (GavinFittingSolver, ("solve", "is_solved")),
        (VarProSolution, ("solution", "best_fit", "diagnostics", "telemetry_counters", "telemetry_timings_ms")),
        (
            VarProSolver,
            (
                "with_parameters",
                "with_initial_parameters",
                "with_data",
                "with_basis",
                "with_basis_str",
                "with_invariant",
                "with_invariant_str",
                "with_tolerances",
                "with_max_iterations",
                "with_max_function_evaluations",
                "with_step_bound",
                "with_telemetry",
                "solve",
                "is_solved",
            ),
        ),
        (UniversalFitting, ("with_data", "with_unknowns", "with_equation_str", "with_equations_str", "with_polynomial", "with_basis_str", "with_invariant_str", "with_frontend", "with_gavin_solver_controls", "build")),
        (UniversalFittingSolution, ("solution_map", "diagnostics")),
    )
    for exposed_type, method_names in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in method_names:
            method = getattr(exposed_type, method_name)
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} needs a useful Python docstring"
            )


def test_task_and_remaining_ivp_api_docstrings_cover_inspection_surfaces():
    documented = (
        (
            BackwardEulerSolution,
            ("telemetry_counters", "telemetry_timings_ms"),
        ),
        (BackwardEulerSolver, ("status", "telemetry_report")),
        (Lsode2Solution, ("telemetry_counters", "telemetry_timings_ms")),
        (Lsode2Solver, ("provenance", "status", "telemetry_report")),
        (RadauSolver, ("is_prepared",)),
        (SolverOptions, ("as_dict", "band_lower", "band_upper", "sparse_factorization")),
        (
            BvpTaskResult,
            (
                "diagnostics",
                "telemetry_counters",
                "telemetry_report",
                "telemetry_timings_ms",
                "write_telemetry_report",
            ),
        ),
        (
            IvpTaskResult,
            ("times", "states", "diagnostics", "has_continuation", "continuation_info"),
        ),
        (
            TaskDocument,
            ("parse", "from_file", "preview", "check", "route", "options", "run", "run_bvp"),
        ),
    )
    for exposed_type, method_names in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in method_names:
            method = getattr(exposed_type, method_name)
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} needs a useful Python docstring"
            )

    documented_properties = (
        (BvpTaskResult, ("solver", "frontend", "execution", "matrix_layout", "status", "status_code")),
        (IvpTaskResult, ("status", "status_code", "has_continuation")),
    )
    for exposed_type, property_names in documented_properties:
        for property_name in property_names:
            descriptor = exposed_type.__dict__[property_name]
            assert descriptor.__doc__ and len(descriptor.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{property_name} needs a useful Python property docstring"
            )


def test_all_exported_native_public_members_have_docstrings():
    """Every new public PyO3 class member must explain its Python contract."""
    import rustedscithe

    undocumented = []
    for name, member in inspect.getmembers(rustedscithe):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) and not (member.__doc__ and member.__doc__.strip()):
            undocumented.append(f"{name} function")

    for name, exposed_type in inspect.getmembers(rustedscithe, inspect.isclass):
        if name.startswith("_") or exposed_type.__module__ != "rustedscithe":
            continue
        if not (exposed_type.__doc__ and exposed_type.__doc__.strip()):
            undocumented.append(f"{name} class")
        for member_name, member in inspect.getmembers(exposed_type):
            if member_name.startswith("_"):
                continue
            if callable(member) and not (member.__doc__ and member.__doc__.strip()):
                undocumented.append(f"{name}.{member_name}")
        for member_name, descriptor in exposed_type.__dict__.items():
            if member_name.startswith("_"):
                continue
            is_property = isinstance(descriptor, property) or type(descriptor).__name__ in {
                "getset_descriptor",
                "member_descriptor",
            }
            if is_property and not (descriptor.__doc__ and descriptor.__doc__.strip()):
                undocumented.append(f"{name}.{member_name} property")

    assert not undocumented, "Missing Python-facing documentation:\n" + "\n".join(undocumented)


def test_python_type_stubs_parse_and_match_runtime_exports():
    import rustedscithe
    from rustedscithe import _native
    from rustedscithe import facade

    package_dir = Path(__file__).parents[1] / "python" / "rustedscithe"
    native_stub = ast.parse((package_dir / "_native.pyi").read_text(encoding="utf-8"))
    facade_stub = ast.parse((package_dir / "facade.pyi").read_text(encoding="utf-8"))
    public_stub = ast.parse((package_dir / "__init__.pyi").read_text(encoding="utf-8"))

    def declared_exports(module):
        for statement in module.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in statement.targets
            ):
                return set(ast.literal_eval(statement.value))
        raise AssertionError("type stub must declare an explicit __all__")

    def declared_symbols(module):
        return {
            statement.name
            for statement in module.body
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    native_exports = set(name for name in dir(_native) if not name.startswith("_"))
    assert declared_exports(native_stub) == native_exports
    assert native_exports <= declared_symbols(native_stub)

    native_stub_classes = {
        statement.name: {
            member.name
            for member in statement.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        | {
            member.target.id
            for member in statement.body
            if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name)
        }
        for statement in native_stub.body
        if isinstance(statement, ast.ClassDef)
    }
    for name in native_exports:
        exposed_type = getattr(_native, name)
        if not inspect.isclass(exposed_type) or exposed_type.__module__ != "rustedscithe":
            continue
        declared_members = native_stub_classes[name]
        for member_name, member in exposed_type.__dict__.items():
            if member_name.startswith("_"):
                continue
            is_public = callable(member) or type(member).__name__ in {
                "getset_descriptor",
                "member_descriptor",
            }
            assert not is_public or member_name in declared_members, (
                f"{name}.{member_name} is missing from _native.pyi"
            )

    facade_exports = {
        "capabilities", "create_bvp", "create_fit", "create_ivp", "create_nonlinear",
        "fit", "render_table", "run_task_batch", "run_task_batch_report", "solve_bvp",
        "solve_ivp", "solve_nonlinear", "task", "task_dict", "write_table",
    }
    assert facade_exports <= declared_symbols(facade_stub)
    facade_functions = {
        name
        for name, member in inspect.getmembers(facade, inspect.isfunction)
        if not name.startswith("_")
    }
    assert facade_functions <= declared_symbols(facade_stub)
    assert declared_exports(public_stub) == set(rustedscithe.__all__)


def test_native_panic_poisoning_crosses_the_python_boundary():
    """The test-only probe verifies metadata and the no-reuse contract end to end."""
    from rustedscithe import _native

    probe_type = getattr(_native, "_PanicProbeSolver", None)
    if probe_type is None:
        pytest.skip("rebuild with Cargo feature `test-panic-probe` to run this boundary test")

    probe = probe_type()
    with pytest.raises(PanicError) as caught:
        probe.trigger_native_panic()
    assert caught.value.stage == "solve"
    assert caught.value.solver == "panic_probe"
    assert caught.value.route == "test_only"
    assert caught.value.solver_state == "discard"

    with pytest.raises(PoisonedSolverError) as poisoned:
        probe.retry()
    assert poisoned.value.category == "poisoned_solver"


def test_native_solver_guard_keeps_non_panic_failures_retryable():
    """A typed solver error must not poison the wrapper instance."""
    from rustedscithe import _native

    probe_type = getattr(_native, "_PanicProbeSolver", None)
    if probe_type is None:
        pytest.skip("rebuild with Cargo feature `test-panic-probe` to run this boundary test")

    probe = probe_type()
    with pytest.raises(SolverError):
        probe.fail_numerically()
    assert probe.retry() is None


def test_data_processing_spline_resampling_returns_owned_numpy_arrays():
    # Non-contiguous views are intentionally accepted and copied at the ABI.
    x = np.linspace(0.0, 4.0, 9, dtype=np.float64)[::2]
    y = 2.0 * x + 1.0
    x_new, y_new = spline_resample(x, y, 11, kind="linear")
    assert isinstance(x_new, np.ndarray) and x_new.dtype == np.float64
    assert isinstance(y_new, np.ndarray) and y_new.dtype == np.float64
    assert not np.shares_memory(x_new, x) and not np.shares_memory(y_new, y)
    assert len(x_new) == len(y_new) == 11
    np.testing.assert_allclose(y_new, 2.0 * x_new + 1.0, atol=1e-8)
    with pytest.raises(InvalidConfigurationError, match="strictly increasing"):
        spline_resample(np.array([0.0, 1.0, 1.0]), np.ones(3), 5)


def test_data_processing_savgol_filters_and_rejects_invalid_shape():
    sample_count = 9
    samples = np.arange(sample_count, dtype=np.float64) ** 2
    smoothed = savgol_filter(samples, window_length=5, polyorder=2)
    derivative = savgol_filter(samples, window_length=5, polyorder=2, derivative=1)
    np.testing.assert_allclose(smoothed, samples, atol=1e-8)
    np.testing.assert_allclose(derivative[2:-2], 2.0 * np.arange(2, sample_count - 2), atol=1e-8)
    with pytest.raises(InvalidConfigurationError, match="odd"):
        savgol_filter(samples, window_length=4, polyorder=2)


def test_data_processing_lowess_and_quality_metrics_have_defined_edge_semantics():
    x = np.linspace(0.0, 2.0, 21, dtype=np.float64)
    y = 1.5 * x + 0.2 * np.sin(8.0 * x)
    smooth = lowess_smooth(x, y, fraction=0.5, iterations=2, delta=0.0, parallel=False)
    assert smooth.shape == y.shape
    assert smooth.dtype == np.float64
    rmse_value, mae_value, r2_value = fit_quality(y, smooth)
    assert rmse_value >= 0.0 and mae_value >= 0.0
    assert r2_value is not None
    assert fit_quality(np.ones(3), np.ones(3))[2] is None
    with pytest.raises(InvalidConfigurationError, match="same length"):
        fit_quality(np.ones(3), np.ones(2))
    with pytest.raises(InvalidConfigurationError, match="finite"):
        lowess_smooth(x, y, fraction=float("nan"))
    with pytest.raises(SolverError, match="overflowed"):
        fit_quality(np.array([1e308, -1e308]), np.array([0.0, 0.0]))


def test_data_processing_lsq_bspline_dense_and_banded_routes_match():
    x = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x)
    # Cubic clamped knot vector with one interior knot. Strided inputs and
    # optional observation weights exercise the NumPy-to-Rust ownership edge.
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    weights = np.linspace(0.5, 1.5, x.size, dtype=np.float64)
    dense = least_squares_bspline(x[::2], y[::2], knots, degree=3, weights=weights[::2], solver="dense")
    banded = least_squares_bspline(x[::2], y[::2], knots, degree=3, weights=weights[::2], solver="banded")

    assert isinstance(dense, LeastSquaresBSpline)
    assert dense.degree == banded.degree == 3
    assert dense.knots.dtype == dense.coefficients.dtype == np.float64
    assert not np.shares_memory(dense.knots, knots)
    sample = np.linspace(0.0, 1.0, 51, dtype=np.float64)[::2]
    values = dense.evaluate_batch(sample)
    assert values.dtype == np.float64 and not np.shares_memory(values, sample)
    np.testing.assert_allclose(values, banded.evaluate_batch(sample), rtol=1e-8, atol=1e-9)
    assert math.isfinite(dense.evaluate(0.25))


def test_data_processing_lsq_bspline_rejects_unsafe_shapes_and_routes():
    x = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    y = x**2
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    with pytest.raises(InvalidConfigurationError, match="nondecreasing"):
        least_squares_bspline(x, y, np.array([0.0, 0.5, 0.4, 1.0]), 1)
    with pytest.raises(InvalidConfigurationError, match=r"degree \+ 1 times"):
        least_squares_bspline(
            x,
            y,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            3,
        )
    with pytest.raises(InvalidConfigurationError, match="same length"):
        least_squares_bspline(x, y, knots, 2, weights=np.ones(3))
    with pytest.raises(InvalidConfigurationError, match="nonnegative"):
        least_squares_bspline(x, y, knots, 2, weights=-np.ones_like(x))
    with pytest.raises(InvalidConfigurationError, match="at least one weight"):
        least_squares_bspline(x, y, knots, 2, weights=np.zeros_like(x))
    with pytest.raises(InvalidConfigurationError, match="expected 'dense'"):
        least_squares_bspline(x, y, knots, 2, solver="auto")
    with pytest.raises(InvalidConfigurationError, match="base interval"):
        least_squares_bspline(x + 2.0, y, knots, 2)


def test_shooting_bvp_binding_returns_numpy_trajectory_and_opt_in_telemetry():
    from rustedscithe import ShootingBvpSolver

    options = SolverOptions(frontend="atom_native", telemetry="detailed")
    solver = ShootingBvpSolver(
        [Expr.parse("y1"), Expr.parse("-y0")],
        ["y0", "y1"],
        "x",
        0.0,
        math.pi / 2.0,
        "dirichlet",
        0.0,
        "dirichlet",
        1.0,
        initial_guess=0.8,
        tolerance=1e-9,
        max_iterations=30,
        step_size=0.001,
        ivp_method="bdf",
        solver_parameters={"rtol": 1e-9, "atol": 1e-11, "max_step": 0.015},
        options=options,
    )
    result = solver.solve()

    assert result.frontend == "AtomViewNative"
    assert result.preparation_ms >= 0.0
    assert result.mesh_array().ndim == 1
    assert result.states_array().shape == (result.mesh_array().size, 2)
    assert abs(result.states_array()[-1, 0] - 1.0) < 1e-6
    assert abs(result.shooting_value - 1.0) < 1e-5
    assert result.telemetry is not None
    assert result.telemetry["ivp_solves"] >= 2
    assert "total_ms" in result.telemetry
    assert result.telemetry_counters["ivp_solves"] >= 2
    assert isinstance(result.telemetry_counters["ivp_solves"], int)
    assert "total_ms" in result.telemetry_timings_ms
    assert result.preparation_stages_ms


def test_shooting_bvp_errors_keep_native_context_and_reject_aot_explicitly():
    from rustedscithe import ShootingBvpSolver

    with pytest.raises(InvalidConfigurationError) as captured:
        ShootingBvpSolver(
            [Expr.parse("y1")],
            ["y0", "y1"],
            "x",
            0.0,
            1.0,
            "dirichlet",
            0.0,
            "dirichlet",
            1.0,
        )
    assert captured.value.stage == "construct"
    assert captured.value.solver == "shooting_bvp"
    assert captured.value.route == "lambdify"
    assert captured.value.native_kind == "InvalidProblemDimension"

    with pytest.raises(UnsupportedOptionError, match="AOT is not implemented"):
        ShootingBvpSolver(
            [Expr.parse("y1"), Expr.parse("-y0")],
            ["y0", "y1"],
            "x",
            0.0,
            1.0,
            "dirichlet",
            0.0,
            "dirichlet",
            1.0,
            options=SolverOptions(execution="aot"),
        )


def test_solver_options_expose_opt_in_native_logging_levels():
    options = SolverOptions()
    assert options.logging_level == "off"
    options.set_logging_level("INFO")
    assert options.logging_level == "info"
    assert options.as_dict()["logging_level"] == "info"
    with pytest.raises(UnsupportedOptionError, match="does not expose native Rust logging"):
        options.validate_for("radau")

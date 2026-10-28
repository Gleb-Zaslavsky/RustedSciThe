"""Black-box smoke tests for the documented Python contract."""

import ast
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.testing
import pytest
import rustedscithe.facade as python_facade

from rustedscithe import (
    AotArtifactError,
    BandedMatrix,
    BandedRefinementResult,
    BlockTridiagonalLuConsistent,
    BlockTridiagonalRefinementResult,
    BvpError,
    BvpDampSolution,
    BvpDampFrozenSolver,
    BvpDampNaiveSolver,
    BvpDampParameterLoopPoint,
    BvpDampParameterLoopReport,
    BvpDampSolver,
    BvpSciSolution,
    BvpSciParameterLoopPoint,
    BvpSciParameterLoopReport,
    BvpSciSolver,
    BvpSolverRequest,
    PreparedBvpSolver,
    BvpApiSolution,
    BvpPostprocessPlan,
    BvpTaskResult,
    BSplineInterpolator,
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
    IvpParameterLoopPoint,
    FittingSolver,
    GavinFittingSolver,
    GavinFittingSolution,
    Lsode2Solver,
    Lsode2Solution,
    LapackStyleBandedLuFaithful,
    PivotedTridiagonalLu,
    LowessResult,
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
    lowess_fit,
    least_squares_bspline,
    savgol_filter,
    spline_resample,
    symbols,
)


def test_faithful_banded_api_preserves_compact_storage_and_factor_reuse():
    """The Python contract must retain packed bands and reusable LU state."""

    compact = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0, 4.0],
            [2.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = BandedMatrix(4, 1, 1, compact)
    assert matrix.shape == (3, 4)
    assert matrix.storage_bytes == compact.nbytes
    np.testing.assert_array_equal(matrix.values(), compact)

    lu = LapackStyleBandedLuFaithful(4, 1, 1)
    lu.factor(matrix)
    assert lu.is_factorized
    assert lu.ldab == 4
    assert lu.factor_residual_relative(matrix) < 1e-12
    np.testing.assert_allclose(
        lu.solve(np.asarray([6.0, 13.0, 20.0, 22.0])),
        [1.0, 2.0, 3.0, 4.0],
    )
    in_place = np.asarray([6.0, 13.0, 20.0, 22.0], dtype=np.float64)
    lu.solve_in_place(in_place)
    np.testing.assert_allclose(in_place, [1.0, 2.0, 3.0, 4.0])

    rhs = np.asarray(
        [[6.0, 9.0], [13.0, 8.0], [20.0, 2.0], [22.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        lu.solve_multiple(rhs),
        [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]],
    )

    # The factorization owns its numeric copy; changing the input cannot alter
    # a previously factored operator or its subsequent solves.
    compact[1, 1] = 99.0
    np.testing.assert_allclose(
        lu.solve(np.asarray([6.0, 13.0, 20.0, 22.0])),
        [1.0, 2.0, 3.0, 4.0],
    )


def test_faithful_banded_api_supports_separate_bands_and_refinement_report():
    diagonal = np.full(4, 4.0)
    lower = np.asarray([[2.0, 2.0, 2.0, 0.0]], dtype=np.float64)
    upper = np.asarray([[0.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    matrix = BandedMatrix.from_bands(diagonal, lower, upper)
    lu = LapackStyleBandedLuFaithful(4, 1, 1)
    lu.factor(matrix)

    report = lu.solve_with_refinement(
        matrix,
        np.asarray([6.0, 13.0, 20.0, 22.0]),
        max_iters=2,
        tolerance=1e-12,
    )
    assert isinstance(report, BandedRefinementResult)
    assert report.converged
    assert report.relative_residual_inf < 1e-12
    np.testing.assert_allclose(report.solution_array(), [1.0, 2.0, 3.0, 4.0])


def test_faithful_banded_api_supports_diagonal_and_wide_band_systems():
    diagonal = BandedMatrix.from_bands(np.asarray([2.0, 3.0, 4.0]))
    diagonal_lu = LapackStyleBandedLuFaithful(3, 0, 0)
    diagonal_lu.factor(diagonal)
    np.testing.assert_allclose(
        diagonal_lu.solve(np.asarray([2.0, 6.0, 12.0])), [1.0, 2.0, 3.0]
    )

    matrix = BandedMatrix.from_bands(
        np.full(5, 5.0),
        np.asarray(
            [[1.0, 1.0, 1.0, 1.0, 0.0], [0.25, 0.25, 0.25, 0.0, 0.0]],
            dtype=np.float64,
        ),
        np.asarray([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
    )
    wide_lu = LapackStyleBandedLuFaithful(5, 2, 1)
    wide_lu.factor(matrix)
    np.testing.assert_allclose(
        wide_lu.solve(np.asarray([7.0, 14.0, 21.25, 28.5, 29.75])),
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )


def test_faithful_banded_api_rejects_invalid_shape_nonfinite_and_lifecycle():
    with pytest.raises(InvalidConfigurationError):
        BandedMatrix(3, 1, 1, np.ones((2, 3), dtype=np.float64))
    with pytest.raises(InvalidConfigurationError):
        BandedMatrix(3, 1, 1, np.asarray([[1.0, np.nan, 1.0]] * 3))

    lu = LapackStyleBandedLuFaithful(3, 1, 1)
    with pytest.raises(SolverError) as error:
        lu.solve(np.ones(3, dtype=np.float64))
    assert error.value.native_kind == "NotFactorized"
    assert error.value.stage == "solve"

    with pytest.raises(InvalidConfigurationError) as invalid_epsilon:
        LapackStyleBandedLuFaithful(3, 1, 1, pivot_epsilon=-1.0)
    assert getattr(invalid_epsilon.value, "stage", None) is None


def test_faithful_banded_api_maps_native_failure_kinds_without_debug_parsing():
    singular = BandedMatrix(
        3,
        1,
        1,
        np.zeros((3, 3), dtype=np.float64),
    )
    lu = LapackStyleBandedLuFaithful(3, 1, 1)
    with pytest.raises(SolverError) as singular_error:
        lu.factor(singular)
    assert singular_error.value.stage == "factor"
    assert singular_error.value.solver == "lapack_banded"
    assert singular_error.value.native_kind in {"ZeroPivot", "Singular"}

    valid = BandedMatrix(3, 1, 1, np.asarray(
        [[1.0, 1.0, 0.0], [4.0, 4.0, 4.0], [2.0, 2.0, 0.0]],
        dtype=np.float64,
    ))
    lu.factor(valid)
    with pytest.raises(InvalidConfigurationError) as rhs_error:
        lu.solve(np.ones(2, dtype=np.float64))
    assert rhs_error.value.native_kind == "DimensionMismatch"

    with pytest.raises(InvalidConfigurationError):
        lu.solve_multiple(np.ones((2, 1), dtype=np.float64))

    with pytest.raises(InvalidConfigurationError) as layout_error:
        lu.solve_in_place(np.ones((3, 1), dtype=np.float64).reshape(-1)[::2])
    assert layout_error.value.stage == "solve"

    with pytest.raises(InvalidConfigurationError) as band_error:
        valid.set(0, 2, 1.0)
    assert band_error.value.native_kind == "OutOfBounds"


def test_faithful_banded_capabilities_describe_a_kernel_not_a_solver():
    banded = capabilities("lapack_banded")
    assert banded.frontends == []
    assert banded.layouts == ["banded"]
    assert banded.executions == ["factor", "solve", "solve_multiple", "refinement"]
    assert banded.policies == ["sequential"]
    assert banded.telemetry_modes == ["off"]
    assert banded.supports_continuation is False


def test_pivoted_tridiagonal_api_solves_pivot_required_system_without_densifying():
    """The compact pivoted route must handle a tiny leading diagonal safely."""

    lower = np.asarray([2.0, 1.0, 1.0], dtype=np.float64)
    diagonal = np.asarray([1.0e-12, 2.0, 2.0, 2.0], dtype=np.float64)
    upper = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    x_true = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    rhs = np.asarray(
        [
            diagonal[0] * x_true[0] + upper[0] * x_true[1],
            lower[0] * x_true[0] + diagonal[1] * x_true[1] + upper[1] * x_true[2],
            lower[1] * x_true[1] + diagonal[2] * x_true[2] + upper[2] * x_true[3],
            lower[2] * x_true[2] + diagonal[3] * x_true[3],
        ],
        dtype=np.float64,
    )

    lu = PivotedTridiagonalLu(lower, diagonal, upper)
    assert lu.n == 4
    assert lu.pivot_epsilon == pytest.approx(1.0e-14)
    assert len(lu.ipiv) == 4
    assert lu.ipiv[0] == 1
    assert lu.storage_bytes > 0
    np.testing.assert_allclose(lu.solve(rhs), x_true, rtol=1e-12, atol=1e-12)

    rhs_in_place = rhs.copy()
    lu.solve_in_place(rhs_in_place)
    np.testing.assert_allclose(rhs_in_place, x_true, rtol=1e-12, atol=1e-12)


def test_pivoted_tridiagonal_api_owns_bands_and_supports_multiple_rhs_layouts():
    lower = np.asarray([2.0, 1.0], dtype=np.float64)
    diagonal = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
    upper = np.asarray([0.5, 0.25], dtype=np.float64)
    lu = PivotedTridiagonalLu(lower, diagonal, upper)
    lower[:] = 100.0
    diagonal[:] = 100.0
    upper[:] = 100.0

    rhs = np.asarray([[4.0, 1.0], [7.0, 2.0], [8.0, 3.0]], dtype=np.float64)
    expected = np.column_stack([lu.solve(rhs[:, column]) for column in range(2)])
    np.testing.assert_allclose(lu.solve_multiple(rhs), expected)

    stride = 5
    padded = np.full(2 * stride, -99.0, dtype=np.float64)
    padded[0:3] = rhs[:, 0]
    padded[stride : stride + 3] = rhs[:, 1]
    padding_before = padded[[3, 4, 8, 9]].copy()
    lu.solve_multiple_in_place(padded, nrhs=2, stride=stride)
    np.testing.assert_allclose(padded[0:3], expected[:, 0])
    np.testing.assert_allclose(padded[stride : stride + 3], expected[:, 1])
    np.testing.assert_array_equal(padded[[3, 4, 8, 9]], padding_before)


def test_pivoted_tridiagonal_api_reports_typed_failures_and_capabilities():
    with pytest.raises(InvalidConfigurationError):
        PivotedTridiagonalLu(
            np.ones(2), np.ones(3), np.ones(2), pivot_epsilon=-1.0
        )
    with pytest.raises(InvalidConfigurationError):
        PivotedTridiagonalLu(
            np.ones(2), np.asarray([1.0, np.nan, 1.0]), np.ones(2)
        )

    with pytest.raises(SolverError) as singular:
        PivotedTridiagonalLu(np.zeros(2), np.zeros(3), np.zeros(2))
    assert singular.value.native_kind == "ZeroPivot"
    assert singular.value.stage == "factor"
    assert singular.value.solver == "tridiagonal_lu_pp"

    lu = PivotedTridiagonalLu(np.ones(2), np.full(3, 2.0), np.ones(2))
    with pytest.raises(InvalidConfigurationError):
        lu.solve(np.ones(2, dtype=np.float64))
    with pytest.raises(InvalidConfigurationError):
        lu.solve(np.asarray([1.0, np.inf, 2.0], dtype=np.float64))

    capabilities_record = capabilities("pivoted_tridiagonal")
    assert capabilities_record.frontends == []
    assert capabilities_record.executions == ["factor", "solve", "solve_multiple"]
    assert capabilities_record.layouts == ["tridiagonal"]
    assert capabilities_record.telemetry_modes == ["off"]
    assert capabilities("tridiagonal_lu_pp").solver == "pivoted_tridiagonal"


def test_block_tridiagonal_api_solves_compact_system_and_reports_diagnostics():
    lower = np.asarray([[[0.25, 0.0], [0.0, 0.25]]], dtype=np.float64)
    diagonal = np.asarray(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[5.0, 1.0], [1.0, 4.0]],
        ],
        dtype=np.float64,
    )
    upper = np.asarray([[[0.5, 0.0], [0.0, 0.5]]], dtype=np.float64)
    matrix = np.block([[diagonal[0], upper[0]], [lower[0], diagonal[1]]])
    x_true = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    rhs = matrix @ x_true

    lu = BlockTridiagonalLuConsistent(lower, diagonal, upper)
    assert lu.n_blocks == 2
    assert lu.block_size == 2
    assert lu.n == 4
    assert lu.is_factorized is True
    assert lu.storage_bytes > 0
    assert np.isfinite(lu.factor_residual_relative)
    assert len(lu.pivot_diagnostics()) == 2
    assert len(lu.condition_diagnostics()) == 2
    np.testing.assert_allclose(lu.solve(rhs), x_true, rtol=1e-12, atol=1e-12)

    rhs_multiple = np.column_stack([rhs, 2.0 * rhs])
    np.testing.assert_allclose(
        lu.solve_multiple(rhs_multiple), np.column_stack([x_true, 2.0 * x_true])
    )

    report = lu.solve_with_refinement(rhs, refinement_steps=2)
    assert isinstance(report, BlockTridiagonalRefinementResult)
    assert report.requested_steps == 2
    np.testing.assert_allclose(report.solution_array(), x_true, rtol=1e-12, atol=1e-12)


def test_block_tridiagonal_api_owns_inputs_and_maps_typed_failures():
    lower = np.zeros((1, 2, 2), dtype=np.float64)
    diagonal = np.asarray([np.eye(2), 2.0 * np.eye(2)], dtype=np.float64)
    upper = np.zeros((1, 2, 2), dtype=np.float64)
    lu = BlockTridiagonalLuConsistent(lower, diagonal, upper)
    diagonal[:] = 100.0

    np.testing.assert_allclose(lu.solve(np.ones(4)), np.asarray([1.0, 1.0, 0.5, 0.5]))
    with pytest.raises(InvalidConfigurationError) as nonfinite:
        BlockTridiagonalLuConsistent(
            lower, np.asarray([[[1.0, np.nan], [0.0, 1.0]], np.eye(2)]), upper
        )
    assert nonfinite.value.native_kind == "NonFiniteInput"
    assert nonfinite.value.solver == "block_tridiagonal"

    with pytest.raises(InvalidConfigurationError):
        lu.solve(np.ones(3))
    with pytest.raises(InvalidConfigurationError):
        lu.solve(np.asarray([1.0, 2.0, np.inf, 4.0]))

    capabilities_record = capabilities("block_tridiagonal")
    assert capabilities_record.frontends == []
    assert capabilities_record.layouts == ["block_tridiagonal"]
    assert capabilities_record.executions == ["factor", "solve", "solve_multiple", "refinement"]
    assert capabilities_record.telemetry_modes == ["off"]
    assert capabilities("block_tridiagonal_lu_consistent").solver == "block_tridiagonal"


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


def test_animation_worker_module_is_discoverable_in_clean_child_process(tmp_path):
    missing_input = tmp_path / "missing-trajectory.npz"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rustedscithe._animation_worker",
            "--input",
            str(missing_input),
            "--dimension",
            "2",
            "--axis-names",
            "null",
            "--speed",
            "1.0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 2
    assert "RUSTEDSCITHE_ANIMATION_ERROR:" in result.stderr
    assert "missing-trajectory.npz" in result.stderr


def test_animation_renderer_initialization_failure_is_returned_to_python(monkeypatch):
    # wgpu parses this environment variable into an empty backend set. That
    # reliably fails adapter discovery before a window is opened, exercising
    # the real Rust/PyO3 panic -> child marker -> facade RuntimeError path.
    monkeypatch.setenv("WGPU_BACKEND", "rustedscithe-invalid-backend")
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    times = np.asarray([0.0, 1.0], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Unable to find a GPU"):
        animate_trajectory_2d(points, times)


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
    assert result.continuation_telemetry() == {
        "segments": 1,
        "fresh_preparations": 1,
        "prepared_reuses": 0,
        "mesh_restarts": 0,
        "initial_guess_restarts": 0,
        "peak_result_elements": len(result.mesh()),
    }
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
    # This capability describes the shared SolverOptions.log-level mapping;
    # BVP_Damp's independent logging controls remain available on its facade.
    assert damp.supports_logging is False
    assert damp.sparse_factorization_policies == []

    scipy_like = capabilities("bvp_sci")
    assert scipy_like.frontends == ["expr_legacy", "atom_native", "numerical"]
    assert scipy_like.executions == ["lambdify", "aot", "numerical"]
    assert scipy_like.layouts == ["dense", "sparse", "banded"]
    assert scipy_like.methods == ["scipy_like"]
    assert scipy_like.supports_continuation
    assert scipy_like.supports_analytical_callbacks
    assert scipy_like.analytical_callback_layouts == ["dense"]
    assert scipy_like.sparse_factorization_policies == [
        "standard_faer",
        "reusable_faer_experimental",
        "reusable_faer_simplicial_experimental",
        "superlu",
    ]
    assert scipy_like.telemetry_modes == ["off", "counters", "timings", "detailed"]
    assert scipy_like.supports_logging is True
    with pytest.raises(UnsupportedOptionError, match="numerical callbacks require"):
        SolverOptions(frontend="numerical", execution="lambdify").validate_for("bvp_sci")
    with pytest.raises(UnsupportedOptionError, match="matrix_layout='dense'"):
        SolverOptions(
            frontend="numerical", execution="numerical", matrix_layout="sparse"
        ).validate_for("bvp_sci")
    with pytest.raises(UnsupportedOptionError):
        SolverOptions(frontend="numerical", execution="numerical").validate_for("bdf")


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
    assert solver.is_prepared
    result = solver.solve([1.0])
    assert result.solution()[0] == pytest.approx(2.0, rel=1e-6)
    assert result.residual_norm < 1e-8
    assert result.telemetry_counters()["jacobian_evaluations"] >= 1
    assert "total_ms" in result.telemetry_timings_ms()


def test_nonlinear_solver_explicit_parameter_rebind_shortcut():
    system = SymbolicSystem([Expr.parse("a*x - 2")], ["x", "a"])
    solver = NonlinearSolver(
        system,
        parameters=["a"],
        parameter_values=[1.0],
        tolerance=1e-10,
    )
    solver.prepare()
    first = solver.solve_with_parameter_values([1.0], [1.0])
    second = solver.solve_with_parameter_values([1.0], [2.0])
    assert first.solution()[0] == pytest.approx(2.0, abs=1e-8)
    assert second.solution()[0] == pytest.approx(1.0, abs=1e-8)
    with pytest.raises(InvalidConfigurationError, match="parameter"):
        solver.solve_with_parameter_values([1.0], [])


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
    counters = result.telemetry_counters()
    timings = result.telemetry_timings_ms()
    assert result.telemetry_mode == "detailed"
    assert counters["residual_evaluations"] > 0
    assert "state_residual_evaluations" in counters
    assert "trial_jacobian_evaluations" in counters
    assert "trust_region_trials" in counters
    assert "linear_system_solve_ms" in timings
    assert "trust_region_subproblem_ms" in timings
    attempt_rows = result.telemetry_attempt_counters()
    assert attempt_rows
    assert attempt_rows[0]["iteration"] == 0
    assert "residual_evaluations" in attempt_rows[0]
    attempt_timings = result.telemetry_attempt_timings_ms()
    assert attempt_timings
    assert "linear_system_solve_ms" in attempt_timings[0]


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


@pytest.mark.parametrize(
    ("mode", "has_counters", "has_timings"),
    [
        ("off", False, False),
        ("counters", True, False),
        ("timings", False, True),
        ("detailed", True, True),
    ],
)
def test_nonlinear_solver_uses_the_four_mode_telemetry_contract(
    mode, has_counters, has_timings
):
    system = SymbolicSystem([Expr.parse("x-1")], ["x"])
    result = NonlinearSolver(system, telemetry=mode).solve([0.0])
    assert bool(result.telemetry_counters()) is has_counters
    assert bool(result.telemetry_timings_ms()) is has_timings


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
    assert result.telemetry_mode == "detailed"
    assert result.telemetry_counters()
    assert result.telemetry_timings_ms()
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

    unsupported_observation_policy = UniversalFitting("gavin")
    unsupported_observation_policy.with_observation_policy("sequential")
    with pytest.raises(UnsupportedOptionError, match="observation_policy"):
        unsupported_observation_policy.build()

    invalid_observation_policy = UniversalFitting("lm")
    with pytest.raises(InvalidConfigurationError, match="min_observations"):
        invalid_observation_policy.with_observation_policy("parallel", 0)

    invalid_lm = UniversalFitting("lm")
    invalid_lm.with_data(x, [2.0 * value + 1.0 for value in x])
    invalid_lm.with_arg("x")
    invalid_lm.with_unknowns(["a", "b"])
    invalid_lm.with_initial_guess([1.0, 0.0])
    with pytest.raises(InvalidConfigurationError) as fitting_error:
        invalid_lm.build()
    assert fitting_error.value.native_kind == "InvalidInput"
    assert fitting_error.value.stage == "build"

    gavin_counters = UniversalFitting("gavin")
    gavin_counters.with_telemetry("counters")
    gavin_counters.with_data(x, [2.0 * value + 1.0 for value in x])
    gavin_counters.with_arg("x")
    gavin_counters.with_equations_str(["a*x+b"])
    gavin_counters.with_unknowns(["a", "b"])
    gavin_counters.with_initial_guess([1.9, 0.9])
    gavin_result = gavin_counters.build()
    assert gavin_result.telemetry_mode == "counters"
    assert gavin_result.telemetry_counters()
    assert gavin_result.telemetry_timings_ms() == {}

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
    with pytest.raises(InvalidConfigurationError, match="auto is not a native Gavin policy"):
        GavinFittingSolver(
            [Expr.parse("a*x")], [0.0, 1.0], [1.0, 2.0], "x", ["a"], [1.0],
            execution_policy="auto",
        )

    missing_basis = VarProSolver("x")
    missing_basis.with_parameters(["tau"])
    missing_basis.with_initial_parameters([1.0])
    missing_basis.with_data([0.0, 1.0], [1.0, 0.5])
    with pytest.raises(InvalidConfigurationError):
        missing_basis.solve()
    with pytest.raises(SolverError, match="one-shot"):
        missing_basis.solve()


def test_least_squares_timings_mode_excludes_counters_for_all_python_facades():
    fitting = FittingSolver(
        [Expr.parse("a-1")], [0.0], variables=["a"], telemetry="timings"
    ).solve()
    assert fitting.telemetry_mode == "timings"
    assert fitting.telemetry_counters() == {}
    assert fitting.telemetry_timings_ms()

    varpro = VarProSolver("x")
    varpro.with_parameters(["tau"])
    varpro.with_initial_parameters([1.0])
    varpro.with_data([0.0, 1.0, 2.0], [1.0, math.exp(-1.0), math.exp(-2.0)])
    varpro.with_basis_str("tau", "exp(-x/tau)")
    varpro.with_telemetry("timings")
    varpro_result = varpro.solve()
    assert varpro_result.telemetry_mode == "timings"
    assert varpro_result.telemetry_counters() == {}
    assert varpro_result.telemetry_timings_ms()

    universal = UniversalFitting("lm")
    universal.with_data([0.0, 1.0], [1.0, 2.0])
    universal.with_arg("x")
    universal.with_equations_str(["a*x+b"])
    universal.with_unknowns(["a", "b"])
    universal.with_initial_guess([1.0, 1.0])
    universal.with_telemetry("timings")
    result = universal.build()
    assert result.telemetry_counters() == {}
    assert result.telemetry_timings_ms()


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


@pytest.mark.parametrize(
    ("mode", "has_counters", "has_timings"),
    [
        ("off", False, False),
        ("counters", True, False),
        ("timings", False, True),
        ("detailed", True, True),
    ],
)
def test_gavin_telemetry_mode_channels_are_independent(mode, has_counters, has_timings):
    x = np.linspace(0.0, 1.0, 12).tolist()
    solver = GavinFittingSolver(
        [Expr.parse("a*x+b")], x, [1.7 * value + 0.4 for value in x],
        "x", ["a", "b"], [1.6, 0.3], telemetry_mode=mode,
    )
    result = solver.solve()
    assert bool(result.telemetry_counters()) is has_counters
    assert bool(result.telemetry_timings_ms()) is has_timings


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


def test_nonlinear_capabilities_advertise_native_dense_aot():
    nonlinear = capabilities("nonlinear_system")
    assert nonlinear.frontends == ["expr_legacy", "atom_native"]
    assert nonlinear.executions == ["lambdify", "aot"]
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


def test_capability_registry_covers_standalone_and_method_specific_fitting_routes():
    gavin = capabilities("gavin_fitting")
    assert gavin.frontends == ["expr_legacy", "atom_native"]
    assert gavin.executions == ["lambdify"]
    assert gavin.layouts == ["dense"]
    assert gavin.policies == ["sequential", "parallel"]
    assert gavin.methods == ["gavin_lm"]
    assert not gavin.supports_continuation

    varpro = capabilities("varpro")
    assert varpro.frontends == ["expr_legacy"]
    assert varpro.executions == ["lambdify"]
    assert varpro.layouts == ["dense"]
    assert varpro.policies == ["sequential"]
    assert varpro.methods == ["variable_projection"]

    universal = capabilities("universal_fitting")
    assert universal.methods == ["lm", "gavin", "varpro"]
    assert universal.frontends == ["expr_legacy", "atom_native"]
    assert capabilities("universal_fitting_lm").frontends == [
        "expr_legacy", "atom_native"
    ]
    assert capabilities("universal_fitting_gavin").policies == [
        "sequential", "parallel"
    ]
    assert capabilities("universal_fitting_varpro").frontends == ["expr_legacy"]
    assert capabilities("universal_fitting_varpro").policies == ["sequential"]

    for method, frontend, execution_policy in (
        ("lm", "atom_native", "parallel"),
        ("gavin", "atom_native", "parallel"),
        ("varpro", "expr_legacy", "sequential"),
    ):
        facade = UniversalFitting(method)
        facade.with_frontend(frontend)
        facade.with_execution_policy(execution_policy, 1)
        assert facade.method == method

    with pytest.raises(UnsupportedOptionError):
        SolverOptions(frontend="atom_native").validate_for("varpro")
    with pytest.raises(UnsupportedOptionError):
        SolverOptions(execution="aot").validate_for("gavin_fitting")


def test_every_advertised_nonlinear_method_runs_on_both_symbolic_frontends():
    """Keep the advertised method/frontend axes executable, not aspirational."""
    advertised = capabilities("nonlinear_system")
    assert advertised.frontends == ["expr_legacy", "atom_native"]
    assert advertised.methods

    for frontend in advertised.frontends:
        for method in advertised.methods:
            problem = SymbolicSystem([Expr.parse("x - 2")], ["x"])
            solver = NonlinearSolver(
                problem,
                method=method,
                frontend=frontend,
                tolerance=1e-8,
                max_iterations=100,
            )
            result = solver.solve([0.0])

            assert result.solution()[0] == pytest.approx(2.0, abs=1e-4), (
                f"advertised route failed: frontend={frontend}, method={method}"
            )
            assert math.isfinite(result.residual_norm)


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
    assert capabilities("nonstiff").telemetry_modes == ["off"]
    with pytest.raises(UnsupportedOptionError, match="telemetry mode 'timings'"):
        SolverOptions(telemetry="timings").validate_for("nonstiff")
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


def test_scalar_expression_multi_value_substitution_matches_sequential_route():
    expression = Expr.parse("sin(x) + y^2 + exp(z)")
    replacements = {"x": 0.5, "z": -1.0, "unused": 12.0}

    mapped = expression.substitute_values(replacements)
    sequential = expression.substitute("x", 0.5).substitute("z", -1.0)
    assert mapped.variables() == ["y"]
    assert mapped.evaluate([2.0]) == pytest.approx(sequential.evaluate([2.0]))
    assert expression.variables() == ["x", "y", "z"]

    with pytest.raises(InvalidExpressionError, match="must not be empty"):
        expression.substitute_values({"": 1.0})
    with pytest.raises(InvalidExpressionError, match="finite"):
        expression.substitute_values({"x": float("nan")})
    with pytest.raises(TypeError, match="boolean"):
        expression.substitute_values({"x": True})
    with pytest.raises(TypeError, match="keys must be strings"):
        expression.substitute_values({1: 2.0})


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


@pytest.mark.parametrize("frontend", [SymbolicFrontend.ExprLegacy, SymbolicFrontend.AtomNative])
def test_symbolic_system_numpy_batch_and_reusable_out_buffers(frontend):
    x, y = Expr.symbols(["x", "y"])
    system = SymbolicSystem([x**2 + y, x * y], ["x", "y"])
    compiled = system.lambdify(frontend=frontend, diagnostics=True)
    assert compiled.residual_count == 2
    assert compiled.jacobian_shape == (2, 2)
    provenance = compiled.provenance()
    assert provenance["requested_frontend"] == compiled.frontend
    assert provenance["effective_frontend"] == compiled.frontend
    assert provenance["jacobian_structure"] == "dense"
    assert provenance["structural_nnz"] is None
    preparation_stages = compiled.diagnostics()["preparation_stages"]
    assert isinstance(preparation_stages, list)
    assert {stage["name"] for stage in preparation_stages} >= {
        "input_validation",
        "jacobian_differentiation",
    }
    points = np.array([[2.0, -1.0, 3.0], [4.0, -1.0, 5.0]])[:, ::2]

    expected_residuals = np.array([[7.0, 6.0], [21.0, 20.0]])
    expected_jacobians = np.array(
        [[[4.0, 1.0], [3.0, 2.0]], [[8.0, 1.0], [5.0, 4.0]]]
    )
    np.testing.assert_allclose(compiled.evaluate_batch(points), expected_residuals)
    np.testing.assert_allclose(compiled.evaluate_jacobian_batch(points), expected_jacobians)

    residual_out = np.empty_like(expected_residuals)
    jacobian_out = np.empty_like(expected_jacobians)
    assert compiled.evaluate_batch_into(points, residual_out) is residual_out
    assert compiled.evaluate_jacobian_batch_into(points, jacobian_out) is jacobian_out
    np.testing.assert_allclose(residual_out, expected_residuals)
    np.testing.assert_allclose(jacobian_out, expected_jacobians)

    point_out = np.empty(2)
    jacobian_point_out = np.empty((2, 2))
    one_point = np.array([2.0, 3.0])
    assert compiled.evaluate_array_into(one_point, point_out) is point_out
    assert compiled.evaluate_jacobian_array_into(one_point, jacobian_point_out) is jacobian_point_out
    np.testing.assert_allclose(point_out, expected_residuals[0])
    np.testing.assert_allclose(jacobian_point_out, expected_jacobians[0])
    np.testing.assert_allclose(compiled.evaluate_array(one_point), expected_residuals[0])
    np.testing.assert_allclose(compiled.evaluate_jacobian_array(one_point), expected_jacobians[0])

    report = compiled.diagnostics()
    assert report["evaluations"] == 2 * 2 + 2 * 2 * 2 + 2 * 2 + 2 * 2 * 2 + 2 + 4 + 2 + 4
    assert report["callback_time_ms"] >= 0.0


def test_symbolic_system_out_alias_shape_and_writability_contract():
    x, y = Expr.symbols(["x", "y"])
    compiled = SymbolicSystem([x + y, x * y], ["x", "y"]).lambdify()
    assert compiled.diagnostics()["preparation_stages"] is None
    shared = np.array([[2.0, 3.0], [4.0, 5.0]])
    returned = compiled.evaluate_batch_into(shared, shared)
    assert returned is shared
    np.testing.assert_allclose(shared, [[5.0, 6.0], [9.0, 20.0]])

    with pytest.raises(InvalidConfigurationError, match="out must have shape"):
        compiled.evaluate_batch_into(np.ones((3, 2)), np.empty((3, 1)))
    read_only = np.empty((2, 2))
    read_only.flags.writeable = False
    with pytest.raises(InvalidConfigurationError, match="out must be writable"):
        compiled.evaluate_batch_into(np.ones((2, 2)), read_only)
    with pytest.raises(InvalidConfigurationError, match="columns"):
        compiled.evaluate_batch(np.ones((2, 1)))
    with pytest.raises(InvalidConfigurationError, match="finite"):
        compiled.evaluate_batch(np.array([[1.0, np.nan]]))
    with pytest.raises(InvalidConfigurationError, match="float64"):
        compiled.evaluate_batch(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(InvalidConfigurationError, match="float64"):
        compiled.evaluate_batch_into(np.ones((2, 2)), np.empty((2, 2), dtype=np.float32))

    strided_out_storage = np.empty((2, 4), dtype=np.float64)
    strided_out = strided_out_storage[:, ::2]
    assert compiled.evaluate_batch_into(np.array([[2.0, 3.0], [4.0, 5.0]]), strided_out) is strided_out
    np.testing.assert_allclose(strided_out, [[5.0, 6.0], [9.0, 20.0]])


def test_symbolic_system_empty_and_long_reuse_batches_keep_numpy_shapes():
    x = Expr.variable("x")
    compiled = SymbolicSystem([x**2, x + 1.0], ["x"]).lambdify()
    empty_points = np.empty((0, 1), dtype=np.float64)
    assert compiled.evaluate_batch(empty_points).shape == (0, 2)
    assert compiled.evaluate_jacobian_batch(empty_points).shape == (0, 2, 1)

    points = np.linspace(-2.0, 2.0, 17, dtype=np.float64).reshape(-1, 1)
    out = np.empty((len(points), 2), dtype=np.float64)
    for _ in range(200):
        assert compiled.evaluate_batch_into(points, out) is out
    np.testing.assert_allclose(out[:, 0], points[:, 0] ** 2)
    np.testing.assert_allclose(out[:, 1], points[:, 0] + 1.0)


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


def test_bdf_long_continuation_keeps_preparation_and_detached_results_stable():
    y = Expr.variable("y")
    rate = Expr.variable("rate")
    solver = BdfSolver(
        [-rate * y], ["y"], "t", [1.0], 0.0, 0.05, 0.01,
        rtol=1e-5,
        atol=1e-8,
        options=SolverOptions(telemetry="counters"),
        parameters=["rate"],
        parameter_values=[0.5],
    )
    solver.prepare()
    first = solver.solve()
    first_times = first.times()
    first_states = first.states()
    assert solver.is_prepared()

    t_end = 0.05
    for index in range(24):
        t_end += 0.05
        segment = solver.continue_with_parameters([0.5 + index * 0.01], t_end)
        assert segment.status == "finished"
        assert len(segment.times()) == len(segment.states())
        assert all(math.isfinite(value) for row in segment.states() for value in row)
        assert solver.is_prepared

    # Results are detached snapshots: later continuation cannot mutate earlier
    # trajectories. This is a wrapper-visible retention contract, not a claim
    # that Python tracemalloc observes allocations made by Rust.
    assert first.times() == first_times
    assert first.states() == first_states
    assert len(first.states()) <= 32


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
    rebound = solver.solve_with_parameter_values([1.0], [2.0])
    assert rebound.final_state[0] == pytest.approx(math.exp(-2.0), rel=4e-3)


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
        scopes = solution.telemetry_timing_scopes()
        assert any(scope["stage"] == "newton_ms" for scope in scopes)
        assert any(scope["kind"] == "child" for scope in scopes)
        adaptive_steps = solution.telemetry_adaptive_steps()
        if has_timings:
            assert solution.telemetry_initial_step() > 0.0
            assert adaptive_steps
            assert all("accepted" in step for step in adaptive_steps)
        else:
            assert adaptive_steps == []


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


def test_radau_long_parameter_series_retains_preparation_and_detached_results():
    y, rate = Expr.variable("y"), Expr.variable("rate")
    solver = RadauSolver(
        [-rate * y], ["y"], "t", 0.0, 0.05, 0.01,
        parameters=["rate"],
        options=SolverOptions(telemetry="counters"),
    )
    first = solver.solve([1.0], [0.5])
    first_state = first.final_state
    assert solver.is_prepared
    for index in range(16):
        result = solver.solve_with_parameter_values(
            [1.0], [0.5 + index * 0.01]
        )
        assert result.final_time == pytest.approx(0.05)
        assert all(math.isfinite(value) for value in result.final_state)
        assert solver.is_prepared
    assert first.final_state == first_state


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

    # Widths are inferred from the symbolic Jacobian when omitted. The
    # explicit setter remains an optional override for validated hints.
    automatic = RadauSolver(
        [-y], ["y"], "t", 0.0, 0.1, 0.05,
        options=SolverOptions(matrix_layout="banded"),
        jacobian=jacobian,
    )
    assert automatic.solve([1.0]).final_state[0] == pytest.approx(math.exp(-0.1), rel=3e-3)

    narrow = SolverOptions(matrix_layout="banded")
    narrow.set_banded_width(0, 0)
    with pytest.raises(InvalidConfigurationError, match="narrower than the inferred"):
        RadauSolver(
            [Expr.parse("-2*y0 + y1"), Expr.parse("y0 - 2*y1 + y2"), Expr.parse("y1 - 2*y2")],
            ["y0", "y1", "y2"], "t", 0.0, 0.1, 0.05,
            options=narrow,
        ).prepare()


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["dense", "sparse", "banded"])
def test_radau_frontend_layout_capability_cross_product(frontend, layout):
    y = Expr.variable(f"radau_{frontend}_{layout}_y")
    options = SolverOptions(
        frontend=frontend,
        matrix_layout=layout,
        telemetry="counters",
    )
    # Structured routes need an explicit layout-aware Jacobian; this matrix
    # checks automatic bandwidth inference for both symbolic frontends.
    solver = RadauSolver(
        [-y], [str(y)], "t", 0.0, 0.1, 0.05,
        options=options,
        jacobian=[Expr.constant(-1.0)],
    )
    solution = solver.solve([1.0])

    assert solution.final_time == pytest.approx(0.1)
    assert solution.final_state[0] == pytest.approx(math.exp(-0.1), rel=4e-3)
    assert solver.provenance()["frontend"] == frontend
    assert solver.provenance()["matrix_layout"] == layout
    assert solution.telemetry_counters()["accepted_steps"] > 0


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
            telemetry="detailed",
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
    assert bdf_solution.telemetry_aot_metadata()["execution"] == "aot"
    assert bdf_solution.telemetry_aot_counters()["aot_link_attempts"] >= 1
    assert bdf_solution.telemetry_aot_timings_ms()["cold.aot_build.ms"] >= 0.0
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
    assert be_solution.telemetry_aot_metadata()["execution"] == "aot"
    assert be_solution.telemetry_aot_counters()["aot_link_attempts"] >= 1
    assert be_solution.telemetry_aot_timings_ms()["cold.aot_build.ms"] >= 0.0
    assert backward_euler.provenance()["execution"] == "aot"

    yl = Expr.variable("yl")
    lsode2 = Lsode2Solver(
        [-yl], ["yl"], "t", [1.0], 0.0, 0.1, 0.05,
        options=options("lsode2"),
    )
    lsode2_solution = lsode2.solve()
    assert lsode2_solution.status == "finished_native_faithful"
    assert lsode2_solution.telemetry_aot_metadata()["execution"] == "aot"
    assert lsode2_solution.telemetry_aot_counters()["aot_build_attempts"] >= 1
    assert lsode2_solution.telemetry_aot_timings_ms()["cold.aot_link.ms"] >= 0.0


def test_ivp_aot_telemetry_mode_matrix_for_all_adapters(tmp_path):
    compiler_path = shutil.which("tcc") or shutil.which("gcc")
    if compiler_path is None:
        pytest.skip("no supported C compiler is available for the AOT telemetry matrix")
    toolchain = "tcc" if compiler_path.lower().endswith(("tcc", "tcc.exe")) else "gcc"
    modes = ("off", "counters", "timings", "detailed")

    # Reuse each solver's artifact directory across modes. The first route may
    # build; subsequent modes exercise telemetry projection on cache reuse too.
    for solver_name in ("bdf", "backward_euler", "lsode2", "radau"):
        for mode in modes:
            variable = Expr.variable(f"y_{solver_name}")
            options = SolverOptions(
                execution="aot",
                aot_policy="build_if_missing",
                aot_toolchain=toolchain,
                aot_profile="debug",
                aot_output_dir=str(tmp_path / solver_name),
                telemetry=mode,
            )
            if solver_name == "bdf":
                solver = BdfSolver(
                    [-variable], [f"y_{solver_name}"], "t", [1.0], 0.0, 0.05, 0.05,
                    options=options,
                )
                solver.prepare()
                solution = solver.solve()
            elif solver_name == "backward_euler":
                solver = BackwardEulerSolver(
                    [-variable], [f"y_{solver_name}"], "t", [1.0], 0.0, 0.05, h=0.05,
                    options=options,
                )
                solution = solver.solve()
            elif solver_name == "lsode2":
                solver = Lsode2Solver(
                    [-variable], [f"y_{solver_name}"], "t", [1.0], 0.0, 0.05, 0.05,
                    options=options,
                )
                solution = solver.solve()
            else:
                solver = RadauSolver(
                    [-variable], [f"y_{solver_name}"], "t", 0.0, 0.05, 0.05,
                    options=options,
                )
                solver.prepare()
                solution = solver.solve([1.0])

            assert solver.provenance()["execution"] == "aot"
            metadata = solution.telemetry_aot_metadata()
            if mode == "off":
                assert metadata == {}
            else:
                assert metadata["execution"] == "aot"
                assert metadata["mode"] == mode
            counters = solution.telemetry_aot_counters()
            timings = solution.telemetry_aot_timings_ms()
            assert bool(counters) == (mode in {"counters", "detailed"}), (solver_name, mode, counters)
            assert bool(timings) == (mode in {"timings", "detailed"}), (solver_name, mode, timings)
            assert all(math.isfinite(value) and value >= 0.0 for value in timings.values())


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


def test_backward_euler_long_parameter_series_keeps_detached_results_stable():
    y, rate = Expr.variable("y"), Expr.variable("rate")
    solver = BackwardEulerSolver(
        [-rate * y], ["y"], "t", [1.0], 0.0, 0.05,
        h=0.01, parameters=["rate"], parameter_values=[0.5],
        options=SolverOptions(telemetry="counters"),
    )
    first = solver.solve()
    first_times, first_states = first.times(), first.states()
    end_time = 0.05
    for index in range(16):
        end_time += 0.05
        result = solver.continue_to(end_time, [0.5 + index * 0.01])
        assert result.status == "finished"
        assert all(math.isfinite(value) for row in result.states() for value in row)
    assert first.times() == first_times
    assert first.states() == first_states
    assert len(first.states()) <= 8


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


@pytest.mark.parametrize("solver_name", ["bdf", "radau", "backward_euler", "lsode2"])
def test_ivp_atom_native_dense_aot_cross_family_contract(tmp_path, solver_name):
    """Exercise the AtomNative+AOT intersection for every stiff IVP adapter."""
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("AtomNative AOT route needs tcc or gcc")
    toolchain = (
        "tcc" if Path(compiler).name.lower() in {"tcc", "tcc.exe"} else "gcc"
    )
    state_name = f"aot_atom_{solver_name}_y"
    y = Expr.variable(state_name)
    options = SolverOptions(
        frontend="atom_native",
        execution="aot",
        matrix_layout="dense",
        aot_policy="build_if_missing",
        aot_toolchain=toolchain,
        aot_profile="debug",
        aot_output_dir=str(tmp_path / solver_name),
        telemetry="counters",
    )

    if solver_name == "bdf":
        solver = BdfSolver(
            [-y], [state_name], "t", [1.0], 0.0, 0.1, 0.05, options=options
        )
        solution = solver.solve()
        end_time = solution.times()[-1]
        final_state = solution.states()[-1][0]
    elif solver_name == "radau":
        solver = RadauSolver(
            [-y], [state_name], "t", 0.0, 0.1, 0.05, options=options
        )
        solution = solver.solve([1.0])
        end_time = solution.final_time
        final_state = solution.final_state[0]
    elif solver_name == "backward_euler":
        solver = BackwardEulerSolver(
            [-y], [state_name], "t", [1.0], 0.0, 0.1, h=0.05, options=options
        )
        solution = solver.solve()
        end_time = solution.times()[-1]
        final_state = solution.states()[-1][0]
    else:
        solver = Lsode2Solver(
            [-y], [state_name], "t", [1.0], 0.0, 0.1, 0.05, options=options
        )
        solution = solver.solve()
        end_time = solution.final_time
        final_state = solution.final_state[0]

    assert end_time == pytest.approx(0.1)
    assert final_state == pytest.approx(math.exp(-0.1), rel=5e-3)
    assert solver.provenance()["frontend"] == "atom_native"
    assert solver.provenance()["execution"] == "aot"
    assert solution.telemetry_counters()["aot_build_attempts"] >= 1


@pytest.mark.parametrize("solver_name", ["radau", "lsode2"])
@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["sparse", "banded"])
def test_ivp_structured_aot_frontend_contract(tmp_path, solver_name, frontend, layout):
    """Exercise the supported structured-AOT intersections for IVP routes."""
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("structured IVP AOT route needs tcc or gcc")
    toolchain = (
        "tcc" if Path(compiler).name.lower() in {"tcc", "tcc.exe"} else "gcc"
    )
    y_name = f"structured_aot_{solver_name}_{frontend}_{layout}_y"
    z_name = f"structured_aot_{solver_name}_{frontend}_{layout}_z"
    y = Expr.variable(y_name)
    z = Expr.variable(z_name)
    rhs = [-y, -2.0 * z]
    state_names = [y_name, z_name]
    options = SolverOptions(
        frontend=frontend,
        execution="aot",
        matrix_layout=layout,
        aot_policy="build_if_missing",
        aot_toolchain=toolchain,
        aot_profile="debug",
        aot_output_dir=str(tmp_path / f"{solver_name}-{frontend}-{layout}"),
        telemetry="counters",
    )
    if layout == "banded":
        options.set_banded_width(1, 1)

    if solver_name == "radau":
        solver = RadauSolver(
            rhs, state_names, "t", 0.0, 0.1, 0.05,
            options=options,
            jacobian=[
                Expr.constant(-1.0), Expr.constant(0.0),
                Expr.constant(0.0), Expr.constant(-2.0),
            ],
        )
        solution = solver.solve([1.0, 1.0])
        end_time = solution.final_time
        final_state = solution.final_state
    else:
        solver = Lsode2Solver(
            rhs, state_names, "t", [1.0, 1.0], 0.0, 0.1, 0.05, options=options
        )
        solution = solver.solve()
        end_time = solution.final_time
        final_state = solution.final_state

    assert end_time == pytest.approx(0.1)
    assert final_state[0] == pytest.approx(math.exp(-0.1), rel=5e-3)
    assert final_state[1] == pytest.approx(math.exp(-0.2), rel=5e-3)
    assert solver.provenance()["frontend"] == frontend
    assert solver.provenance()["execution"] == "aot"
    assert solver.provenance()["matrix_layout"] == layout
    assert solution.telemetry_counters()["aot_build_attempts"] >= 1


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

    expected_modes = {
        "off": (False, False),
        "counters": (True, False),
        "timings": (False, True),
        "detailed": (True, True),
    }
    for mode, (has_counters, has_timings) in expected_modes.items():
        for solution in solve_family(mode):
            assert solution.telemetry_mode == mode
            assert bool(solution.telemetry_counters()) is has_counters
            assert bool(solution.telemetry_timings_ms()) is has_timings


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


def test_backend_specific_options_are_rejected_when_the_route_cannot_consume_them():
    y = Expr.variable("y_backend_options")

    assert SolverOptions().as_dict()["sparse_factorization"] == "standard_faer"
    dense_with_width = SolverOptions()
    dense_with_width.set_banded_width(1, 1)
    with pytest.raises(UnsupportedOptionError, match="only valid for matrix_layout='banded'"):
        BdfSolver(
            [-y], ["y_backend_options"], "t", [1.0], 0.0, 0.1, 0.05,
            options=dense_with_width,
        )

    sparse_factorization = SolverOptions(matrix_layout="sparse")
    sparse_factorization.set_sparse_factorization("superlu")
    assert sparse_factorization.as_dict()["sparse_factorization"] == "superlu"
    with pytest.raises(UnsupportedOptionError, match="only by the BVP_sci Sparse route"):
        Lsode2Solver(
            [-y], ["y_backend_options"], "t", [1.0], 0.0, 0.1, 0.05,
            options=sparse_factorization,
        )

    # The same policy is meaningful when the actual BVP_sci route is Sparse.
    sparse_factorization.validate_for("bvp_sci")


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
    assert solution.evaluation_scope in {
        "bridge_bdf_callbacks",
        "native_faithful_inner_loop",
    }
    assert "evaluations" in solution.evaluation_counter_note
    assert "evaluation_scope" in solution.diagnostics()
    assert solution.telemetry_counters()["evaluation_residual_calls"] >= 0
    assert solution.telemetry_counters()["evaluation_jacobian_calls"] >= 0
    assert solution.telemetry_counters()["evaluation_linear_solves"] >= 0
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


def test_lsode2_effective_telemetry_is_scoped_and_distinguishes_disabled_modes():
    y = Expr.variable("y")
    solver = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.025,
        options=SolverOptions(telemetry="detailed"),
    )
    solver.prepare()
    result = solver.solve()

    counters = result.telemetry_counters()
    timings = result.telemetry_timings_ms()
    assert result.evaluation_scope in {
        "bridge_bdf_callbacks",
        "native_faithful_inner_loop",
    }
    assert {
        "evaluation_residual_calls",
        "evaluation_jacobian_calls",
        "evaluation_linear_solves",
    } <= counters.keys()
    assert {
        "evaluation_residual_ms",
        "evaluation_jacobian_ms",
        "evaluation_linear_solve_ms",
    } <= timings.keys()
    assert "evaluation_counter_note=" in solver.telemetry_report()

    off = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.025,
        options=SolverOptions(telemetry="off"),
    )
    off.prepare()
    off_result = off.solve()
    assert off_result.evaluation_scope == result.evaluation_scope
    assert off_result.telemetry_counters() == {}
    assert off_result.telemetry_timings_ms() == {}


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

    option_width = SolverOptions(matrix_layout="banded")
    option_width.set_banded_width(1, 1)
    option_width_solver = Lsode2Solver(
        [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1, options=option_width
    )
    assert option_width_solver.provenance()["matrix_layout"] == "banded"

    with pytest.raises(InvalidConfigurationError, match="conflicting band_lower"):
        Lsode2Solver(
            [-y], ["y"], "t", [1.0], 0.0, 0.1, 0.1,
            options=option_width, band_lower=2, band_upper=1,
        )

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


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("controller", ["bdf", "adams", "auto"])
def test_lsode2_advertised_controller_frontend_cross_product(frontend, controller):
    state = Expr.variable(f"lsode2_{frontend}_{controller}_state")
    rate = Expr.variable("rate")
    options = SolverOptions(
        frontend=frontend,
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="counters",
    )
    solver = python_facade.create_ivp(
        "lsode2",
        [-rate * state],
        [str(state)],
        "t",
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.1,
        0.025,
        parameters=["rate"],
        parameter_values=np.asarray([0.8], dtype=np.float64),
        options=options,
        controller=controller,
        rtol=1e-8,
        atol=1e-10,
    )
    result = solver.solve()

    assert result.status.startswith("finished")
    assert result.final_time == pytest.approx(0.1)
    assert result.final_state[0] == pytest.approx(math.exp(-0.08), rel=2e-5)
    assert solver.provenance()["frontend"] == frontend
    assert solver.provenance()["method"] == controller
    assert solver.provenance()["execution"] == "lambdify"
    assert result.telemetry_counters()["solve_calls"] >= 1


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


def test_lsode2_long_parameter_series_retains_preparation_and_detached_results():
    y, rate = Expr.variable("y"), Expr.variable("rate")
    solver = Lsode2Solver(
        [-rate * y], ["y"], "t", [1.0], 0.0, 0.05, 0.01,
        parameters=["rate"], parameter_values=[0.5],
        options=SolverOptions(telemetry="counters"),
    )
    first = solver.solve()
    first_times, first_states = first.times(), first.states()
    assert solver.is_prepared
    final_time = 0.05
    for index in range(16):
        final_time += 0.05
        result = solver.continue_with_parameters([0.5 + index * 0.01], final_time)
        assert result.final_time == pytest.approx(final_time)
        assert all(math.isfinite(value) for row in result.states() for value in row)
        assert solver.is_prepared
    assert first.times() == first_times
    assert first.states() == first_states


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


def test_lsode2_symbolic_banded_layout_infers_width_and_zero_width_is_explicit():
    y = Expr.variable("lsode2_banded_y")
    automatic = Lsode2Solver(
        [-y], [str(y)], "t", [1.0], 0.0, 0.2, 0.05,
        options=SolverOptions(matrix_layout="banded", telemetry="counters"),
    )
    automatic_solution = automatic.solve()
    assert automatic_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=5e-3)
    assert automatic.provenance()["matrix_layout"] == "banded"

    explicit_diagonal = Lsode2Solver(
        [-y], [str(y)], "t", [1.0], 0.0, 0.2, 0.05,
        options=SolverOptions(matrix_layout="banded", telemetry="counters"),
        band_lower=0,
        band_upper=0,
    )
    explicit_solution = explicit_diagonal.solve()
    assert explicit_solution.final_state[0] == pytest.approx(math.exp(-0.2), rel=5e-3)

    with pytest.raises(UnsupportedOptionError, match="requires symbolic expressions"):
        Lsode2Solver(
            [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.1,
            options=SolverOptions(matrix_layout="banded"),
            residual_callback=lambda _t, state: [-state[0]],
            jacobian_callback=lambda _t, _state: [[-1.0]],
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


def test_bvp_api_typed_request_builds_native_sci_route_and_keeps_telemetry():
    """The generic facade should route to Rust and return node-major arrays."""
    y = Expr.variable("y")
    request = BvpSolverRequest.sci(
        [-y], ["y"], "x", [0.0, 0.5, 1.0], [(0, "left", 1.0)],
        [[1.0], [math.exp(-0.5)], [math.exp(-1.0)]],
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            telemetry="counters",
        ),
        tolerance=1e-5,
    )
    solver = request.prepare()
    assert isinstance(solver, PreparedBvpSolver)
    assert solver.route == "sci"
    solution = solver.solve()
    assert isinstance(solution, BvpApiSolution)
    assert solution.route == "sci"
    assert solution.values_array().shape[1] == 1
    assert solution.values()[0][0] == pytest.approx(1.0, abs=1e-7)
    telemetry = solver.telemetry_snapshot()
    assert telemetry["route"] == "sci"
    assert telemetry["mode"] == "counters"
    assert telemetry["counters"]["rhs_calls"] > 0
    assert telemetry["timings_ms"] == {}


def test_bvp_api_sci_external_parameter_loop_uses_native_route_report():
    """Fixed inputs are prepared once and rebound before each continued solve."""
    y = Expr.variable("y")
    request = BvpSolverRequest.sci(
        [Expr.variable("rate") * y],
        ["y"],
        "x",
        [index / 20.0 for index in range(21)],
        [(0, "left", 1.0)],
        [[1.0] for _ in range(21)],
        options=SolverOptions(frontend="expr_legacy", telemetry="counters"),
        tolerance=1e-6,
        external_parameters=["rate"],
        external_parameter_values=[0.5],
    )
    solver = request.prepare()
    report = solver.solve_parameter_loop([[0.5], [1.0]])

    assert isinstance(report, BvpSciParameterLoopReport)
    assert len(report.points) == 2
    assert [point.external_parameters for point in report.points] == [[0.5], [1.0]]
    assert all(point.succeeded for point in report.points)
    assert report.points[0].solution.values()[-1][0] == pytest.approx(math.exp(0.5), rel=2e-4)
    assert report.points[1].solution.values()[-1][0] == pytest.approx(math.e, rel=2e-4)
    assert all(point.telemetry_counters["rhs_calls"] > 0 for point in report.points)
    assert solver.telemetry_snapshot()["counters"]["rhs_calls"] > 0


def test_bvp_api_frozen_parameter_loop_returns_native_damped_report():
    """The facade does not replace the Frozen/Naive point report contract."""
    request = BvpSolverRequest.frozen(
        [Expr.variable("rate")],
        ["y"],
        "x",
        [index / 4 for index in range(5)],
        [(0, "left", 0.0)],
        [[0.0] for _ in range(5)],
        parameters=["rate"],
        parameter_values=[1.0],
        options=SolverOptions(frontend="atom_native", telemetry="counters"),
    )
    report = request.prepare().solve_parameter_loop([[1.0], [2.0]])

    assert isinstance(report, BvpDampParameterLoopReport)
    assert [point.parameters for point in report.points] == [[1.0], [2.0]]
    assert all(point.status == "converged" for point in report.points)
    first_solution = report.points[0].solution
    second_solution = report.points[1].solution
    assert first_solution is not None and second_solution is not None
    # The native BVP_Damp report keeps state-major [state][mesh] values.
    assert first_solution.values()[0][-1] == pytest.approx(1.0, abs=1e-7)
    assert second_solution.values()[0][-1] == pytest.approx(2.0, abs=1e-7)
    assert report.points[1].telemetry_counters()["residual_calls"] > 0


@pytest.mark.parametrize(
    ("route", "guess"),
    [
        ("damped", [[1.0], [math.exp(-0.5)]]),
        ("frozen", [[1.0], [math.exp(-0.5)], [math.exp(-1.0)]]),
    ],
)
def test_bvp_api_damped_and_frozen_requests_keep_route_specific_guess_shapes(route, guess):
    y = Expr.variable("y")
    options = SolverOptions(frontend="atom_native", telemetry="counters")
    if route == "damped":
        request = BvpSolverRequest.damped(
            [-y], ["y"], "x", [0.0, 0.5, 1.0], [(0, "left", 1.0)], guess,
            options=options, bounds={"y": (-10.0, 10.0)}, tolerance=1e-5,
        )
    else:
        request = BvpSolverRequest.frozen(
            [-y], ["y"], "x", [0.0, 0.5, 1.0], [(0, "left", 1.0)], guess,
            options=options, tolerance=1e-5,
        )
    solver = request.prepare()
    assert solver.route == route
    solution = solver.solve()
    assert solution.route == route
    assert solution.values_array().shape == (3, 1)
    terminal_value = solution.values()[-1][0]
    assert math.isfinite(terminal_value)
    assert 0.0 < terminal_value < 1.0
    if route == "damped":
        # The default forward discretization on two intervals yields (1 - h)^2.
        assert terminal_value == pytest.approx(0.25, abs=2e-3)
    telemetry = solver.telemetry_snapshot()
    assert telemetry["route"] == route
    assert telemetry["counters"]


def test_bvp_api_frozen_accepts_shared_text_logging_level():
    y = Expr.variable("y")
    options = SolverOptions(frontend="atom_native")
    options.set_logging_level("info")
    request = BvpSolverRequest.frozen(
        [-y],
        ["y"],
        "x",
        [0.0, 0.5, 1.0],
        [(0, "left", 1.0)],
        [[1.0], [math.exp(-0.5)], [math.exp(-1.0)]],
        options=options,
        tolerance=1e-5,
    )
    assert request.prepare().route == "frozen"


def test_bvp_api_rejects_unsupported_numerical_symbolic_facade_route():
    y = Expr.variable("y")
    with pytest.raises(UnsupportedOptionError, match="numerical callbacks"):
        BvpSolverRequest.sci(
            [-y], ["y"], "x", [0.0, 1.0], [(0, "left", 1.0)], [[1.0], [math.exp(-1.0)]],
            options=SolverOptions(execution="numerical"),
        )


def test_bvp_api_sci_free_parameter_callback_and_native_postprocessing(tmp_path):
    """Unknown BVP parameters and output actions remain native facade features."""
    p = Expr.variable("p")
    y = Expr.variable("y")
    request = BvpSolverRequest.sci(
        [p], ["y"], "x", [0.0, 0.5, 1.0], [(0, "left", 0.0)],
        [[0.0], [0.5], [1.0]],
        options=SolverOptions(frontend="expr_legacy", telemetry="counters"),
        tolerance=1e-6,
        free_parameter_names=["p"],
        free_parameter_guess=[0.7],
        boundary_residual=lambda _ya, yb, _params: [yb[0] - 1.0],
        boundary_jacobian=lambda _ya, _yb, _params: (
            [[0.0]], [[1.0]], [[0.0]],
        ),
    )
    solver = request.prepare()
    solution = solver.solve()
    assert solution.sci_details is not None
    assert solution.sci_details.status_code == 0
    assert solver.parameter_guess() == pytest.approx([1.0], rel=1e-5)

    output_path = tmp_path / "facade.csv"
    report = solver.execute_postprocessing(BvpPostprocessPlan().save_csv(str(output_path)))
    assert output_path.exists()
    assert report[0]["status"] == "done"
    assert report[0]["path"] == str(output_path)


def test_bvp_api_sci_callback_failure_keeps_typed_native_context():
    y = Expr.variable("y")

    def fail_boundary(_ya, _yb, _parameters):
        raise LookupError("original BVP_api boundary failure")

    request = BvpSolverRequest.sci(
        [y], ["y"], "x", [0.0, 0.5, 1.0], [(0, "left", 0.0)],
        [[0.0], [0.5], [1.0]],
        free_parameter_names=["p"],
        free_parameter_guess=[1.0],
        boundary_residual=fail_boundary,
    )

    with pytest.raises(CallbackError) as failure:
        request.prepare().solve()

    assert failure.value.stage == "solve"
    assert failure.value.solver == "bvp_api"
    assert failure.value.route == "sci"
    assert failure.value.native_kind == "Callback"
    assert "original BVP_api boundary failure" in str(failure.value)
    assert isinstance(failure.value.__cause__, LookupError)
    assert str(failure.value.__cause__) == "original BVP_api boundary failure"


@pytest.mark.parametrize("route", ["damped", "frozen", "sci"])
@pytest.mark.parametrize("mode", ["off", "counters", "timings", "detailed"])
def test_bvp_api_telemetry_modes_preserve_native_channel_semantics(route, mode):
    y = Expr.variable("y")
    mesh = [0.0, 0.5, 1.0]
    common = ([-y], ["y"], "x", mesh, [(0, "left", 1.0)])
    guess = [[1.0], [math.exp(-0.5)], [math.exp(-1.0)]]
    options = SolverOptions(frontend="expr_legacy", telemetry=mode)
    if route == "damped":
        request = BvpSolverRequest.damped(
            *common, guess[:-1], options=options, tolerance=1e-5,
            bounds={"y": (-10.0, 10.0)},
        )
    elif route == "frozen":
        request = BvpSolverRequest.frozen(
            *common, guess, options=options, tolerance=1e-5,
        )
    else:
        request = BvpSolverRequest.sci(
            *common, guess, options=options, tolerance=1e-5,
        )

    solver = request.prepare()
    solution = solver.solve()
    snapshot = solver.telemetry_snapshot()
    assert snapshot["mode"] == mode
    if route in {"damped", "frozen"}:
        assert set(snapshot["scopes"]) == {
            "solve", "mesh_revision", "iteration", "damping_trial",
        }
        if mode in {"counters", "detailed"}:
            assert set(snapshot["counters"]) == {
                "iterations", "scalar_evaluations", "residual_calls",
                "residual_requests", "jacobian_requests", "jacobian_recalculations",
                "residual_chunks", "jacobian_chunks", "conversions", "copies",
                "damping_trials", "damping_rejections", "linear_solves",
                "factorizations", "factorization_cache_hits",
                "factorization_invalidations", "rhs_solves", "grid_refinements",
            }
        if mode in {"timings", "detailed"}:
            assert {
                "total", "residual", "jacobian", "linear_system",
                "factorization", "rhs_solve", "symbolic_operations", "grid_refinement",
            } <= set(snapshot["timings_ms"])
        expected_scope_fields = {
            "events" if mode in {"counters", "detailed"} else None,
            "elapsed_ms" if mode in {"timings", "detailed"} else None,
        } - {None}
        assert all(set(scope) == expected_scope_fields for scope in snapshot["scopes"].values())
    else:
        # The generic projection and the retained route-native BVP_sci view
        # must expose identical counter/timing vocabularies and measured values.
        details = solution.sci_details
        assert details is not None
        assert snapshot["counters"] == details.telemetry_counters()
        assert snapshot["timings_ms"] == details.telemetry_timings_ms()
    if mode in {"off", "timings"}:
        assert snapshot["counters"] == {}
    else:
        assert snapshot["counters"]
    if mode in {"off", "counters"}:
        assert snapshot["timings_ms"] == {}
    else:
        assert snapshot["timings_ms"]


def test_bvp_api_factory_defaults_match_native_iteration_budget():
    assert inspect.signature(BvpSolverRequest.damped).parameters["max_iterations"].default == 25
    assert inspect.signature(BvpSolverRequest.frozen).parameters["max_iterations"].default == 25


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("toolchain", ["gcc", "tcc", "zig", "rust"])
def test_bvp_api_aot_require_prebuilt_forwards_frontend_and_toolchain(
    tmp_path, frontend, toolchain,
):
    """A missing strict artifact must fail natively, never downgrade to Lambdify."""
    state = f"missing_{frontend}_{toolchain}_{tmp_path.name.replace('-', '_')}"
    y = Expr.variable(state)
    options = SolverOptions(
        frontend=frontend,
        execution="aot",
        aot_policy="require_prebuilt",
        aot_toolchain=toolchain,
        aot_output_dir=str(tmp_path / "artifacts"),
    )
    request = BvpSolverRequest.sci(
        [-y], [state], "x", [0.0, 1.0], [(0, "left", 1.0)], [[1.0], [math.exp(-1.0)]],
        options=options,
    )

    with pytest.raises((AotArtifactError, PreparationError)) as failure:
        request.prepare()

    assert failure.value.stage == "construct"
    assert failure.value.solver == "bvp_api"
    assert failure.value.route == "sci"
    assert failure.value.native_kind in {"AotPreparation", "AotHandoff"}


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
    assert solution.telemetry_preparation_scope() == "bvp"
    assert timings["reported_preparation_ms"] >= 0.0
    scopes = solution.telemetry_timing_scopes()
    assert scopes
    assert all("kind" in scope and "parent" in scope for scope in scopes)
    mesh_passes = solution.telemetry_mesh_passes()
    if telemetry == "timings":
        assert solution.telemetry_counters() == {}
        assert mesh_passes == []
        assert solution.telemetry_newton_trace() == []
    else:
        assert mesh_passes
        assert mesh_passes[0]["linear_solves"] >= 1
        assert solution.telemetry_counters()["linear_solves"] >= 1
        assert solution.telemetry_newton_trace()
        assert "accepted" in solution.telemetry_newton_trace()[0]
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


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
def test_bvp_sci_banded_layout_infers_width_from_symbolic_equations(frontend):
    y = Expr.variable("y")
    solver = BvpSciSolver(
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0],
        [1.0, 0.7, math.exp(-1.0)],
        options=SolverOptions(
            frontend=frontend,
            execution="lambdify",
            matrix_layout="banded",
            telemetry="counters",
        ),
        tolerance=1e-5,
    )
    result = solver.solve()
    assert solver.matrix_layout == "banded"
    assert result.status_code == 0
    assert result.values()[-1][0] == pytest.approx(math.exp(-1.0), rel=3e-3)

    narrow = SolverOptions(
        frontend=frontend,
        execution="lambdify",
        matrix_layout="banded",
    )
    narrow.set_banded_width(0, 0)
    with pytest.raises(InvalidConfigurationError, match="narrower than the inferred"):
        BvpSciSolver(
            [Expr.parse("y + z"), Expr.parse("y")],
            ["y", "z"],
            [Expr.parse("ya_y - 1"), Expr.parse("yb_z - 1")],
            [0.0, 0.5, 1.0],
            [1.0, 1.0, 0.7, 0.7, math.exp(-1.0), math.exp(-1.0)],
            options=narrow,
        )


def test_bvp_sci_long_parameter_series_reuses_plan_and_detaches_results():
    rate = Expr.variable("rate")
    solver = BvpSciSolver(
        [rate],
        ["y"],
        [Expr.parse("ya_y"), Expr.parse("yb_y - 1")],
        [index / 8.0 for index in range(9)],
        [0.0] * 9,
        parameters=["rate"],
        parameter_guess=[1.0],
        options=SolverOptions(telemetry="detailed"),
        tolerance=1e-5,
        max_nodes=64,
        max_mesh_refinements=6,
    )
    first = solver.solve()
    first_mesh = first.mesh()
    first_values = first.values_flat()
    first_counters = first.telemetry_counters()
    assert first.status_code == 0
    assert first_counters["preparation_calls"] == 1

    for index in range(20):
        parameter_guess = 0.5 + index * 0.05
        solver.set_parameter_guess([parameter_guess])
        result = solver.solve()
        assert result.status_code == 0
        assert result.parameters == pytest.approx([1.0], abs=1e-6)
        assert solver.parameter_guess() == pytest.approx([1.0], abs=1e-6)
        assert result.values()[-1][0] == pytest.approx(1.0, abs=1e-6)
        assert len(result.mesh()) <= 64
        assert result.telemetry_counters()["preparation_calls"] == 1

    # Same mesh/state dimensions should reuse the prepared plan and buffers.
    # Keep the assertion tolerant of a one-time solver capacity expansion.
    final_counters = result.telemetry_counters()
    assert final_counters["workspace_resizes"] <= first_counters["workspace_resizes"] + 1
    assert first.mesh() == first_mesh
    assert first.values_flat() == first_values


def test_bvp_sci_external_parameter_loop_keeps_failures_and_continues():
    rate = Expr.variable("rate")
    seen_parameter_pairs = []

    def boundary(ya, yb, runtime_parameters):
        # Free Newton unknowns precede fixed external inputs in callback data.
        seen_parameter_pairs.append(tuple(runtime_parameters))
        slope, rate_value = runtime_parameters
        if rate_value < 0.0:
            raise RuntimeError("intentional parameter-point failure")
        return [ya[0], yb[0] - slope]

    solver = BvpSciSolver(
        [rate], ["y"], [], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["slope"], parameter_guess=[0.0],
        external_parameters=["rate"], external_parameter_values=[1.0],
        boundary_callback=boundary,
        options=SolverOptions(frontend="expr_legacy", execution="lambdify", telemetry="detailed"),
        tolerance=1e-6,
    )
    report = solver.solve_parameter_loop([[1.0], [-1.0], [2.0]])

    assert isinstance(report, BvpSciParameterLoopReport)
    assert all(isinstance(point, BvpSciParameterLoopPoint) for point in report.points)
    assert [point.succeeded for point in report.points] == [True, False, True]
    assert report.stopped_early is False
    assert report.points[0].external_parameters == pytest.approx([1.0])
    assert {(0.0, 1.0), (1.0, -1.0), (1.0, 2.0)}.issubset(set(seen_parameter_pairs))
    assert report.points[1].native_kind is not None
    assert "intentional parameter-point failure" in report.points[1].error_message
    with pytest.raises(CallbackError, match="intentional parameter-point failure") as failure:
        report.points[1].raise_if_failed()
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert report.points[0].solution.parameters == pytest.approx([1.0], abs=1e-6)
    assert report.points[2].solution.values()[-1][0] == pytest.approx(2.0, abs=1e-6)
    assert report.points[2].solution.parameters == pytest.approx([2.0], abs=1e-6)
    assert report.points[0].solution.values()[0][0] == pytest.approx(0.0, abs=1e-6)
    assert report.points[0].telemetry_mode == "detailed"
    assert report.points[0].telemetry_counters["residual_evaluations"] > 0
    assert solver.external_parameter_values() == pytest.approx([2.0])
    assert solver.parameter_guess() == pytest.approx([2.0], abs=1e-6)


@pytest.mark.slow
def test_bvp_sci_external_parameter_loop_runs_on_atom_aot(tmp_path):
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no supported C compiler is available for BVP_sci AOT loop")
    toolchain = "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"
    rate = Expr.variable("rate")

    def boundary(ya, yb, runtime_parameters):
        slope, rate_value = runtime_parameters
        if rate_value < 0.0:
            raise RuntimeError("intentional AOT parameter-point failure")
        return [ya[0], yb[0] - slope]

    solver = BvpSciSolver(
        [rate], ["y"], [], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        parameters=["slope"], parameter_guess=[0.0],
        external_parameters=["rate"], external_parameter_values=[1.0],
        boundary_callback=boundary,
        options=SolverOptions(
            frontend="atom_native", execution="aot", aot_toolchain=toolchain,
            aot_profile="debug", aot_output_dir=str(tmp_path / "external-input-aot"),
            telemetry="detailed",
        ),
        tolerance=1e-6,
    )
    report = solver.solve_parameter_loop([[1.0], [-1.0], [2.0]])

    assert [point.succeeded for point in report.points] == [True, False, True]
    assert report.points[0].solution.external_parameters == pytest.approx([1.0])
    assert report.points[0].solution.parameters == pytest.approx([1.0], abs=1e-6)
    assert report.points[2].solution.values()[-1][0] == pytest.approx(2.0, abs=1e-6)
    assert report.points[2].solution.parameters == pytest.approx([2.0], abs=1e-6)
    assert "intentional AOT parameter-point failure" in report.points[1].error_message
    assert report.points[0].solution.telemetry_aot_metadata()["execution"] == "aot"


def test_bvp_sci_constructor_rejects_ambiguous_parameter_guess_aliases():
    rate = Expr.variable("rate")
    with pytest.raises(InvalidConfigurationError, match="pass only parameter_guess"):
        BvpSciSolver(
            [rate],
            ["y"],
            [Expr.parse("ya_y"), Expr.parse("yb_y - 1")],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0],
            parameters=["rate"],
            parameter_guess=[0.7],
            parameter_values=[0.7],
        )


def test_bvp_sci_numerical_callbacks_support_analytic_state_and_parameter_jacobians():
    solver = BvpSciSolver(
        [],
        ["y"],
        [Expr.parse("ya_y"), Expr.parse("yb_y - 1")],
        [0.0, 0.5, 1.0],
        [0.0, 0.4, 0.8],
        parameters=["rate"],
        parameter_values=[0.8],
        options=SolverOptions(
            frontend="numerical", execution="numerical", matrix_layout="dense",
            telemetry="detailed",
        ),
        rhs_callback=lambda _x, _y, parameters: [parameters[0]],
        rhs_jacobian_callback=lambda _x, _y, _p: [[0.0]],
        rhs_parameter_jacobian_callback=lambda _x, _y, _p: [[1.0]],
    )
    assert solver.frontend == "numerical"
    result = solver.solve()
    assert result.status_code == 0
    assert result.parameters == pytest.approx([1.0], abs=1e-7)
    assert result.values()[-1][0] == pytest.approx(1.0, abs=1e-7)
    counters = result.telemetry_counters()
    assert counters["residual_evaluations"] > 0
    assert counters["jacobian_evaluations"] > 0


def test_bvp_sci_numerical_callbacks_use_fd_jacobian_and_preserve_callback_errors():
    solver = BvpSciSolver(
        [], ["y"], [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
        options=SolverOptions(frontend="numerical", execution="numerical", telemetry="counters"),
        output="dense",
        rhs_callback=lambda _x, y, _p: [-y[0]],
    )
    result = solver.solve()
    assert solver.frontend == "numerical"
    assert result.status_code == 0
    assert result.sample(1.0)[0] == pytest.approx(math.exp(-1.0), rel=3e-3)
    assert result.telemetry_counters()["jacobian_evaluations"] > 0

    def failing_rhs(_x, _y, _p):
        raise RuntimeError("numerical RHS fixture failure")

    failing = BvpSciSolver(
        [], ["y"], [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
        options=SolverOptions(frontend="numerical", execution="numerical"),
        rhs_callback=failing_rhs,
    )
    with pytest.raises(CallbackError, match="numerical RHS fixture failure") as error:
        failing.solve()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "numerical RHS fixture failure"

    def failing_jacobian(_x, _y, _p):
        raise RuntimeError("numerical Jacobian fixture failure")

    failing_jacobian_solver = BvpSciSolver(
        [], ["y"], [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
        options=SolverOptions(frontend="numerical", execution="numerical"),
        rhs_callback=lambda _x, y, _p: [-y[0]],
        rhs_jacobian_callback=failing_jacobian,
    )
    with pytest.raises(CallbackError, match="numerical Jacobian fixture failure") as error:
        failing_jacobian_solver.solve()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "numerical Jacobian fixture failure"


def test_bvp_sci_numerical_callback_route_rejects_mixed_and_non_dense_options():
    base = dict(
        expressions=[], state_variables=["y"],
        boundary_expressions=[Expr.parse("ya_y - 1")],
        mesh=[0.0, 0.5, 1.0], initial_state=[1.0, 0.6, 0.37],
        rhs_callback=lambda _x, y, _p: [-y[0]],
    )
    with pytest.raises(UnsupportedOptionError, match="dense"):
        BvpSciSolver(
            **base,
            options=SolverOptions(frontend="numerical", execution="numerical", matrix_layout="sparse"),
        )
    with pytest.raises(InvalidConfigurationError, match="cannot be mixed"):
        BvpSciSolver(
            **base,
            options=SolverOptions(frontend="expr_legacy", execution="lambdify"),
        )
    with pytest.raises(InvalidConfigurationError, match="require rhs_callback"):
        BvpSciSolver(
            [-Expr.variable("y")], ["y"], [Expr.parse("ya_y - 1")], [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
            options=SolverOptions(frontend="numerical", execution="numerical"),
        )

    handoff_options = SolverOptions(frontend="numerical", execution="numerical")
    handoff_options.set_aot_handoff_path("unused-registry.txt")
    with pytest.raises(UnsupportedOptionError, match="require execution='aot'"):
        BvpSciSolver(**base, options=handoff_options)


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
def test_nonlinear_dense_aot_uses_native_generated_backend(tmp_path, frontend):
    x = Expr.variable("x")
    solver = NonlinearSolver(
        SymbolicSystem([x * x - 2.0], ["x"]),
        frontend=frontend,
        execution="aot",
        aot_policy="build_if_missing",
        aot_profile="debug",
        aot_output_dir=str(tmp_path / "nonlinear-aot"),
        telemetry="detailed",
        tolerance=1e-9,
    )
    solution = solver.solve([1.0])
    assert solution.residual_norm < 1e-8
    assert solution.solution()[0] == pytest.approx(math.sqrt(2.0), rel=1e-8)
    assert solver.backend() == "aot"
    assert "Aot" in solver.prepare_report()


def test_nonlinear_aot_requires_artifact_or_explicit_build_directory(tmp_path):
    x = Expr.variable("x")
    solver = NonlinearSolver(
        SymbolicSystem([x - 1.0], ["x"]),
        execution="aot",
        aot_policy="require_prebuilt",
        aot_output_dir=str(tmp_path / "missing-artifact"),
    )
    with pytest.raises(PreparationError) as failure:
        solver.prepare()
    assert failure.value.stage == "prepare"
    assert failure.value.solver == "nonlinear"
    assert "aot" in failure.value.route
    assert failure.value.native_kind
    assert "aot" in capabilities("nonlinear_system").executions


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


def test_bvp_sci_aot_telemetry_exposes_lifecycle_and_nested_scopes(tmp_path):
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no supported C compiler is available for BVP_sci AOT telemetry")
    toolchain = "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"
    y = Expr.variable("y")
    options = SolverOptions(
        execution="aot",
        telemetry="detailed",
        aot_toolchain=toolchain,
        aot_profile="debug",
        aot_output_dir=str(tmp_path / "bvp-sci-aot-telemetry"),
    )
    solver = BvpSciSolver(
        [-y],
        ["y"],
        [Expr.parse("ya_y - 1")],
        [0.0, 0.5, 1.0],
        [1.0, 0.6, 0.37],
        options=options,
        tolerance=1e-4,
    )
    solution = solver.solve()
    assert solution.status_code == 0
    assert solution.telemetry_aot_metadata()["execution"] == "aot"
    assert solution.telemetry_aot_counters()["aot_build_attempts"] >= 1
    assert solution.telemetry_aot_counters()["aot_link_attempts"] >= 1
    assert solution.telemetry_aot_artifact_keys()
    aot_times = solution.telemetry_aot_timings_ms()
    assert aot_times["cold.aot_build.ms"] >= 0.0
    assert aot_times["cold.aot_link.ms"] >= 0.0
    stages = solution.telemetry_aot_stages()
    build_stage = next(stage for stage in stages if stage["stage"] == "aot_build")
    assert build_stage["calls"] >= 1
    assert isinstance(build_stage["calls"], int)
    assert solution.telemetry_timings_ms()["reported_preparation_ms"] >= 0.0
    assert solution.telemetry_preparation_scope() == "aot_ivp"


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
    counters = solution.telemetry_counters()
    assert "superlu_fallbacks" in counters
    assert "sparse_symbolic_analyses" in counters
    assert "sparse_symbolic_reuses" in counters

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
    assert "scalar_evaluations" in counters
    assert "residual_requests" in counters
    assert "factorization_invalidations" in counters
    assert timings["total_ms"] >= 0.0
    assert timings["symbolic_operations_ms"] >= 0.0
    assert timings["linear_system_ms"] >= 0.0
    assert solution.telemetry_scope_timings_ms()["solve_ms"] >= 0.0
    assert solution.telemetry_scope_events()["iteration_events"] == counters["iterations"]
    storage = solution.telemetry_storage_bytes()
    assert isinstance(storage, dict)
    assert (solution.telemetry_storage_is_estimated() is not None) is bool(storage)
    if storage:
        assert solution.telemetry_storage_is_estimated() is True
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["dense", "sparse", "banded"])
@pytest.mark.parametrize("execution", ["lambdify", "aot"])
def test_bvp_damp_facade_frontend_execution_layout_matrix(
    frontend, layout, execution
):
    """Gate the main damped route across its advertised symbolic axes."""
    aot_toolchain = None
    if execution == "aot":
        compiler = shutil.which("tcc") or shutil.which("gcc")
        if compiler is None:
            pytest.skip("BVP_Damp AOT route needs tcc or gcc")
        aot_toolchain = (
            "tcc" if Path(compiler).name.lower() in {"tcc", "tcc.exe"} else "gcc"
        )

    options = SolverOptions(
        frontend=frontend,
        execution=execution,
        matrix_layout=layout,
        execution_policy="sequential",
        telemetry="counters",
        aot_toolchain=aot_toolchain,
    )
    if layout == "banded":
        options.set_banded_width(1, 1)

    mesh = np.linspace(0.0, 1.0, 5)
    solver = python_facade.create_bvp(
        "bvp_damp",
        [Expr.constant(1.0)],
        ["y"],
        {"y": [(0, 0.0)]},
        mesh,
        np.zeros(mesh.size),
        options=options,
        bounds={"y": (-2.0, 2.0)},
        max_iterations=8,
    )
    solution = solver.solve()

    assert solver.frontend == frontend
    assert solver.execution == execution
    assert solver.matrix_layout == layout
    assert solution.status == "finished"
    assert solution.mesh()[0] == pytest.approx(0.0)
    assert solution.mesh()[-1] == pytest.approx(1.0)
    assert solution.values()[0][-1] == pytest.approx(1.0, abs=1e-7)
    assert solution.telemetry_counters()["linear_solves"] > 0


@pytest.mark.parametrize("strategy", ["naive", "frozen"])
@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("layout", ["dense", "sparse", "banded"])
def test_bvp_damp_fixed_mesh_aot_frontend_layout_matrix(
    strategy, frontend, layout, tmp_path
):
    """Cover AOT intersections for fixed-mesh BVP_Damp strategies."""
    compiler = shutil.which("tcc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("BVP_Damp AOT route needs tcc or gcc")
    toolchain = (
        "tcc" if Path(compiler).name.lower() in {"tcc", "tcc.exe"} else "gcc"
    )
    rate = Expr.variable(f"fixed_aot_{strategy}_{frontend}_{layout}_rate")
    options = SolverOptions(
        frontend=frontend,
        execution="aot",
        matrix_layout=layout,
        aot_policy="build_if_missing",
        aot_toolchain=toolchain,
        aot_profile="debug",
        telemetry="counters",
    )

    common = (
        [rate],
        ["y"],
        {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
    )
    kwargs = {
        "parameters": [str(rate)],
        "parameter_values": [1.0],
        "options": options,
    }
    if frontend == "atom_native" and layout == "dense":
        with pytest.raises(UnsupportedOptionError, match=r"AtomNative\+AOT\+Dense"):
            if strategy == "naive":
                BvpDampNaiveSolver(*common, **kwargs)
            else:
                BvpDampFrozenSolver(
                    *common, **kwargs, policy="Frozen_naive"
                )
        return

    if strategy == "naive":
        solver = BvpDampNaiveSolver(*common, **kwargs)
    else:
        solver = BvpDampFrozenSolver(
            *common, **kwargs, policy="Frozen_naive"
        )

    solution = solver.solve()
    assert solver.frontend == frontend
    assert solver.execution == "aot"
    assert solver.matrix_layout == layout
    assert solver.provenance()["execution"] == "aot"
    assert solution.values()[0][-1] == pytest.approx(1.0, abs=1e-7)
    assert solution.telemetry_counters()["linear_solves"] >= 1


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
    assert bool(solution.telemetry_scope_events()) is has_counters
    assert bool(solution.telemetry_scope_timings_ms()) is has_timings
    assert bool(solution.telemetry_callback_counters()) is has_counters
    assert bool(solution.telemetry_callback_timings_ms()) is has_timings
    assert has_timings or not solution.telemetry_preparation_timings_ms()

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
    assert bool(solution.telemetry_scope_events()) is has_counters
    assert bool(solution.telemetry_scope_timings_ms()) is has_timings
    assert bool(solution.telemetry_callback_counters()) is has_counters
    assert bool(solution.telemetry_callback_timings_ms()) is has_timings
    assert has_timings or not solution.telemetry_preparation_timings_ms()


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

    warm = solver.solve_with_parameter_values([2.0])
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
    second = solver.solve_with_parameter_values([2.0])
    assert first.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert second.values()[0][-1] == pytest.approx(2.0, abs=1e-8)
    assert second.parameters == pytest.approx([2.0])
    assert second.telemetry_timings_ms()["total_ms"] >= 0.0


def test_bvp_damp_frozen_parameter_loop_keeps_typed_failures_and_continues():
    solver = BvpDampFrozenSolver(
        [Expr.parse("alpha^0.5")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0] * 6,
        parameters=["alpha"], parameter_values=[1.0],
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="sparse",
            telemetry="detailed",
        ),
        policy="Frozen_naive",
    )

    report = solver.solve_parameter_loop([[1.0], [-1.0], [4.0]])
    assert isinstance(report, BvpDampParameterLoopReport)
    assert report.completed_count() == 3
    assert report.stop_error is None
    first, failed, last = report.points
    assert all(isinstance(point, BvpDampParameterLoopPoint) for point in report.points)
    assert [point.index for point in report.points] == [0, 1, 2]
    assert [point.status for point in report.points] == ["converged", "failed", "converged"]
    assert first.parameters == [1.0]
    assert first.solution.values()[0][-1] == pytest.approx(1.0, abs=1e-8)
    assert first.solution.telemetry_counters()["iterations"] >= 1
    assert first.telemetry_timings_ms()["total_ms"] >= 0.0
    assert failed.solution is None
    assert isinstance(failed.error, SolverError)
    assert failed.error.solver == "bvp_damp_frozen"
    assert failed.error.stage == "solve"
    assert failed.error.route == "expr_legacy:lambdify:sparse"
    assert failed.error.native_kind is not None
    assert failed.telemetry_counters()["iterations"] >= 1
    assert failed.telemetry_timings_ms()["total_ms"] >= 0.0
    assert failed.telemetry_scope_timings_ms()
    assert last.solution.values()[0][-1] == pytest.approx(2.0, abs=1e-8)
    assert last.parameters == [4.0]
    assert solver.parameter_values == [4.0]


def test_bvp_damp_frozen_parameter_loop_rejects_invalid_input_before_prepare():
    solver = BvpDampFrozenSolver(
        [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.5, 1.0], [0.0] * 3,
        parameters=["rate"], parameter_values=[1.0],
    )
    with pytest.raises(InvalidConfigurationError, match="at least one"):
        solver.solve_parameter_loop([])
    with pytest.raises(InvalidConfigurationError, match="expected 1 values"):
        solver.solve_parameter_loop([[1.0], [2.0, 3.0]])
    assert solver.provenance()["prepared"] == "false"


def test_bvp_damp_naive_parameter_loop_uses_same_ordered_failure_contract():
    solver = BvpDampNaiveSolver(
        [Expr.parse("alpha^0.5")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
        parameters=["alpha"], parameter_values=[1.0],
        options=SolverOptions(telemetry="counters"),
    )
    report = solver.solve_parameter_loop([[1.0], [-1.0], [4.0]])
    assert [point.status for point in report.points] == ["converged", "failed", "converged"]
    assert report.points[2].solution.values()[0][-1] == pytest.approx(2.0, abs=1e-8)
    assert isinstance(report.points[1].error, SolverError)
    assert report.points[1].error.solver == "bvp_damp_naive"
    assert report.points[1].telemetry_counters()


def test_bvp_damp_adaptive_parameter_loop_recovers_after_failed_parameter():
    solver = BvpDampSolver(
        [Expr.parse("alpha^0.5")], ["y"], {"y": [(0, 0.0)]},
        [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
        parameters=["alpha"], parameter_values=[1.0],
        options=SolverOptions(
            frontend="expr_legacy", execution="lambdify", matrix_layout="dense",
            telemetry="counters",
        ),
        tolerance=1e-9, max_refinements=1, grid_method="double_points",
        bounds={"y": (-10.0, 10.0)},
    )

    report = solver.solve_parameter_loop([[1.0], [-1.0], [4.0]])
    assert report.stop_error is None
    assert [point.index for point in report.points] == [0, 1, 2]
    assert [point.status for point in report.points] == ["converged", "failed", "converged"]
    first, failed, last = report.points
    assert len(first.solution.mesh()) > 5
    assert first.solution.values()[0][-1] == pytest.approx(1.0, abs=1e-7)
    assert isinstance(failed.error, SolverError)
    assert failed.error.solver == "bvp_damp"
    assert failed.error.stage == "solve"
    assert failed.error.route == "expr_legacy:lambdify:dense"
    assert failed.error.native_kind is not None
    assert last.parameters == [4.0]
    assert last.solution.values()[0][-1] == pytest.approx(2.0, abs=1e-7)
    assert len(last.solution.mesh()) >= len(first.solution.mesh())
    assert last.telemetry_counters()["iterations"] > 0
    assert solver.parameter_values == [4.0]


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

    solver.set_parameter_values([5.0])
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
        solver.set_parameter_values([])
    with pytest.raises(InvalidConfigurationError, match="finite"):
        solver.set_parameter_values([math.nan])
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
        (
            BdfSolver,
            (
                "prepare",
                "solve",
                "solve_with_parameter_values",
                "solve_with_parameter_values_in_interval",
                "restart",
                "continue_with_parameters",
            ),
        ),
        (
            BackwardEulerSolver,
            (
                "validate",
                "solve",
                "solve_with_parameter_values",
                "solve_with_parameter_values_in_interval",
                "continue_to",
                "restart",
            ),
        ),
        (
            RadauSolver,
            (
                "prepare",
                "solve",
                "solve_with_parameter_values",
                "solve_with_parameter_values_in_interval",
                "restart",
            ),
        ),
        (
            Lsode2Solver,
            (
                "prepare",
                "solve",
                "solve_with_parameter_values",
                "solve_with_parameter_values_in_interval",
                "continue_with_parameters",
                "restart",
            ),
        ),
    )
    for exposed_type, expected_terms in documented:
        assert exposed_type.__doc__ and len(exposed_type.__doc__.strip()) > 40
        for method_name in expected_terms:
            method = getattr(exposed_type, method_name, None)
            assert method is not None, f"{exposed_type.__name__}.{method_name} is not exposed"
            assert method.__doc__ and len(method.__doc__.strip()) > 20, (
                f"{exposed_type.__name__}.{method_name} lacks a Python-visible contract"
            )


def test_ivp_solver_fresh_parameterized_solve_shortcuts():
    from rustedscithe import (
        BackwardEulerSolver,
        BdfSolver,
        Lsode2Solver,
        RadauSolver,
    )

    make_options = lambda: SolverOptions(telemetry="counters")
    solvers = [
        (
            BdfSolver(
                [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
                parameters=["rate"], parameter_values=[1.0], options=make_options(),
            ),
            lambda solver: solver.solve_with_parameter_values([2.0], [1.5]),
            lambda solver: solver.solve_with_parameter_values_in_interval(
                0.2, [3.0], 0.4, [0.8],
            ),
        ),
        (
            BackwardEulerSolver(
                [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
                h=0.001, parameters=["rate"], parameter_values=[1.0],
                options=make_options(),
            ),
            lambda solver: solver.solve_with_parameter_values([2.0], [1.5]),
            lambda solver: solver.solve_with_parameter_values_in_interval(
                0.2, [3.0], 0.4, [0.8],
            ),
        ),
        (
            Lsode2Solver(
                [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.01,
                parameters=["rate"], parameter_values=[1.0], options=make_options(),
            ),
            lambda solver: solver.solve_with_parameter_values([2.0], [1.5]),
            lambda solver: solver.solve_with_parameter_values_in_interval(
                0.2, [3.0], 0.4, [0.8],
            ),
        ),
        (
            RadauSolver(
                [Expr.parse("-rate*y")], ["y"], "t", 0.0, 0.1, 0.02,
                parameters=["rate"], parameter_values=[1.0], options=make_options(),
            ),
            lambda solver: solver.solve_with_parameter_values([2.0], [1.5]),
            lambda solver: solver.solve_with_parameter_values_in_interval(
                0.2, [3.0], 0.4, [0.8],
            ),
        ),
    ]

    def terminal_state(solution):
        if hasattr(solution, "states_array"):
            return float(solution.states_array()[-1, 0])
        return float(solution.final_state_array()[0])

    def end_time(solution):
        if hasattr(solution, "times_array"):
            return float(solution.times_array()[-1])
        return float(solution.final_time)

    for solver, solve_configured_interval, solve_new_interval in solvers:
        prepare = getattr(solver, "prepare", None)
        if prepare is not None:
            prepare()
        first = solve_configured_interval(solver)
        assert terminal_state(first) == pytest.approx(
            2.0 * math.exp(-1.5 * 0.1), rel=3e-3,
        )
        second = solve_new_interval(solver)
        assert terminal_state(second) == pytest.approx(
            3.0 * math.exp(-0.8 * 0.2), rel=3e-3,
        )
        assert end_time(second) == pytest.approx(0.4)


def test_radau_fresh_solve_api_has_no_continuation_aliases():
    from rustedscithe import PreparedStiffSolver, RadauCallbackSolver, RadauSolver

    assert not hasattr(RadauSolver, "continue_with_parameters")
    assert not hasattr(RadauCallbackSolver, "continue_with_initial_state")
    assert not hasattr(PreparedStiffSolver, "continue_radau_with_parameters")


def test_stiff_api_fresh_parameterized_solve_shortcuts_dispatch_all_routes():
    from rustedscithe import StiffSolverRequest

    expression = [Expr.parse("-rate*y")]
    common = dict(
        expressions=expression,
        state_variables=["y"],
        time_variable="t",
        parameters=["rate"],
        parameter_values=[1.0],
        options=SolverOptions(telemetry="counters"),
    )
    requests = [
        StiffSolverRequest.bdf(
            **common, y0=[1.0], t0=0.0, t_end=0.1, max_step=0.02,
        ),
        StiffSolverRequest.backward_euler(
            **common, y0=[1.0], t0=0.0, t_end=0.1, h=0.001,
        ),
        StiffSolverRequest.lsode2(
            **common, y0=[1.0], t0=0.0, t_end=0.1, max_step=0.01,
        ),
        StiffSolverRequest.radau(
            **common, y0=[1.0], t0=0.0, t_end=0.1, max_step=0.02,
        ),
    ]

    def terminal_state(solution):
        if hasattr(solution, "states_array"):
            return float(solution.states_array()[-1, 0])
        return float(solution.final_state_array()[0])

    def end_time(solution):
        if hasattr(solution, "times_array"):
            return float(solution.times_array()[-1])
        return float(solution.final_time)

    for request in requests:
        prepared = request.prepare()
        first = prepared.solve_with_parameter_values([2.0], [1.5])
        assert terminal_state(first) == pytest.approx(
            2.0 * math.exp(-1.5 * 0.1), rel=3e-3,
        )
        second = prepared.solve_with_parameter_values_in_interval(
            0.2, [3.0], 0.4, [0.8],
        )
        assert terminal_state(second) == pytest.approx(
            3.0 * math.exp(-0.8 * 0.2), rel=3e-3,
        )
        assert end_time(second) == pytest.approx(0.4)


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
            ("solve", "restart", "provenance"),
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


def test_python_example_inventory_and_explanatory_headers_stay_complete():
    examples_dir = Path(__file__).parents[1] / "examples"
    scripts = {path.name for path in examples_dir.glob("*.py")}
    readme = (examples_dir / "README.md").read_text(encoding="utf-8")
    indexed = set(re.findall(r"\]\(([^)]+\.py)\)", readme))
    indexed = {Path(item).name for item in indexed}

    assert indexed == scripts, (
        f"unindexed examples={sorted(scripts - indexed)}, "
        f"stale README entries={sorted(indexed - scripts)}"
    )
    for script in sorted(scripts):
        module = ast.parse((examples_dir / script).read_text(encoding="utf-8"))
        header = ast.get_docstring(module)
        assert header is not None and len(header.strip()) >= 60, (
            f"{script} needs a useful opening explanation of its problem and route"
        )


def test_python_type_stubs_parse_and_match_runtime_exports():
    import rustedscithe
    from rustedscithe import _native
    from rustedscithe import facade

    # Resolve stubs beside the imported package rather than from the checkout.
    # This makes the same test validate the actual installed wheel payload.
    package_dir = Path(rustedscithe.__file__).resolve().parent
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
    native_stub_nodes = {
        statement.name: statement
        for statement in native_stub.body
        if isinstance(statement, ast.ClassDef)
    }
    damp_constructor = next(
        member
        for member in native_stub_nodes["BvpDampSolver"].body
        if isinstance(member, ast.FunctionDef) and member.name == "__init__"
    )
    damp_argument_types = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in damp_constructor.args.args + damp_constructor.args.kwonlyargs
        if argument.annotation is not None
    }
    assert damp_argument_types["boundary_conditions"] == (
        "Mapping[str, Sequence[tuple[int, float]]]"
    )
    assert damp_argument_types["initial_state"] == "Sequence[float]"
    assert damp_argument_types["grid_parameters"] == "Sequence[float] | None"
    damp_restart = next(
        member
        for member in native_stub_nodes["BvpDampSolver"].body
        if isinstance(member, ast.FunctionDef) and member.name == "restart"
    )
    assert ast.unparse(damp_restart.returns) == "None"
    sci_restart = next(
        member
        for member in native_stub_nodes["BvpSciSolver"].body
        if isinstance(member, ast.FunctionDef) and member.name == "restart"
    )
    sci_restart_types = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in sci_restart.args.args
        if argument.annotation is not None
    }
    assert sci_restart_types["initial_state"] == "Sequence[float]"
    assert ast.unparse(sci_restart.returns) == "None"

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

        constructor = next(
            (
                member
                for member in native_stub_nodes[name].body
                if isinstance(member, ast.FunctionDef) and member.name == "__init__"
            ),
            None,
        )
        if constructor is not None:
            runtime_parameters = inspect.signature(exposed_type).parameters.values()
            runtime_contract = [
                (
                    parameter.name,
                    "keyword_only"
                    if parameter.kind is inspect.Parameter.KEYWORD_ONLY
                    else "positional",
                    parameter.default is inspect.Parameter.empty,
                )
                for parameter in runtime_parameters
            ]

            stub_args = constructor.args
            positional_names = [
                argument.arg for argument in stub_args.args if argument.arg != "self"
            ]
            default_count = len(stub_args.defaults)
            positional_default_names = set(
                positional_names[len(positional_names) - default_count :]
                if default_count
                else ()
            )
            stub_contract = [
                (name, "positional", name not in positional_default_names)
                for name in positional_names
            ]
            stub_contract.extend(
                (
                    argument.arg,
                    "keyword_only",
                    default is None,
                )
                for argument, default in zip(stub_args.kwonlyargs, stub_args.kw_defaults)
            )
            assert runtime_contract == stub_contract, (
                f"{name} constructor differs between the installed PyO3 API and _native.pyi: "
                f"runtime={runtime_contract}, stub={stub_contract}"
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


def test_data_processing_lowess_fit_returns_detached_coordinate_and_value_arrays():
    x = np.linspace(0.0, 2.0, 21, dtype=np.float64)
    y = 1.5 * x + 0.2 * np.sin(8.0 * x)
    result = lowess_fit(x, y, fraction=0.5, iterations=2, delta=0.0, parallel=False)
    assert isinstance(result, LowessResult)
    assert result.x.dtype == result.y.dtype == np.float64
    assert result.x.shape == result.y.shape == x.shape
    assert result.iterations_used is None or result.iterations_used > 0
    assert result.fraction_used == pytest.approx(0.5)
    assert not np.shares_memory(result.x, x)
    assert not np.shares_memory(result.y, y)
    np.testing.assert_allclose(result.x, x)
    np.testing.assert_allclose(result.y, lowess_smooth(x, y, fraction=0.5, iterations=2, delta=0.0, parallel=False))


def test_data_processing_bspline_interpolator_supports_reuse_and_validates_native_panics():
    points = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    knots = np.array([0.0, 0.0, 1.0, 2.0, 2.0], dtype=np.float64)
    spline = BSplineInterpolator(1, points, knots)
    assert spline.degree == 1
    assert spline.knot_domain == pytest.approx((0.0, 2.0))
    assert spline.interpolate(0.5) == pytest.approx(0.5)
    params = np.array([0.25, 1.0, 1.75], dtype=np.float64)
    values = spline.interpolate_batch(params)
    assert values.dtype == np.float64 and not np.shares_memory(values, params)
    np.testing.assert_allclose(values, params, atol=1e-12)

    uniform = BSplineInterpolator.with_uniform_knots(2, np.array([0.0, 1.0, 0.0]))
    assert uniform.degree == 2
    assert np.isfinite(uniform.interpolate_batch(np.array([0.1, 0.5, 0.9]))).all()
    with pytest.raises(InvalidConfigurationError, match="knot domain"):
        spline.interpolate(-0.1)
    with pytest.raises(InvalidConfigurationError, match="nondecreasing"):
        BSplineInterpolator(1, points, np.array([0.0, 1.0, 0.5, 2.0, 2.0]))


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
    assert capabilities("bdf").supports_logging is True
    assert capabilities("nonstiff").supports_logging is False


def test_stiff_preset_catalog_composition_and_native_validation():
    from rustedscithe import StiffPreset

    catalog = StiffPreset.catalog()
    assert {item["name"] for item in catalog} >= {
        "solve_1d",
        "expr_dense",
        "atom_sparse",
        "aot_tcc",
    }

    preset = (
        StiffPreset.solve_1d()
        .with_telemetry("detailed")
        .with_logging("info")
        .with_callback_policy("sequential")
    )
    details = preset.describe()
    assert details["required_state_dimension"] == 1
    assert "symbolic_frontend" in details["fields_changed"]
    assert "telemetry_mode" in details["fields_changed"]
    assert {item["solver"] for item in details["solvers"]} == {
        "bdf",
        "radau",
        "backward_euler",
        "lsode2",
    }

    composed = StiffPreset.expr_dense().compose(StiffPreset.lambdify())
    assert "symbolic_execution" in composed.describe()["fields_changed"]
    with pytest.raises(InvalidConfigurationError, match="frontend"):
        StiffPreset.expr_dense().compose(StiffPreset.atom_dense())
    with pytest.raises(UnsupportedOptionError, match="frontend"):
        StiffPreset.expr_dense().with_frontend("numerical")


def test_stiff_api_bdf_native_request_prepares_and_dispatches_without_wrapper_copy():
    from rustedscithe import BdfSolution, StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.bdf(
        [Expr.parse("-y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.1,
        0.025,
        options=SolverOptions(telemetry="counters"),
    )
    prepared = request.prepare(StiffPreset.solve_1d())
    assert prepared.solver == "bdf"
    assert prepared.telemetry_mode == "counters"
    with pytest.raises(InvalidConfigurationError, match="not available before"):
        prepared.trajectory_arrays()
    result = prepared.solve()
    assert isinstance(result, BdfSolution)
    assert prepared.status == result.status
    assert result.states()[-1][0] == pytest.approx(math.exp(-0.1), rel=2e-3)
    times, states = prepared.trajectory_arrays()
    np.testing.assert_array_equal(times, result.times_array())
    np.testing.assert_array_equal(states, result.states_array())
    snapshot = prepared.telemetry_snapshot()
    assert snapshot["solver"] == "bdf"
    assert snapshot["mode"] == "counters"
    assert snapshot["bdf"]["counters"] == result.telemetry_counters()
    restarted = prepared.restart(0.0, [2.0], 0.1)
    assert isinstance(restarted, BdfSolution)
    assert restarted.states()[-1][0] == pytest.approx(2.0 * math.exp(-0.1), rel=2e-3)
    with pytest.raises(InvalidConfigurationError, match="already been consumed"):
        request.prepare()


def test_bdf_parameter_loop_retains_typed_failures_and_solution_telemetry():
    solver = BdfSolver(
        [Expr.parse("-rate*y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.02,
        options=SolverOptions(telemetry="counters"),
        parameters=["rate"],
        parameter_values=[1.0],
    )
    solver.prepare()

    points = solver.solve_parameter_loop([1.0], [[1.0], [], [2.0]])

    assert all(isinstance(point, IvpParameterLoopPoint) for point in points)
    assert [point.index for point in points] == [0, 1, 2]
    assert [point.succeeded for point in points] == [True, False, True]
    assert points[0].parameter_values == [1.0]
    first, last = points[0].solution, points[2].solution
    assert first is not None and last is not None
    np.testing.assert_allclose(first.states_array()[0], [1.0])
    np.testing.assert_allclose(last.states_array()[0], [1.0])
    assert first.telemetry_mode == "counters"
    assert first.telemetry_counters()["solve_calls"] > 0
    assert points[1].solution is None
    assert isinstance(points[1].error, InvalidConfigurationError)
    assert points[1].error.stage == "solve"
    assert points[1].error.solver == "bdf"
    assert points[1].error.route.startswith("expr_legacy:")


@pytest.mark.parametrize(
    "solver_factory,output",
    [
        (
            lambda: BackwardEulerSolver(
                [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
                h=0.01, parameters=["rate"], parameter_values=[1.0],
                options=SolverOptions(telemetry="counters"),
            ),
            "states_array",
        ),
        (
            lambda: Lsode2Solver(
                [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
                0.02, parameters=["rate"], parameter_values=[1.0],
                options=SolverOptions(telemetry="counters"),
            ),
            "states_array",
        ),
        (
            lambda: RadauSolver(
                [Expr.parse("-rate*y")], ["y"], "t", 0.0, 0.1, 0.02,
                parameters=["rate"], parameter_values=[1.0], output="dense",
                options=SolverOptions(telemetry="counters"),
            ),
            "trajectory_arrays",
        ),
    ],
)
def test_direct_ivp_parameter_loops_keep_success_failure_success(solver_factory, output):
    solver = solver_factory()
    if hasattr(solver, "prepare"):
        solver.prepare()

    points = solver.solve_parameter_loop([1.0], [[0.5], [], [2.0]])

    assert [point.succeeded for point in points] == [True, False, True]
    assert [point.index for point in points] == [0, 1, 2]
    assert isinstance(points[1].error, InvalidConfigurationError)
    assert points[1].error.stage == "solve"
    for point in (points[0], points[2]):
        solution = point.solution
        if output == "trajectory_arrays":
            times, states = solution.trajectory_arrays()
        else:
            times = solution.times_array()
            states = solution.states_array()
        assert states.shape == (len(times), 1)
        assert times[0] == pytest.approx(0.0)
        assert states[0, 0] == pytest.approx(1.0)
        assert solution.telemetry_counters()


def test_stiff_api_parameter_loop_preserves_each_route_and_failure():
    from rustedscithe import StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.backward_euler(
        [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
        h=0.01, parameters=["rate"], parameter_values=[1.0],
        options=SolverOptions(telemetry="counters"),
    )
    prepared = request.prepare(StiffPreset.solve_1d())
    points = prepared.solve_parameter_loop([1.0], [[0.5], [], [2.0]])

    assert [point.succeeded for point in points] == [True, False, True]
    assert points[1].error is not None
    assert points[1].error.stage == "solve"
    assert points[1].error.solver == "backward_euler"
    assert points[1].error.route == "stiff_api"
    times, states = points[2].solution.times_array(), points[2].solution.states_array()
    assert states.shape == (len(times), 1)
    assert states[0, 0] == pytest.approx(1.0)
    assert points[2].solution.telemetry_counters()


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda Expr, SolverOptions, StiffSolverRequest: StiffSolverRequest.bdf(
            [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[1.0],
            options=SolverOptions(telemetry="counters"),
        ),
        lambda Expr, SolverOptions, StiffSolverRequest: StiffSolverRequest.radau(
            [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[1.0], output="dense",
            options=SolverOptions(telemetry="counters"),
        ),
        lambda Expr, SolverOptions, StiffSolverRequest: StiffSolverRequest.lsode2(
            [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
            parameters=["rate"], parameter_values=[1.0],
            options=SolverOptions(telemetry="counters"), native_execution="bridge",
        ),
    ],
    ids=["bdf", "radau", "lsode2"],
)
def test_stiff_api_parameter_loop_continues_after_failure_for_each_route(request_factory):
    from rustedscithe import StiffPreset, StiffSolverRequest

    request = request_factory(Expr, SolverOptions, StiffSolverRequest)
    prepared = request.prepare(StiffPreset.solve_1d())
    points = prepared.solve_parameter_loop([1.0], [[0.5], [], [2.0]])

    assert [point.succeeded for point in points] == [True, False, True]
    assert points[1].error is not None
    assert points[1].error.stage == "solve"
    assert points[1].error.solver == prepared.solver
    assert points[1].error.route
    for point in (points[0], points[2]):
        if hasattr(point.solution, "trajectory_arrays"):
            times, states = point.solution.trajectory_arrays()
        else:
            times = point.solution.times_array()
            states = point.solution.states_array()
        assert states.shape == (len(times), 1)
        assert states[0, 0] == pytest.approx(1.0)
        assert point.solution.telemetry_counters()


def test_stiff_api_backward_euler_preserves_fixed_step_and_parameter_binding():
    from rustedscithe import BackwardEulerSolution, StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.backward_euler(
        [Expr.parse("-rate*y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        h=0.01,
        parameters=["rate"],
        parameter_values=[2.0],
        options=SolverOptions(telemetry="counters"),
    )
    prepared = request.prepare(StiffPreset.solve_1d())
    assert prepared.solver == "backward_euler"
    assert prepared.telemetry_mode == "counters"
    result = prepared.solve()
    assert isinstance(result, BackwardEulerSolution)
    assert prepared.status == result.status
    assert result.states()[-1][0] == pytest.approx((1.0 + 2.0 * 0.01) ** -20, rel=1e-10)
    times, states = prepared.trajectory_arrays()
    np.testing.assert_array_equal(times, result.times_array())
    np.testing.assert_array_equal(states, result.states_array())
    assert len(result.times()) == 21
    restarted = prepared.restart(0.0, [2.0], 0.2)
    assert isinstance(restarted, BackwardEulerSolution)
    assert restarted.states()[-1][0] == pytest.approx(2.0 * (1.0 + 0.02) ** -20, rel=1e-10)

    with pytest.raises(InvalidConfigurationError, match="parameter values"):
        StiffSolverRequest.backward_euler(
            [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
            parameters=["rate"], parameter_values=[1.0, 2.0],
        )


def test_stiff_api_radau_preserves_initial_state_output_and_native_telemetry():
    from rustedscithe import RadauSolution, StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.radau(
        [Expr.parse("-y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.1,
        options=SolverOptions(telemetry="counters"),
        output="dense",
    )
    prepared = request.prepare(StiffPreset.solve_1d())
    assert prepared.solver == "radau"
    assert prepared.telemetry_mode == "counters"
    assert prepared.status is None
    result = prepared.solve()
    assert isinstance(result, RadauSolution)
    assert result.final_time is not None
    assert result.final_time == pytest.approx(0.2)
    assert result.final_state[0] == pytest.approx(math.exp(-0.2), rel=3e-3)
    assert result.sample(0.1)[0] == pytest.approx(math.exp(-0.1), rel=3e-3)
    assert result.telemetry_counters()
    restarted = prepared.restart(0.0, [2.0], 0.2)
    assert isinstance(restarted, RadauSolution)
    assert restarted.final_state[0] == pytest.approx(2.0 * math.exp(-0.2), rel=3e-3)
    with pytest.raises(InvalidConfigurationError, match="dimension") as captured:
        prepared.restart(0.0, [2.0, 3.0], 0.2)
    assert captured.value.stage == "restart"
    assert captured.value.solver == "radau"
    assert captured.value.route == "stiff_api"
    assert captured.value.native_kind == "InitialStateDimension"
    with pytest.raises(UnsupportedOptionError, match="RadauSolution returned by solve"):
        prepared.telemetry_snapshot()
    with pytest.raises(InvalidConfigurationError, match="belongs to the RadauSolution"):
        prepared.trajectory_arrays()

    with pytest.raises(UnsupportedOptionError, match="logging"):
        logging_options = SolverOptions()
        logging_options.set_logging_level("info")
        StiffSolverRequest.radau(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            options=logging_options,
        )


def test_stiff_api_lsode2_keeps_method_native_execution_and_parameter_values():
    from rustedscithe import Lsode2Solution, StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.lsode2(
        [Expr.parse("-rate*y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.05,
        parameters=["rate"],
        parameter_values=[0.7],
        method="bdf",
        native_execution="native",
        options=SolverOptions(telemetry="counters"),
    )
    prepared = request.prepare(StiffPreset.solve_1d())
    assert prepared.solver == "lsode2"
    assert prepared.telemetry_mode == "counters"
    result = prepared.solve()
    assert isinstance(result, Lsode2Solution)
    assert prepared.status == result.status
    assert result.final_time is not None
    assert result.final_time == pytest.approx(0.2)
    assert result.states()[-1][0] == pytest.approx(math.exp(-0.14), rel=3e-2)
    assert result.method == "bdf"
    assert result.telemetry_counters()
    times, states = prepared.trajectory_arrays()
    np.testing.assert_array_equal(times, result.times_array())
    np.testing.assert_array_equal(states, result.states_array())
    restarted = prepared.restart(0.0, [2.0], 0.2)
    assert isinstance(restarted, Lsode2Solution)
    assert restarted.states()[-1][0] == pytest.approx(2.0 * math.exp(-0.14), rel=3e-2)

    with pytest.raises(UnsupportedOptionError, match="method"):
        StiffSolverRequest.lsode2(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            method="radau",
        )


def test_stiff_api_lsode2_analytical_callbacks_restore_original_python_exception():
    from rustedscithe import StiffSolverRequest

    def residual(_time, _state):
        raise ValueError("residual callback sentinel")

    def jacobian(_time, _state):
        return [[-1.0]]

    request = StiffSolverRequest.lsode2(
        [Expr.parse("-y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.05,
        residual_callback=residual,
        jacobian_callback=jacobian,
        options=SolverOptions(telemetry="counters"),
    )
    prepared = request.prepare()
    with pytest.raises(CallbackError, match="residual_callback raised") as captured:
        prepared.solve()

    assert captured.value.category == "callback"
    assert captured.value.stage == "solve"
    assert captured.value.solver == "lsode2"
    assert captured.value.route.endswith("lambdify:dense")
    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == "residual callback sentinel"


def test_stiff_api_lsode2_rejects_unpaired_python_callbacks():
    from rustedscithe import StiffSolverRequest

    with pytest.raises(InvalidConfigurationError, match="must be provided together"):
        StiffSolverRequest.lsode2(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            residual_callback=lambda _time, state: [-state[0]],
        )


def test_stiff_api_lsode2_python_callbacks_reject_aot_and_inferred_bands():
    from rustedscithe import StiffSolverRequest

    callbacks = {
        "residual_callback": lambda _time, state: [-state[0]],
        "jacobian_callback": lambda _time, _state: [[-1.0]],
    }
    with pytest.raises(UnsupportedOptionError, match="require execution='lambdify'"):
        StiffSolverRequest.lsode2(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            options=SolverOptions(execution="aot"), **callbacks,
        )

    with pytest.raises(UnsupportedOptionError, match="explicit band width"):
        StiffSolverRequest.lsode2(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            options=SolverOptions(matrix_layout="banded"), **callbacks,
        )


def test_stiff_api_prepare_preserves_typed_unsupported_capability_context():
    from rustedscithe import StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.radau(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
    )
    preset = StiffPreset.expr_dense().with_logging("info")
    with pytest.raises(UnsupportedOptionError, match="logging") as captured:
        request.prepare(preset)

    error = captured.value
    assert error.category == "unsupported_option"
    assert error.stage == "configuration"
    assert error.solver == "radau"
    assert error.route == "stiff_api:preset"
    assert error.native_kind == "UnsupportedCapability"


def test_stiff_api_radau_forwards_sampled_output_policy():
    from rustedscithe import RadauSolution, StiffPreset, StiffSolverRequest

    request = StiffSolverRequest.radau(
        [Expr.parse("-y")],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.1,
        output="sampled",
        sample_times=[0.1, 0.2],
    )
    solution = request.prepare(StiffPreset.solve_1d()).solve()
    assert isinstance(solution, RadauSolution)
    assert solution.sample(0.1)[0] == pytest.approx(math.exp(-0.1), rel=3e-3)
    assert solution.sample(0.2)[0] == pytest.approx(math.exp(-0.2), rel=3e-3)

    with pytest.raises(InvalidConfigurationError, match="sample_times are required"):
        StiffSolverRequest.radau(
            [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.05,
            output="sampled",
        )


def test_stiff_api_postprocess_plan_exports_retained_and_radau_trajectories(tmp_path):
    from rustedscithe import StiffPostprocessPlan, StiffSolverRequest

    bdf = StiffSolverRequest.bdf(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
        options=SolverOptions(telemetry="counters"),
    ).prepare()
    bdf.solve()
    csv_path = tmp_path / "bdf.csv"
    report_path = tmp_path / "bdf.md"
    plan = StiffPostprocessPlan().save_csv(str(csv_path)).write_report(str(report_path))
    result = bdf.postprocess(plan, "time", ["y"])
    assert result["all_done"] is True
    assert [entry["status"] for entry in result["entries"]] == ["done", "done"]
    assert csv_path.is_file() and report_path.is_file()
    csv_rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_rows[0].split(",") == ["time", "y"]
    assert float(csv_rows[-1].split(",")[0]) == pytest.approx(0.2)

    times = [0.0, 0.1, 0.2]
    radau_solution = StiffSolverRequest.radau(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
        output="sampled", sample_times=times,
    ).prepare().solve()
    radau_csv = tmp_path / "radau.csv"
    radau_report = radau_solution.postprocess(
        StiffPostprocessPlan().save_csv(str(radau_csv)),
        times,
        "time",
        ["y"],
    )
    assert radau_report["all_done"] is True
    assert radau_report["entries"][0]["action"] == "save_csv"
    assert radau_csv.is_file()


def test_stiff_api_postprocess_preserves_typed_missing_trajectory_error():
    from rustedscithe import StiffPostprocessPlan, StiffSolverRequest

    prepared = StiffSolverRequest.bdf(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
    ).prepare()
    with pytest.raises(InvalidConfigurationError, match="not available") as captured:
        prepared.postprocess(StiffPostprocessPlan().save_csv("unused.csv"), "time", ["y"])
    assert captured.value.stage == "postprocess"
    assert captured.value.solver == "bdf"
    assert captured.value.native_kind == "ResultUnavailable"


def test_stiff_api_time_segments_and_fresh_radau_solve_keep_native_semantics():
    from rustedscithe import (
        BackwardEulerSolution,
        BdfSolution,
        Lsode2Solution,
        RadauSolution,
        StiffSolverRequest,
    )

    bdf = StiffSolverRequest.bdf(
        [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
        parameters=["rate"], parameter_values=[0.5],
    ).prepare()
    bdf_first = bdf.solve()
    bdf_next = bdf.continue_bdf_with_parameters([1.2], 0.2)
    assert isinstance(bdf_first, BdfSolution) and isinstance(bdf_next, BdfSolution)
    assert bdf_next.states()[-1][0] == pytest.approx(
        bdf_first.states()[-1][0] * math.exp(-1.2 * 0.1), rel=2e-6
    )

    backward_euler = StiffSolverRequest.backward_euler(
        [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1,
        h=0.001, parameters=["rate"], parameter_values=[0.5],
    ).prepare()
    be_first = backward_euler.solve()
    be_next = backward_euler.continue_backward_euler(0.2, [1.2])
    assert isinstance(be_first, BackwardEulerSolution) and isinstance(be_next, BackwardEulerSolution)
    assert be_next.states()[-1][0] == pytest.approx(
        be_first.states()[-1][0] * math.exp(-1.2 * 0.1), rel=2e-3
    )

    radau = StiffSolverRequest.radau(
        [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.025,
        parameters=["rate"], parameter_values=[0.5], output="dense",
    ).prepare()
    radau.solve()
    radau_next = radau.solve_with_parameter_values([0.8], [1.2])
    assert isinstance(radau_next, RadauSolution)
    assert radau_next.sample(0.1)[0] == pytest.approx(0.8 * math.exp(-1.2 * 0.1), rel=2e-6)

    lsode2 = StiffSolverRequest.lsode2(
        [Expr.parse("-rate*y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
        parameters=["rate"], parameter_values=[0.5],
    ).prepare()
    lsode_first = lsode2.solve()
    lsode_next = lsode2.continue_lsode2_with_parameters([1.2], 0.2)
    assert isinstance(lsode_first, Lsode2Solution) and isinstance(lsode_next, Lsode2Solution)
    assert lsode_next.states()[-1][0] == pytest.approx(
        lsode_first.states()[-1][0] * math.exp(-1.2 * 0.1), rel=2e-5
    )


def test_stiff_api_continuation_rejects_wrong_solver_method_with_context():
    from rustedscithe import StiffSolverRequest

    prepared = StiffSolverRequest.bdf(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
    ).prepare()
    with pytest.raises(UnsupportedOptionError, match="only available for Backward Euler") as captured:
        prepared.continue_backward_euler(0.2)
    assert captured.value.stage == "continuation"
    assert captured.value.solver == "stiff_api"
    assert captured.value.route == "stiff_api:continuation"
    assert captured.value.native_kind == "WrongSolverRoute"


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
def test_stiff_api_matches_direct_solver_routes_and_exact_decay(frontend):
    from rustedscithe import StiffSolverRequest

    options = SolverOptions(frontend=frontend, telemetry="counters")
    expression = [Expr.parse("-rate*y")]
    shared = dict(
        expressions=expression,
        state_variables=["y"],
        time_variable="t",
        y0=[1.0],
        t0=0.0,
        t_end=0.1,
        parameters=["rate"],
        parameter_values=[0.8],
        options=options,
    )

    direct_bdf = BdfSolver(max_step=0.02, **shared).solve()
    stiff_bdf = StiffSolverRequest.bdf(max_step=0.02, **shared).prepare().solve()
    assert stiff_bdf.states()[-1] == pytest.approx(direct_bdf.states()[-1], rel=1e-10)
    assert stiff_bdf.states()[-1][0] == pytest.approx(math.exp(-0.08), rel=2e-6)

    direct_be = BackwardEulerSolver(h=0.001, **shared).solve()
    stiff_be = StiffSolverRequest.backward_euler(h=0.001, **shared).prepare().solve()
    assert stiff_be.states()[-1] == pytest.approx(direct_be.states()[-1], rel=1e-12)
    assert stiff_be.states()[-1][0] == pytest.approx(math.exp(-0.08), rel=5e-4)

    direct_radau = RadauSolver(
        expression, ["y"], "t", 0.0, 0.1, 0.02,
        options=options, parameters=["rate"], parameter_values=[0.8], output="dense",
    ).solve([1.0])
    stiff_radau = StiffSolverRequest.radau(
        expression, ["y"], "t", [1.0], 0.0, 0.1, 0.02,
        options=options, parameters=["rate"], parameter_values=[0.8], output="dense",
    ).prepare().solve()
    assert stiff_radau.final_state == pytest.approx(direct_radau.final_state, rel=1e-12)
    assert stiff_radau.final_state[0] == pytest.approx(math.exp(-0.08), rel=2e-6)

    direct_lsode2 = Lsode2Solver(max_step=0.02, **shared).solve()
    stiff_lsode2 = StiffSolverRequest.lsode2(max_step=0.02, **shared).prepare().solve()
    assert stiff_lsode2.states()[-1] == pytest.approx(direct_lsode2.states()[-1], rel=1e-10)
    assert stiff_lsode2.states()[-1][0] == pytest.approx(math.exp(-0.08), rel=2e-5)


def test_stiff_api_aot_missing_artifact_context_is_preserved_for_each_factory(tmp_path):
    from rustedscithe import PreparationError, SolverOptions, StiffSolverRequest

    # Keep each request's compiler input unique: RequirePrebuilt may reuse an
    # already-registered native artifact even when this test's output folder
    # is empty.
    token = tmp_path.name.replace("-", "_")
    options = SolverOptions(
        execution="aot",
        aot_policy="require_prebuilt",
        aot_output_dir=str(tmp_path / "missing-artifacts"),
    )
    for solver_name in ("bdf", "backward_euler", "radau", "lsode2"):
        state_name = f"missing_{solver_name}_{token}"
        expressions = [Expr.parse(f"-{state_name}")]
        symbols = [state_name]
        if solver_name == "bdf":
            request = StiffSolverRequest.bdf(
                expressions, symbols, "t", [1.0], 0.0, 0.1, 0.02,
                options=options,
            )
        elif solver_name == "backward_euler":
            request = StiffSolverRequest.backward_euler(
                expressions, symbols, "t", [1.0], 0.0, 0.1,
                options=options,
            )
        elif solver_name == "radau":
            request = StiffSolverRequest.radau(
                expressions, symbols, "t", [1.0], 0.0, 0.1, 0.02,
                options=options,
            )
        else:
            request = StiffSolverRequest.lsode2(
                expressions, symbols, "t", [1.0], 0.0, 0.1, 0.02,
                options=options,
            )
        try:
            prepared = request.prepare()
            prepared.solve()
        except (AotArtifactError, PreparationError) as error:
            captured_error = error
        else:
            pytest.fail(f"{solver_name} unexpectedly solved without a required AOT artifact")
        error = captured_error
        assert error.solver == solver_name
        assert error.stage in {"prepare", "preparation", "solve"}
        assert error.route is not None
        assert error.native_kind is not None


def test_stiff_api_postprocess_io_failure_does_not_erase_solved_result(tmp_path):
    from rustedscithe import SolverError, StiffPostprocessPlan, StiffSolverRequest

    prepared = StiffSolverRequest.bdf(
        [Expr.parse("-y")], ["y"], "t", [1.0], 0.0, 0.1, 0.02,
    ).prepare()
    prepared.solve()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file", encoding="utf-8")

    with pytest.raises(SolverError) as captured:
        prepared.postprocess(
            StiffPostprocessPlan().save_csv(str(blocker / "result.csv")),
            "time",
            ["y"],
        )
    assert captured.value.stage == "postprocess"
    assert captured.value.solver == "bdf"
    assert captured.value.native_kind == "PostprocessPlanExecution"
    times, states = prepared.trajectory_arrays()
    assert times[-1] == pytest.approx(0.1)
    assert states[-1, 0] == pytest.approx(math.exp(-0.1), rel=2e-6)

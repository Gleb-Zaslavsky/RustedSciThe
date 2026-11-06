"""Native continuation ownership, failure recovery and detached Python reports."""

import numpy as np
import pytest

from rustedscithe import (
    BvpCorrectionNewtonParameterLoopPoint, BvpCorrectionNewtonParameterLoopReport,
    BvpSolverRequest, Expr, SolverOptions, InvalidConfigurationError,
)


def prepare(frontend="expr_legacy", telemetry="counters"):
    return BvpSolverRequest.correction_newton(
        [Expr.parse("1/rate")], ["y"], "x",
        [i / 16 for i in range(17)], [(0, "left", 1.0)], [[1.0]] * 16,
        options=SolverOptions(frontend=frontend, matrix_layout="banded", telemetry=telemetry),
        external_parameters=["rate"], external_parameter_values=[1.0],
    ).prepare()


@pytest.mark.parametrize("frontend", ["expr_legacy", "atom_native"])
@pytest.mark.parametrize("mode", ["off", "counters", "detailed"])
def test_correction_newton_loop_success_failure_success(frontend, mode):
    solver = prepare(frontend, mode)
    report = solver.solve_parameter_loop([[1.0], [0.0], [2.0]])
    assert isinstance(report, BvpCorrectionNewtonParameterLoopReport)
    assert report.stop_error is None
    assert report.completed_count() == 3
    points = report.points
    assert all(isinstance(point, BvpCorrectionNewtonParameterLoopPoint) for point in points)
    assert [p.status for p in points] == ["converged", "failed", "converged"]
    assert [p.guess_source_index for p in points] == [None, 0, 0]
    assert points[1].solution is None
    assert points[1].error is not None
    assert points[1].error.native_kind == "ResidualCallback"
    assert points[1].error.operation == "parameter_loop"
    for index, endpoint in [(0, 2.0), (2, 1.5)]:
        solution = points[index].solution
        assert solution.route == "correction_newton"
        assert solution.values_array().shape == (17, 1)
        assert solution.mesh_array().shape == (17,)
        assert solution.values_array()[-1, 0] == pytest.approx(endpoint, abs=1e-10)
    for point in points:
        assert bool(point.telemetry_counters()) == (mode != "off")
        assert bool(point.telemetry_timings_ms()) == (mode == "detailed")
    detached = points[0].solution.values_array()
    detached[:] = 123.0
    assert points[0].solution.values_array()[0, 0] == 1.0
    np.testing.assert_allclose(solver.solve_with_parameter_values([4.0]).values_array()[-1], [1.25])


def test_correction_newton_loop_preflight_and_all_failures_do_not_poison_solver():
    solver = prepare()
    first = solver.solve().values_array()
    for invalid in [[], [[1.0], []], [[float("nan")]]]:
        with pytest.raises(InvalidConfigurationError):
            solver.solve_parameter_loop(invalid)
    failed = solver.solve_parameter_loop([[0.0], [0.0]])
    assert all(p.status == "failed" and p.solution is None for p in failed.points)
    np.testing.assert_allclose(solver.solve_with_parameter_values([1.0]).values_array(), first)

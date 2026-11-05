"""Shared runner for the two public-facade Cantera continuation examples."""

import argparse

import numpy as np

from rustedscithe import BvpSolverRequest, SolverOptions


def run(*, names, expressions, right, boundaries, initial, parameter, parameters, rhs, check):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256, help="fixed intervals (default: 256)")
    args = parser.parse_args()
    if args.steps < 16:
        parser.error("--steps must be at least 16")
    steps = args.steps
    mesh = [right * node / steps for node in range(steps + 1)]
    # Reduced node-major rows, NOT a full profile. Each left-constrained state
    # omits node 0; each right-constrained state omits node `steps`.
    guess = [list(initial) for _ in range(steps)]
    solver = BvpSolverRequest.cantera(
        expressions=expressions,
        state_variables=names,
        independent_variable="x",
        mesh=mesh,
        boundary_conditions=boundaries,
        initial_guess=guess,
        options=SolverOptions(
            frontend="atom_native", execution="lambdify",
            matrix_layout="banded", telemetry="timings",
        ),
        tolerance=1e-8,
        max_iterations=100,
        bounds={name: (-10.0, 10.0) for name in names},
        external_parameters=[parameter],
        external_parameter_values=parameters[0],
    ).prepare()
    report = solver.solve_parameter_loop(parameters)
    if report.stop_error is not None:
        raise report.stop_error
    assert len(report.points) == 10, "the whole schedule must be attempted"
    last_success = None
    successful = 0
    for index, point in enumerate(report.points):
        assert point.index == index
        assert point.parameters == parameters[index]
        assert point.guess_source_index == last_success, "only successful profiles become checkpoints"
        print(f"point={index} {parameter}={point.parameters[0]:.6f}, "
              f"status={point.status}, guess={point.guess_source_index}")
        print(f"  counters: {point.telemetry_counters()}")
        print(f"  timings (ms): {point.telemetry_timings_ms()}")
        if point.status == "failed":
            assert point.solution is None
            assert point.error is not None
            print(f"  {point.error}; checkpoint remains {last_success}")
            # Do not advance last_success: the following point must reuse it.
            continue
        assert point.status == "converged"
        assert point.error is None
        solution = point.solution
        assert solution is not None and solution.route == "cantera"
        actual_mesh = solution.mesh_array()
        values = solution.values_array()  # full node-major (steps + 1, states)
        np.testing.assert_allclose(actual_mesh, mesh, rtol=0, atol=1e-14)
        assert values.shape == (steps + 1, len(names))
        assert np.isfinite(values).all() and (np.abs(values) <= 10.0).all()
        for state, side, expected in boundaries:
            assert abs(values[0 if side == "left" else -1, state] - expected) < 1e-12
        # Independently evaluate the fixed-grid trapezoidal equations.
        slopes = np.asarray([rhs(point.parameters[0], row) for row in values])
        defect = np.diff(values, axis=0) - 0.5 * np.diff(actual_mesh)[:, None] * (slopes[:-1] + slopes[1:])
        residual = np.max(np.abs(defect))
        assert residual < 1e-7, f"discrete residual: {residual:.3e}"
        check(point.parameters[0], actual_mesh, values)
        print(f"  nodes={len(actual_mesh)}, residual={residual:.3e}")
        successful += 1
        last_success = index
    assert successful == 10, "the default sweep should converge at every point"
    assert report.completed_count() == 10


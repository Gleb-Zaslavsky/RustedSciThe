"""Compact SciPy 1.18.1 BVP reference rows for matched Rust debug stories."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from telemetry import BvpTelemetry, instrument_module


HERE = Path(__file__).resolve().parent


def _load_instrumented_scipy_bvp():
    spec = importlib.util.spec_from_file_location(
        "rustedscithe_scipy_bvp_reference", HERE / "bvp.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load vendored SciPy _bvp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return instrument_module(module)


def solve_reference(case: str) -> dict[str, object]:
    solve_bvp = _load_instrumented_scipy_bvp()
    telemetry = BvpTelemetry(capture_newton_trace=case == "shock-layer")
    mesh = np.linspace(0.0, 1.0, 6)
    sample = np.linspace(0.0, 1.0, 9)

    if case == "validation":
        def fun(x, y):
            return np.ones_like(y)

        def bc(ya, yb):
            return np.array([ya[0]])

        def bad_bc(ya, yb):
            return np.empty(0)

        failures = {}
        for name, x, y, boundary in (
            ("non_increasing_mesh", np.array([0.0, 0.5, 0.4]), np.ones((1, 3)), bc),
            ("state_shape", np.array([0.0, 0.5, 1.0]), np.ones((1, 2)), bc),
            ("boundary_shape", np.array([0.0, 0.5, 1.0]), np.ones((1, 3)), bad_bc),
        ):
            try:
                solve_bvp(fun, boundary, x, y)
            except Exception as error:
                failures[name] = type(error).__name__
            else:
                raise AssertionError(f"SciPy accepted invalid input {name}")
        return {"case": case, "validation_failures": failures}

    if case == "global-jacobian":
        # Exercise SciPy's matrix assembly directly, independently of Newton
        # convergence and sparse factorization behavior.
        n, k = 2, 1
        scipy_module = solve_bvp.__wrapped__.__globals__
        x = np.array([0.0, 0.4, 1.0])
        h = np.diff(x)
        y = np.array([[1.0, 0.9, 0.7], [0.25, 0.3, -0.1]])
        parameters = np.array([1.3])

        def fun(_, values, p):
            return np.vstack((values[1], p[0] * values[0]))

        _, y_middle, f, f_middle = scipy_module["collocation_fun"](
            fun, y, parameters, x, h
        )
        df_dy = np.zeros((n, n, x.size))
        df_dy[0, 1, :] = 1.0
        df_dy[1, 0, :] = parameters[0]
        df_dy_middle = np.zeros((n, n, x.size - 1))
        df_dy_middle[0, 1, :] = 1.0
        df_dy_middle[1, 0, :] = parameters[0]
        # Match the current Lambdify contract: state Jacobians and boundary
        # blocks are analytical, while parameter columns use forward FD.
        _, df_dp = scipy_module["estimate_fun_jac"](fun, x, y, parameters, f)
        x_middle = x[:-1] + 0.5 * h
        _, df_dp_middle = scipy_module["estimate_fun_jac"](
            fun, x_middle, y_middle, parameters, f_middle
        )
        dbc_dya = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        dbc_dyb = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        dbc_dp = np.array([[0.0], [0.0], [-1.0]])
        i_jac, j_jac = scipy_module["compute_jac_indices"](n, x.size, k)
        jacobian = scipy_module["construct_global_jac"](
            n, x.size, k, i_jac, j_jac, h, df_dy, df_dy_middle,
            df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp,
        )
        dense = np.asarray(jacobian.toarray(), dtype=float)
        collocation_residual = y[:, 1:] - y[:, :-1] - h / 6 * (
            f[:, :-1] + f[:, 1:] + 4 * f_middle
        )
        return {
            "case": case,
            "scipy_version": "1.18.1",
            "shape": list(dense.shape),
            "global_jacobian": dense.ravel().tolist(),
            # Rust's collocation residual buffer is interval-major; transpose
            # SciPy's state-major array to serialize the same logical order.
            "collocation_residual": collocation_residual.T.ravel().tolist(),
        }
    elif case == "linear":
        def fun(x, y):
            return np.ones_like(y)

        def bc(ya, yb):
            return np.array([ya[0]])

        result = solve_bvp(fun, bc, mesh, np.zeros((1, mesh.size)),
                           tol=1e-8, max_nodes=100, telemetry=telemetry)
    elif case in {"unknown-parameter", "unknown-parameter-analytic"}:
        def fun(x, y, p):
            return np.ones_like(y) * p[0]

        def bc(ya, yb, p):
            return np.array([ya[0], yb[0] - 1.0])

        initial = (0.8 * mesh).reshape(1, -1)
        kwargs = {}
        if case.endswith("-analytic"):
            def fun_jac(x, y, p):
                points = x.size
                return np.zeros((1, 1, points)), np.ones((1, 1, points))

            # The second boundary residual is yb - 1; expose both endpoint
            # derivatives and the (zero) parameter derivative.
            def bc_jac(ya, yb, p):
                return (
                    np.array([[1.0], [0.0]]),
                    np.array([[0.0], [1.0]]),
                    np.zeros((2, 1)),
                )

            kwargs.update(fun_jac=fun_jac, bc_jac=bc_jac)
        result = solve_bvp(fun, bc, mesh, initial, p=np.array([0.8]),
                           tol=1e-8, max_nodes=100, telemetry=telemetry,
                           **kwargs)
    elif case in {"exponential", "exponential-analytic", "exponential-tight"}:
        def fun(x, y):
            return np.vstack((y[1], y[0]))

        def bc(ya, yb):
            return np.array([ya[0] - 1.0, yb[0]])

        kwargs = {}
        if case.endswith("-analytic"):
            def fun_jac(x, y):
                values = np.zeros((2, 2, x.size))
                values[0, 1, :] = 1.0
                values[1, 0, :] = 1.0
                return values

            def bc_jac(ya, yb):
                return np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])

            kwargs.update(fun_jac=fun_jac, bc_jac=bc_jac)
        exponential_mesh = np.linspace(0.0, 1.0, 5)
        result = solve_bvp(fun, bc, exponential_mesh,
                           np.zeros((2, exponential_mesh.size)),
                           # Match SciPy's official test_no_params defaults.
                           tol=1e-8 if case == "exponential-tight" else 1e-3,
                           max_nodes=100, telemetry=telemetry,
                           **kwargs)
    elif case in {"sturm-liouville", "sturm-liouville-analytic"}:
        interval_end = np.pi
        sample = np.linspace(0.0, interval_end, 9)
        parameter_mesh = np.linspace(0.0, interval_end, 5)

        def fun(x, y, p):
            return np.vstack((y[1], -(p[0] ** 2) * y[0]))

        def bc(ya, yb, p):
            return np.array([ya[0], yb[0], ya[1] - p[0]])

        kwargs = {}
        if case.endswith("-analytic"):
            def fun_jac(x, y, p):
                points = x.size
                df_dy = np.zeros((2, 2, points))
                df_dy[0, 1, :] = 1.0
                df_dy[1, 0, :] = -(p[0] ** 2)
                df_dp = np.zeros((2, 1, points))
                df_dp[1, 0, :] = -2.0 * p[0] * y[0]
                return df_dy, df_dp

            def bc_jac(ya, yb, p):
                return (
                    np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
                    np.array([[0.0], [0.0], [-1.0]]),
                )

            kwargs.update(fun_jac=fun_jac, bc_jac=bc_jac)
        result = solve_bvp(
            fun,
            bc,
            parameter_mesh,
            np.ones((2, parameter_mesh.size)),
            p=np.array([0.5]),
            # Match SciPy's official test_with_params default tolerance.
            tol=1e-3,
            max_nodes=100,
            telemetry=telemetry,
            **kwargs,
        )
    elif case == "nonlinear-boundary":
        interval_end = 0.1
        sample = np.linspace(0.0, interval_end, 5)
        boundary_mesh = np.linspace(0.0, interval_end, 5)

        def fun(x, y):
            return np.vstack((y[1], np.zeros_like(x)))

        def bc(ya, yb):
            kappa, io_a, io_c, voltage, factor = 1.64, 0.01, 1.0e-4, 0.5, 38.9
            h_a = -ya[0]
            current_a = io_a * (np.exp(factor * h_a) - np.exp(-factor * h_a))
            h_c = voltage - yb[0] - 1.0
            current_c = io_c * (np.exp(factor * h_c) - np.exp(-factor * h_c))
            return np.array([current_a + kappa * ya[1], current_c - kappa * yb[1]])

        result = solve_bvp(
            fun,
            bc,
            boundary_mesh,
            np.zeros((2, boundary_mesh.size)),
            tol=1e-3,
            max_nodes=100,
            telemetry=telemetry,
        )
    elif case in {"big-parameterized", "big-parameterized-analytic"}:
        systems = 30
        parameter_mesh = np.linspace(0.0, np.pi, 5)
        parameter_sample = np.linspace(0.0, np.pi, 100)
        sample = parameter_sample

        def fun(x, y, p):
            values = np.zeros_like(y)
            values[::2] = y[1::2]
            values[1::4] = -(p[0] ** 2) * y[::4]
            values[3::4] = -(p[1] ** 2) * y[2::4]
            return values

        def bc(ya, yb, p):
            return np.hstack((ya[::2], yb[::2], ya[1] - p[0], ya[3] - p[1]))

        kwargs = {}
        if case.endswith("-analytic"):
            def fun_jac(x, y, p):
                n, points = y.shape
                df_dy = np.zeros((n, n, points))
                df_dy[np.arange(0, n, 2), np.arange(1, n, 2)] = 1.0
                df_dy[np.arange(1, n, 4), np.arange(0, n, 4)] = -(p[0] ** 2)
                df_dy[np.arange(3, n, 4), np.arange(2, n, 4)] = -(p[1] ** 2)
                df_dp = np.zeros((n, 2, points))
                df_dp[np.arange(1, n, 4), 0] = -2.0 * p[0] * y[np.arange(0, n, 4)]
                df_dp[np.arange(3, n, 4), 1] = -2.0 * p[1] * y[np.arange(2, n, 4)]
                return df_dy, df_dp

            def bc_jac(ya, yb, p):
                n = ya.size
                dya = np.zeros((n + 2, n))
                dyb = np.zeros((n + 2, n))
                dya[np.arange(n // 2), np.arange(0, n, 2)] = 1.0
                dyb[np.arange(n // 2, n), np.arange(0, n, 2)] = 1.0
                dp = np.zeros((n + 2, 2))
                dp[n, 0] = -1.0
                dya[n, 1] = 1.0
                dp[n + 1, 1] = -1.0
                dya[n + 1, 3] = 1.0
                return dya, dyb, dp

            kwargs.update(fun_jac=fun_jac, bc_jac=bc_jac)
        result = solve_bvp(
            fun,
            bc,
            parameter_mesh,
            np.ones((2 * systems, parameter_mesh.size)),
            p=np.array([0.5, 0.5]),
            telemetry=telemetry,
            **kwargs,
        )
    elif case == "big-problem":
        system_count = 30
        big_mesh = np.linspace(0.0, 1.0, 5)

        def fun(x, y):
            values = np.zeros_like(y)
            values[::2] = y[1::2]
            return values

        def bc(ya, yb):
            return np.hstack((ya[::2], yb[::2] - 1.0))

        result = solve_bvp(
            fun,
            bc,
            big_mesh,
            np.zeros((2 * system_count, big_mesh.size)),
            tol=1e-3,
            max_nodes=100,
            telemetry=telemetry,
        )
    elif case == "boundary-tolerance-exhaustion":
        exhaustion_mesh = np.linspace(0.0, 1.0, 5)
        roundoff_offset = np.spacing(1.0) / 4.0

        def fun(x, y):
            return np.zeros_like(y)

        def bc(ya, yb):
            return np.array([(ya[0] - 1.0) + roundoff_offset])

        result = solve_bvp(
            fun,
            bc,
            exhaustion_mesh,
            np.ones((1, exhaustion_mesh.size)),
            tol=1e-3,
            bc_tol=roundoff_offset / 4.0,
            max_nodes=100,
            telemetry=telemetry,
        )
    elif case == "singular-jacobian":
        failure_mesh = np.linspace(0.0, 1.0, 2)

        def fun(x, y):
            return np.zeros_like(y)

        def bc(ya, yb):
            return np.array([ya[0], yb[0] - 1.0])

        result = solve_bvp(
            fun,
            bc,
            failure_mesh,
            np.zeros((2, failure_mesh.size)),
            tol=1e-5,
            max_nodes=5,
            telemetry=telemetry,
        )
    elif case == "shock-layer":
        shock_mesh = np.linspace(-1.0, 1.0, 5)
        sample = np.linspace(-1.0, 1.0, 100)
        epsilon = 1e-3

        def fun(x, y):
            return np.vstack((
                y[1],
                -(x * y[1] + epsilon * np.pi**2 * np.cos(np.pi * x)
                  + np.pi * x * np.sin(np.pi * x)) / epsilon,
            ))

        def bc(ya, yb):
            return np.array([ya[0] + 2.0, yb[0]])

        result = solve_bvp(
            fun,
            bc,
            shock_mesh,
            np.zeros((2, shock_mesh.size)),
            telemetry=telemetry,
        )
    elif case in {"bratu", "bratu-refine", "bratu-exhaustion"}:
        def fun(x, y):
            return np.vstack((y[1], -2.0 * np.exp(y[0])))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        tolerance = 1e-5 if case == "bratu" else 1e-8
        max_nodes = 8 if case == "bratu-exhaustion" else 200
        result = solve_bvp(fun, bc, mesh, np.zeros((2, mesh.size)),
                           tol=tolerance, max_nodes=max_nodes,
                           telemetry=telemetry)
    elif case == "singular":
        def fun(x, y):
            return np.vstack((y[1], -y[0] ** 5))

        def bc(ya, yb):
            return np.array([ya[1], yb[0] - np.sqrt(3.0 / 4.0)])

        singular_mesh = np.linspace(0.0, 1.0, 10)
        initial = np.empty((2, singular_mesh.size))
        initial[0] = np.sqrt(3.0 / 4.0)
        initial[1] = 1e-4
        result = solve_bvp(fun, bc, singular_mesh, initial,
                           S=np.array([[0.0, 0.0], [0.0, -2.0]]),
                           tol=1e-3, max_nodes=100, telemetry=telemetry)
    else:
        raise ValueError(f"unknown workload {case!r}")

    values = np.asarray(result.sol(sample), dtype=float)
    derivatives = np.asarray(result.sol(sample, 1), dtype=float)
    counters = telemetry.snapshot()["counters"]
    return {
        "case": case,
        "scipy_version": "1.18.1",
        "success": bool(result.success),
        "status": int(result.status),
        "niter": int(result.niter),
        "nodes": int(result.x.size),
        "parameters": [] if result.p is None else np.asarray(result.p).tolist(),
        "sample_x": sample.tolist(),
        "sample_y": values.tolist(),
        "sample_dy": derivatives.tolist(),
        "rms_residuals": np.asarray(result.rms_residuals, dtype=float).tolist(),
        "rms_residual_max": float(np.max(result.rms_residuals)),
        "telemetry": telemetry.snapshot(),
        "newton_trace": telemetry.newton_trace,
        "newton_convergence_trace": telemetry.newton_convergence_trace,
        # Stable, named counters make the Rust story output easy to parse
        # without serializing the full diagnostic timing payload into TSV.
        "newton_passes": counters.get("solve_newton_calls", 0),
        "jacobian_builds": counters.get("construct_global_jac_calls", 0),
        "factorizations": counters.get("factorization_calls", 0),
        "linear_solves": counters.get("linear_solve_calls", 0),
        "mesh_modify_calls": counters.get("modify_mesh_calls", 0),
        "rhs_callback_calls": counters.get("rhs_callback_calls", 0),
        "boundary_callback_calls": counters.get("boundary_callback_calls", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=("global-jacobian", "linear", "unknown-parameter", "unknown-parameter-analytic",
                 "exponential", "exponential-analytic", "exponential-tight",
                 "sturm-liouville",
                 "sturm-liouville-analytic", "bratu", "bratu-refine",
                 "bratu-exhaustion", "singular", "nonlinear-boundary",
                 "big-problem", "big-parameterized", "big-parameterized-analytic",
                 "shock-layer", "singular-jacobian",
                 "boundary-tolerance-exhaustion", "validation"),
    )
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()
    row = solve_reference(args.case)
    if args.format == "json":
        print(json.dumps(row, separators=(",", ":")))
    elif args.case == "global-jacobian":
        fields = (
            ("case", row["case"]),
            ("shape", ",".join(map(str, row["shape"]))),
            ("global_jacobian", ",".join(map(str, row["global_jacobian"]))),
            ("collocation_residual", ",".join(map(str, row["collocation_residual"]))),
        )
        print("\t".join(f"{key}={value}" for key, value in fields))
    elif args.case == "validation":
        fields = (
            ("case", row["case"]),
            ("validation_failures", ",".join(
                f"{name}:{kind}"
                for name, kind in row["validation_failures"].items()
            )),
        )
        print("\t".join(f"{key}={value}" for key, value in fields))
    else:
        fields = (
            ("case", row["case"]),
            ("status", row["status"]),
            ("success", int(row["success"])),
            ("niter", row["niter"]),
            ("nodes", row["nodes"]),
            ("parameters", ",".join(map(str, row["parameters"]))),
            ("sample_x", ",".join(map(str, row["sample_x"]))),
            ("sample_y", ";".join(
                ",".join(map(str, component)) for component in row["sample_y"]
            )),
            ("sample_dy", ";".join(
                ",".join(map(str, component)) for component in row["sample_dy"]
            )),
            ("rms_residuals", ",".join(map(str, row["rms_residuals"]))),
            ("newton_passes", row["newton_passes"]),
            ("jacobian_builds", row["jacobian_builds"]),
            ("factorizations", row["factorizations"]),
            ("linear_solves", row["linear_solves"]),
            ("mesh_modify_calls", row["mesh_modify_calls"]),
            ("rhs_callback_calls", row["rhs_callback_calls"]),
            ("boundary_callback_calls", row["boundary_callback_calls"]),
            ("newton_trace", ";".join(
                f'{entry["iteration"]}:{entry["trial"]}:a{entry["alpha"]:.4g}:'
                f'{entry["cost"]:.4g}>{entry["cost_new"]:.4g}:'
                f'{int(entry["accepted"])}'
                for entry in row["newton_trace"]
            )),
            ("newton_convergence_trace", ";".join(
                f'{entry["iteration"]}:r{entry["max_normalized_collocation_residual"]:.4g}:'
                f'b{entry["normalized_boundary_residual"]:.4g}:'
                f'{int(entry["converged"])}'
                for entry in row["newton_convergence_trace"]
            )),
        )
        print("\t".join(f"{key}={value}" for key, value in fields))


if __name__ == "__main__":
    main()

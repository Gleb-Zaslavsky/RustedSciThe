"""Fast debug contracts for the vendored SciPy BVP reference adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import scipy

REFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFERENCE_DIR))

from telemetry import BvpTelemetry, instrument_module


SCIPY_VERSION = "1.18.1"
EXPECTED_BVP_SHA256 = (
    "f8c4125dabc65bfcffbb9dcd2d1f438f6fc34089a4117af741a85ccc25506e92"
)


def load_bvp_module(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REFERENCE_DIR / "bvp.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load vendored SciPy BVP source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_linear(solve_bvp, telemetry=None):
    def fun(x, y):
        return np.ones_like(y)

    def bc(ya, yb):
        return np.array([ya[0]])

    kwargs = {"tol": 1e-7, "max_nodes": 100}
    if telemetry is not None:
        kwargs["telemetry"] = telemetry
    return solve_bvp(fun, bc, np.linspace(0.0, 1.0, 5),
                     np.zeros((1, 5)), **kwargs)


class BvpReferenceTelemetryTests(unittest.TestCase):
    def test_runtime_versions_and_vendored_source_hash_are_pinned(self):
        source = (REFERENCE_DIR / "bvp.py").read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), EXPECTED_BVP_SHA256)
        self.assertEqual(scipy.__version__, SCIPY_VERSION)
        self.assertEqual(np.__version__, "2.5.4")

    def test_instrumentation_is_observational_and_reports_stages(self):
        plain = load_bvp_module("bvp_reference_plain").solve_bvp
        instrumented_module = load_bvp_module("bvp_reference_instrumented")
        instrumented = instrument_module(instrumented_module)
        plain_result = solve_linear(plain)
        telemetry = BvpTelemetry()
        measured_result = solve_linear(instrumented, telemetry)

        np.testing.assert_array_equal(measured_result.x, plain_result.x)
        np.testing.assert_allclose(measured_result.y, plain_result.y, rtol=0, atol=0)
        self.assertEqual(measured_result.status, plain_result.status)
        self.assertEqual(measured_result.niter, plain_result.niter)
        self.assertGreater(telemetry.counters["solve_newton_calls"], 0)
        self.assertGreater(telemetry.counters["rhs_callback_calls"], 0)
        self.assertGreater(telemetry.timings_ms["solve_bvp"], 0.0)

    def test_disabled_instrumentation_preserves_result_without_collecting_data(self):
        module = load_bvp_module("bvp_reference_disabled")
        solve = instrument_module(module)
        telemetry = BvpTelemetry(enabled=False)
        result = solve_linear(solve, telemetry)
        self.assertTrue(result.success)
        self.assertEqual(telemetry.snapshot(), {"counters": {}, "timings_ms": {}})

    def test_opt_in_newton_trace_records_each_scipy_trial(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"), "shock-layer"],
            check=True, capture_output=True, text=True,
        )
        row = json.loads(completed.stdout)
        trace = row["newton_trace"]
        self.assertEqual(len(trace), row["linear_solves"] - row["factorizations"])
        self.assertEqual(len(row["newton_convergence_trace"]), len(trace))
        self.assertTrue(all("iteration" in trial and "cost_new" in trial
                            for trial in trace))
        self.assertTrue(all("max_normalized_collocation_residual" in entry
                            for entry in row["newton_convergence_trace"]))
        self.assertTrue(all(trial["accepted"] for trial in trace))

    def test_reference_runner_emits_compact_parameterized_parity_row(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"),
             "unknown-parameter"],
            check=True, capture_output=True, text=True,
        )
        row = json.loads(completed.stdout)
        self.assertTrue(row["success"])
        self.assertEqual(row["status"], 0)
        self.assertAlmostEqual(row["parameters"][0], 1.0, places=7)
        self.assertEqual(len(row["sample_x"]), len(row["sample_y"][0]))
        self.assertEqual(len(row["sample_dy"]), len(row["sample_y"]))
        self.assertEqual(len(row["sample_dy"][0]), len(row["sample_x"]))
        self.assertEqual(len(row["rms_residuals"]), row["nodes"] - 1)
        self.assertGreater(row["telemetry"]["counters"]["factorization_calls"], 0)
        counters = row["telemetry"]["counters"]
        self.assertEqual(row["newton_passes"], counters["solve_newton_calls"])
        self.assertEqual(row["jacobian_builds"], counters["construct_global_jac_calls"])

    def test_diagnostic_tsv_separates_analytic_callback_cost_and_vectorized_work(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"),
             "big-parameterized-analytic", "--format", "diagnostic-tsv"],
            check=True, capture_output=True, text=True,
        )
        fields = dict(
            part.split("=", 1)
            for part in completed.stdout.strip().split("\t")
        )
        self.assertEqual(fields["success"], "1")
        self.assertEqual(fields["status"], "0")
        self.assertEqual(fields["nodes"], "17")
        self.assertGreater(int(fields["counter.rhs_jacobian_callback_calls"]), 0)
        self.assertGreater(int(fields["counter.rhs_jacobian_callback_state_points"]), 0)
        self.assertGreater(int(fields["counter.boundary_jacobian_callback_calls"]), 0)
        self.assertGreater(int(fields["counter.newton_trials"]), 0)
        self.assertEqual(
            int(fields["counter.newton_trials"]),
            int(fields["counter.newton_accepted_trials"])
            + int(fields["counter.newton_rejected_trials"]),
        )
        self.assertGreater(float(fields["time_ms.rhs_jacobian_callback"]), 0.0)
        self.assertGreater(float(fields["time_ms.boundary_jacobian_callback"]), 0.0)
        self.assertNotIn("counter.rhs_finite_difference_callback_calls", fields)
        self.assertNotIn("counter.boundary_finite_difference_callback_calls", fields)
        mesh_count = int(fields["mesh_pass_count"])
        self.assertGreaterEqual(mesh_count, 2)
        self.assertEqual(fields["mesh.0.nodes"], "5")
        self.assertGreater(int(fields[f"mesh.{mesh_count - 1}.nodes"]), 5)
        for index in range(mesh_count):
            self.assertGreater(int(fields[f"mesh.{index}.dimension"]), 0)
            self.assertIn(f"mesh.{index}.counter.solve_newton_calls", fields)
            self.assertIn(f"mesh.{index}.time_ms.solve_newton", fields)

    def test_reference_runner_exposes_direct_global_jacobian_fixture(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"),
             "global-jacobian"],
            check=True, capture_output=True, text=True,
        )
        row = json.loads(completed.stdout)
        self.assertEqual(row["shape"], [7, 7])
        self.assertEqual(len(row["global_jacobian"]), 49)
        self.assertTrue(all(np.isfinite(row["global_jacobian"])))
        self.assertEqual(len(row["collocation_residual"]), 4)

    def test_reference_runner_pins_official_input_validation_categories(self):
        completed = subprocess.run(
            [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"),
             "validation", "--format", "tsv"],
            check=True, capture_output=True, text=True,
        )
        fields = dict(part.split("=", 1) for part in completed.stdout.strip().split("\t"))
        self.assertEqual(
            fields["validation_failures"],
            "non_increasing_mesh:ValueError,state_shape:ValueError,"
            "boundary_shape:ValueError",
        )

    def test_analytic_jacobian_route_preserves_controller_work(self):
        for stem in ("unknown-parameter", "sturm-liouville"):
            with self.subTest(problem=stem):
                rows = {}
                for suffix in ("", "-analytic"):
                    case = stem + suffix
                    completed = subprocess.run(
                        [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"), case],
                        check=True, capture_output=True, text=True,
                    )
                    rows[case] = json.loads(completed.stdout)

                finite_difference = rows[stem]["telemetry"]["counters"]
                analytic = rows[stem + "-analytic"]["telemetry"]["counters"]
                for key in ("construct_global_jac_calls", "factorization_calls",
                            "linear_solve_calls", "solve_newton_calls"):
                    self.assertEqual(finite_difference[key], analytic[key], key)
                self.assertGreater(finite_difference["rhs_finite_difference_callback_calls"], 0)
                self.assertNotIn("rhs_finite_difference_callback_calls", analytic)
                self.assertGreater(finite_difference["boundary_finite_difference_callback_calls"], 0)
                self.assertNotIn("boundary_finite_difference_callback_calls", analytic)
                np.testing.assert_allclose(
                    rows[stem]["sample_y"],
                    rows[stem + "-analytic"]["sample_y"],
                    rtol=2e-6,
                    atol=2e-8,
                )

    def test_nonlinear_mesh_refinement_and_singular_reference_cases(self):
        rows = {}
        for case in ("bratu", "bratu-refine", "bratu-exhaustion", "singular",
                     "exponential", "exponential-analytic", "sturm-liouville",
                     "exponential-tight", "sturm-liouville-analytic", "nonlinear-boundary",
                     "big-problem", "big-parameterized-analytic", "shock-layer",
                     "boundary-tolerance-exhaustion"):
            with self.subTest(case=case):
                completed = subprocess.run(
                    [sys.executable, str(REFERENCE_DIR / "bvp_parity.py"), case],
                    check=True, capture_output=True, text=True,
                )
                row = json.loads(completed.stdout)
                rows[case] = row
                self.assertIn(row["status"], (0, 1, 2, 3))
                self.assertGreaterEqual(row["nodes"], 5)
                self.assertTrue(all(np.isfinite(row["sample_y"][0])))
        self.assertEqual(rows["bratu"]["status"], 0)
        self.assertEqual(rows["singular"]["status"], 0)
        self.assertGreater(rows["bratu-refine"]["telemetry"]["counters"].get(
            "modify_mesh_calls", 0
        ), 0)
        self.assertEqual(rows["boundary-tolerance-exhaustion"]["status"], 3)
        self.assertEqual(rows["shock-layer"]["status"], 0)
        self.assertEqual(rows["big-parameterized-analytic"]["status"], 0)
        self.assertEqual(len(rows["big-parameterized-analytic"]["parameters"]), 2)


if __name__ == "__main__":
    unittest.main()

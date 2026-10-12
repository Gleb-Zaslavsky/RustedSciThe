import unittest

import numpy as np
from scipy.integrate._ivp.bdf import BDF as InstalledSciPyBDF
from scipy.sparse import csc_matrix

from ivp_scipy_reference import BDF, BdfTelemetry


def solve_scalar(telemetry=None, jac=None):
    solver = BDF(
        lambda _t, y: -10.0 * y,
        0.0,
        np.array([1.0]),
        0.4,
        first_step=0.02,
        max_step=0.05,
        rtol=1e-7,
        atol=1e-10,
        jac=jac,
        telemetry=telemetry,
    )
    while solver.status == "running":
        solver.step()
    return solver


class BdfTelemetrySemanticsTests(unittest.TestCase):
    def test_instrumented_reference_matches_installed_scipy_step_trace(self):
        telemetry = BdfTelemetry()
        rhs = lambda _t, y: -10.0 * y
        kwargs = dict(
            t0=0.0,
            y0=np.array([1.0]),
            t_bound=0.4,
            first_step=0.02,
            max_step=0.05,
            rtol=1e-7,
            atol=1e-10,
        )
        reference = InstalledSciPyBDF(rhs, **kwargs)
        instrumented = BDF(rhs, **kwargs, telemetry=telemetry)

        while reference.status == "running":
            reference.step()
            instrumented.step()
            self.assertEqual(instrumented.status, reference.status)
            self.assertEqual(instrumented.t, reference.t)
            np.testing.assert_array_equal(instrumented.y, reference.y)
            self.assertEqual(instrumented.order, reference.order)
            self.assertEqual(instrumented.h_abs, reference.h_abs)

        self.assertEqual(
            (instrumented.nfev, instrumented.njev, instrumented.nlu),
            (reference.nfev, reference.njev, reference.nlu),
        )

    def test_native_counter_contract_for_finite_difference_jacobian(self):
        telemetry = BdfTelemetry()
        solver = solve_scalar(telemetry)
        counters = telemetry.counters

        self.assertGreater(counters["fd_baseline_rhs_calls"], 0)
        self.assertGreater(counters["fd_rhs_callback_calls"], 0)
        self.assertEqual(
            solver.nfev,
            counters["rhs_callback_calls"]
            - counters["fd_baseline_rhs_calls"]
            - counters["fd_rhs_callback_calls"],
        )
        self.assertEqual(solver.njev, counters["jacobian_refreshes"])
        self.assertEqual(solver.nlu, counters["factorization_calls"])
        self.assertEqual(
            counters["accepted_steps"],
            sum(1 for attempt in telemetry.controller_trace if attempt[5]),
        )
        self.assertEqual(
            counters["step_attempts"], len(telemetry.controller_trace)
        )
        self.assertEqual(
            counters["newton_iterations"],
            sum(attempt[3] for attempt in telemetry.controller_trace),
        )
        self.assertGreater(counters["linear_solve_calls"], 0)
        self.assertGreater(counters["newton_iterations"], 0)
        self.assertGreater(counters["error_estimates"], 0)

    def test_analytic_jacobian_callbacks_are_counted_separately(self):
        telemetry = BdfTelemetry()
        solver = solve_scalar(telemetry, jac=lambda _t, _y: np.array([[-10.0]]))

        self.assertEqual(solver.njev, telemetry.counters["jacobian_callback_calls"])
        self.assertEqual(solver.njev, telemetry.counters["jacobian_refreshes"])
        self.assertEqual(solver.nlu, telemetry.counters["factorization_calls"])
        self.assertGreater(telemetry.timings_ms["jacobian_callback"], 0.0)

    def test_sparse_lu_path_uses_the_same_native_counter_contract(self):
        telemetry = BdfTelemetry()
        solver = solve_scalar(
            telemetry,
            jac=lambda _t, _y: csc_matrix([[-10.0]]),
        )

        self.assertEqual(solver.nlu, telemetry.counters["factorization_calls"])
        self.assertEqual(
            solver.njev, telemetry.counters["jacobian_callback_calls"]
        )
        self.assertGreater(telemetry.counters["linear_solve_calls"], 0)

    def test_disabled_telemetry_preserves_native_results_and_counters(self):
        plain = solve_scalar()
        disabled = BdfTelemetry(enabled=False)
        observed = solve_scalar(disabled)

        np.testing.assert_array_equal(observed.y, plain.y)
        self.assertEqual(observed.t, plain.t)
        self.assertEqual(
            (observed.nfev, observed.njev, observed.nlu),
            (plain.nfev, plain.njev, plain.nlu),
        )
        self.assertEqual(disabled.counters, {})
        self.assertEqual(disabled.timings_ms, {})
        self.assertEqual(disabled.controller_trace, [])

    def test_enabled_telemetry_does_not_change_numerical_trace(self):
        plain = solve_scalar()
        telemetry = BdfTelemetry()
        observed = solve_scalar(telemetry)

        np.testing.assert_array_equal(observed.y, plain.y)
        self.assertEqual(observed.t, plain.t)
        self.assertEqual(
            (observed.nfev, observed.njev, observed.nlu),
            (plain.nfev, plain.njev, plain.nlu),
        )

    def test_controller_trace_is_bounded_and_snapshot_keeps_native_counts(self):
        telemetry = BdfTelemetry(max_trace_entries=1)
        solver = solve_scalar(telemetry)
        snapshot = telemetry.snapshot(solver)

        self.assertEqual(len(telemetry.controller_trace), 1)
        self.assertTrue(telemetry.trace_truncated)
        self.assertEqual(snapshot["scipy_nfev"], solver.nfev)
        self.assertEqual(snapshot["scipy_njev"], solver.njev)
        self.assertEqual(snapshot["scipy_nlu"], solver.nlu)
        self.assertGreaterEqual(telemetry.timings_ms["step_total"], 0.0)
        self.assertGreaterEqual(telemetry.timings_ms["newton_solve"], 0.0)

    def test_dense_output_construction_and_evaluation_are_timed(self):
        telemetry = BdfTelemetry()
        solver = BDF(
            lambda _t, y: -y,
            0.0,
            np.array([1.0]),
            0.1,
            first_step=0.02,
            telemetry=telemetry,
        )
        solver.step()
        dense = solver.dense_output()
        values = dense(np.array([solver.t_old, solver.t]))

        self.assertTrue(np.all(np.isfinite(values)))
        self.assertIn("dense_output_construction", telemetry.timings_ms)
        self.assertIn("dense_output_evaluation", telemetry.timings_ms)


if __name__ == "__main__":
    unittest.main()

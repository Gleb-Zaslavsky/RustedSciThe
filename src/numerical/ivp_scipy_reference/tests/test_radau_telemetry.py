import unittest

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize._numdiff import group_columns
from scipy.sparse import csc_matrix, eye, issparse
from scipy.sparse.linalg import splu

from ivp_scipy_reference.radau import Radau
from ivp_scipy_reference.telemetry import RadauTelemetry


class RadauTelemetrySemanticsTests(unittest.TestCase):
    def test_scipy_nfev_excludes_finite_difference_probes(self):
        telemetry = RadauTelemetry()
        solver = Radau(
            lambda _t, y: -10.0 * y,
            0.0,
            np.array([1.0]),
            0.2,
            first_step=0.02,
            max_step=0.05,
            rtol=1e-7,
            atol=1e-10,
            jac=None,
            telemetry=telemetry,
        )
        while solver.status == "running":
            solver.step()

        self.assertGreater(telemetry.counters["fd_rhs_callback_calls"], 0)
        self.assertEqual(
            solver.nfev,
            telemetry.counters["rhs_callback_calls"]
            - telemetry.counters["fd_rhs_callback_calls"],
        )
        self.assertEqual(solver.njev, telemetry.counters["jacobian_refreshes"])
        self.assertEqual(solver.nlu, telemetry.counters["lu_factorization_calls"])
        self.assertEqual(
            solver.nlu,
            2 * telemetry.counters["factorization_pair_attempts"],
        )

    def test_telemetry_disabled_preserves_scipy_counters(self):
        solver = Radau(
            lambda _t, y: -10.0 * y,
            0.0,
            np.array([1.0]),
            0.2,
            first_step=0.02,
            max_step=0.05,
            rtol=1e-7,
            atol=1e-10,
            jac=lambda _t, _y: np.array([[-10.0]]),
            telemetry=RadauTelemetry(enabled=False),
        )
        while solver.status == "running":
            solver.step()
        self.assertGreater(solver.nfev, 0)
        self.assertGreater(solver.njev, 0)
        self.assertGreater(solver.nlu, 0)


class SciPyHelperApiContractTests(unittest.TestCase):
    def test_fd_column_coloring_contract(self):
        pattern = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        )
        groups = group_columns(pattern)

        self.assertEqual(groups.shape, (pattern.shape[1],))
        self.assertTrue(np.issubdtype(groups.dtype, np.integer))
        for group in np.unique(groups):
            columns = np.flatnonzero(groups == group)
            occupied_rows = pattern[:, columns]
            self.assertTrue(np.all(occupied_rows.sum(axis=1) <= 1))

    def test_dense_and_sparse_lu_contracts(self):
        dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([1.0, 2.0])
        dense_solution = lu_solve(lu_factor(dense), rhs)
        np.testing.assert_allclose(dense @ dense_solution, rhs)

        sparse = eye(2, format="csc") * 2.0 + csc_matrix([[1.0, 0.0], [0.0, 3.0]])
        self.assertTrue(issparse(sparse))
        sparse_solution = splu(sparse).solve(rhs)
        np.testing.assert_allclose(sparse @ sparse_solution, rhs)


if __name__ == "__main__":
    unittest.main()

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splu_benchmark import benchmark_fixture


class SpluFixtureTests(unittest.TestCase):
    def run_fixture(self, content: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.fixture"
            path.write_text(content, encoding="ascii")
            return benchmark_fixture(path, repeats=1)

    def test_shared_rust_fixture_includes_known_solution(self) -> None:
        # Header: n, mesh nodes, state block size, border size, triplet count.
        result = self.run_fixture(
            "2 2 1 0 4\n"
            "3 4\n"
            "1 1\n"
            "0 0 2\n0 1 1\n1 0 1\n1 1 3\n"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["csc_nnz"], "4")
        self.assertLess(float(result["known_solution_max_diff"]), 1e-14)

    def test_legacy_bvp_fixture_remains_supported(self) -> None:
        result = self.run_fixture("2 2 4\n3 4\n0 0 2\n0 1 1\n1 0 1\n1 1 3\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["csc_stored_entries"], "4")
        self.assertEqual(result["state_dimension"], "0")
        self.assertLess(float(result["relative_residual_inf"]), 1e-14)

    def test_bvp_compatibility_entry_point_delegates_to_shared_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "legacy.fixture"
            fixture.write_text(
                "2 2 4\n3 4\n0 0 2\n0 1 1\n1 0 1\n1 1 3\n",
                encoding="ascii",
            )
            repository = Path(__file__).resolve().parents[4]
            entry_point = repository / "src/numerical/BVP_sci/scipy_reference/linear_kernel_benchmark.py"
            output = subprocess.run(
                [sys.executable, str(entry_point), str(fixture), "--repeats", "1"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("status=ok", output.stdout)
        self.assertIn("factor_median_ms=", output.stdout)
        self.assertIn("relative_residual_inf=", output.stdout)


if __name__ == "__main__":
    unittest.main()

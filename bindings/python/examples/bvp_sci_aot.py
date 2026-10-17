"""Compare BVP_sci Lambdify and AOT on the same sparse exact BVP.

The equation ``y'=-y`` with left boundary value ``y(0)=1`` has the exact
profile ``exp(-x)``. Both runs use AtomNative, Sparse storage, the same mesh,
guess and tolerances; only callback execution changes. AOT needs ``tcc`` or
``gcc`` on PATH and includes code generation/compilation, so this is a
correctness and lifecycle illustration, not a speed benchmark.
"""

import shutil
import tempfile

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_bvp


compiler = shutil.which("tcc") or shutil.which("gcc")
if compiler is None:
    raise SystemExit("This example requires tcc or gcc on PATH.")
toolchain = "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"
mesh = np.linspace(0.0, 1.0, 18)
# A small smooth perturbation keeps the initial guess distinct from the exact
# solution while preserving both endpoint values.
guess = np.exp(-mesh) * (1.0 + 0.02 * np.sin(np.pi * mesh))
y = Expr.variable("y")
results = {}

for execution in ("lambdify", "aot"):
    with tempfile.TemporaryDirectory(
        prefix=f"rustedscithe-bvp-sci-{execution}-",
        ignore_cleanup_errors=True,
    ) as artifacts:
        options = SolverOptions(
            frontend="atom_native",
            execution=execution,
            matrix_layout="sparse",
            telemetry="timings",
            aot_policy="build_if_missing" if execution == "aot" else None,
            aot_toolchain=toolchain if execution == "aot" else None,
            aot_profile="debug" if execution == "aot" else None,
            aot_output_dir=artifacts if execution == "aot" else None,
        )
        # The AOT branch builds a fresh artifact in this directory. Lambdify
        # has no compiler lifecycle, so its AOT-only fields remain unset.
        solution = solve_bvp(
            "bvp_sci",
            [-y],
            ["y"],
            [Expr.parse("ya_y - 1")],
            mesh,
            guess,
            options=options,
            tolerance=1e-7,
            max_nodes=100,
            output="dense",
        )
        results[execution] = np.asarray(
            [solution.sample(float(x))[0] for x in mesh], dtype=np.float64
        )
        exact = np.exp(-mesh)
        exact_error = float(np.max(np.abs(results[execution] - exact)))
        if solution.status != "success" or exact_error > 1e-6:
            raise AssertionError(
                f"{execution}: status={solution.status!r}, "
                f"maximum exact-profile error={exact_error:.3e}"
            )
        print(
            f"execution={execution:8} "
            f"max_exact_error={exact_error:.3e} "
            f"status={solution.status}"
        )
        print("telemetry counters:", solution.telemetry_counters())
        # Core stages overlap the Python call wall time; do not sum them into
        # an alleged total or treat this smoke example as a benchmark.
        print("stage timings:", solution.telemetry_timings_ms())
        del solution

route_difference = float(
    np.max(np.abs(results["lambdify"] - results["aot"]))
)
print(f"max_lambdify_aot_difference={route_difference:.3e}")
if route_difference > 2e-6:
    raise SystemExit("Lambdify and AOT profiles differ beyond the smoke tolerance.")

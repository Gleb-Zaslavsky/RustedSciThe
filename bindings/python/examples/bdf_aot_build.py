"""Build and execute a small BDF AOT model using a local C compiler.

This optional example requires ``tcc`` or ``gcc`` on PATH. Generated files
live in a temporary directory; cleanup is best-effort on Windows because a
loaded DLL may remain locked until process exit. AOT startup includes code
generation and compilation; use Lambdify for one-off small solves and
benchmark continuation before choosing a route for a workload.
"""

import shutil
import tempfile

from rustedscithe import Expr, SolverOptions, create_ivp


compiler = shutil.which("tcc") or shutil.which("gcc")
if compiler is None:
    raise SystemExit("This example requires tcc or gcc on PATH.")
toolchain = "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"

y = Expr.variable("y")
with tempfile.TemporaryDirectory(
    prefix="rustedscithe-bdf-aot-",
    ignore_cleanup_errors=True,
) as artifacts:
    solver = create_ivp(
        "bdf",
        [-y],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.2,
        0.05,
        options=SolverOptions(
            execution="aot",
            aot_policy="build_if_missing",
            aot_toolchain=toolchain,
            aot_profile="debug",
            aot_output_dir=artifacts,
            telemetry="counters",
        ),
    )
    solver.prepare()
    solution = solver.solve()
    print("toolchain:", toolchain)
    print("route:", solver.provenance())
    print("final state:", solution.states()[-1])
    print("AOT counters:", solution.telemetry_counters())
    # Release the solver before cleanup. Windows may still retain the loaded
    # generated DLL until process exit, so cleanup is best-effort there.
    del solution
    del solver

"""Build one LSODE2 AOT model and reuse it for a parameter continuation.

The piecewise decay problem uses ``y'=-alpha*y`` with ``y(0)=1``. First,
``alpha=1`` is integrated to ``t=0.5``; then the same prepared symbolic model
continues with ``alpha=2`` to ``t=1``. The exact final value is ``exp(-1.5)``.
This illustrates artifact reuse and numeric rebinding, not a performance
benchmark. The first preparation includes code generation and compilation.

Requires ``tcc`` or ``gcc`` on ``PATH``. Build artifacts are placed in a
temporary directory; on Windows, a loaded DLL can delay cleanup until exit.
"""

import math
import shutil
import tempfile
from typing import Literal

from rustedscithe import Expr, Lsode2Solver, SolverOptions


compiler = shutil.which("tcc") or shutil.which("gcc")
if compiler is None:
    raise SystemExit("This example requires tcc or gcc on PATH.")
toolchain: Literal["tcc", "gcc"] = (
    "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"
)

y = Expr.variable("y")
alpha = Expr.variable("alpha")
with tempfile.TemporaryDirectory(
    prefix="rustedscithe-lsode2-aot-",
    ignore_cleanup_errors=True,
) as artifacts:
    solver = Lsode2Solver(
        [-alpha * y],
        ["y"],
        "t",
        [1.0],
        0.0,
        0.5,
        0.1,
        method="bdf",
        parameters=["alpha"],
        parameter_values=[1.0],
        # This must stay identical for the full continuation series. Only
        # `parameter_values` and the requested terminal time are rebound.
        options=SolverOptions(
            frontend="atom_native",
            execution="aot",
            matrix_layout="dense",
            telemetry="detailed",
            aot_policy="build_if_missing",
            aot_toolchain=toolchain,
            aot_profile="debug",
            aot_output_dir=artifacts,
        ),
        rtol=1e-8,
        atol=1e-10,
    )
    solver.prepare()
    first = solver.solve()
    continued = solver.continue_with_parameters([2.0], 1.0)

    assert continued.final_state is not None
    final_value = float(continued.final_state[0])
    exact_value = math.exp(-1.5)
    print("toolchain:", toolchain)
    print("AOT provenance:", solver.provenance())
    print("first segment endpoint:", first.final_state)
    print("continued endpoint:", final_value)
    print("exact endpoint:", exact_value)
    print("continuation counters:", continued.telemetry_counters())
    print("AOT stages:", continued.telemetry_aot_timings_ms())
    assert abs(final_value - exact_value) < 2e-7
    # Drop references before the temporary directory is removed; Windows may
    # keep a generated library loaded until the solver object is destroyed.
    del continued, first, solver

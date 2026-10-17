"""Solve a parameterized nonlinear equation through AtomNative AOT.

The equation ``x**2-a=0`` has the positive root ``sqrt(a)``. The unknown is
``x`` while ``a`` is a numeric parameter, so one prepared symbolic problem can
be solved for several values without rebuilding its expression graph. This
example demonstrates the nonlinear facade's compiler-backed route; the first
solve includes AOT preparation and is not a timing comparison.

Requires ``tcc`` or ``gcc`` on ``PATH``. Generated artifacts use a temporary
directory; Windows can defer removal of a loaded DLL until process exit.
"""

import math
import shutil
import tempfile

from rustedscithe import Expr, NonlinearSolver, SymbolicSystem


compiler = shutil.which("tcc") or shutil.which("gcc")
if compiler is None:
    raise SystemExit("This example requires tcc or gcc on PATH.")
toolchain = "tcc" if compiler.lower().endswith(("tcc", "tcc.exe")) else "gcc"

x = Expr.variable("x")
a = Expr.variable("a")
system = SymbolicSystem([x**2 - a], ["x", "a"])

with tempfile.TemporaryDirectory(
    prefix="rustedscithe-nonlinear-aot-",
    ignore_cleanup_errors=True,
) as artifacts:
    solver = NonlinearSolver(
        system,
        method="damped_newton",
        frontend="atom_native",
        execution="aot",
        parameters=["a"],
        aot_policy="build_if_missing",
        aot_profile="debug",
        aot_output_dir=artifacts,
        telemetry="detailed",
        tolerance=1e-10,
    )
    # Explicit preparation makes the symbolic/AOT lifecycle visible. The
    # subsequent calls change only numeric parameter values and initial guesses.
    solver.prepare()
    print("backend:", solver.backend())
    print("preparation:", solver.prepare_report())

    for value in (2.0, 5.0, 10.0):
        result = solver.solve([1.0], parameters=[value])
        root = float(result.solution()[0])
        expected = math.sqrt(value)
        print(
            f"a={value:4.1f} x={root:.10f} expected={expected:.10f} "
            f"residual_norm={result.residual_norm:.3e}"
        )
        print("  solve counters:", result.telemetry_counters())
        assert abs(root - expected) < 1e-8
        assert result.residual_norm < 1e-9

    del solver

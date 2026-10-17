"""Optional AOT Radau5 solve for a mass-conserving Robertson variant.

This uses the same slow-recycling Robertson variant as
``radau_robertson_routes.py`` but selects AtomNative+AOT+Sparse explicitly. It
requires a local ``tcc`` or
``gcc`` and keeps generated artifacts in a temporary directory. The script
prints lifecycle provenance so a cold build is not confused with a cache hit.
"""

import shutil
import tempfile

import numpy as np

from rustedscithe import Expr, SolverOptions, solve_ivp


compiler_path = shutil.which("tcc") or shutil.which("gcc")
if compiler_path is None:
    raise SystemExit("Optional example: install tcc or gcc and retry.")
toolchain = "tcc" if compiler_path.lower().endswith(("tcc", "tcc.exe")) else "gcc"

y1, y2, y3 = (Expr.variable(name) for name in ("y1", "y2", "y3"))
rhs = [
    Expr.constant(-0.04) * y1 + Expr.constant(1.0e4) * y2 * y3
    + Expr.constant(0.01) * y3,
    Expr.constant(0.04) * y1 - Expr.constant(1.0e4) * y2 * y3
    - Expr.constant(3.0e7) * y2 * y2,
    Expr.constant(3.0e7) * y2 * y2 - Expr.constant(0.01) * y3,
]
jacobian = [
    Expr.constant(-0.04), Expr.constant(1.0e4) * y3,
    Expr.constant(1.0e4) * y2 + Expr.constant(0.01),
    Expr.constant(0.04), Expr.constant(-1.0e4) * y3 - Expr.constant(6.0e7) * y2,
    Expr.constant(-1.0e4) * y2,
    Expr.constant(0.0), Expr.constant(6.0e7) * y2, Expr.constant(-0.01),
]

with tempfile.TemporaryDirectory(
    prefix="rustedscithe-radau-robertson-aot-",
    ignore_cleanup_errors=True,
) as artifacts:
    # Use a fresh location so provenance reports a real build, not an artifact
    # left by an earlier run. The `debug` profile favors quick compilation.
    options = SolverOptions(
        frontend="atom_native",
        execution="aot",
        matrix_layout="sparse",
        aot_policy="build_if_missing",
        aot_toolchain=toolchain,
        aot_profile="debug",
        aot_output_dir=artifacts,
        telemetry="timings",
    )
    solution = solve_ivp(
        "radau5", rhs, ["y1", "y2", "y3"], "t",
        np.array([1.0, 0.0, 0.0]), 0.0, 100.0, 100.0,
        options=options,
        # Supplying the exact Jacobian avoids conflating AOT callback execution
        # with finite-difference Jacobian work in this route demonstration.
        jacobian=jacobian,
        rtol=1e-7,
        atol=1e-10,
        max_steps=20_000,
    )
    final_state = solution.final_state_array()
    mass_error = abs(float(final_state.sum()) - 1.0)
    # The slow recycle transfers mass between species but creates or destroys
    # none; this guards the AOT callback route independently of its timings.
    if not np.isfinite(final_state).all() or mass_error > 1e-7:
        raise AssertionError(f"AOT Robertson mass/finiteness gate failed: {mass_error:.3e}")
    print("toolchain:", toolchain)
    print("final state:", final_state)
    print(f"mass conservation error: {mass_error:.3e}")
    print("provenance:", solution.diagnostics())
    print("stage timings:", solution.telemetry_timings_ms())

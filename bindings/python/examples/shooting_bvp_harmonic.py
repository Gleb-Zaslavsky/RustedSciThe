"""Solve a small exact BVP with the shooting method and inspect its phase path.

The second-order equation ``y''=-y`` is rewritten as the first-order system
``y'=v, v'=-y``. Dirichlet conditions ``y(0)=0`` and ``y(pi/2)=1`` imply the
unknown shooting slope ``v(0)=1`` and exact solution ``y=sin(x)``. Shooting is
best suited to scalar and very small systems (roughly two or three states),
not large coupled BVPs; each root trial invokes an IVP solve. The optional
animation plots the state-space curve ``(y, y')`` rather than the spatial
profile.
"""

from __future__ import annotations

import math

import numpy as np

from rustedscithe import (
    Expr,
    ShootingBvpSolver,
    SolverOptions,
    animate_solution_2d,
)


def main() -> None:
    # Rewrite y'' = -y as y' = v and v' = -y. The exact solution satisfying
    # y(0)=0, y(pi/2)=1 is y=sin(x), with initial slope y'(0)=1.
    options = SolverOptions(frontend="atom_native", telemetry="timings")
    # Uncomment to create a timestamped Rust log. RST_SOLVER_LOG_DIR can select
    # the directory before Python starts; logging is off by default.
    # options.set_logging_level("info")
    solver = ShootingBvpSolver(
        equations=[Expr.parse("v"), Expr.parse("-y")],
        state_variables=["y", "v"],
        independent_variable="x",
        t_start=0.0,
        t_end=math.pi / 2.0,
        left_boundary_type="dirichlet",
        left_boundary_value=0.0,
        right_boundary_type="dirichlet",
        right_boundary_value=1.0,
        initial_guess=0.5,
        tolerance=1e-9,
        max_iterations=50,
        step_size=0.01,
        ivp_method="bdf",
        solver_parameters={"rtol": 1e-9, "atol": 1e-11, "max_step": 0.015},
        options=options,
    )
    solution = solver.solve()

    mesh = solution.mesh_array()
    states = solution.states_array()
    exact_error = float(np.max(np.abs(states[:, 0] - np.sin(mesh))))
    slope_error = abs(solution.shooting_value - 1.0)
    # The IVP tolerances and shooting root tolerance are both tight here; this
    # gate checks the demonstrated small-system case, not large-system fitness.
    if exact_error > 1e-6 or slope_error > 1e-6:
        raise AssertionError(
            f"harmonic shooting error is too large: profile={exact_error:.3e}, "
            f"slope={slope_error:.3e}"
        )
    print(f"inferred initial slope: {solution.shooting_value:.12g}")
    print(f"maximum |y - sin(x)|: {exact_error:.3e}")
    print(f"preparation time: {solution.preparation_ms:.3f} ms")
    print("solve telemetry:")
    for name, value in sorted((solution.telemetry or {}).items()):
        print(f"  {name}: {value:.6g}")

    # The state vector is [y, y']; animate its phase portrait if desired.
    # animate_solution_2d(solution, columns=(0, 1), axis_names=("y", "y'"))


if __name__ == "__main__":
    main()

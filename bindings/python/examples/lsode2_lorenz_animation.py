"""Solve Lorenz with LSODE2 and show its 3D trajectory by default.

Use ``--animate 2d`` / ``--animate 3d`` for one view, ``both`` for sequential
views, or ``--no-animate`` for a headless numerical run.
"""

import argparse

import numpy as np

from rustedscithe import (
    Expr,
    SolverOptions,
    animate_solution_2d,
    animate_solution_3d,
    solve_ivp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    animation = parser.add_mutually_exclusive_group()
    animation.add_argument(
        "--animate", choices=("2d", "3d", "both"), default="3d",
        help="show one projection or both sequentially (default: 3d)",
    )
    animation.add_argument(
        "--no-animate", dest="animate", action="store_const", const=None,
        help="solve and print diagnostics without opening a renderer window",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    args = parser.parse_args()

    x, y, z = (Expr.variable(name) for name in ("x", "y", "z"))
    rhs = [
        10.0 * (y - x),
        x * (28.0 - z) - y,
        x * y - (8.0 / 3.0) * z,
    ]
    # Auto lets LSODE2 choose/switch Adams or BDF for this stiff, nonlinear
    # attractor; dense storage is appropriate for its three coupled states.
    solution = solve_ivp(
        "lsode2",
        rhs,
        ["x", "y", "z"],
        "t",
        [1.0, 1.0, 1.0],
        0.0,
        25.0,
        0.01,
        controller="auto",
        options=SolverOptions(
            frontend="atom_native",
            execution="lambdify",
            matrix_layout="dense",
            telemetry="timings",
        ),
        rtol=1e-8,
        atol=1e-10,
    )
    states = solution.states_array()
    times = solution.times_array()
    # Row i of `states` belongs to `times[i]`; animation consumes this native
    # trajectory and does not rerun the differential equation.
    assert np.isfinite(times).all() and np.isfinite(states).all()
    print("samples:", len(times))
    print("controller/family:", solution.controller_mode, solution.executed_family)
    print("final state:", states[-1])

    if args.animate in ("2d", "both"):
        # State order is [x,y,z]. The (x,z) projection exposes the familiar
        # two-lobe attractor; it is a view of the same computed trajectory.
        print("Opening 2D animation; close the window or press Esc to finish.")
        animate_solution_2d(solution, (0, 2), ("x", "z"), args.speed)
    if args.animate in ("3d", "both"):
        print("Opening 3D animation; close the window or press Esc to finish.")
        animate_solution_3d(solution, (0, 1, 2), ("x", "y", "z"), args.speed)


if __name__ == "__main__":
    main()

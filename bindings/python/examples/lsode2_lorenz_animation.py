"""Solve Lorenz with LSODE2 and optionally animate a 2D/3D projection.

Run ``python examples/lsode2_lorenz_animation.py`` for compute-only mode.
Add ``--animate 2d`` or ``--animate 3d`` to open the interactive Bevy window.
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
    parser.add_argument("--animate", choices=("2d", "3d"), default=None)
    parser.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    args = parser.parse_args()

    x, y, z = (Expr.variable(name) for name in ("x", "y", "z"))
    rhs = [
        10.0 * (y - x),
        x * (28.0 - z) - y,
        x * y - (8.0 / 3.0) * z,
    ]
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
    assert np.isfinite(times).all() and np.isfinite(states).all()
    print("samples:", len(times))
    print("controller/family:", solution.controller_mode, solution.executed_family)
    print("final state:", states[-1])

    if args.animate == "2d":
        # The (x,z) projection exposes the familiar two-lobe Lorenz attractor.
        animate_solution_2d(solution, (0, 2), ("x", "z"), args.speed)
    elif args.animate == "3d":
        animate_solution_3d(solution, (0, 1, 2), ("x", "y", "z"), args.speed)


if __name__ == "__main__":
    main()

"""Solve the planar three-body example and animate one body's XY path.

The interactive animation is shown by default. Pass ``--no-animate`` for a
headless numerical run suitable for batch checks.
"""

import argparse

import numpy as np

from rustedscithe import Expr, SolverOptions, animate_solution_2d, solve_ivp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    animation = parser.add_mutually_exclusive_group()
    animation.add_argument("--animate", dest="animate", action="store_true")
    animation.add_argument("--no-animate", dest="animate", action="store_false")
    parser.set_defaults(animate=True)
    parser.add_argument("--body", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.01,
        help="playback rate multiplier; the short default interval is slowed down",
    )
    args = parser.parse_args()

    names = [
        "x0", "vx0", "y0", "vy0",
        "x1", "vx1", "y1", "vy1",
        "x2", "vx2", "y2", "vy2",
    ]
    v = {name: Expr.variable(name) for name in names}
    aliases = {
        "R01": Expr.parse("((x0-x1)^2 + (y0-y1)^2)^0.5"),
        "R02": Expr.parse("((x0-x2)^2 + (y0-y2)^2)^0.5"),
        "R12": Expr.parse("((x1-x2)^2 + (y1-y2)^2)^0.5"),
        "k": Expr.constant(39.47841760435743),
        "m0": Expr.constant(1.0),
        "m1": Expr.constant(0.5),
        "m2": Expr.constant(0.75),
    }
    acceleration = [
        Expr.parse("-k*(m1*(x0-x1)/R01^3 + m2*(x0-x2)/R02^3)"),
        Expr.parse("-k*(m1*(y0-y1)/R01^3 + m2*(y0-y2)/R02^3)"),
        Expr.parse("-k*(m0*(x1-x0)/R01^3 + m2*(x1-x2)/R12^3)"),
        Expr.parse("-k*(m0*(y1-y0)/R01^3 + m2*(y1-y2)/R12^3)"),
        Expr.parse("-k*(m0*(x2-x0)/R02^3 + m1*(x2-x1)/R12^3)"),
        Expr.parse("-k*(m0*(y2-y0)/R02^3 + m1*(y2-y1)/R12^3)"),
    ]
    forces = [expr.where_substitute(aliases) for expr in acceleration]
    rhs = [
        v["vx0"], forces[0], v["vy0"], forces[1],
        v["vx1"], forces[2], v["vy1"], forces[3],
        v["vx2"], forces[4], v["vy2"], forces[5],
    ]
    initial = np.asarray([
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 5.0,
        0.0, -0.4, 33.0, 0.0,
    ])
    solution = solve_ivp(
        "lsode2", rhs, names, "t", initial, 0.0, 0.02, 0.001,
        controller="bdf",
        options=SolverOptions(
            frontend="atom_native", execution="lambdify",
            matrix_layout="dense", telemetry="timings",
        ),
        rtol=1e-8,
        atol=1e-10,
    )
    times = solution.times_array()
    states = solution.states_array()
    assert np.isfinite(times).all() and np.isfinite(states).all()
    x_column, y_column = 4 * args.body, 4 * args.body + 2
    print("samples:", len(times))
    print(f"body {args.body} final XY:", states[-1, [x_column, y_column]])
    if args.animate:
        print("Opening 2D animation; close the window or press Esc to finish.")
        animate_solution_2d(
            solution,
            (x_column, y_column),
            (f"body {args.body} x", f"body {args.body} y"),
            args.speed,
        )


if __name__ == "__main__":
    main()

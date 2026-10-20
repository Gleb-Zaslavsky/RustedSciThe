"""Solve a small structured system with the faithful banded API.

The example deliberately uses compact NumPy storage instead of constructing a
dense coefficient matrix.  It demonstrates the lifecycle used by larger
applications: build the banded operator once, factor once, then solve several
right-hand sides and optionally request an explicit residual report.
"""

from __future__ import annotations

import numpy as np

from rustedscithe import BandedMatrix, LapackStyleBandedLuFaithful


def main() -> None:
    # A tridiagonal operator with one lower and one upper diagonal.  The
    # compact row index is ku + i - j, not a dense row index.
    compact = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0, 4.0],
            [2.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = BandedMatrix(n=4, kl=1, ku=1, values=compact)

    factor = LapackStyleBandedLuFaithful(n=4, kl=1, ku=1)
    factor.factor(matrix)

    rhs = np.asarray([6.0, 13.0, 20.0, 22.0], dtype=np.float64)
    solution = factor.solve(rhs)
    print("single RHS:", solution)
    np.testing.assert_allclose(solution, [1.0, 2.0, 3.0, 4.0])

    # Multiple right-hand sides reuse the same factorization.  The returned
    # NumPy array is owned by Python and has shape (n, number_of_rhs).
    multiple_rhs = np.asarray(
        [[6.0, 9.0], [13.0, 8.0], [20.0, 2.0], [22.0, 0.0]],
        dtype=np.float64,
    )
    multiple_solution = factor.solve_multiple(multiple_rhs)
    print("multiple RHS:\n", multiple_solution)
    np.testing.assert_allclose(
        multiple_solution,
        [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]],
    )

    report = factor.solve_with_refinement(matrix, rhs, max_iters=2, tolerance=1e-12)
    print(
        "refinement:",
        {
            "iterations": report.iterations,
            "residual_inf": report.residual_inf,
            "relative_residual_inf": report.relative_residual_inf,
            "converged": report.converged,
        },
    )


if __name__ == "__main__":
    main()

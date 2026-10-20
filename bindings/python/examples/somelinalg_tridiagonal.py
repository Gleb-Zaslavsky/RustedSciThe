"""Solve a pivot-required tridiagonal system without dense matrix storage.

The tiny leading diagonal is intentional: it demonstrates why this route
uses partial pivoting instead of a no-pivot Thomas-style elimination. The
example also shows factor reuse, detached NumPy output and a padded workspace
for several right-hand sides.
"""

from __future__ import annotations

import numpy as np

from rustedscithe import PivotedTridiagonalLu, capabilities


def main() -> None:
    lower = np.asarray([2.0, 1.0, 1.0], dtype=np.float64)
    diagonal = np.asarray([1.0e-12, 2.0, 2.0, 2.0], dtype=np.float64)
    upper = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    expected = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)

    # Build the RHS directly from the three bands. No dense coefficient matrix
    # is needed even to construct this known-solution correctness check.
    rhs = np.asarray(
        [
            diagonal[0] * expected[0] + upper[0] * expected[1],
            lower[0] * expected[0]
            + diagonal[1] * expected[1]
            + upper[1] * expected[2],
            lower[1] * expected[1]
            + diagonal[2] * expected[2]
            + upper[2] * expected[3],
            lower[2] * expected[2] + diagonal[3] * expected[3],
        ],
        dtype=np.float64,
    )

    lu = PivotedTridiagonalLu(lower, diagonal, upper)
    solution = lu.solve(rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)

    # The same factorization can serve multiple RHS values. The padded form is
    # useful when a larger native workspace already owns fixed-size blocks.
    rhs_matrix = np.column_stack([rhs, 2.0 * rhs])
    solutions = lu.solve_multiple(rhs_matrix)
    np.testing.assert_allclose(solutions[:, 0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(solutions[:, 1], 2.0 * expected, rtol=1e-12, atol=1e-12)

    stride = 6
    padded = np.full(2 * stride, -1.0, dtype=np.float64)
    padded[:4] = rhs
    padded[stride : stride + 4] = 2.0 * rhs
    lu.solve_multiple_in_place(padded, nrhs=2, stride=stride)
    np.testing.assert_allclose(padded[:4], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        padded[stride : stride + 4], 2.0 * expected, rtol=1e-12, atol=1e-12
    )
    assert np.all(padded[[4, 5, 10, 11]] == -1.0)

    print("capabilities:", capabilities("pivoted_tridiagonal").solver)
    print("n:", lu.n, "pivot indices:", lu.ipiv)
    print("storage_bytes:", lu.storage_bytes)
    print("solution:", solution)


if __name__ == "__main__":
    main()

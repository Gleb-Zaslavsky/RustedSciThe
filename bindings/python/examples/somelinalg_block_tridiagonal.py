"""Solve a compact block-tridiagonal system without global densification.

The two dense blocks per axis are small by design: this is a readable contract
example, not a performance benchmark. Real applications should choose this
route only when their global system has the same block-tridiagonal structure.
"""

from __future__ import annotations

import numpy as np

from rustedscithe import BlockTridiagonalLuConsistent, capabilities


def main() -> None:
    lower = np.asarray([[[0.25, 0.0], [0.0, 0.25]]], dtype=np.float64)
    diagonal = np.asarray(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[5.0, 1.0], [1.0, 4.0]],
        ],
        dtype=np.float64,
    )
    upper = np.asarray([[[0.5, 0.0], [0.0, 0.5]]], dtype=np.float64)
    expected = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)

    # Build the known RHS directly from the three block bands. This keeps the
    # example faithful to the production contract: no global dense matrix is
    # needed even for the correctness fixture.
    expected_blocks = expected.reshape(2, 2)
    rhs_blocks = np.empty_like(expected_blocks)
    rhs_blocks[0] = diagonal[0] @ expected_blocks[0] + upper[0] @ expected_blocks[1]
    rhs_blocks[1] = lower[0] @ expected_blocks[0] + diagonal[1] @ expected_blocks[1]
    rhs = rhs_blocks.reshape(-1)

    lu = BlockTridiagonalLuConsistent(lower, diagonal, upper)
    solution = lu.solve(rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)

    rhs_matrix = np.column_stack([rhs, 2.0 * rhs])
    solutions = lu.solve_multiple(rhs_matrix)
    np.testing.assert_allclose(solutions[:, 0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(solutions[:, 1], 2.0 * expected, rtol=1e-12, atol=1e-12)

    report = lu.solve_with_refinement(rhs, refinement_steps=2)
    np.testing.assert_allclose(report.solution_array(), expected, rtol=1e-12, atol=1e-12)

    print("capabilities:", capabilities("block_tridiagonal").solver)
    print("shape:", lu.n_blocks, "blocks x", lu.block_size, "x", lu.block_size)
    print("storage_bytes:", lu.storage_bytes)
    print("factor_residual_relative:", lu.factor_residual_relative)
    print("pivot_diagnostics:", lu.pivot_diagnostics())
    print("refinement accepted:", report.accepted_steps)
    print("solution:", solution)


if __name__ == "__main__":
    main()

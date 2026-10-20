# Block-tridiagonal LU

`rustedscithe` exposes the validated `somelinalg` block-tridiagonal route as
`BlockTridiagonalLuConsistent`. It is intended for systems whose global matrix
has dense square blocks on the main, lower and upper block diagonals. The
factorization is structured: the Python API accepts three compact arrays and
does not create an `n x n` coefficient matrix.

## Compact NumPy representation

For `n_blocks` blocks of size `block_size`, pass:

- `diagonal`: shape `(n_blocks, block_size, block_size)`;
- `lower`: shape `(n_blocks - 1, block_size, block_size)`;
- `upper`: shape `(n_blocks - 1, block_size, block_size)`.

Each array is copied at construction and must contain finite `float64`
values. The native factorization is completed by the constructor, so the
object is immediately reusable:

```python
import numpy as np
from rustedscithe import BlockTridiagonalLuConsistent

lower = np.asarray([[[0.25, 0.0], [0.0, 0.25]]], dtype=np.float64)
diagonal = np.asarray(
    [
        [[4.0, 1.0], [2.0, 3.0]],
        [[5.0, 1.0], [1.0, 4.0]],
    ],
    dtype=np.float64,
)
upper = np.asarray([[[0.5, 0.0], [0.0, 0.5]]], dtype=np.float64)
lu = BlockTridiagonalLuConsistent(lower, diagonal, upper)
```

Mutating `lower`, `diagonal` or `upper` after construction cannot mutate the
native factorization. `storage_bytes`, `n_blocks`, `block_size`, `n` and
`is_factorized` make this ownership and lifecycle visible without exposing
internal vectors.

## Solves and diagnostics

`solve(rhs)` accepts and returns shape `(n,)`. `solve_multiple(rhs)` uses the
natural NumPy shape `(n, nrhs)` and performs only the RHS layout conversion;
the coefficient matrix remains structured. `solve_in_place` and
`solve_multiple_in_place` are available for callers that already own
contiguous workspaces. The padded multiple-RHS form uses one `stride` block per
RHS and leaves padding untouched.

The factorization exposes `factor_residual_relative`,
`condition_diagnostics()` and `pivot_diagnostics()`. These are diagnostics, not
a promise that every ill-conditioned matrix is rejected: callers should use
the reported values to decide whether a workload is suitable for this route.
`solve_with_refinement(rhs, refinement_steps)` applies the native guarded
iterative-refinement policy and returns a detached
`BlockTridiagonalRefinementResult`.

## Errors and scope

Wrong block shapes, dimension overflow, non-finite matrix/RHS values and
invalid RHS layouts raise `InvalidConfigurationError`. Singular pivots,
non-finite factors or solutions raise `SolverError`. Exceptions preserve
`stage`, `solver`, `route` and explicit `native_kind`; application code should
branch on these fields rather than parse error text.

This standalone kernel has no native logging or telemetry contract, so its
capability record reports `telemetry_modes == ["off"]`. Enclosing BVP/IVP
solvers remain responsible for solve-level counters and timings. The older
block LU implementation and the bordered block solver are intentionally not
part of this Python surface. Use `capabilities("block_tridiagonal")` to inspect
the supported contract.

The maintained example is
[`somelinalg_block_tridiagonal.py`](../examples/somelinalg_block_tridiagonal.py).

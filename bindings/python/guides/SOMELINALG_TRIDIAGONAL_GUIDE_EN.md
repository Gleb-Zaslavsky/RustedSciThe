# Pivoted tridiagonal LU

`rustedscithe` exposes the stable pivoted tridiagonal LU kernel from
`somelinalg` as `PivotedTridiagonalLu`. It is a small numerical linear
algebra API, not a symbolic solver. The wrapper keeps the matrix in three
one-dimensional `float64` bands and never builds an `n x n` NumPy matrix.

## Compact storage and pivoting

For an `n x n` tridiagonal matrix, pass `lower` and `upper` with shape
`(n - 1,)`, and `diagonal` with shape `(n,)`. The native factorization owns
copies of these arrays, so changing an input array after construction cannot
change a factorization that is already in use.

```python
import numpy as np
from rustedscithe import PivotedTridiagonalLu

lower = np.asarray([2.0, 1.0, 1.0], dtype=np.float64)
diagonal = np.asarray([1.0e-12, 2.0, 2.0, 2.0], dtype=np.float64)
upper = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
lu = PivotedTridiagonalLu(lower, diagonal, upper)
```

This example deliberately has a tiny first diagonal entry. Partial pivoting
uses the neighbouring lower entry instead of assuming that the diagonal is a
safe pivot. The `ipiv` property records the native pivot decisions. The
optional `pivot_epsilon` is an absolute safety threshold for accepting a
pivot; it must be finite and non-negative.

## One and multiple right-hand sides

`solve()` returns an owned vector with shape `(n,)`. `solve_in_place()` avoids
that result allocation when a contiguous writable `float64` array is already
available. Multiple right-hand sides use the natural NumPy shape `(n, nrhs)`:

```python
rhs = np.asarray(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    dtype=np.float64,
)
solutions = lu.solve_multiple(rhs)
```

The padded in-place form is for callers that own a larger column-major-like
workspace. Only the first `n` values of each `stride` block are modified;
padding is preserved. The normal `solve_multiple()` route returns a regular
`(n, nrhs)` array and performs the layout conversion at the Python boundary,
while the native factorization itself remains structured.

## Errors and scope

Invalid band lengths, non-finite input, invalid tolerances, short RHS buffers
and invalid padded layouts raise `InvalidConfigurationError`. A zero pivot,
non-finite factor or non-finite solution raises `SolverError`. Both classes
retain `stage`, `solver`, `route` and explicit `native_kind` context; callers
should branch on these fields rather than parse error text.

The kernel has no independent logging or telemetry contract. Its capability
record is therefore `telemetry_modes == ["off"]`; enclosing IVP/BVP solvers
remain responsible for native stage timing and counters. The public route is
`float64` only, sequential, reusable after construction and intentionally
does not expose the unfinished non-pivoted tridiagonal implementation.

Use `capabilities("pivoted_tridiagonal")` to inspect this contract. The
maintained executable example is
[`somelinalg_tridiagonal.py`](../examples/somelinalg_tridiagonal.py).

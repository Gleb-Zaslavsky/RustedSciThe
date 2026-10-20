# Faithful Banded Linear Algebra

`rustedscithe` exposes the stable LAPACK-style banded LU path from
`somelinalg` as `BandedMatrix` and `LapackStyleBandedLuFaithful`. This API is
for applications that already know the bandwidth of their matrix. It keeps
the coefficient matrix in compact band storage and does not build a dense
`n x n` NumPy array during ordinary factorization or solve.

## Compact storage

For a matrix with `n` rows, `kl` lower diagonals and `ku` upper diagonals,
the input shape is `(kl + ku + 1, n)`. The native-compatible indexing rule is

```text
ab[ku + i - j, j] = A[i, j]
```

Entries outside the band are not part of the matrix. For a tridiagonal
matrix (`kl=ku=1`), the rows are upper diagonal, main diagonal and lower
diagonal. The compact array is copied at construction, so later mutation of
the caller's NumPy array cannot change the `BandedMatrix`.

```python
import numpy as np
from rustedscithe import BandedMatrix

compact = np.asarray(
    [
        [0.0, 1.0, 1.0, 1.0],  # upper diagonal; first slot is edge padding
        [4.0, 4.0, 4.0, 4.0],  # main diagonal
        [2.0, 2.0, 2.0, 0.0],  # lower diagonal
    ],
    dtype=np.float64,
)
matrix = BandedMatrix(4, kl=1, ku=1, values=compact)
```

The separate-band constructor is useful when an application naturally stores
the diagonals independently. `diagonal` has shape `(n,)`; `lower` and
`upper`, when present, have shapes `(kl, n)` and `(ku, n)`. Their rows are
ordered from the closest diagonal outwards. The edge entries that fall
outside the matrix are ignored, matching compact LAPACK storage.

```python
matrix = BandedMatrix.from_bands(
    diagonal=np.full(4, 4.0),
    lower=np.asarray([[2.0, 2.0, 2.0, 0.0]]),
    upper=np.asarray([[0.0, 1.0, 1.0, 1.0]]),
)
```

All arrays are `float64`, two-dimensional where documented, finite and shape
checked before entering the native solver. Invalid dimensions, non-finite
values and impossible bandwidths raise `InvalidConfigurationError` rather
than being silently reshaped or densified.

## Factor once, solve many times

`LapackStyleBandedLuFaithful` owns the factorization and can be reused for
multiple right-hand sides. The factorization is independent of the source
matrix after `factor()` returns. `solve()` accepts shape `(n,)` and
`solve_multiple()` accepts a NumPy matrix of shape `(n, nrhs)` and returns the
same shape. `solve_in_place()` is available for a contiguous writable vector
when avoiding the result allocation matters.

```python
from rustedscithe import LapackStyleBandedLuFaithful

lu = LapackStyleBandedLuFaithful(n=4, kl=1, ku=1)
lu.factor(matrix)

x = lu.solve(np.asarray([6.0, 13.0, 20.0, 22.0]))
# x == [1.0, 2.0, 3.0, 4.0]

rhs = np.asarray(
    [[6.0, 9.0], [13.0, 8.0], [20.0, 2.0], [22.0, 0.0]],
    dtype=np.float64,
)
solutions = lu.solve_multiple(rhs)
```

The object exposes `is_factorized`, `ipiv`, `n`, `kl`, `ku`, `kv` and `ldab`
for diagnostics. `factor_residual_relative(matrix)` checks the factorization
against the original banded operator without changing the normal solve path.

## Iterative refinement

Refinement is explicit, not an invisible policy change. Use
`solve_with_refinement()` when the application wants a residual report:

```python
report = lu.solve_with_refinement(
    matrix,
    np.asarray([6.0, 13.0, 20.0, 22.0]),
    max_iters=2,
    tolerance=1e-12,
)
print(report.solution_array())
print(report.iterations, report.residual_inf, report.converged)
```

`BandedRefinementResult` owns a detached solution array and reports the
absolute and relative infinity-norm residuals. It does not expose mutable
native workspace memory.

## Errors and scope

The wrapper preserves the native distinction between configuration errors,
missing factorization, invalid RHS layout, singular/zero pivots and non-finite
results through the stable Python exception hierarchy. The exception context
identifies the `lapack_banded` stage and operation.

This is a linear-algebra API, not a replacement for a BVP solver. It has no
solver logging or telemetry of its own; callers should use the surrounding
solver's native telemetry when this kernel is embedded in BVP or IVP work.
The API intentionally does not accept a dense fallback input and does not
claim that banded storage is faster for every matrix. Choose it when the
problem's structure and bandwidth justify the compact representation.

The executable example is
[`somelinalg_banded.py`](../examples/somelinalg_banded.py). Its timings are
illustrative only. Release performance evidence belongs to dedicated Rust
core and Python-boundary benchmark reports.

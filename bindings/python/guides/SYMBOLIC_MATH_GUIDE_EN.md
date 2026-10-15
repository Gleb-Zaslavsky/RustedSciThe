# Symbolic Mathematics in Python

`rustedscithe.Expr` is a Python handle to the crate's native symbolic
expression tree. Parsing, differentiation, substitution, simplification and
prepared callback evaluation run in Rust; this package does not use SymPy.

## Build expressions

```python
from rustedscithe import Expr, symbols

x, y = symbols(["x", "y"])
f = x**2 + 3 * x * y + y.sin()
print(f)
print(f.pretty_print())
print(f.variables())  # sorted free-variable order: ["x", "y"]
```

`Expr.variable(name)`, `Expr.symbols(names)`, `Expr.indexed(prefix, count)`
and the module-level `symbol`, `symbols`, and `indexed_symbols` helpers create
symbols. Numeric scalar operands are accepted by arithmetic operators; boolean
and non-finite operands are rejected. Expressions are immutable values, so
transformations return new handles.

## Derivatives and substitutions

```python
df_dx = f.diff("x")
d2f_dx2 = f.diff_n("x", 2)
gradient = f.gradient()  # follows f.variables() order

rate = Expr.variable("rate")
model = (rate * x + y).where_substitute(
    {"rate": Expr.constant(2.0)}
)
```

`substitute_expr` replaces one variable; `substitute_many` and
`where_substitute` recursively resolve definitions independent of mapping
order. Cycles and duplicate names are errors. Unmapped variables stay free.
`rename_variable` changes a symbol name in a new expression, and
`contains_variable` checks whether a free symbol occurs.

## Simplification and domains

`simplify()` invokes the native algebraic simplifier. It is a symbolic rewrite,
not a proof assistant or a domain-preserving transformation. For example,
`(x / x).simplify()` becomes `1`, while the original is undefined at `x = 0`.
Structural equality, algebraic equivalence under stated assumptions, and
numerical agreement at sampled points are different claims. Rewrites involving
logarithms, powers or quotients can have domain restrictions. Preserve the
original expression or enforce its domain separately when singularities matter.
Numerical probes are useful regression checks, not a proof of equivalence over
a domain.

`taylor_series(variable, center, order)` constructs a formal Taylor polynomial;
other free variables remain symbolic. It does not certify convergence or a
radius of convergence, and the binding limits the order to 32 to bound tree
growth.

## Integration

`integrate(variable)` requests an indefinite primitive from the native
elementary integrator. Its supported pattern set is intentionally limited;
unsupported forms raise `UnsupportedSymbolicOperationError`, with
`stage == "symbolic_integration"` and `operation == "integrate"`. The result
has no integration constant.

The current core rule maps `1/x` to `ln(x)`. This is not a globally valid real
primitive across negative values or the singularity at zero. The caller must
state and respect the applicable domain. Definite integration and numerical
quadrature are not currently exposed by this symbolic API; do not infer
convergence from endpoint substitution alone.

## Symbolic vectors and matrices

`ExprVector([...])` and `ExprMatrix([[...], ...])` hold immutable collections
of `Expr` values. Matrix rows must be rectangular. Both expose sorted
`variables()` and evaluate using that exact order:

```python
import numpy as np
from rustedscithe import Expr, ExprMatrix, ExprVector

x, y = Expr.symbols(["x", "y"])
v = ExprVector([x + y, x * y])
A = ExprMatrix([[x, y], [x * y, x**2]])

v_values = v.evaluate([2.0, 3.0])  # owned NumPy shape (2,)
A_values = A.evaluate([2.0, 3.0])  # owned NumPy shape (2, 2), row-major
assert isinstance(A_values, np.ndarray)
At = A.transpose()
Av = A.matvec(v)
det_A = A.determinant()  # native determinant is limited to sizes 1..3
```

Vectors support indexing, differentiation, substitution, simplification and
dot products. Matrices support shape/index access, rows/columns, transpose,
differentiation, substitution, simplification, matrix-vector products, trace
and determinants up to 3x3. Invalid indexes, ragged rows, incompatible shapes
and unsupported determinant sizes are rejected before reaching native
assertions. Matrix-matrix products and advanced block operations are not part
of this Python contract yet.

## Prepared scalar callables

`Expr.lambdify()` compiles once and reuses the returned callable. The default
frontend is `SymbolicFrontend.ExprLegacy`; choose `AtomNative` for a one-time
conversion to the packed Atom representation. Both routes then evaluate in
Rust. For numerical arrays, `evaluate_batch()` expects shape
`(samples, variables)` and returns an owned `(samples,)` NumPy array.

Diagnostics are opt-in: `lambdify(diagnostics=True)` records preparation time,
callback count and callback time. Disabled preparation timing is `None`, not
zero. Callback timing excludes Python/NumPy conversion; it is useful for
attribution, not a substitute for a controlled benchmark.

```python
compiled = f.lambdify(
    ["x", "y"], frontend=SymbolicFrontend.AtomNative, diagnostics=True
)
values = compiled.evaluate_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
print(compiled.diagnostics())
```

See [`symbolic_calculus.py`](../examples/symbolic_calculus.py) and
[`symbolic_arrays.py`](../examples/symbolic_arrays.py) for runnable examples.

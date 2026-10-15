"""Native symbolic calculus, substitutions and reusable NumPy callbacks."""

import numpy as np

from rustedscithe import (
    Expr,
    SymbolicFrontend,
    UnsupportedSymbolicOperationError,
    symbols,
)


def main() -> None:
    # A two-variable model with exact native derivatives.
    x, y = symbols(["x", "y"])
    model = x**2 + 3 * x * y + y.sin()
    # `str()` stays friendly to legacy Windows console encodings; notebooks
    # and UTF-8 terminals can display the Unicode pretty printer directly.
    print("model:", model)
    print("variables:", model.variables())
    print("gradient:", [str(item) for item in model.gradient()])

    # `where_substitute` resolves the rate symbol before numerical preparation.
    rate = Expr.variable("rate")
    residual = (rate * x + y).where_substitute({"rate": Expr.constant(2.0)})
    primitive = (x**3 + 2 * x).integrate("x")
    print("substituted residual:", residual)
    print("primitive:", primitive)

    # AtomNative converts once during preparation; the batch loop stays in Rust.
    callback = model.lambdify(
        ["x", "y"], frontend=SymbolicFrontend.AtomNative, diagnostics=True
    )
    points = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    print("batch values:", callback.evaluate_batch(points))
    print("callback diagnostics:", callback.diagnostics())

    # Unsupported forms use a typed error instead of a misleading primitive.
    try:
        Expr.parse("exp(x^2)").integrate("x")
    except UnsupportedSymbolicOperationError as error:
        print("unsupported integral:", error, error.category, error.stage)


if __name__ == "__main__":
    main()

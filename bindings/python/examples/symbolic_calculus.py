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
    quadratic_integral = (x**2).integrate_simpson(0.0, 1.0, 100)
    quadratic_gauss = (x**2).integrate_gauss_legendre(0.0, 1.0, 4)
    hermite_constant = Expr.constant(1.0).integrate_gauss_hermite(8)
    laguerre_cubic = (x**3).integrate_gauss_laguerre(4, alpha=0.0)
    print("substituted residual:", residual)
    print("primitive:", primitive)
    print("Simpson integral of x^2 on [0, 1]:", quadratic_integral)
    print("Gauss-Legendre integral of x^2 on [0, 1]:", quadratic_gauss)
    print("Gauss-Hermite integral of exp(-x^2):", hermite_constant)
    print("Gauss-Laguerre weighted integral of x^3:", laguerre_cubic)

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

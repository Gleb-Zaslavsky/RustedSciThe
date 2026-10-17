"""Explore symbolic calculus and prepare a reusable numerical callback.

The model ``x**2 + 3*x*y + sin(y)`` demonstrates variable discovery and exact
symbolic gradients. The rest contrasts recursive symbolic substitution and a
symbolic antiderivative with numerical quadrature, then lowers the same model
through AtomNative for batched NumPy evaluation. Unsupported antiderivatives
are reported with a typed exception instead of silently approximated.
"""

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
    # These moments have known closed forms, keeping the quadrature examples
    # self-checking without relying on visual inspection of printed values.
    if not np.allclose([quadratic_integral, quadratic_gauss], [1.0 / 3.0] * 2, atol=1e-12):
        raise AssertionError("quadrature of x^2 on [0, 1] is incorrect")
    if abs(hermite_constant - np.sqrt(np.pi)) > 1e-12 or abs(laguerre_cubic - 6.0) > 1e-12:
        raise AssertionError("Gaussian quadrature moment check failed")
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
    batch_values = callback.evaluate_batch(points)
    expected_values = points[:, 0] ** 2 + 3.0 * points[:, 0] * points[:, 1] + np.sin(points[:, 1])
    np.testing.assert_allclose(batch_values, expected_values, rtol=1e-13, atol=1e-13)
    print("batch values:", batch_values)
    print("callback diagnostics:", callback.diagnostics())

    # Unsupported forms use a typed error instead of a misleading primitive.
    try:
        Expr.parse("exp(x^2)").integrate("x")
    except UnsupportedSymbolicOperationError as error:
        if error.category != "unsupported_symbolic_operation":
            raise AssertionError("unsupported integral lost its typed error category")
        print("unsupported integral:", error, error.category, error.stage)
    else:
        raise AssertionError("unsupported antiderivative was accepted unexpectedly")


if __name__ == "__main__":
    main()

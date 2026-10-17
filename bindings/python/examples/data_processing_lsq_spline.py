"""Fit one noisy signal with both native least-squares B-spline solvers.

The knot vector includes the repeated boundary knots required for a clamped
cubic spline. ``dense`` uses weighted QR; ``banded`` uses the native banded
normal-equation/Cholesky path. The API keeps this selection explicit and does
not silently retry with another backend.
"""

import numpy as np

from rustedscithe import fit_quality, least_squares_bspline


def main() -> None:
    rng = np.random.default_rng(20261016)
    x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x) + 0.035 * rng.standard_normal(x.size)
    weights = np.linspace(0.75, 1.25, x.size, dtype=np.float64)

    # Four repeated endpoint knots plus internal knots define the cubic basis.
    knots = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    sample = np.linspace(0.0, 1.0, 401, dtype=np.float64)

    for route in ("dense", "banded"):
        spline = least_squares_bspline(
            x,
            y,
            knots,
            degree=3,
            weights=weights,
            solver=route,
        )
        fitted_observations = spline.evaluate_batch(x)
        rmse, mae, r_squared = fit_quality(y, fitted_observations)
        print(
            f"{route:>6}: degree={spline.degree}, "
            f"coefficients={spline.coefficients.size}, "
            f"RMSE={rmse:.6g}, MAE={mae:.6g}, R2={r_squared}"
        )
        print(f"       sample spline values: {spline.evaluate_batch(sample)[:5]}")


if __name__ == "__main__":
    main()

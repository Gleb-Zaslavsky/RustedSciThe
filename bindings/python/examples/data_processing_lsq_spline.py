"""Fit one noisy signal with both native least-squares B-spline solvers.

The knot vector includes the repeated boundary knots required for a clamped
cubic spline. ``dense`` uses weighted QR; ``banded`` uses the native banded
normal-equation/Cholesky path. The API keeps this selection explicit and does
not silently retry with another backend.
"""

import numpy as np

from rustedscithe import fit_quality, least_squares_bspline, render_table


def main() -> None:
    rng = np.random.default_rng(20261016)
    x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x) + 0.035 * rng.standard_normal(x.size)
    # Nonuniform weights model different observation confidence while the
    # fixed seed keeps the comparison reproducible across runs.
    weights = np.linspace(0.75, 1.25, x.size, dtype=np.float64)

    # Four repeated endpoint knots plus internal knots define the cubic basis.
    knots = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    sample = np.linspace(0.0, 1.0, 401, dtype=np.float64)

    rows = []
    reference_values = None
    # These are two solver routes for the same weighted fit, not two different
    # spline models. The explicit choice prevents hidden backend fallback.
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
        sample_values = spline.evaluate_batch(sample)
        if reference_values is None:
            reference_values = sample_values.copy()
        else:
            # Compare on a shared dense grid, not only at the fitted samples;
            # agreement at the data points could hide interpolation drift.
            route_difference = float(np.max(np.abs(sample_values - reference_values)))
            if route_difference > 1e-8:
                raise AssertionError(
                    f"Dense and Banded spline profiles differ by {route_difference:.3e}"
                )
        rows.append(
            {
                "solver": route,
                "degree": spline.degree,
                "coefficients": spline.coefficients.size,
                "RMSE": f"{rmse:.6g}",
                "MAE": f"{mae:.6g}",
                "R2": f"{r_squared:.6g}" if r_squared is not None else "n/a",
            }
        )
    print("Weighted least-squares B-spline backend comparison (not a benchmark)")
    print(render_table(rows))
    print("sample values on 401-point grid:", sample_values[:5])


if __name__ == "__main__":
    main()

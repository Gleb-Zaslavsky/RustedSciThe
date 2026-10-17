"""Reuse a native scalar B-spline interpolator over a sampled domain.

This is interpolation through control points, not least-squares fitting of
observations. See ``data_processing_lsq_spline.py`` for the latter and its
explicit DenseQR/Banded solver comparison.
"""

import numpy as np

from rustedscithe import BSplineInterpolator


def main() -> None:
    # The uniform-knot factory creates a clamped knot vector in Rust and owns
    # a copy of these control values; later mutation cannot alter the spline.
    control = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float64)
    curve = BSplineInterpolator.with_uniform_knots(3, control)

    # Evaluation is restricted to the valid knot domain and returns an owned
    # NumPy array suitable for plotting or downstream analysis.
    parameters = np.linspace(*curve.knot_domain, 200, dtype=np.float64)
    values = curve.interpolate_batch(parameters)
    endpoint_values = curve.interpolate_batch(np.asarray(curve.knot_domain))
    # A clamped spline passes through its first and last control values at the
    # domain endpoints; batch evaluation must also preserve the requested size.
    if (
        values.shape != parameters.shape
        or not np.isfinite(values).all()
        or not np.allclose(endpoint_values, [control[0], control[-1]], atol=1e-12)
    ):
        raise AssertionError("B-spline interpolation shape or endpoint check failed")
    print(f"degree={curve.degree}, domain={curve.knot_domain}")
    print(f"sample count={values.size}, first values={values[:5]}")


if __name__ == "__main__":
    main()

"""Smooth and resample a noisy signal with native Rust utilities.

The example composes the Rust Savitzky-Golay filter, CPU LOWESS and spline
resampler. Arrays are NumPy float64 values at the Python boundary; each native
operation returns an owned array. These routines are for data preparation and
post-processing, not a substitute for the solver's dense-output interpolant.
"""

import numpy as np

from rustedscithe import (
    fit_quality,
    lowess_fit,
    lowess_smooth,
    render_table,
    savgol_filter,
    spline_resample,
)


x = np.linspace(0.0, 4.0 * np.pi, 81, dtype=np.float64)
clean = np.sin(x)
# Add a faster oscillation as deterministic measurement noise, so smoothing
# quality can be compared against a known signal rather than judged by eye.
noisy = clean + 0.08 * np.sin(17.0 * x)
raw_rmse, raw_mae, raw_r2 = fit_quality(clean, noisy)

# Savitzky-Golay uses a local cubic fit over an odd 9-sample window. LOWESS
# uses a narrow 5% neighborhood to follow the slowly varying signal without
# averaging across its broad curvature; it stays serial in this small example.
sg = savgol_filter(noisy, window_length=9, polyorder=3)
lowess = lowess_smooth(x, noisy, fraction=0.05, iterations=2, delta=0.0, parallel=False)
lowess_result = lowess_fit(x, noisy, fraction=0.05, iterations=2, delta=0.0, parallel=False)
# Resample the smoothed curve on a denser cosine-spaced grid for plotting; this
# is post-processing, not the IVP solver's dense-output interpolation.
x_dense, y_dense = spline_resample(
    lowess_result.x, lowess_result.y, n_points=241, kind="cosine"
)

sg_rmse, sg_mae, sg_r2 = fit_quality(clean, sg)
lowess_rmse, lowess_mae, lowess_r2 = fit_quality(clean, lowess)
if sg_rmse >= raw_rmse or lowess_rmse >= raw_rmse:
    raise AssertionError("a configured smoother did not improve on the raw signal")
assert sg_r2 is not None
assert lowess_r2 is not None
assert raw_r2 is not None
assert sg.shape == clean.shape and lowess.shape == clean.shape
assert x_dense.shape == y_dense.shape and np.isfinite(y_dense).all()
print(
    "Smoothing quality on the same synthetic signal (diagnostic, not a benchmark)"
)
print(
    render_table(
        [
            {
                "method": "raw",
                "RMSE": f"{raw_rmse:.5f}",
                "MAE": f"{raw_mae:.5f}",
                "R2": f"{raw_r2:.5f}",
            },
            {
                "method": "Savitzky-Golay",
                "RMSE": f"{sg_rmse:.5f}",
                "MAE": f"{sg_mae:.5f}",
                "R2": f"{sg_r2:.5f}",
            },
            {
                "method": "LOWESS",
                "RMSE": f"{lowess_rmse:.5f}",
                "MAE": f"{lowess_mae:.5f}",
                "R2": f"{lowess_r2:.5f}",
            },
        ]
    )
)
print(
    "LOWESS returned arrays:", lowess_result.x.shape, lowess_result.y.shape,
    "iterations:", lowess_result.iterations_used,
)
print("resampled grid shape:", x_dense.shape, "value shape:", y_dense.shape)

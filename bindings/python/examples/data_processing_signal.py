"""Smooth and resample a noisy signal with native Rust utilities.

The example composes the Rust Savitzky-Golay filter, CPU LOWESS and spline
resampler. Arrays are NumPy float64 values at the Python boundary; each native
operation returns an owned array. These routines are for data preparation and
post-processing, not a substitute for the solver's dense-output interpolant.
"""

import numpy as np

from rustedscithe import fit_quality, lowess_smooth, savgol_filter, spline_resample


x = np.linspace(0.0, 4.0 * np.pi, 81, dtype=np.float64)
clean = np.sin(x)
noisy = clean + 0.08 * np.sin(17.0 * x)

sg = savgol_filter(noisy, window_length=9, polyorder=3)
lowess = lowess_smooth(x, noisy, fraction=0.18, iterations=2, delta=0.0, parallel=False)
x_dense, y_dense = spline_resample(x, lowess, n_points=241, kind="cosine")

sg_rmse, sg_mae, sg_r2 = fit_quality(clean, sg)
lowess_rmse, lowess_mae, lowess_r2 = fit_quality(clean, lowess)
assert sg_r2 is not None
assert lowess_r2 is not None
print(f"Savitzky-Golay RMSE: {sg_rmse:.5f}")
print(f"LOWESS RMSE/MAE/R2: {lowess_rmse:.5f} / {lowess_mae:.5f} / {lowess_r2:.5f}")
print("resampled grid shape:", x_dense.shape, "value shape:", y_dense.shape)

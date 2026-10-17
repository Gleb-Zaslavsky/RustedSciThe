# Data Processing Guide

The Python data-processing API is a thin interface to the native Rust utilities
in `numerical::data_processing`. Inputs are copied from NumPy `float64` arrays
before entering Rust; returned arrays own their memory. Non-contiguous NumPy
views are supported. Invalid shapes, non-finite data, and unsupported options
raise typed RustedSciThe exceptions rather than invoking a Python reimplementation.

## Signal Smoothing and Resampling

```python
import numpy as np
from rustedscithe import fit_quality, lowess_fit, savgol_filter, spline_resample

x = np.linspace(0.0, 4.0 * np.pi, 81, dtype=np.float64)
signal = np.sin(x) + 0.05 * np.sin(17.0 * x)

filtered = savgol_filter(signal, window_length=9, polyorder=3)
fit = lowess_fit(x, signal, fraction=0.2, iterations=2, delta=0.0)
x_fine, y_fine = spline_resample(fit.x, fit.y, 241, kind="cosine")
rmse, mae, r_squared = fit_quality(signal, filtered)
```

`lowess_fit` returns a detached `LowessResult` with NumPy `x` and `y` arrays,
`fraction_used`, and the native `iterations_used` value when reported.
`lowess_smooth` is the smaller API when only smoothed ordinates are needed.
The currently exposed LOWESS route is CPU-only; the unstable/unavailable GPU
backend is rejected rather than silently replaced.

`savgol_filter` uses unit sample spacing for derivatives; it does not claim
SciPy's configurable `delta` behavior. `spline_resample` supports `linear`,
`cosine`, and `bezier` interpolation and returns a uniform grid strictly within
the input domain. Bezier `tension` is used only for that interpolation kind.
`fit_quality` returns `(RMSE, MAE, R-squared)`; R-squared is `None` for a
constant reference signal.

## B-Splines

`BSplineInterpolator` is a reusable scalar interpolator. Supply a full knot
vector or request the native clamped uniform-knot construction:

```python
from rustedscithe import BSplineInterpolator, least_squares_bspline

x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
signal = np.sin(2.0 * np.pi * x)
control = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
curve = BSplineInterpolator.with_uniform_knots(3, control)
samples = np.linspace(*curve.knot_domain, 100, dtype=np.float64)
values = curve.interpolate_batch(samples)

knots = np.array(
    [0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=np.float64
)
fit = least_squares_bspline(x, signal, knots, degree=3, solver="banded")
```

`BSplineInterpolator` validates knot count, ordering, multiplicity, and domain
before entering the legacy native constructor (which can panic on malformed
input). Evaluation is restricted to the knot domain and returns owned arrays.
`least_squares_bspline` fits observations and supports the explicit `dense`
(QR) and `banded` routes; it does not silently switch solvers. These two APIs
have different purposes: interpolation uses supplied control points, while
least-squares fitting estimates spline coefficients from observations.

For solver trajectory output, prefer the solver's own dense-output or sampling
API. These general-purpose utilities are for signal preparation and
post-processing, not substitutes for a solver's error-controlled interpolant.

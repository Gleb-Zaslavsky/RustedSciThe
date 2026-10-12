# Gavin Levenberg-Marquardt Python Reference

This small Python project preserves Gavin H. P. (2020)'s reference solver for
algorithm-parity checks against the Rust implementation. The solver source is
kept separate from the example driver; the numerical algorithm is not adapted
to Rust-specific behavior.

The reference model is

```text
y(t) = p0 * exp(-t / p1) + p2 * sin(t / p3)
```

Install the project into the repository's Windows virtual environment and run
the deterministic example from the repository root:

```powershell
.\.venv-gavin-win\Scripts\python.exe -m pip install -e .\src\numerical\optimization\gavin_reference
.\.venv-gavin-win\Scripts\python.exe .\src\numerical\optimization\gavin_reference\examples\exponential_sinusoidal_fit.py
```

NumPy is the only required runtime dependency. Install the optional plotting
dependencies only when using `make_lm_plots`:

```powershell
.\.venv-gavin-win\Scripts\python.exe -m pip install matplotlib seaborn
```

The upstream attribution and source URL are retained in
`levenberg_marquardt.py`. Plotting imports are local to the plotting function,
so numerical parity checks do not require Matplotlib or Seaborn.

The original finite-difference routine assigned `ps = p`, aliasing the
perturbed parameter array and therefore failing to restore its input. The
reference copy uses `p.copy()` so each finite-difference column is evaluated
around the same accepted point. It also uses a one-dimensional sample vector,
as required by the implementation despite the old docstring saying m-by-1,
and extracts a scalar convergence-history value with `.item()` for current
NumPy. The optional `update_type` argument exposes the source's three existing
controller branches (Marquardt, Quadratic, Nielsen); its default remains
Marquardt. A Quadratic trial residual is reshaped to `(m, 1)` before the
quadratic update, avoiding a NumPy rank error without changing the formula.
These compatibility/correctness fixes do not change the intended
finite-difference, damping, or accept/reject formulas.

The Python example and Rust tests share deterministic fit fixtures and full
controller histories for all three update policies, including selected
rejection/refresh paths. See `../GAVIN_STORY_TESTS.md` for the one known
periodic-refresh objective-cache defect in the Python source and the resulting
Rust correction. Callback counts intentionally differ because Rust reuses
trial model outputs when building the next linearization.

## Optional Reference Telemetry

Telemetry is opt-in: pass a `ReferenceTelemetry()` instance to `lm`. With no
telemetry argument the numerical reference follows its original uninstrumented
path. The example reports objective/FD/Jacobian and linear-solve counts plus
timings for objective evaluation, linearization, linear solves, covariance,
and total wall time. The stage scopes intentionally overlap: objective and FD
work are included in linearization, and final covariance work invokes the
linearization routine.

To save the compact Markdown table without redirecting compiler or console
noise, set an explicit report path:

```powershell
$env:GAVIN_REFERENCE_REPORT = "test_reports/Optimization_Gavin/python_reference.md"
.\.venv-gavin-win\Scripts\python.exe .\src\numerical\optimization\gavin_reference\examples\exponential_sinusoidal_fit.py
```

The Rust matched frontend benchmark is `cargo bench --no-default-features
--bench gavin_fitting`; it writes its own compact report under
`test_reports/Optimization_Gavin/<profile>/`.

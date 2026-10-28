# Stiff API

`StiffSolverRequest` is a Python facade over the Rust StiffAPI. It centralizes
common route selection and native solve dispatch without converting the four
solver families into a lowest-common-denominator solver. Each request is built
with the selected solver's own numerical controls, then prepared and solved by
Rust. Results retain their ordinary solver-specific Python type.

## Shared options and presets

Use `SolverOptions` for the shared symbolic frontend, Lambdify/AOT execution,
matrix layout, callback scheduling, telemetry and logging controls. Use a
`StiffPreset` when a reusable profile should override only selected shared
axes. Preset composition and applicability are validated by the Rust core;
unsupported combinations raise typed exceptions rather than being silently
changed.

Inspect the built-in profiles with `StiffPreset.catalog()` and the exact
overrides/applicability of a composed profile with `preset.describe()`. The
`fields_changed` list is the authoritative set of changed axes; unset axes keep
the selected native solver's defaults. Composition rejects two profiles that
set the same axis instead of implementing a hidden last-writer-wins rule.

```python
from rustedscithe import Expr, SolverOptions, StiffPreset, StiffSolverRequest

request = StiffSolverRequest.bdf(
    [Expr.parse("-rate*y")],
    ["y"],
    "t",
    [1.0],
    0.0,
    1.0,
    max_step=0.05,
    parameters=["rate"],
    parameter_values=[0.7],
    options=SolverOptions(telemetry="detailed"),
)
prepared = request.prepare(StiffPreset.solve_1d())
solution = prepared.solve()
print(solution.states()[-1])
print(solution.telemetry_report())
```

`prepare()` consumes a request exactly once. It returns a native prepared
facade; `solve()` performs native dispatch and returns the selected solver's
detached result. In a configuration-driven application, choose the
solver-specific request factory when constructing the calculation, then keep
the prepared object behind the application's solver boundary. Python does not
reimplement numerical stepping or callback scheduling.

After a solve, `PreparedStiffSolver.status` exposes the selected solver's
native status when available. `trajectory_arrays()` explicitly copies the
retained BDF, Backward Euler or LSODE2 trajectory into owned NumPy arrays with
shapes `(time_points,)` and `(time_points, state_variables)`. Radau keeps output
on `RadauSolution`; use its sampling methods instead. Calling
`trajectory_arrays()` before a trajectory exists raises a typed configuration
error.

`telemetry_snapshot()` returns a solver-tagged mapping: `solver` and `mode` are
at the top level, while the selected solver's native counters, timings, report
and AOT details remain nested under its solver key. Metric names are not merged
across solver families. Radau telemetry is obtained from the `RadauSolution`
returned by `solve()`.

`StiffPostprocessPlan` is a thin Python builder over the core Rust output plan.
It can write CSV/text, a compact report, Plotters/Gnuplot images or a terminal
plot. `prepared.postprocess(plan, axis_name, variable_names)` applies it to
retained BDF, Backward Euler or LSODE2 data. For Radau, call
`RadauSolution.postprocess(plan, sample_times, axis_name, variable_names)`;
the caller chooses explicit sample times because Radau owns dense/sampled
output on its detached solution. The return mapping reports each action's
`status`, path and optional message; optional Gnuplot absence is reported as
`skipped`, not treated as a solve failure.

`restart(t0, y0, t_end)` starts a new segment on the same prepared solver and
returns the ordinary solver-specific solution object. Symbolic preparation,
AOT artifacts and shared policy remain attached to the native instance.

Fresh parameterized solves use the common `solve_with_parameter_values(y0,
values)` or `solve_with_parameter_values_in_interval(t0, y0, t_end, values)`
methods. Both start a new initial-value solve and retain prepared symbolic/AOT
resources. The first keeps the configured interval; the second changes it.

Time-segment continuation remains solver-specific:

- `continue_bdf_with_parameters(values, t_end)` advances from BDF's last
  accepted point and resets its multistep numerical history.
- `continue_backward_euler(t_end, parameter_values=None)` extends the current
  trajectory and optionally rebinds parameters.
- `continue_lsode2_with_parameters(values, t_end)` starts at LSODE2's last
  successful endpoint, rebinds values and delegates the segment reset to its
  native restart path.

Each time-segment method returns the ordinary solver-specific result. Calling
one on the wrong prepared solver raises a typed unsupported-route error. Radau
fresh solves use the same facade-level `solve_with_parameter_values` methods as
the other solver families.

## Solver-specific controls

- `StiffSolverRequest.bdf(...)` keeps BDF order, first-step, maximum-step and
  work-budget controls. Its result is `BdfSolution`.
- `StiffSolverRequest.radau(...)` takes the initial state as part of the
  request, supports an optional analytic Jacobian, `output="dense"` for
  arbitrary interpolation, and `output="sampled", sample_times=[...]` to
  retain only requested output points. Sampled output can be queried only at
  those times. Its result is `RadauSolution`; Radau logging is currently
  unsupported and a non-off logging request is rejected.
- `StiffSolverRequest.backward_euler(...)` retains the fixed-step `h` and
  Newton controls. If `h` is omitted, the native solver's existing heuristic
  remains in effect. Its result is `BackwardEulerSolution`.
- `StiffSolverRequest.lsode2(...)` keeps `method="bdf"`, `"adams"` or
  `"auto"` independent from `native_execution="native"`, `"bridge"` or
  `"probe_before_bridge"`. Its result is `Lsode2Solution`. The factory accepts
  either symbolic equations or a pair of `residual_callback(time, state)` and
  `jacobian_callback(time, state)` callables. Python callbacks are Lambdify-only;
  AOT cannot embed Python callables, and inferred band width requires symbolic
  equations. For a banded callback route, provide both `band_lower` and
  `band_upper` in `SolverOptions`. Callback exceptions are surfaced as
  `CallbackError` with the original Python exception as `__cause__`.

All symbolic factories validate declared state, time and parameter names before
native preparation. Declared parameters require matching finite values where
the selected solver needs an initial binding. AOT toolchain, artifact policy
and output location remain explicit in `SolverOptions`; selecting AOT may
invoke an external compiler during native preparation.

## Telemetry and errors

`prepared.telemetry_mode` reports the effective native collection mode. The
detached result exposes that solver's counters, timing data, AOT provenance and
diagnostics using its established API. These fields are intentionally not
flattened into a synthetic common schema because native counters and timing
scopes differ by solver. Typed native failures are mapped through the existing
Python solver exception hierarchy with solver/route context preserved.

`restart()` changes the interval and initial state without rebinding parameters.
Use the fresh parameterized solve shortcuts when both parameter values and
initial conditions need to change.

Runnable application examples are `examples/stiff_api_solver_selection.py`
(four-family selection and output plan) and `examples/stiff_api_restart.py`
(prepared BDF restart). `examples/stiff_api_solver_lifecycle.py` demonstrates native
time-segment continuation for BDF/BE/LSODE2 and a fresh parameterized Radau
solve, while `examples/stiff_api_reaction_chain.py`
selects among the stiff solvers for an exact two-species reaction chain. They
print results for orientation; they are not performance benchmarks.

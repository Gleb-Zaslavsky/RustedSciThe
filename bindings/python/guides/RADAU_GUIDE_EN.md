# Radau5 Guide for Python

This guide mirrors the maintained Rust Radau guide while showing Python syntax.
The solver is the fixed-order, fifth-order Radau IIA implementation. The
numerical algorithm runs in Rust; Python supplies expressions/options and
receives detached results.

## When to choose Radau

Radau5 is an implicit stiff IVP method and is a reasonable first choice for
strongly stiff kinetics or when robust stiff integration matters more than a
minimal per-step cost. It is not a universal winner for smooth nonstiff
systems; compare it with LSODE2 Adams/Auto on the same model when that choice
matters.

## One-shot and prepared APIs

`solve_ivp("radau5", ...)` is convenient for one solve. Use `create_ivp` or
`RadauSolver` when you need explicit preparation, parameter rebinds, restart,
or solver-specific configuration. The direct wrapper binds `y0` at `solve`:

```python
from rustedscithe import Expr, RadauSolver, SolverOptions

y = Expr.variable("y")
k = Expr.variable("k")
solver = RadauSolver(
    [-k * y], ["y"], "t", 0.0, 1.0, 0.05,
    parameters=["k"], options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
    rtol=1e-7, atol=1e-9, output="dense",
)
solver.prepare()
first = solver.solve([1.0], [0.7])
second = solver.solve_with_parameter_values([0.8], [1.2])
```

Each call is a fresh solve over the configured interval. The caller supplies
the initial state and parameter values explicitly; no state is inherited from
the previous result. Use `solve_with_parameter_values_in_interval(t0, y0,
t_end, values)` to change the interval as well.

## Frontend, execution and layout

These are independent configuration axes. `expr_legacy` and `atom_native`
select symbolic preparation; `lambdify` and `aot` select callback execution;
`dense`, `sparse` and `banded` select the linear-system layout. Use
`capabilities("radau5")` to inspect the installed contract. There is no
universally fastest frontend or matrix layout.

For symbolic Banded routes, Radau infers the half-bandwidth from the residual
or supplied symbolic Jacobian. `SolverOptions.set_banded_width(lower, upper)`
is optional; when supplied, the widths are checked against the symbolic pattern
so an undersized band cannot silently drop derivatives.

Structured layouts need a symbolic Jacobian. When no Jacobian is supplied,
Radau's finite-difference fallback is Dense-only. The callback wrapper is a
separate Dense, sequential route; its Jacobian callback is optional and native
finite differences are used when omitted. Unsupported combinations raise a
typed `UnsupportedOptionError`, not an implicit conversion.

## Controls, output and diagnostics

`rtol`/`atol` control local error, `first_step` optionally chooses the initial
step, `max_step` caps it, and `max_steps`, `max_newton_iterations` and
`max_retries` bound work. `output="dense"` retains a continuous interpolant;
otherwise use the returned accepted trajectory. Result arrays are owned
NumPy `float64` arrays, not views into a reusable Rust workspace.

Choose `telemetry="counters"`, `"timings"` or `"detailed"` in `SolverOptions`.
`counters` collects discrete operation counts only, `timings` collects durations
only, and `detailed` collects both. Then inspect `telemetry_counters()`,
`telemetry_timings_ms()` and `telemetry_report()`. A
timing snapshot describes only the scopes the native solver measured; absent
and zero-valued scopes must not be treated as equivalent. Catch the typed
configuration, preparation, callback, AOT and exhaustion exceptions. Native
errors expose stage/solver/route metadata where available.

## Examples and evidence

Run `examples/radau_basic.py` for a minimal solve and
`examples/radau_robertson_routes.py` for matched frontend/layout routes on the
same stiff Robertson equations. `examples/radau_robertson_aot.py` is an
optional compiler-backed lifecycle example. The core crate's Radau guide and
story reports remain the source for SciPy numerical-fidelity evidence; Python
examples are not performance benchmarks.

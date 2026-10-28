# BDF Guide for Python

The Python BDF adapter follows the crate's deliberate dense-solver scope. It
does not expose Sparse or Banded storage: for those IVP layouts, use LSODE2.
The BDF numerical core is Rust; this document translates the native BDF guide
and examples into the Python API.

## Choose the route

`BdfSolver` is the full symbolic Python entry point. Select `expr_legacy` or
`atom_native` independently from `lambdify` or `aot` using `SolverOptions`.
The matrix layout is Dense. Native residual/Jacobian callbacks are available
through the separate `BdfCallbackSolver`, also Dense/sequential; if only a
residual is supplied, native finite differences provide the Jacobian.

```python
from rustedscithe import BdfSolver, Expr, SolverOptions

y = Expr.variable("y")
rate = Expr.variable("rate")
solver = BdfSolver(
    [-rate * y], ["y"], "t", [1.0], 0.0, 0.5, 0.05,
    parameters=["rate"], parameter_values=[0.7], rtol=1e-8, atol=1e-10,
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", telemetry="timings",
    ),
)
solver.prepare()
result = solver.solve()
print(result.final_state_array())
```

## Controls and repeated solves

`rtol` and `atol` set error tolerances; `first_step`, `max_step`,
`max_bdf_order` and `max_steps` control the integration. The native default
order ceiling is appropriate for ordinary use; lower it only for a justified
problem-specific reason. Always check the returned status/diagnostics rather
than inferring success from a finite endpoint alone.

`solve_with_parameter_values(y0, values)` starts a fresh solve with a new
initial state and parameter values on the configured interval;
`solve_with_parameter_values_in_interval(t0, y0, t_end, values)` also changes
the interval. Both reuse the prepared symbolic route. `continue_with_parameters`
is different: it advances a time segment from BDF's last accepted state.
`restart(t0, y0, t_end)` starts a fresh segment without rebinding parameters.
Use a new solver when the equation structure or dimensions change.

## AOT, telemetry and errors

For one cold solve, AOT compilation may cost more than it saves. The shared
native lifecycle supports BuildIfMissing, RequirePrebuilt and RebuildAlways;
configure its compiler, profile and artifact directory through `SolverOptions`.
Inspect provenance and AOT counters before claiming an artifact was reused.

Telemetry is detached from mutable solver workspaces. Choose `off`, `counters`,
`timings`, or `detailed` explicitly: counters and timings are separate
collections, and `detailed` enables both. Compare preparation, callback and
numerical solve scopes separately. The callback API does not make Python callbacks a parallel hot
path. Invalid options and numerical exhaustion produce typed errors.

## Examples

`examples/bdf_continuation.py` shows warm parameter continuation;
`examples/bdf_aot_build.py` demonstrates an optional compiler-backed lifecycle;
`examples/bdf_robertson.py` exercises a stiff chemical IVP. The native BDF
guide contains the algorithm/fidelity details and explains why structured
matrix support belongs to LSODE2 instead.

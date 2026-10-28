# LSODE2 Guide for Python

LSODE2 is the broad IVP workhorse when the problem may be nonstiff or stiff,
or when Sparse/Banded matrix structure is important. Its automatic controller
can switch between Adams and BDF families. It is distinct from the smaller
SciPy-mirroring BDF and Radau wrappers.

## Family versus controller

Choose the solver family as `"lsode2"`; choose its method using the separate
`controller="adams"`, `"bdf"` or `"auto"` option. `"lsoda"` is a family alias
for the automatic Adams/BDF controller. Do not pass `"adams"` as the solver
family.

```python
from rustedscithe import Expr, SolverOptions, create_ivp

y = Expr.variable("y")
rate = Expr.variable("rate")
solver = create_ivp(
    "lsode2", [-rate * y], ["y"], "t", [1.0], 0.0, 2.0, 0.05,
    parameters=["rate"], parameter_values=[0.7], controller="auto",
    options=SolverOptions(
        frontend="atom_native", execution="lambdify",
        matrix_layout="dense", execution_policy="sequential",
        telemetry="timings",
    ),
    rtol=1e-8, atol=1e-10,
)
solution = solver.solve()
print(solution.controller_mode, solution.final_state_array())
```

## Layout and symbolic routes

LSODE2 exposes ExprLegacy and AtomNative preparation, Lambdify and AOT
execution, and Dense/Sparse/Banded linear-system routes. These choices are
independent. Dense is a natural baseline for small fully coupled systems;
Sparse or Banded may be more suitable when the Jacobian structure supports
them. There is no universally fastest layout. Compare matched workloads and
inspect the selected linear solver and provenance.

For symbolic expressions, `matrix_layout="banded"` infers the Jacobian
bandwidth automatically when both width hints are omitted. Supply both
`band_lower` and `band_upper` only to constrain or declare the structure
explicitly; `(0, 0)` means a diagonal Jacobian, not automatic inference.
Numerical Python callbacks have no symbolic structure to inspect and must
provide both widths when using Banded layout.

| Control | Python API | Notes |
| --- | --- | --- |
| Controller | `create_ivp(..., controller=...)` | `"bdf"`, `"adams"` or `"auto"`; `"lsoda"` defaults to `"auto"` |
| Tolerances | `rtol`, `atol` | Local error control for the IVP solver |
| Parameters | `parameters`, `parameter_values` | Names and order must match the expressions |
| Banded widths | `band_lower`, `band_upper` | Optional together for symbolic Banded; required together for numerical callbacks |
| Callbacks | `residual_callback`, `jacobian_callback` | Native callback path; omitted Jacobian uses FD |
| Execution | `SolverOptions(execution=...)` | `lambdify` or `aot` |
| Scheduling | `SolverOptions(execution_policy=...)` | `sequential`, `parallel` or `auto` where supported |

The direct `Lsode2Solver` adapter accepts the controller as its `method`
constructor argument; the common `create_ivp` facade calls the same setting
`controller` because its `method` argument selects the solver family. The
direct adapter also accepts `band_lower`/`band_upper` constructor keywords or
`SolverOptions.set_banded_width(lower, upper)`. If both width forms are
provided, matching values are accepted and conflicting values raise a typed
configuration error. Widths are rejected for non-Banded layouts; providing
only one width is invalid.

The Python wrapper also exposes native analytical residual/Jacobian callbacks
for supported layouts. These callbacks cross into Python and retain the GIL;
Rust callback worker settings do not make Python callbacks equivalent to
Rust-native callbacks or promise Python-thread scalability. For a numerical
hot path, prefer symbolic preparation when possible.

## Preparation, fresh solves and time segments

Use `create_ivp` for explicit lifecycle control. `prepare()` builds the
selected symbolic route once. `solve_with_parameter_values(y0, values)` starts
a fresh solve on the configured interval; its `_in_interval(t0, y0, t_end,
values)` counterpart also changes the interval. `continue_with_parameters(...)`
is a separate time-segment operation from the previous accepted endpoint, and
`restart(...)` starts a new segment with a supplied state and interval.

## Telemetry and AOT

Choose `telemetry="off"`, `"counters"`, `"timings"`, or `"detailed"` in
`SolverOptions`. `counters` collects discrete counts only; `timings` collects
durations only; `detailed` collects both. Read the detached snapshot with
`solution.telemetry_counters()`, `solution.telemetry_timings_ms()`, and
`solution.telemetry_report()`. Stage durations may overlap and therefore
should not always be summed. AOT supports the shared BuildIfMissing, RequirePrebuilt and
RebuildAlways lifecycle, including process handoff and provenance. A cold
compile may only amortize over repeated work; report preparation, callbacks
and full-solve time separately.

## Larger examples

`examples/lsode2_controller_matrix.py` holds an equation fixed while comparing
controllers. `examples/lsode2_lorenz.py` demonstrates a nonlinear chaotic
three-state system; `examples/lsode2_three_body.py` adapts the crate's
12-state gravitational example, including symbolic distance aliases. These
are numerical demonstrations, not controlled speed comparisons.

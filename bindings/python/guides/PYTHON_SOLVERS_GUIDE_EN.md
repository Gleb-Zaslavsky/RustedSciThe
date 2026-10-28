# RustedSciThe Python Solvers Guide

This package is a thin Python boundary over the RustedSciThe symbolic and
numerical core. It does not reimplement numerical algorithms in Python.
`Expr` is parsed, differentiated and simplified by the Rust symbolic engine;
the selected solver owns preparation, callbacks, workspaces and telemetry.

## Installation on Windows

Create a virtual environment, install NumPy and build tools, then install the
wheel or use an editable development build:

```powershell
py -3.12 -m venv .venv-rustedscithe
.\.venv-rustedscithe\Scripts\python.exe -m pip install -U pip numpy maturin
& .\.venv-rustedscithe\Scripts\maturin.exe develop --manifest-path bindings/python/Cargo.toml
```

The package currently targets `abi3-py310` and requires NumPy 2.x. The native
extension is `rustedscithe._native`; normal users should import the stable
facade from `rustedscithe`.

## Route vocabulary

These are independent choices:

| Dimension | Values | Meaning |
| --- | --- | --- |
| symbolic frontend | `expr_legacy`, `atom_native` | expression preparation and differentiation route |
| execution | `lambdify`, `aot` | in-process closure or generated compiled artifact |
| matrix layout | `dense`, `sparse`, `banded` | linear-system storage and solve policy |
| callback policy | `sequential`, `parallel`, `auto` | callback scheduling where the native solver supports it |

Use `capabilities("solver")` rather than assuming that every combination is
valid. The metadata lists supported values per axis, not a Cartesian product;
`SolverOptions.validate_for()` checks shared cross-axis constraints and the
concrete constructor checks route-specific details. `telemetry_modes` reports
the modes actually collected by that family. In particular,
`ode_api2_nonstiff` supports only `off`, so set that explicitly when passing
`SolverOptions` to the nonstiff adapter. The `supports_logging` flag reports
whether `SolverOptions.logging_level` maps to the common Rust `log` API; a
solver's separate logging controls are documented on its own facade.

## IVP

`solve_ivp(...)` is the concise route. Use `create_ivp(...)` for explicit
preparation, fresh parameterized solves, time-segment continuation and restart. Radau5 receives `y0` at solve time;
BDF, Backward Euler and LSODE2 retain the initial state in the solver object.

```python
import numpy as np
from rustedscithe import Expr, SolverOptions, solve_ivp

y = Expr.variable("y")
result = solve_ivp(
    "radau5", [-y], ["y"], "t", np.array([1.0]), 0.0, 1.0, 0.1,
    options=SolverOptions(frontend="expr_legacy", telemetry="timings"),
    rtol=1e-5, atol=1e-7, output="dense",
)
print(result.final_state_array())
```

For repeated independent solves, prepare once and rebind parameters:

```python
from rustedscithe import BdfSolver

solver = BdfSolver(
    [-Expr.variable("a") * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
    parameters=["a"], parameter_values=[1.0],
)
solver.prepare()
first = solver.solve()
second = solver.solve_with_parameter_values([0.8], [2.0])
```

`solve_with_parameter_values(y0, values)` starts a fresh solve on the configured
interval; `solve_with_parameter_values_in_interval(t0, y0, t_end, values)` also
sets a new interval. Both reuse symbolic preparation. Time-segment continuation
is a separate operation from the last accepted state. AOT may be slower for
one cold solve and amortize over a longer series; measure preparation, callback
and full-solve stages separately.

The common facade selects the solver family with its first argument. LSODE2's
controller is a separate keyword, so Adams or automatic mode is not confused
with the choice of solver:

```python
from rustedscithe import Expr, create_ivp

y = Expr.variable("y")
solver = create_ivp(
    "lsode2", [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
    controller="adams",  # also "bdf" or "auto"
)
result = solver.solve()
```

`"lsoda"` is accepted as an LSODE2 solver-family alias for the automatic
Adams/BDF controller. `controller=` is only valid for the LSODE2 route.

The nonstiff part of `ODE_api2` is exposed separately as
`UniversalNonStiffSolver`; `create_ivp` and `solve_ivp` also accept `"rk45"`,
`"dopri"` and `"ab4"`. This route is intentionally narrower than the stiff
adapters: it supports ExprLegacy/Lambdify, Dense, sequential execution, a
positive step size, optional named-state stop conditions and a neighborhood
tolerance. It has no native telemetry, AOT, AtomNative, parameter continuation
or restart contract; non-default telemetry and unsupported backend selections
are rejected. Its fixed-step loop stops after reaching or crossing `t_end`,
so the final time can overshoot by less than one step. Stiff methods continue
to use their dedicated solver APIs.

### Convenience facade typing

The family-selecting helpers intentionally accept solver-specific keywords
through `**solver_options`: valid names and types differ substantially
between BDF, Radau, LSODE2 and the nonstiff adapter. Runtime validation is
route-aware and rejects unsupported options rather than silently ignoring
them. For complete IDE/mypy checking of every solver setting, construct the
corresponding public class directly (for example `BdfSolver` or
`RadauSolver`); their `.pyi` signatures mirror the native constructor.

## BVP

`solve_bvp("bvp_sci", ...)` and `solve_bvp("bvp_damp", ...)` share only the
outer lifecycle. Their boundary data are intentionally different:

- BVP_sci receives a list of symbolic boundary residual expressions.
- BVP_Damp receives a state-to-boundary mapping and requires explicit bounds.

Do not compare Dense, Sparse and Banded on different equations. A layout that
wins on a fully coupled problem may lose on a structured diffusion or
collocation Jacobian. BVP_sci also has opt-in SuperLU and reusable-faer sparse
policies; SuperLU availability and fallback are reported in telemetry.

## NumPy ownership

Results expose owned arrays such as `times_array()`, `states_array()`,
`mesh_array()` and `values_array()`. Array shapes are documented by each
result: IVP states are `(time, state)`, BVP_sci values are `(mesh, state)`,
and BVP_Damp values preserve its native `(state, mesh)` orientation. The
facade accepts NumPy vectors and converts them before entering Rust. No
zero-copy view is promised because prepared solvers reuse mutable workspaces.

The result API follows a shared ownership and naming convention, but it does
not invent interpolation for solvers whose native contract has none:

| Result family | Detached samples | Continuous point evaluation |
| --- | --- | --- |
| IVP trajectories | `times_array()`, `states_array()` | Radau only, with `output="dense"`, via `RadauSolution.sample(t)` |
| BVP_sci | `mesh_array()`, `values_array()` | With `output="dense"`, via `BvpSciSolution.sample(x)` |
| BVP_Damp, ShootingBVP, ODE_api2 nonstiff | Native mesh/trajectory samples | No continuous interpolator is exposed by these wrappers |

Dense output must be requested before solving where the solver supports it.
Sampling outside the retained interval or calling `sample()` without dense
output is a typed error. In particular, do not infer a shared `sample()` method
from the common array naming convention.

## Errors and telemetry

Catch typed exceptions: `InvalidConfigurationError`, `UnsupportedOptionError`,
`PreparationError`, AOT lifecycle errors, `CallbackError`, `SolverError` and
`ExhaustionError`. Native-mapped errors also expose `stage`, `solver` and
`route`; do not parse error messages for control flow.

Telemetry is disabled or minimized according to the selected native policy.
Use `telemetry="counters"` or `telemetry="timings"` when diagnosing a route.
Detached solutions expose structured mappings and compact report-file methods.

## Text tasks and examples

`TaskDocument.parse(...)`, `task(...)` and `run_task_batch(...)` reuse the Rust
task parser. Batch execution continues after failures and can be rendered with
`render_table(...)`. The runnable examples in `bindings/python/examples/`
cover Radau, BDF time-segment continuation, BVP_sci, nonlinear/fitting and task batches.
Use `task_dict(source_or_path)` when application code needs plain Python data:
it serializes the already parsed native specification, including equations,
solver options, continuation and output settings, without executing the task.
`check_dependencies()` returns structured PATH-tool and optional SuperLU
loadability results without printing.

Native `log` output is distinct from telemetry. It is disabled by default;
solvers exposing the Rust logger accept `options.set_logging_level("info")`
with `error`, `warn`, `info`, or `debug`. Logs are written to a timestamped
process-level file, not returned as solver data. Set `RST_SOLVER_LOG_DIR`
before starting Python to choose the destination directory. Solvers without
native logging reject an enabled common logging level instead of silently
ignoring it.

## Per-solver guides

The overview above is intentionally concise. Detailed guides preserve the
configuration breadth of each native solver while translating code snippets
to Python. Start with the relevant route:

- [Radau5](RADAU_GUIDE_EN.md): stiff IVPs, frontend/layout selection, output
  and telemetry.
- [BDF](BDF_GUIDE_EN.md): dense-only BDF, order controls, callbacks and
  continuation.
- [Backward Euler](BACKWARD_EULER_GUIDE_EN.md): fixed-step semantics and
  supported dense execution routes.
- [LSODE2](LSODE2_GUIDE_EN.md): Adams/BDF controllers, symbolic routes,
  matrix layouts and parallel callback policy.
- [BVP_sci](BVP_SCI_GUIDE_EN.md): boundary residuals, dense/sparse/banded
  layouts, SuperLU and AOT.
- [BVP_Damp](BVP_DAMP_GUIDE_EN.md): observations, bounds, mesh refinement,
  layout and symbolic/AOT choices.
- [ShootingBVP](SHOOTING_BVP_GUIDE_EN.md): small scalar second-order BVPs,
  two symbolic frontends, IVP selection and solve telemetry.
- [Nonlinear systems and fitting](NONLINEAR_FITTING_GUIDE_EN.md): root,
  least-squares and fitting APIs.

Runnable, commented Python scripts are indexed in
[`examples/README.md`](../examples/README.md).

# BVP_Damp Guide for Python

BVP_Damp solves first-order boundary-value systems with a damped nonlinear
iteration and mesh/refinement policies. It is not the same formulation as
BVP_sci: its boundary input is a mapping from state names to observed endpoint
values, with optional bounds. The Python package exposes separate damped,
fixed-mesh Naive and fixed-mesh Frozen routes. These select distinct native
algorithms rather than aliases for one another.

## Model and boundary observations

The equations are the first-order state derivatives. The Python adapter
currently requires one total boundary observation per unknown; this differs
from BVP_sci's complete left/right boundary-residual vector. For example,
`y'=0` with `y(0)=1` has the exact constant profile `y=1`. The initial state
is a node-major guess; bounds constrain iterates rather than define the
boundary conditions.

```python
import numpy as np
from rustedscithe import Expr, SolverOptions, solve_bvp

mesh = np.linspace(0.0, 1.0, 16)
options = SolverOptions(
    frontend="atom_native", execution="lambdify",
    matrix_layout="banded", telemetry="timings",
)
options.set_banded_width(1, 1)
solution = solve_bvp(
    "bvp_damp", [Expr.constant(0.0)], ["y"],
    {"y": [(0, 1.0)]}, mesh, np.full(mesh.size, 0.6),
    options=options,
    bounds={"y": (-0.1, 1.2)}, tolerance=1e-7,
    max_iterations=40, max_refinements=4,
    logging_mode="off", loglevel="error",
)
```

The constructor preserves BVP_Damp's state-oriented boundary semantics; it does
not accept BVP_sci's residual-expression list. The output values follow the
native BVP_Damp `(state, mesh)` orientation; use `values_array()` and inspect
its shape rather than assuming BVP_sci's node-major array convention.

## Solver policies

Choose ExprLegacy/AtomNative and Lambdify/AOT independently. Dense, Sparse and
Banded routes, damping, Jacobian refresh, grid refinement, linear-system
policy, iterative refinement and bounds are forwarded through the direct
wrapper where supported. Read `capabilities("bvp_damp")`; a convenience
function does not silently replace solver-specific controls with defaults.
No matrix layout is the universal winner: compare matched equations and mesh
sizes, and consider the global Jacobian structure.

`BvpDampNaiveSolver` is a separate fixed-mesh route backed by the native Naive
implementation; it does not pass through the damped or Frozen solver.
It supports the same symbolic frontend and Lambdify/AOT selection with Dense,
Sparse or Banded linear layouts, plus native telemetry and prepared parameter
rebinding. Its mesh must be uniform. Damping controls, bounds, adaptive grid
refinement, and explicit Banded widths are not part of this native route and
are not accepted as if they had an effect. The prepared API currently supports
parameter rebinding, but not replacing the initial guess or interval in place.

```python
from rustedscithe import BvpDampNaiveSolver, Expr, SolverOptions

solver = BvpDampNaiveSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
    parameters=["rate"], parameter_values=[1.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="timings"),
)
solver.prepare()
first = solver.solve_prepared()
solver.set_parameters([2.0])
second = solver.solve_prepared()
```

`BvpDampFrozenSolver` uses a frozen Jacobian with a user-selected native reuse
policy. The mesh remains fixed and uniform; parameters can be rebound after
`prepare()`, but the initial guess and interval cannot be replaced in place.
The supported policies are `Frozen_naive` (reuse after the first Jacobian),
`every_m` (one positive integer period), `at_high_morm` (one positive norm
threshold), `at_low_speed` (one speed-rate threshold no greater than 1), and
`complex` (`[period, norm_threshold, speed_rate]`). `policy_parameters` is
omitted for `Frozen_naive` and otherwise follows the listed shape. The class
reports the chosen policy through `provenance()` and returns the same detached
solution telemetry contract as the other BVP_Damp routes.

```python
from rustedscithe import BvpDampFrozenSolver, Expr, SolverOptions

solver = BvpDampFrozenSolver(
    [Expr.parse("rate")], ["y"], {"y": [(0, 0.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5,
    parameters=["rate"], parameter_values=[1.0],
    policy="every_m", policy_parameters=[2.0],
    options=SolverOptions(frontend="atom_native", execution="lambdify",
                          matrix_layout="dense", telemetry="counters"),
)
solver.prepare()
for rate in (1.0, 2.0):
    solver.set_parameters([rate])
    solution = solver.solve_prepared()
    print(rate, solution.values()[0][-1], solution.telemetry_counters())
```

The direct constructor exposes the main numerical and structural controls:

| Area | Python keyword(s) | Notes |
| --- | --- | --- |
| Discretization | `scheme="forward"` or `"trapezoid"` | Selects the first-order scheme |
| Strategy | `BvpDampSolver`, `BvpDampNaiveSolver` or `BvpDampFrozenSolver` | Each selects a separate native algorithm; none is silently mapped to another |
| Nonlinear work | `tolerance`, `max_iterations`, `max_jacobian_refreshes`, `max_damping_iterations`, `damping_factor` | Newton/damping policy |
| Mesh | `max_refinements`, `grid_method`, `grid_parameters` | `grid_parameters` must match the selected method |
| Linear system | `linear_sys_method`, `banded_linear_solver`, `banded_fallback`, `iterative_refinement_steps` | Backend-specific native policies |
| Bounds/logging | `bounds`, `logging_mode`, `loglevel`, `logging_max_events` | Logging defaults off; bounds are user-specified |
| Banded layout | `SolverOptions.set_banded_width(lower, upper)` | Required by the damped route; fixed-mesh Naive/Frozen infer structure and reject explicit widths |

## Lifecycle, logging and telemetry

`prepare()` separates symbolic preparation from repeated numerical solves.
Use `solve_prepared()` for a prepared iterate and `restart(...)` when replacing
the iterate. Parameter changes use `set_parameters(...)` and should preserve
prepared symbolic work where the native lifecycle allows it. Logging is off
unless explicitly enabled; telemetry mode is independently configured through
`SolverOptions` and detached into the result.

Read counters and stage timings separately. A successful status, final mesh,
residual/iteration diagnostics and bounds should all be inspected. Catch typed
configuration, preparation, linear-backend and convergence exceptions; do not
parse human-readable log output for program control.

Telemetry modes are disjoint: `off` collects neither counters nor timings;
`counters` records discrete events without starting clocks; `timings` measures
stage durations without collecting discrete counters; and `detailed` collects
both. The selected mode is available as `solution.telemetry_mode`, while the
two result accessors return empty dictionaries for a disabled collection.
The detached result also exposes `telemetry_scope_timings_ms()` and
`telemetry_scope_events()` for solve/mesh/iteration/damping scopes,
`telemetry_callback_counters()` and `telemetry_callback_timings_ms()` for the
selected symbolic callback route, and `telemetry_preparation_timings_ms()`
when the native solver published preparation snapshots. Available storage
estimates are reported separately by `telemetry_storage_bytes()`; a missing
key means that the backend does not expose that storage quantity, not zero
bytes. `telemetry_storage_is_estimated()` distinguishes estimates from direct
measurements and returns `None` when no values are available. Scope durations
can overlap and must not be summed as independent work.

## Examples

`examples/bvp_damp_observations.py` is the compact single-route example;
`examples/bvp_damp_layout_matrix.py` runs the exact constant profile across
both frontends and all three layouts. The Rust BVP_Damp guides remain
authoritative for the full numerical scheme and options that are not exposed
by the Python adapter.

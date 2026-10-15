# BVP_Damp Guide for Python

BVP_Damp solves first-order boundary-value systems with a damped nonlinear
iteration and mesh/refinement policies. It is not the same formulation as
BVP_sci: its boundary input is a mapping from state names to observed endpoint
values, with optional bounds. Frozen routes remain deferred in this Python
wrapper; do not infer support from the broader native Rust crate.

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

The direct constructor exposes the main numerical and structural controls:

| Area | Python keyword(s) | Notes |
| --- | --- | --- |
| Discretization | `scheme="forward"` or `"trapezoid"` | Selects the first-order scheme |
| Strategy | `strategy="damped"` | `"frozen"` is explicitly unsupported by this wrapper |
| Nonlinear work | `tolerance`, `max_iterations`, `max_jacobian_refreshes`, `max_damping_iterations`, `damping_factor` | Newton/damping policy |
| Mesh | `max_refinements`, `grid_method`, `grid_parameters` | `grid_parameters` must match the selected method |
| Linear system | `linear_sys_method`, `banded_linear_solver`, `banded_fallback`, `iterative_refinement_steps` | Backend-specific native policies |
| Bounds/logging | `bounds`, `logging_mode`, `loglevel`, `logging_max_events` | Logging defaults off; bounds are user-specified |
| Banded layout | `SolverOptions.set_banded_width(lower, upper)` | Width is required; never inferred from the RHS |

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

## Examples

`examples/bvp_damp_observations.py` is the compact single-route example;
`examples/bvp_damp_layout_matrix.py` runs the exact constant profile across
both frontends and all three layouts. The Rust BVP_Damp guides remain
authoritative for the full numerical scheme and options that are not exposed
by the Python adapter.

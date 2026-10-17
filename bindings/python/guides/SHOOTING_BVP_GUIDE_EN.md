# ShootingBVP Python Guide

`ShootingBvpSolver` is a symbolic, Lambdify-backed interface to the core
ShootingBVP algorithm. It is intentionally scoped to a scalar second-order
boundary-value problem written as a two-state first-order system
`u = [y, y']`. Shooting is useful for small problems, but it is not a general
replacement for collocation: it can become ill-conditioned for difficult
boundary conditions and is not recommended for large systems.

## Problem and Boundary Conditions

Pass two right-hand-side expressions in state order, two state names, the
independent variable, the interval and one boundary condition at each endpoint.
Each boundary type is `"dirichlet"` (`y = value`) or `"neumann"`
(`y' = value`). `initial_guess` is the trial value for the unconstrained left
state component. The outer shooting iteration is a secant method.

```python
import math

from rustedscithe import Expr, ShootingBvpSolver, SolverOptions

solver = ShootingBvpSolver(
    equations=[Expr.parse("v"), Expr.parse("-y")],
    state_variables=["y", "v"],
    independent_variable="x",
    t_start=0.0,
    t_end=math.pi / 2.0,
    left_boundary_type="dirichlet",
    left_boundary_value=0.0,
    right_boundary_type="dirichlet",
    right_boundary_value=1.0,
    initial_guess=0.5,
    tolerance=1e-8,
    max_iterations=100,
    step_size=0.01,
    ivp_method="rk45",
    options=SolverOptions(frontend="atom_native", telemetry="timings"),
)
solution = solver.solve()
print("inferred initial slope:", solution.shooting_value)
print("trajectory shape:", solution.states_array().shape)
print("telemetry:", solution.telemetry)
```

The example solves `y'' = -y`, `y(0)=0`, `y(pi/2)=1`; its exact solution is
`sin(x)` and the inferred initial slope is approximately one. `states_array()`
is an owned NumPy array with shape `(mesh_points, 2)` and columns `[y, y']`.

## Frontend, IVP Method and Parameters

The symbolic frontend is selected with `SolverOptions(frontend=...)`:

- `expr_legacy` is the default symbolic route.
- `atom_native` uses the AtomView symbolic frontend.

The callback execution route is Lambdify. AOT is not supported by this solver.
The linear layout is Dense; Sparse and Banded are not relevant to this small
shooting interface. `execution_policy` accepts `sequential` or `parallel` for
symbolic callback preparation/evaluation. IVP methods accepted by
`ivp_method` are `rk45`, `dopri`, `ab4`, `radau`, `bdf`, `backward_euler`, and
`lsode2`. Method-specific options can be supplied in `solver_parameters` as a
Python dictionary of scalar values or rectangular list-of-lists matrices;
the selected native IVP adapter remains responsible for validating each key.

## Telemetry and Logging

Telemetry is disabled with `SolverOptions(telemetry="off")`. `counters` and
`timings` both request the native ShootingBVP snapshot because the core exposes
them together. The returned `solution.telemetry` distinguishes secant updates,
boundary residual evaluations and IVP solves, and reports residual-IVP,
final-IVP, secant-phase and total solve milliseconds. `preparation_ms` is the
separate symbolic preparation wall time.

Rust logging is independently disabled by default. Enable it with
`options.set_logging_level("info")` (or `error`, `warn`, `debug`). The core
creates one timestamped log file per process; set `RST_SOLVER_LOG_DIR` before
Python starts to choose its directory.

## Scope and Guidance

Shooting integrates from one boundary and adjusts the free initial state until
the other boundary is met. This can be very effective for a single equation
or a tiny system with a well-behaved IVP, but error growth and unstable modes
can make the method unsuitable for some BVPs. For larger or more difficult
systems, prefer `BvpDampSolver` or `BvpSciSolver` and compare matched problems.

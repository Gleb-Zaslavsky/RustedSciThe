# ShootingBVP User Guide

ShootingBVP solves a boundary-value problem by repeatedly integrating an
initial-value problem and adjusting one unknown initial state with a secant
iteration. The maintained symbolic route uses the crate's Lambdify
infrastructure and supports `ExprLegacy` and `AtomViewNative`.

## When To Use Shooting

The current implementation targets a **single scalar second-order ODE**
written as the two-state system `[y, y']`. It is a reasonable choice for an
isolated equation and, conceptually, for only very small coupled systems if a
future API explicitly supports them. The current API requires exactly two
state variables and two equations; it does not yet implement coupled vector
second-order systems.

Shooting is not a general-purpose method for large systems. Each secant trial
requires a complete IVP integration, and sensitivity to the guessed initial
value can make convergence fragile. For large coupled systems, difficult
boundary layers, or problems where an IVP trajectory is unstable, prefer a
collocation BVP solver such as BVP_Damp or BVP_sci. There is no universal
fastest solver: choose based on problem structure and verify against known
solutions or an independent solver.

## Typed Builder API

The builder keeps the left and right boundary conditions explicit and uses a
typed IVP method selector. This example solves `y'' = y`, `y(0)=0`,
`y(1)=sinh(1)`, whose exact solution is `sinh(x)`:

```rust
use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType};
use RustedSciThe::numerical::ShootingBVP::Shooting_simple::{
    BoundaryCondition, BoundaryConditionType,
};
use RustedSciThe::numerical::ShootingBVP::{ShootingSymbolicOptions, shooting_bvp};
use RustedSciThe::numerical::SolverTelemetryMode;
use RustedSciThe::symbolic::symbolic_engine::Expr;

let problem = shooting_bvp(
    vec![Expr::parse_expression("dy"), Expr::parse_expression("y")],
    vec!["y".into(), "dy".into()],
    "x",
)
.interval(0.0, 1.0)
.boundary_conditions(
    BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet },
    BoundaryCondition {
        value: 1.0_f64.sinh(),
        bc_type: BoundaryConditionType::Dirichlet,
    },
)
.initial_guess(0.8)
.tolerance(1e-8)
.max_iterations(30)
.step_size(0.002)
.ivp_solver(SolverType::NonStiff(NonStiffMethod::Rk45))
.symbolic_options(ShootingSymbolicOptions::default())
.build()?;

let report = problem.solve(SolverTelemetryMode::Detailed)?;
let solution = report.solve.solution;
assert!((solution.bound_values[0] - 1.0_f64.sinh()).abs() < 1e-5);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The equivalent, fully commented example is runnable with:

```text
cargo run --example shooting_bvp_builder_guide
```

## Telemetry

The builder's `solve` method takes the shared `SolverTelemetryMode`:
`Off` collects nothing, `Counters` collects discrete work counts without
reading the clock, `Timings` records durations without updating counters, and
`Detailed` collects both. The returned `ShootingTelemetry.mode` records which
channels were measured, so zero-valued fields in an unrequested channel are
not mistaken for measured zeros.

The state order is `[y, y']`; the equations must therefore be `[y', f(x,y,y')]`.
Each endpoint supplies exactly one scalar condition. `Dirichlet` constrains
`y`, and `Neumann` constrains `y'`. Mixed conditions are supported, for example
`y(a)=alpha` and `y'(b)=beta`.

The shooting `tolerance` controls the outer boundary residual, not the accuracy
of each IVP trajectory. Set suitable tolerances/options on the selected IVP
solver as well; a tightly converged secant iteration cannot recover accuracy
lost inside a loosely integrated IVP.

## IVP Solver And Symbolic Frontend

The typed builder accepts `SolverType`, including non-stiff methods, BDF,
Radau, Backward Euler, and LSODE2 when supported by the configured crate
features and method API. The default is RK45. Method-specific options can be
passed with `solver_parameter(name, SolverParam::...)`; use option names and
types documented by the selected IVP solver. For example, BDF accepts `rtol`,
`atol`, and `max_step` options in the native adapter.

Frontend selection is independent of the numerical IVP solver:

```rust
use RustedSciThe::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;

let options = ShootingSymbolicOptions {
    frontend: SymbolicLambdifyFrontend::AtomViewNative,
    ..ShootingSymbolicOptions::default()
};
```

`ExprLegacy` is the default and can be useful for small expressions. `AtomViewNative`
keeps symbolic processing on the Atom representation. Neither route uses AOT;
both prepare a Lambdify callback. Frontend choice does not imply a universal
performance winner, so compare them on the same equation and IVP settings.

## Results, Telemetry, And Logging

`solve(true)` returns the trajectory, selected frontend, symbolic preparation
report, preparation wall time, and opt-in numerical telemetry. Numerical
counters include secant iterations, residual evaluations, and IVP solves;
timings distinguish the secant phase, residual integrations, final trajectory
integration, and total solve. These scopes overlap by design and should not be
summed as independent costs.

Telemetry is binary opt-in: use `solve(false)` when it is not needed. Logging
is off by default. Enable it with `.logging_level(SolverLogLevel::Info)`; the
shared logging layer writes to a timestamped file. Failure to initialize that
file is returned as a typed error by the fallible solve API. Numerical failures
are also typed (`ShootingBvpError`), and a failed solve clears the solver's
previous stored trajectory rather than presenting stale output.

The builder currently performs symbolic preparation for each `solve` call; it
is not yet a prepared continuation interface. Repeated parameter-family solves
should use a solver with an explicit prepared/rebind lifecycle until that
capability is added here.

## Compatibility API

`BVPShooting` and the older `HashMap` boundary-condition interface remain
available for existing code. New code should prefer `shooting_bvp(...)`, which
validates required fields and exposes typed endpoint conditions and typed
solver selection without making callers construct the legacy boundary map.

## Independent Correctness Check

The debug test `shooting_matches_independent_damped_bvp_on_common_mesh`
compares ShootingBVP with BVP_Damp on the same harmonic-oscillator problem
`y'' + y = 0`, `y(0)=0`, `y(pi/2)=1`. Since the solvers choose different meshes,
both trajectories are linearly interpolated onto 81 shared coordinates; their
maximum component-wise difference must be below `2e-3`. Each solver is also
checked independently against the exact `y=sin(x)`, `y'=cos(x)` solution, so
agreement between two inaccurate trajectories cannot pass by itself. This is a
correctness gate, not a performance comparison, and it intentionally does not
compare solver-specific node counts or iteration counters.

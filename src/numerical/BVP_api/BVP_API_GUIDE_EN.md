# Typed BVP Facade

`numerical::BVP_api` is a thin selector over production BVP solvers. It
validates shared problem data, routes requests to native APIs, and retains the
selected native solver, result, and telemetry. It does not replace symbolic
preparation, solver configuration, Newton/collocation algorithms, or linear
algebra.

```rust
use nalgebra::DMatrix;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciSparseFactorization,
    BvpSolution, BvpSolver, BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = -y, with y(0) = 1.
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?;

    let config = BvpSolverConfig::Sci {
        options: BvpSciOptions::default()
            .with_assembly(BvpSciAssembly::ExprLegacy)
            .with_execution(BvpSciExecution::Lambdify)
            .with_telemetry(BvpSciTelemetryMode::Counters),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
        // State-major: one initial value per state at every mesh node.
        initial_guess: DMatrix::from_row_slice(1, 5, &[1.0, 0.78, 0.61, 0.47, 0.37]),
    };
    let mut solver = BvpSolver::new(problem, config)?;

    if let BvpSolution::Sci(result) = solver.solve()? {
        println!("y(1) = {:.8}", result.y[4]);
    }
    let dataset = solver.postprocess_dataset()?;
    println!("node-major dataset: {:?}", dataset.values.shape());

    if let BvpSolverTelemetrySnapshot::Sci(snapshot) = solver.telemetry_snapshot() {
        println!("native RHS calls: {}", snapshot.rhs_calls);
    }
    Ok(())
}
```

The complete runnable version is in
[`examples/bvp_api_facade_sci.rs`](../../../../examples/bvp_api_facade_sci.rs).
For fixed-input rebinding on Damped, see
[`examples/bvp_api_parameter_rebind.rs`](../../../../examples/bvp_api_parameter_rebind.rs).
For a side-by-side Damped/Frozen comparison, including their different guess
shapes, see
[`examples/bvp_api_damped_frozen.rs`](../../../../examples/bvp_api_damped_frozen.rs).
For a prepared BVP_sci external-input parameter loop, see
[`examples/bvp_api_parameter_loop.rs`](../../../../examples/bvp_api_parameter_loop.rs).
For fixed-input loops through both BVP_Damp routes, including their distinct
checkpoint and failure-report behavior, see
[`examples/bvp_api_damped_frozen_parameter_loop.rs`](../../../../examples/bvp_api_damped_frozen_parameter_loop.rs).

## Routes And Data

`BvpProblem` stores equations in `state_names` order, the independent-variable
name, a strictly increasing mesh, and scalar boundary conditions. Each
condition specifies a state index, an endpoint, and a finite value. State
names must be unique and must not collide with the independent variable.
The facade requires one scalar condition per equation in total, but permits
both endpoints to constrain the same state when that matches the mathematical
problem; the native solver remains responsible for solvability.

Initial-guess layout is solver-specific by design. Damped expects
`state_count x interval_count`; Frozen/Naive and BVP_sci expect
`state_count x mesh_point_count`. Damped and Frozen accept only uniform meshes
because their native constructors take interval endpoints and a point/step
count. BVP_sci receives the explicit mesh directly.

BVP_Damp's adaptive refinement is solver-managed inside the solve lifecycle;
callers do not manually restart it when refinement criteria are unmet. The
facade's explicit `restart(mesh, guess)` operation is specific to BVP_sci,
which can preserve its prepared symbolic model while starting from a supplied
mesh and profile.

The facade does not invent a Cartesian capability matrix. It forwards native
`DampedSolverOptions`, `FrozenSolverOptions`, or `BvpSciOptions`; BVP_sci also
retains its sparse-factorization selection and optional AOT configuration. For
fixed model inputs on Damped/Frozen, call `bind_parameters(names, values)`
before the first solve, then use `solve_with_parameter_values(values)` to reuse
the prepared route and previous Newton iterate. BVP_sci `parameters` are not
fixed inputs: they are free Newton unknowns, so the BVP_Damp rebinding methods
reject that route. Fixed BVP_sci model inputs can be declared on
`BvpProblem::with_external_parameters(names, initial_values)` before symbolic
preparation; `solve_parameter_loop` then rebinds each supplied vector and
warm-starts from the last successful profile. This external-input schema is
distinct from free parameter unknowns and their parameter-dependent boundary
residuals. For numerical RHS or Jacobian callbacks, use the native BVP_Damp API.

`solve_parameter_loop` preserves each route's native loop semantics. Damped
may carry a newly refined mesh together with the last successful profile;
Frozen/Naive retains its fixed mesh and carries the successful profile only.
Point failures are recorded and do not replace the last successful checkpoint.
The facade does not convert mesh refinement into parameter continuation or
share a fabricated common result type.

The facade also exposes the native unknown-parameter route through
`BvpSolverConfig::SciWithFreeParameters`. Keep the `n` ordinary scalar endpoint
conditions in `BvpProblem`; provide the additional `p` residual equations in
`BvpSciFreeParameters`. Its optional Jacobian callback supplies the extra
rows with respect to left state, right state, and free parameters. If omitted,
BVP_sci uses its native finite-difference boundary Jacobian.
`parameter_guess()` reads the current Newton guess,
`set_parameter_guess(...)` updates it, and
`solve_with_parameter_guess(...)` updates and solves in one call. These are
free Newton unknowns, not fixed-input rebinding. A successful solve retains its
solved parameter vector as the current guess for a subsequent warm solve. See
[`examples/bvp_api_sci_free_parameter.rs`](../../../examples/bvp_api_sci_free_parameter.rs)
for a complete minimal solve of `y' = p`, `y(0) = 0`, with an extra equation
that identifies `p`.

### Capability Boundaries

The shared `BvpProblem` is symbolic-only. Damped and Frozen/Naive use their
native generated-backend configuration, but require a uniform mesh because
their constructors take endpoints and a node/interval count. Native numerical
RHS/Jacobian callback constructors are not represented by this facade. For the
same reason, a BVP_sci facade request with `BvpSciExecution::Numerical` or
`BvpSciAssembly::Numerical` returns `UnsupportedRoute`; use the native
BVP_sci numerical builder for callbacks. Note that `BvpSciOptions::default()`
selects Numerical execution, so symbolic facade requests should explicitly
select `Lambdify` (assembly then defaults to ExprLegacy) or `Aot` with its
generated-backend configuration.

The solver families deliberately have different defaults. Damped defaults to
forward differencing, Sparse, `1e-6` absolute tolerance, 25 nonlinear
iterations, AtomView symbolic assembly, and AOT-if-available with Lambdify
fallback; solver counters are on while callback timings and logging are off.
Frozen defaults to its `Frozen_naive` strategy, Sparse, `1e-6` tolerance and
25 iterations. BVP_sci native options default to Numerical/Dense, `1e-3`
tolerance, telemetry Off, and sequential callback dispatch. Because the facade
is symbolic-only, the BVP_sci native default must be overridden as described
above. These are native defaults, not a shared facade preset.

The facade forwards native options but uses its own narrow
`BvpSciSparseFactorization` enum: only `StandardFaer` and `SuperLu` are exposed.
Experimental reusable-faer variants remain available through the direct native
BVP_sci API only; they have not shown a measured performance win. `SuperLu` is
an explicit optional route and keeps the native bridge and fallback behavior.
Dense-output sampling likewise remains a native BVP_sci operation; shared
postprocessing exports retained mesh rows only.

The facade does not normalize every symbolic route into a Cartesian product.
For the supported BVP_sci Lambdify routes, ExprLegacy and AtomViewNative work
with Dense, Sparse and BandedAuto storage and sequential or parallel callback
policies. BandedAuto is resolved during preparation; inspect
`native().options().matrix_layout` for the effective native band. AOT requires
an explicit generated-backend config and preserves native compiler/build
errors; the facade never silently changes AOT to Lambdify.

After a successful solve, `solution_view()` borrows the actual mesh and
provides uniform node/state indexing through `mesh()`, `state_count()`, and
`value(node, state)`. It allocates no common dense buffer. The original
route-specific solution and diagnostics remain available through `solution()`.
`execute_postprocessing(&plan)` applies the shared export/plot plan to those
retained mesh rows and returns its native report. Export failures are typed and
do not discard the solution. BVP_sci dense-output sampling is intentionally
not implied; use the native result API for that route.

## Errors, Telemetry, And Logs

`BvpApiError` distinguishes invalid problem data, unsupported routes, missing
results, and native solver failures. Native errors remain typed sources rather
than being stringified. `solve()` clears the facade result before each attempt,
so a failed retry cannot expose the previous result as if it were new. Use
`native()` and `native_mut()` for the complete solver-specific lifecycle,
diagnostics, and runtime policy controls.

`telemetry_snapshot()` returns `BvpSolverTelemetrySnapshot`, with one variant
per solver family. Native scopes and measured-zero versus absent-measurement
semantics are preserved; the facade does not merge or rename metrics. Select a
telemetry mode through native options. A numeric zero is an observed zero; an
optional stage snapshot set to `None` means the route did not install or
measure that stage. BVP_Damp snapshots retain their native counters, timing
scopes, storage and optional symbolic/AOT callback snapshots. BVP_sci retains
its callback, Newton, mesh-pass and linear-backend counters/timings; timing
scopes may be inclusive or nested and are not necessarily additive. Consult the
native `BvpTelemetrySnapshot` and `BvpSciTelemetrySnapshot` docs for
field-level semantics; the facade does not define a second telemetry schema.
Logging is also configured through
native options or native solver APIs; the facade installs no second logger and
does not return text logs as telemetry. An enabled logging level must create
the shared timestamped file sink before BVP preparation/solve proceeds; setup
failure is returned as a typed native error rather than silently running with
logging disabled. The shared log path is available from
`numerical::logging::solver_log_path()`.

The process-wide `log` interface currently cannot return file write/flush
failures to the solver call site. Such a failure produces one stderr
diagnostic; making it a typed solve result requires a crate-wide logger
health/error contract and is not implemented as a facade-specific workaround.

The task-document parser is a separate subsystem. Use
`command_interpreter::task_parser_bvp` to read and execute task documents; it
constructs native BVP solvers directly and does not depend on this facade.

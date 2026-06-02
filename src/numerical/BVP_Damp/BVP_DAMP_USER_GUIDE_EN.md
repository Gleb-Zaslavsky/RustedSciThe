# BVP Damped/Frozen in RustedSciThe: user guide

This guide is written for a Rust developer who opens the BVP part of RustedSciThe for the first time and wants a practical mental model rather than a scavenger hunt through tests. We will cover what data the solver expects, how Damped and Frozen Newton differ, when Sparse or Banded is the right matrix route, what Numerical, Lambdify, and AOT actually mean, and what the current story/performance measurements tell us.

RustedSciThe's BVP solvers handle boundary value problems for systems of ordinary differential equations. You provide a first-order continuous system, a mesh, an initial guess, and boundary conditions. The solver discretizes the system, turns it into a large nonlinear algebraic problem, and solves that problem with a modified Newton method. In practice, configuring a BVP solve means configuring the whole path from continuous RHS to residual vector, Jacobian, linear solve, damping, and diagnostics.

## 1. Damped and Frozen: two Newton strategies

The main production path is Damped Newton, implemented in [`NR_Damp_solver_damped.rs`](NR_Damp_solver_damped.rs). It supports damping, bounds, Jacobian refreshes, diagnostics, generated backend handoff, and the broadest set of tested configurations. If you are not sure which strategy to use, start with Damped.

Frozen Newton, implemented in [`NR_Damp_solver_frozen.rs`](NR_Damp_solver_frozen.rs), is a specialized route with frozen linearization behavior. It is useful when you intentionally want to reuse or compare against a fixed-Jacobian style process. It is not the default "newer" or automatically faster route. A good workflow is: first make the problem converge and pass correctness checks with Damped, then use Frozen deliberately when its behavior matches the experiment.

Both solvers support symbolic problem setup through `Expr` and generated backend configuration. The pure numerical RHS-closure route is currently implemented for Damped.

## 2. What the BVP solver receives

The preferred constructor for new Damped code is `NRBVP::new_with_options(...)`:

```rust
pub fn new_with_options(
    eq_system: Vec<Expr>,
    initial_guess: DMatrix<f64>,
    values: Vec<String>,
    arg: String,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    t0: f64,
    t_end: f64,
    n_steps: usize,
    options: DampedSolverOptions,
) -> NRBVP
```

`eq_system` is the first-order RHS system. For example, the scalar equation `y'' + y = 0` is usually rewritten as `y' = z`, `z' = -y`, so `eq_system = ["z", "-y"]` and `values = ["y", "z"]`. The order of `values` defines the state ordering and must match the RHS ordering.

`arg` is the independent variable name, usually `"x"`. The interval is given by `t0` and `t_end`, while `n_steps` is the number of mesh intervals.

`initial_guess` is the reduced unknown guess. This is an important detail in the current BVP Damped/Frozen API: the matrix is usually `values.len() x n_steps`, not `values.len() x (n_steps + 1)`. Boundary values are stored separately in `border_conditions` and are reconstructed into the full mesh state internally.

```rust
let initial_guess = DMatrix::from_vec(values.len(), n_steps, guess_values);
```

Boundary conditions use `HashMap<String, Vec<(usize, f64)>>`. The side flag `0` means the left boundary and `1` means the right boundary. For example, `y(0)=0` and `z(L)=1` are written as:

```rust
let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(1usize, 1.0)]),
]);
```

The Damped numeric route expects a well-posed first-order BVP: the total number of fixed endpoint conditions must equal `values.len()`. Those conditions do not have to be distributed one-per-variable. For example, the oscillator written as `y' = z`, `z' = -y` can be posed numerically with `y(0)=0` and `y(pi/2)=1`, leaving the derivative variable `z` free at both endpoints. Frozen does not use the numeric route.

## 3. `DampedSolverOptions`: the main configuration surface

`DampedSolverOptions` groups the values that used to be passed through long positional constructors. The explicit constructor is:

```rust
pub fn new(
    scheme: String,
    strategy: String,
    strategy_params: Option<SolverParams>,
    linear_sys_method: Option<String>,
    method: String,
    abs_tolerance: f64,
    rel_tolerance: Option<HashMap<String, f64>>,
    max_iterations: usize,
    bounds: Option<HashMap<String, (f64, f64)>>,
    loglevel: Option<String>,
) -> Self
```

In everyday code, start from one of the presets:

```rust
let options = DampedSolverOptions::sparse_damped();
let options = DampedSolverOptions::banded_damped();
let options = DampedSolverOptions::dense_damped();
```

Then refine tolerances, bounds, nonlinear strategy, and backend:

```rust
let strategy = SolverParams {
    max_jac: Some(6),
    max_damp_iter: Some(6),
    damp_factor: Some(0.5),
    adaptive: None,
};

let options = DampedSolverOptions::banded_damped()
    .with_strategy_params(Some(strategy))
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(HashMap::from([
        ("y".to_string(), 1e-8),
        ("z".to_string(), 1e-8),
    ]))
    .with_bounds(HashMap::from([
        ("y".to_string(), (-2.0, 2.0)),
        ("z".to_string(), (-2.0, 2.0)),
    ]))
    .with_max_iterations(40)
    .with_loglevel(Some("none".to_string()));
```

`scheme` selects the discretization scheme. `strategy` is usually `"Damped"`. `linear_sys_method` can normally stay `None` when the matrix route is selected through `method` and generated backend configuration. `method` is the broad matrix route: `"Dense"`, `"Sparse"`, or `"Banded"`.

`FrozenSolverOptions` is similar but smaller. It does not expose the same bounds and per-variable relative tolerance surface as Damped:

```rust
let options = FrozenSolverOptions::banded_frozen()
    .with_tolerance(1e-8)
    .with_max_iterations(40);
```

## 4. Dense, Sparse, Banded: a cost model, not a style preference

`Dense` is the small/debug path. It is useful for tiny systems and tests where simplicity matters more than memory or asymptotic cost.

`Sparse` is the general production route for large systems with sparse but not necessarily narrow Jacobians. It is a safe default for chemistry-like systems, irregular coupling, and cases where you do not want to assert banded structure too early.

`Banded` is the right route when nonzeros are concentrated near the diagonal. This is common after finite-difference discretization of first-order BVP/PDE-like systems: each node is coupled mostly to its neighbors. RustedSciThe's Banded route uses a LAPACK-style banded LU solver, so both storage and factorization exploit the bandwidth. On current combustion BVP story tests, Banded consistently reduces linear-solve time compared with Sparse when the problem really is narrow-band.

The warning is simple: Banded is excellent when the structure is truly banded. If the problem has long-range or irregular coupling, forcing Banded can hurt both robustness and performance.

## 5. Three backend routes: Numerical, Lambdify, AOT

For BVP Damped/Frozen, it helps to separate the Newton method from the evaluator backend. The nonlinear algorithm can remain the same while residual and Jacobian evaluation take different routes.

### Numerical path

The Damped Numerical path bypasses symbolic assembly. You provide a Rust RHS closure `f(x, y, params)`, and [`numeric_discretization.rs`](numeric_discretization.rs) builds the discrete residual directly from that closure. This route is meant for models whose source of truth is already Rust code, not `Expr`.

There are now two explicit Damped numerical entry points. `NRBVP::new_numeric_fd_with_options(...)` accepts the RHS closure and asks the solver to approximate the Newton Jacobian by finite differences. `NRBVP::new_numeric_with_jacobian_options(...)` accepts both the RHS closure and the small continuous Jacobian closure `df/dy`; the large discretized BVP Jacobian is assembled internally from that local Jacobian, the mesh, and the boundary-condition layout. This is the important ergonomic difference from the older low-level pattern: users no longer need to pass an empty `eq_system` just to say "this is a pure numeric problem".

Boundary handling follows the same first-order BVP rule as the symbolic route: provide exactly as many fixed endpoint values as there are state variables. They may be attached to different variables or to the same variable at two endpoints, as long as the resulting reduced system is square. A common two-point setup is:

```rust
let border_conditions = HashMap::from([(
    "y".to_string(),
    vec![(0usize, 0.0), (1usize, 1.0)],
)]);
```

Here `z = y'` is still part of the state vector and still appears in the RHS and Jacobian closures, but it has no direct endpoint value.

For the common sparse case there are shorter wrappers, `NRBVP::new_numeric_fd(...)` and `NRBVP::new_numeric_with_jacobian(...)`. They still ask for `bounds` and per-variable `rel_tolerance`, because those are not optional decoration in the damped Newton algorithm: bounds limit the admissible step and tolerances define the weighted convergence test. If you need Banded, custom nonlinear strategy parameters, or custom logging, use the `*_with_options` constructors.

Frozen is different. `BackendSelectionPolicy::NumericOnly` is intentionally rejected for the Frozen solver rather than silently falling back to Lambdify or finite differences. Frozen currently expects a symbolic Lambdify/AOT route with a prepared Jacobian callback.

### Lambdify

Lambdify means symbolic equations (`Expr`) are assembled into the BVP residual/Jacobian and lowered into executable Rust callbacks without an external compiler. It is the best correctness baseline: low friction, no dynamic artifact build, no `gcc`, `tcc`, or `zig` dependency.

Lambdify remains the best first run for a new formulation: at that stage the priority is checking equations, boundary conditions, and the initial guess without depending on an external toolchain. For small one-off solves it often wins end-to-end. After the `AtomView` optimizations, however, this should not be generalized to large problems: in the measured `n_steps = 1000` combustion BVP, cold `tcc` AOT beat Lambdify on both Sparse and Banded routes.

### AOT

AOT means ahead-of-time generated backend. Symbolic expressions are lowered through an intermediate representation, code is generated, a separate artifact is compiled, and the solver links that artifact as runtime callbacks. The philosophy is to pay preparation cost once and reduce per-call cost later.

RustedSciThe supports several AOT toolchains: C through `gcc` or `tcc`, Zig, and a Rust backend in the codegen layer. The corresponding compiler must be installed and visible in `PATH`. In current Windows release story runs, `tcc` is more than a cheap artifact compiler: on large `AtomView` problems it is competitive with Lambdify in full time-to-solution and, in several cold measurements, already wins. `gcc` can be attractive for runtime throughput but usually pays a higher startup price, while Zig sometimes has high or unstable build cost. Treat this as measured guidance, not a universal law; recheck ranking for another machine or equation structure.

AOT is unquestionably useful when you solve the same large symbolic problem repeatedly: parameter sweeps, continuation, batch runs, or production workflows where artifacts are reused. The boundary is friendlier than expected, though: with `AtomView` and `tcc`, AOT can be a reasonable choice even for one large solve, especially on the Banded route. For a small BVP, artifact preparation will still usually not pay for itself.

## 6. ExprLegacy and AtomView

The symbolic BVP pipeline has two assembly backends.

`ExprLegacy` is the older, established expression route. It remains valuable as a compatibility baseline.

`AtomView` is the newer route designed for efficient assembly and code generation. Separate process-isolated release measurements on the combustion BVP family show that it removes a particularly expensive symbolic-Jacobian construction cost in both Sparse and Banded routes while preserving the solution to the expected numerical tolerance. For that reason, the Sparse and Banded production presets for both Damped and Frozen solvers now select `AtomView` by default.

For a banded Lambdify solve the production spelling is now intentionally short:

```rust
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;

let options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify();

// Compatibility/control route when reproducing an ExprLegacy run:
let legacy_options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify()
    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
```

The same default is inherited by `FrozenSolverOptions::{sparse_frozen, banded_frozen}()` and by the Sparse/Banded AOT presets. `ExprLegacy` remains a supported explicit override for validation, compatibility runs, and investigation of frontend-sensitive behavior.

## 7. Generated backend config: Lambdify, AOT, and build policy

`GeneratedBackendConfig` is the central generated-backend configuration object. It stores backend selection policy, AOT build policy, codegen backend, C compiler, symbolic assembly backend, matrix backend override, and chunking policy.

High-level modes:

```rust
SparseGeneratedBackendMode::Defaults
SparseGeneratedBackendMode::RequirePrebuilt
SparseGeneratedBackendMode::BuildIfMissingRelease

BandedGeneratedBackendMode::Defaults
BandedGeneratedBackendMode::Lambdify
BandedGeneratedBackendMode::BuildIfMissingRelease
```

`Defaults` prefers compiled AOT when an artifact is already available and otherwise falls back to Lambdify. For both Sparse and Banded production modes it now also means `AtomView` symbolic assembly; Banded additionally selects the faithful LAPACK-style banded linear solver. `RequirePrebuilt` is useful in production because a missing artifact becomes a clear typed error instead of an unexpected build. `BuildIfMissingRelease` is convenient during development and experiments: the first run builds the artifact if needed.

Repeated-solve presets:

```rust
let sparse_aot = GeneratedBackendConfig::sparse_atomview_for_repeated_solves();
let banded_aot = GeneratedBackendConfig::banded_atomview_for_repeated_solves();
```

Explicit toolchain choices:

```rust
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc();
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
let cfg = GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig();
```

`BuildIfMissing` exists so artifacts can be reused. The first run may pay for generation and compilation; later runs can use the compiled backend. In CI or deployment, use `RequirePrebuilt` when implicit compilation would be a bug.

A useful workflow for a new large symbolic problem has two passes. First run `AtomView + Lambdify` to establish a simple solution and matrix-structure control. Then compare it with `AtomView + tcc AOT` on the same `Sparse` or `Banded` route. In the recorded honest cold `combustion-1000` run, AOT `tcc/whole` completed in about `2.46 s` versus `2.84 s` for Banded Lambdify, and in about `2.68 s` versus `4.33 s` for Sparse Lambdify. This is strong enough evidence to test TCC AOT early, rather than reserving it only for late repeated-solve optimization.

There is now also a controlled warm result for the production lifecycle. In the
Damped Banded AtomView `combustion-1000` story, one setup solve built the `tcc`
artifact, and five later strict `RequirePrebuilt` solves were compared with five
Lambdify solves using alternating order and an equal five-second cooldown. Warm
AOT took `431.4 +/- 10.9 ms` versus `468.3 +/- 12.5 ms` for Lambdify, a modest
but repeatable improvement of about `7.9%`. The setup build cost means the
measured benefit pays back after roughly eight subsequent solves. This is the
intended meaning of the repeated-solve preset: not an unconditional one-shot
victory, but a compiled route that becomes useful when the same large model is
solved again.

## 8. Chunking and parallel execution: the honest measurement story

AOT can split residual and sparse Jacobian value evaluation into chunks. The relevant user-level types are `AotChunkingPolicy`, `ResidualChunkingStrategy`, `SparseChunkingStrategy`, `AotExecutionPolicy`, and `ParallelExecutorConfig`.

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use RustedSciThe::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;

let chunking = AotChunkingPolicy::with_parts(
    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
);

let exec = ParallelExecutorConfig {
    jobs_per_worker: 1,
    max_residual_jobs: None,
    max_sparse_jobs: None,
    fallback_policy: ParallelFallbackPolicy::Auto,
};

let cfg = GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
    .with_aot_chunking_policy(chunking)
    .with_aot_execution_policy(AotExecutionPolicy::Parallel(exec));
```

Current BVP combustion story tests show a nuanced result: chunking is correct and the true parallel path exists, but its value depends on scale and on the dominant stage. For medium `n_steps = 200` and `n_steps = 1000` problems, callback value evaluation can take only a few milliseconds, so parallel execution may barely move full wall clock. On the large Banded `AtomView` problem with `n_steps = 3000`, `tcc/chunk4` did execute four residual and four Jacobian jobs without fallback and reduced the hot callback intervals: residual values from about `13.8` to `7.1 ms`, and Jacobian values from about `6.7` to `1.7 ms`.

Even that large run does not make forced chunking a universal default: full cold times for `tcc/whole` and `tcc/chunk4` were statistically close, because hot callback work is only one part of the solve. This is why `Auto` fallback matters. If the chunk/job workload is too small, the runtime may intentionally use the sequential path to avoid scheduling overhead. That is not a hidden failure; it is the desired production behavior. If you want to study break-even points on your machine, use the diagnostic tests and inspect `actual_jobs`, `fallback`, `residual_ms`, `jacobian_ms`, and `linear_ms`. In production, leave chunking on `Auto` unless a measured model-specific configuration earns an explicit override.

The most direct production spelling is to leave chunking at its default and ask
the compiled backend to decide at runtime:

```rust
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
    .with_aot_execution_policy(AotExecutionPolicy::Auto);
```

After `solver.try_solver()?`, inspect the diagnostics map rather than guessing
from the requested configuration:

```rust
let stats = solver.get_statistics();
let d = &stats.diagnostics;

let auto_mode = d.get("aot.auto.execution_mode");
let auto_residual_reason = d.get("aot.auto.residual.reason");
let auto_jacobian_reason = d.get("aot.auto.sparse_jacobian.reason");
let residual_jobs = d.get("aot.runtime.residual.actual_jobs");
let jacobian_jobs = d.get("aot.runtime.sparse_jacobian.actual_jobs");
let residual_fallback = d.get("aot.runtime.residual.fallback_reason");
let jacobian_fallback = d.get("aot.runtime.sparse_jacobian.fallback_reason");
let parallel_requested = d.get("aot.runtime.parallel_requested");

println!(
    "Auto={auto_mode:?}, residual={auto_residual_reason:?}/{residual_jobs:?}/{residual_fallback:?}, \
     jacobian={auto_jacobian_reason:?}/{jacobian_jobs:?}/{jacobian_fallback:?}, \
     parallel_requested={parallel_requested:?}"
);
```

The important point is semantic, not cosmetic: `aot.auto.*` describes the plan,
while `aot.runtime.*` describes what the linked callback actually did. In the
12-core `combustion-1000` AtomView+tcc story, `Auto` chose `Sequential` for both
Sparse and Banded. The plan said `work_per_chunk_too_small` for the Jacobian
stage and `single_chunk_or_job` for the residual stage; runtime diagnostics
matched that with `actual_jobs = 1`, `parallel_requested = false`, and
`single_requested_job`. That is a successful Auto decision. Forced chunking
should only be used when a larger model, or a break-even test, proves that there
is enough callback work per job to pay for scheduling.

### Comparing cold and warm runs

`total_ms` is meaningful only when the lifecycle included in a run is clear. A cold wall-clock test starts a fresh process and includes symbolic frontend construction, code generation, compilation, linking, and the Newton solve; it answers "how long from pressing the button until a result?" A warm or `RequirePrebuilt` run starts with an existing artifact and answers a different question: "how expensive is one more solve in a series?"

Both views matter. Cold measurements protect against an AOT pipeline that has excellent callbacks but takes too long before the first result. Warm or prebuilt measurements show its value in parameter sweeps, continuation, and batch workflows. Do not compare cold Lambdify with warm AOT, or draw conclusions from a table in which one row reused an artifact while another built it.

`RequirePrebuilt` is also a correctness contract, not merely a performance
switch. In the current Damped Sparse/Banded and Frozen Sparse/Banded lifecycle
stories, strict reuse remains on `AotCompiled`, reports no compiler/linker
interval, and preserves the Lambdify solution to roundoff. A prebuilt row that
compiles again or silently falls back to Lambdify should be treated as a
regression. In particular, Frozen combustion-1000 now has release-confirmed
artifact reuse on both matrix routes; the Sparse reuse rows agree with
Lambdify to `4.44e-16`.

## 9. End-to-end example: Symbolic Lambdify + Banded

This example solves `y' = z`, `z' = -y` with `y(0)=0`, `z(0)=1`. The exact solution is `y=sin(x)`, `z=cos(x)`, which makes it a good sanity check.

```rust
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

let n_steps = 80;
let values = vec!["y".to_string(), "z".to_string()];
let eq_system = vec![
    Expr::parse_expression("z"),
    Expr::parse_expression("-y"),
];

let t0 = 0.0;
let t_end = std::f64::consts::FRAC_PI_2;
let h = (t_end - t0) / n_steps as f64;
let mut guess = Vec::with_capacity(values.len() * n_steps);
for i in 0..n_steps {
    let x = t0 + i as f64 * h;
    guess.push(x.sin());
    guess.push(x.cos());
}
let initial_guess = DMatrix::from_column_slice(
    values.len(),
    n_steps,
    DVector::from_vec(guess).as_slice(),
);

let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(0usize, 1.0)]),
]);

let bounds = HashMap::from([
    ("y".to_string(), (-2.0, 2.0)),
    ("z".to_string(), (-2.0, 2.0)),
]);

let rel_tol = HashMap::from([
    ("y".to_string(), 1e-8),
    ("z".to_string(), 1e-8),
]);

let options = DampedSolverOptions::banded_damped()
    .with_banded_lambdify()
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(rel_tol)
    .with_bounds(bounds)
    .with_max_iterations(40)
    .with_loglevel(Some("none".to_string()));

let mut solver = NRBVP::new_with_options(
    eq_system,
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
);

solver.dont_save_log(true);
solver.try_solve()?;
let result = solver.get_result().expect("BVP result should be available");
let stats = solver.get_statistics();

println!("solution shape = {} x {}", result.nrows(), result.ncols());
println!("timers = {:?}", stats.timers);
# Ok::<(), Box<dyn std::error::Error>>(())
```

This is a strong first example because it uses symbolic equations but does not require an external compiler.

## 10. End-to-end example: Numerical path

The Numerical path is useful when the RHS closure is the source of truth. The constructor below does not receive symbolic equations at all. The discrete residual is built from the Rust closure, and the Jacobian choice is explicit: either finite differences through `new_numeric_fd_with_options`, or a user-provided continuous Jacobian through `new_numeric_with_jacobian_options`.

```rust
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP,
};

let n_steps = 80;
let values = vec!["y".to_string(), "z".to_string()];
let t0 = 0.0;
let t_end = std::f64::consts::FRAC_PI_2;
let h = (t_end - t0) / n_steps as f64;

let mut guess = Vec::with_capacity(values.len() * n_steps);
for i in 0..n_steps {
    let x = t0 + i as f64 * h;
    guess.push(x.sin());
    guess.push(x.cos());
}
let initial_guess = DMatrix::from_column_slice(
    values.len(),
    n_steps,
    DVector::from_vec(guess).as_slice(),
);

let border_conditions = HashMap::from([
    ("y".to_string(), vec![(0usize, 0.0)]),
    ("z".to_string(), vec![(0usize, 1.0)]),
]);

let bounds = HashMap::from([
    ("y".to_string(), (-2.0, 2.0)),
    ("z".to_string(), (-2.0, 2.0)),
]);

let rel_tol = HashMap::from([
    ("y".to_string(), 1e-8),
    ("z".to_string(), 1e-8),
]);

let options = DampedSolverOptions::sparse_damped()
    .with_abs_tolerance(1e-8)
    .with_rel_tolerance(rel_tol)
    .with_bounds(bounds)
    .with_loglevel(Some("none".to_string()));

let mut solver = NRBVP::new_numeric_fd_with_options(
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
    |_x, y, _params| {
    DVector::from_vec(vec![y[1], -y[0]])
    },
);

solver.dont_save_log(true);
solver.try_solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

If you have an analytical local Jacobian, use the second numerical constructor. The closure returns the `2 x 2` continuous Jacobian of `f(x, y)` at one mesh node; the solver handles the global BVP matrix.

```rust
let mut solver = NRBVP::new_numeric_with_jacobian_options(
    initial_guess,
    values,
    "x".to_string(),
    border_conditions,
    t0,
    t_end,
    n_steps,
    options,
    |_x, y, _params| DVector::from_vec(vec![y[1], -y[0]]),
    |_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]),
);
```

This route is intentionally free of symbolic semantics. If it fails, the likely source is numeric discretization, Newton choreography, or linear algebra rather than expression parsing or codegen.

## 11. End-to-end example: AtomView + AOT + Banded

For a large structured BVP that will be solved repeatedly, the typical production candidate is Banded + AtomView + AOT. The first run may be expensive because it builds the artifact, but repeated runs reuse compiled callbacks and the LAPACK-style banded linear solver.

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;

let generated = GeneratedBackendConfig::banded_atomview_for_repeated_solves();

let options = DampedSolverOptions::banded_damped()
    .with_generated_backend_config(generated)
    .with_abs_tolerance(1e-8)
    .with_max_iterations(60)
    .with_loglevel(Some("none".to_string()));
```

Explicit toolchain selection:

```rust
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc();
let generated = GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig();
```

Production runs can require a prebuilt artifact:

```rust
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, GeneratedBackendConfig,
};

let generated = GeneratedBackendConfig::banded_atomview_for_repeated_solves()
    .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
```

`RequirePrebuilt` surfaces a typed error when the artifact is missing. That is usually preferable to an unexpected compilation step in production.

## 12. Practical backend recommendations

When writing a new BVP model, start with Lambdify. It is the simplest way to check equations, boundary conditions, bounds, and initial guesses.

If your model already exists as Rust code, use the Numerical path. Choose `new_numeric_fd_with_options` when maintaining an analytical Jacobian is not worth it, and choose `new_numeric_with_jacobian_options` when the continuous `df/dy` is available and you want to avoid finite-difference noise. For very stiff or badly scaled symbolic models, Lambdify/AOT may still be preferable because they give a mechanically derived Jacobian from the `Expr` source of truth.

If the problem is large and symbolic, test AOT with `tcc` alongside Lambdify rather than waiting until a repeated-solve use case appears. On the corrected `AtomView` path, TCC AOT can already match or beat Lambdify in honest cold end-to-end measurements. For local-stencil BVP/PDE-like problems, test Banded early; for irregular sparse coupling, Sparse is safer. For a single small solve, still begin with Lambdify because build/bootstrap can easily consume any runtime saving.

If the model will be solved repeatedly, distinguish artifact creation from
artifact use explicitly: create or validate it with `BuildIfMissing`, and use
`RequirePrebuilt` in the measured/production loop. On the measured Damped
Banded combustion-1000 case this warm route is about `1.09x` faster per solve
than Lambdify, but requires several reuses before the first build is amortized.

When selecting an AOT toolchain, `tcc` is now the sensible first candidate: it delivered the best practical cold profile in our large BVP runs. `gcc`, Zig, and the Rust backend remain important platform-dependent alternatives and controls, but should not silently be treated as the performance default.

For chunking, do not assume parallel is better. Measure. Inspect callback value time, matrix assembly time, linear solve time, total solver time, actual jobs, and fallback decisions. Current measurements establish both facts: real parallel execution exists, and on Banded `n_steps = 3000` it materially accelerates callback values; full cold wall clock can nevertheless remain nearly unchanged. Therefore `Auto`, rather than forced `chunk4`, is the appropriate library default.

## 13. Reading statistics and story tables

`solver.get_statistics()` returns counters and timers. This interface is available on both the Damped and Frozen solvers. Frozen deliberately remains a different nonlinear strategy, but it now reports the same generated-backend facts needed for honest Lambdify/AOT comparisons: selected backend, symbolic frontend, handoff preparation stages, callback diagnostics, Newton iterations, Jacobian rebuilds, and linear solves. In BVP story tests, the most useful entries are:

`linear_ms` for Newton linear solves, `jac_ms` for Jacobian preparation/evaluation at solver level, `fun_ms` for residual evaluation, `Callback Residual Values` and `Callback Jacobian Values` for low-level callback arithmetic, `Callback Jacobian Matrix Assembly` for matrix construction from values, and counters such as `iters`, `linsys`, and `jac_re`.

For generated AOT routes the same statistics object also carries a `diagnostics` map. The most useful keys are `generated.selected_backend`, `generated.handoff.initial_generate_wall_ms`, `generated.handoff.build_policy_wall_ms`, `generated.handoff.post_build_rebind_wall_ms`, `aot.runtime.execution_policy`, `aot.runtime.parallel_requested`, `aot.runtime.residual.actual_jobs`, `aot.runtime.residual.fallback_reason`, `aot.runtime.sparse_jacobian.actual_jobs`, `aot.runtime.sparse_jacobian.fallback_reason`, and the corresponding `work_per_job` / `work_per_chunk` entries. These fields are intentionally runtime facts, not just requested configuration: if `Auto` or a threshold policy falls back to sequential execution, the statistics will say so directly.

Older tables may still show `Symbolic Operations`; newer code also exports the same timer as `Backend Preparation`. The alias is deliberate. In a Lambdify/AOT route the stage is genuinely symbolic/codegen preparation, while in Numerical path it is numeric discretization and backend handoff rather than symbolic work.

The current combustion BVP measurements suggest the following working picture. For `n_steps = 200`, cold AOT can still lose time to preparation. For `n_steps = 1000`, after switching to `AtomView`, cold TCC AOT beat Lambdify in the recorded comparison on both Sparse and Banded routes; in a separate cooldown-controlled warm Damped Banded run, strict prebuilt TCC also beat Lambdify by about `7.9%`. For Banded `n_steps = 3000`, the production `AtomView` frontend preserved the solution to roundoff and placed TCC AOT at roughly equal or better full cold wall-clock time, although the Lambdify row in that experiment was noisy. Banded remains the natural candidate for a narrow Jacobian band, while chunking should be selected by `Auto` or by a dedicated measurement: it improves hot callback work but does not guarantee proportional whole-solve acceleration.

These are engineering conclusions, not marketing promises. Re-run the story tests on your own hardware when changing toolchain, thread count, mesh size, or equation structure.

## 14. Runnable guide examples

The `examples/` directory contains small executable companions to this chapter. Each example solves a boundary-value problem with an analytical reference solution, so it demonstrates not only the configuration syntax but also an immediate correctness check. The Damped examples cover the pure numerical route, symbolic Lambdify and generated AOT; the Frozen examples cover its supported symbolic Lambdify and AOT routes. The separate Frozen numerical-route note explains why a closure-defined problem should be routed through Damped rather than represented by an empty symbolic placeholder.

```powershell
cargo run --example bvp_damped_numerical_guide
cargo run --example bvp_damped_lambdify_guide
cargo run --example bvp_damped_aot_guide
cargo run --example bvp_frozen_lambdify_guide
cargo run --example bvp_frozen_aot_guide
cargo run --example bvp_frozen_numerical_route_guide
```

The AOT examples deliberately demonstrate both phases of the artifact lifecycle: a first `BuildIfMissing` solve followed by a `RequirePrebuilt` solve. They require `tcc` on `PATH`; if it is not installed, they report that the AOT demonstration was skipped instead of disguising a toolchain failure as a solver failure.

## 15. Where to look next

Basic correctness tests live in `BVP_Damp_tests.rs`. Heavier story and performance runs live in `BVP_Damp_tests3.rs` and `BVP_Damp_tests4.rs`. The running story-test ledger is [`BVP_DAMP_STORY_TESTS.md`](BVP_DAMP_STORY_TESTS.md).

The lower-level codegen/performance layer is documented in [`../../symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md`](../../symbolic/codegen/tests/BVP_CODEGEN_STORY_TESTS.md). Use it when you need to distinguish solver-loop cost from generated callback cost.

Prefer `try_solve()` in production code. The older `solve()` wrapper is kept for compatibility and panics on failure, while `try_solve()` returns typed errors that can be logged, retried, or surfaced to users.

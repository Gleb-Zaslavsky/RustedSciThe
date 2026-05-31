# IVP User Guide (RustedSciThe)

This guide is for a Rust developer who wants to solve initial value problems in RustedSciThe without reverse-engineering the crate from source files. The goal is practical: understand what each solver family does, how backend routing changes behavior and performance, and how to choose a configuration that is correct first and fast second.

If you already use LSODE2, keep this in mind from the start. LSODE2 has its own dedicated guide and its own algorithmic story (LSODE/LSODA mirroring, Adams/BDF switching telemetry, parity gates). In this document we focus on the other IVP paths: Native BDF, Radau, Backward Euler, and the non-stiff explicit family exposed through the universal facade.

## 1. Solver Landscape in This Guide

RustedSciThe exposes two levels of IVP API.

The first level is solver-specific APIs. You work directly with `BDF::ODEsolver`, `Radau`, or `BE`, configure their options in detail, and call them explicitly. This gives maximum control and is often preferable in production code where behavior should be fully explicit.

The second level is `UniversalODESolver` from [`ODE_api2.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/ODE_api2.rs). This is a unified facade that allows you to switch methods without rewriting your setup code. It is ideal for experiments, comparative tests, and orchestration pipelines where method selection is data-driven.

The methods discussed in this guide are:

- stiff-focused: BDF, Radau IIA, Backward Euler;
- non-stiff explicit: RK45, Dormand–Prince (`DOPRI`), Adams–Bashforth 4 (`AB4`).

About RK4: a standalone “classic fixed-step RK4” entrypoint is not currently exposed as a dedicated method in `UniversalODESolver`. In practice, users typically choose RK45 or DOPRI in that slot, and AB4 when they want a multi-step explicit family.

## 2. Stiff vs Non-Stiff: Practical Selection

In many real systems the mathematically hardest part is not writing equations, but choosing the right integration family before numerical pathologies appear.

When dynamics include widely separated time scales, fast transient relaxation, or strongly dissipative chemistry-like terms, start from a stiff method. BDF is usually the default workhorse for large stiff runs, Radau is often the best choice when robust high-order implicit Runge–Kutta behavior matters, and Backward Euler is the conservative baseline for “must converge” coarse implicit marching.

When trajectories are smooth and stiffness indicators are absent, explicit methods usually give lower setup overhead and simpler tuning. RK45 is a strong default for general non-stiff use, DOPRI is often used in the same niche, and AB4 can be useful when you want explicit multistep behavior.

If you are uncertain, the universal API makes it cheap to test multiple methods on the same task definition and compare final error, number of calls, and runtime statistics.

## 3. Backend Philosophy: Numerical, Lambdify, AOT

For stiff solvers in RustedSciThe, “method” and “backend route” are separate decisions. This separation is central and worth understanding.

`Numerical` means you provide callbacks directly: residual `f(t, y)` and optionally Jacobian `df/dy`. If Jacobian is omitted, finite differences are used. This route is ideal when you already own a trusted model implementation, or when symbolic preprocessing is not wanted.

`Lambdify` means equations are symbolic (`Expr`), then transformed into executable closures during preparation. This route has low operational friction and avoids external compiler dependencies. It is often the best baseline for correctness and developer speed.

`AOT` (ahead-of-time generated backend) means symbolic expressions are converted to generated code, compiled, and linked as artifacts. You pay setup cost and usually gain lower per-call cost in long runs or repeated solves. This route is a performance engineering tool, not merely a syntax variant.

The important conceptual point is that all three routes feed the same numerical integrator. You are not changing mathematics, you are changing evaluator infrastructure.

### 3.1 LSODE2 backend configuration syntax

LSODE2 has its own dedicated guide, but one API convention is important for IVP
users because examples may show two similar-looking styles:

```rust
.with_backend(
    Lsode2BackendConfig::native_banded_faithful()
        .with_generated_backend_target_chunks(4, 4),
)
```

and:

```rust
.with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
```

They are not interchangeable. `with_backend(...)` is the high-level route
selector. It describes the complete LSODE2 backend preset: dense/sparse/banded
matrix structure, concrete linear algebra backend, symbolic/generated backend
lifecycle, AOT toolchain, and generated callback chunking knobs. This is the
recommended syntax for user-facing examples and production code because the
whole execution route is visible in one place.

```rust
let config = base.with_backend(
    Lsode2BackendConfig::native_sparse_faer()
        .with_generated_backend_target_chunks(4, 4),
);
```

`with_linear_solver_policy(...)` is lower-level. It only controls how the linear
solver is selected after the linear system structure is already known. `Auto`
maps dense systems to dense LU, sparse systems to faer sparse LU, and banded
systems to the faithful LAPACK-style banded LU. `Force(...)` is useful in parity
tests, diagnostics, and legacy compatibility examples where you explicitly want
to prove that a particular linear solver was selected.

Practical rule: use `with_backend(...)` when you are choosing a real solver
route; use `with_linear_solver_policy(...)` when a test or diagnostic is about
the linear-policy resolver itself.

## 4. UniversalODESolver: One Surface, Many Methods

`UniversalODESolver` lets you keep one integration shell and vary method/backend policy with minimal code movement.

### 4.1 Minimal explicit example (non-stiff)

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

let eq = vec![Expr::parse_expression("-y")];
let vars = vec!["y".to_string()];

let mut solver = UniversalODESolver::rk45(
    eq,
    vars,
    "t".to_string(),
    0.0,
    DVector::from_vec(vec![1.0]),
    1.0,
    1e-4,
);

solver.solve();
let (t, y) = solver.get_result();
println!("status = {:?}", solver.get_status());
println!("grid points = {}", t.unwrap().len());
println!("y(1) = {}", y.unwrap()[(y.unwrap().nrows() - 1, 0)]);
```

### 4.2 Native stiff path with user Jacobian (BDF)

```rust
use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

let eq = vec![
    Expr::parse_expression("-1000.0*x0"),
    Expr::parse_expression("-x1"),
];
let vars = vec!["x0".to_string(), "x1".to_string()];

let mut solver = UniversalODESolver::bdf(
    eq,
    vars,
    "t".to_string(),
    0.0,
    DVector::from_vec(vec![1.0, 1.0]),
    0.02,
    1e-4,
    1e-7,
    1e-9,
)
.with_native_ode_callbacks(
    |_t, y| DVector::from_vec(vec![-1000.0 * y[0], -y[1]]),
    Some(|_t, _y| DMatrix::from_row_slice(2, 2, &[-1000.0, 0.0, 0.0, -1.0])),
);

solver.solve();
if let Some(stats) = solver.get_statistics() {
    println!("{}", stats.table_report());
}
```

### 4.3 Native stiff path with finite-difference Jacobian fallback

The only change is Jacobian argument:

```rust
.with_native_ode_callbacks(
    |_t, y| DVector::from_vec(vec![-1000.0 * y[0], -y[1]]),
    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
);
```

This is often convenient for quick model iteration, but analytical Jacobian remains preferable for heavy stiff workloads.

## 5. Solver-Specific APIs (Direct Control)

In production, many teams prefer direct APIs because the method boundary is explicit in code review and refactoring.

### 5.1 Native BDF API

Core module: [`BDF_api.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/BDF/BDF_api.rs).

Typical flow is:

1. build `BdfSolverOptions`;
2. create solver via `ODEsolver::new_with_options(...)`;
3. optionally install native callbacks with `set_native_ode_callbacks(...)`;
4. solve and inspect results/statistics.

`BDF` supports symbolic routes with generated backend config and also pure numerical callbacks. If native Jacobian callback is omitted, finite differences are used.

### 5.2 Radau API

Core module: [`Radau_main.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/Radau/Radau_main.rs).

Radau uses `RadauSolverOptions` with order (`Order3/Order5/Order7`), tolerances, and generated backend config. Native callbacks are available through `set_native_ode_callbacks(...)`, again with finite-difference fallback when Jacobian is absent.

Radau has rich runtime counters (Newton solves, Jacobian calls, LU usage), which makes it suitable for serious diagnostics, not only for “did it converge”.

### 5.3 Backward Euler API

Core module: [`BE.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/BE.rs).

`BE` is the simplest stiff implicit baseline in this family. It supports generated backends and pure numerical callbacks, including finite-difference Jacobian fallback. It is valuable when you need robust coarse stepping and transparent Newton behavior.

## 6. Backends in Stiff Solvers: What to Choose and When

### 6.1 Lambdify baseline

Use Lambdify first when:

- you are validating a new model;
- you need low setup friction;
- you want fewer infrastructure moving parts while debugging.

It gives a strong correctness baseline and usually makes backend regressions easier to isolate.

### 6.2 AOT route

Use AOT when:

- the same model is solved repeatedly;
- solve loop dominates total workflow;
- you can amortize compile/link setup over long runtime.

In practice, RustedSciThe supports four dense AOT targets in this IVP stack: `C+tcc`, `C+gcc`, `Zig`, and `Rust`. For quick local iteration, `C+tcc` is often the most comfortable. For repeated production solves, `C+gcc` is a frequent default because runtime throughput is usually better. `Zig` and `Rust` are fully valid routes and useful in environments where those toolchains are preferred.

Build lifecycle is controlled by `DenseIvpGeneratedBackendMode` and the underlying `SymbolicIvpAotBuildPolicy`. The common modes are:

- `BuildIfMissingRelease`: build once (if needed), then reuse artifacts;
- `RequirePrebuilt`: fail fast unless the artifact is already built;
- `Defaults`/`UseIfAvailable`: prefer prebuilt artifacts but keep lambdify fallback.

The low-friction API is solver-facing:

```rust
// BDF-style fluent route selection:
let solver = ODEsolver::new_with_options(opts)
    .with_dense_generated_backend_c_tcc("target/generated-ivp-tests")
    .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);
```

When you need explicit control (policy, backend, output path), configure `SymbolicIvpGeneratedBackendConfig` directly:

```rust
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpGeneratedBackendConfig,
};

let gen_cfg = SymbolicIvpGeneratedBackendConfig::from_mode(
    DenseIvpGeneratedBackendMode::BuildIfMissingRelease
)
.with_c_gcc()
.with_output_parent_dir(Some("target/generated-ivp-tests".into()));
```

AOT chunking is configured through `SymbolicIvpAotOptions` (residual + dense Jacobian chunking strategies). This is where parallel decomposition is defined:

```rust
use RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};

let chunked = SymbolicIvpAotOptions {
    residual_strategy: ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 },
    jacobian_strategy: DenseJacobianChunkingStrategy::ByTargetChunkCount { target_chunks: 2 },
};

let gen_cfg = gen_cfg.with_aot_options(chunked);
```

Operationally, treat AOT in two phases: first warm up the artifact (`build_if_missing`), then benchmark steady-state solve time separately. Otherwise build noise will mask method/backend differences.

### 6.3 Native numerical route

Use native callbacks when:

- equations already exist in optimized Rust callbacks;
- symbolic preparation is unnecessary or undesirable;
- you need exact runtime control over Jacobian logic.

This is also the cleanest route for solver benchmarking against external references where symbolic preprocessing should be excluded.

## 7. Statistics: Treat Them as First-Class Outputs

One of the strongest practical features in this stack is built-in statistics reporting. Instead of wrapping solvers with ad-hoc timers, you can extract call counters and durations from native instrumentation.

For `UniversalODESolver`, use:

```rust
if let Some(stats) = solver.get_statistics() {
    println!("{}", stats.table_report());
}
```

For direct solver APIs, use their native statistics accessors (`get_statistics()` where available). In stiff studies, you usually care about at least residual calls, Jacobian calls, LU counts, nonlinear iterations, and aggregate setup/solve timing.

## 8. End-to-End Examples in This Repository

If you want runnable scenarios instead of snippets, start with these files:

- [`bdf_solver_examples.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/bdf_solver_examples.rs): multiple BDF end-to-end cases, including stiff and nonlinear systems.
- [`universal_ode_example.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/universal_ode_example.rs): side-by-side usage of RK45, DOPRI, Radau, BDF, and BE through one facade.
- [`radau_backends_guide.rs`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/examples/radau_backends_guide.rs): backend-oriented Radau usage patterns.

For LSODE2-specific workflows, read:

- [`LSODE2_USER_GUIDE_EN.md`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/LSODE2/LSODE2_USER_GUIDE_EN.md)
- [`LSODE2_USER_GUIDE_RU.md`](/f:/RUST/RustProjects_/RustedSciThe_experimental4/src/numerical/LSODE2/LSODE2_USER_GUIDE_RU.md)

## 9. Practical Decision Sequence

A stable production routine for new IVP models usually looks like this.

First, get correctness with Lambdify and conservative tolerances. Second, validate stiff vs non-stiff family behavior and inspect statistics. Third, choose native callbacks or AOT only when data shows a bottleneck worth optimization. Finally, freeze method/backend policy in tests so future changes cannot silently shift numerical behavior.

This sequence sounds boring, but it prevents most expensive debugging cycles in real projects.

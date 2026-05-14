# LSODE2 in RustedSciThe: user guide

This guide is written for a Rust developer who opens LSODE2 for the first time and wants to move quickly from “what is this solver?” to a practical, reproducible configuration. We will not only list options, but explain why those options exist, what trade-offs they encode, and when one route is better than another.

It helps to set the scope up front. LSODE2 in RustedSciThe combines two goals: it mirrors the ODEPACK family philosophy (especially LSODA-like Adams/BDF auto-switching), and it integrates with the modern symbolic/codegen/linear-algebra pipeline of the crate. So LSODE2 has an algorithmic side and an infrastructure side. If you mix them mentally, configuration feels chaotic; if you separate them, the system becomes straightforward.

## Where to start: two decision axes

Most LSODE2 setups can be described along two axes. The first axis is step mathematics: fixed BDF, fixed Adams, or LSODA-like automatic family switching. The second axis is backend execution: how residual/Jacobian are evaluated and how Newton linear systems are solved.

That second axis leads to the three familiar routes: pure numerical (`Numerical`), symbolic lambdify (`Lambdify`), and symbolic ahead-of-time generation (`AOT`). The key idea is that these are not three unrelated solvers; they are three ways to feed the same integration algorithm with high-quality function and matrix evaluators.

## What `Lsode2ProblemConfig::new` takes, and why it is the core syntax

`Lsode2ProblemConfig::new(...)` is the central entry point. Its signature is:

```rust
pub fn new(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    max_step: f64,
    rtol: f64,
    atol: f64,
) -> Self
```

You can read it as: “here is my ODE system, my state-variable names, my independent variable, my initial state/time, my integration horizon, and my baseline step/accuracy policy.”

`eq_system` is the RHS in symbolic form (`Expr`).  
`values` are state-variable names in the same order as `y0`.  
`arg` is typically `"t"`.  
`t0`, `y0`, `t_bound` are the usual initial-value triple.  
`max_step`, `rtol`, `atol` control step ceiling and error tolerances.

Even if you later switch to fully analytical callbacks, this constructor remains the backbone: it defines the global geometry and tolerance policy of the solve.

## Dense, Sparse, Banded: not style, but cost model

Jacobian structure is a computational economics choice.

`Dense` is natural for small systems, debugging, and cases where simplicity matters most.  
`Sparse` is the practical default for large unstructured systems: less memory, cheaper factorization.  
`Banded` is ideal when nonzeros are concentrated near the diagonal, common in discretized transport/diffusion and many BVP/PDE-like systems.

If the structure is wrong, the solver may still run, but performance usually degrades and stability can suffer. In practice, structure is the first thing to get right, before any tolerance micro-tuning.

## LSODE vs LSODA: historical context and LSODE2 behavior

In classic Fortran ODEPACK, LSODE and LSODA are not the same thing. LSODE assumes method-family choice is user-fixed (Adams or BDF). LSODA adds automatic stiffness sensing and method switching.

LSODE2 follows that philosophy explicitly. A manual controller (`bdf_only` or `adams_only`) is LSODE-style behavior. `automatic_adams_bdf` is LSODA-style behavior under the same modern API.

## Core configuration types and their roles

### `Lsode2BackendConfig`

This is the low-level backend container. It groups Jacobian backend (`jacobian_backend`), linear backend (`linear_solver_backend`), and generated symbolic/AOT backend settings (`generated_backend`).

Most users should not build it from scratch every time. Presets like `native_sparse_faer()`, `native_banded_faithful()`, or `dense_aot_c_gcc(...)` are usually the right starting point, followed by minimal targeted overrides.

### `Lsode2JacobianBackend`

`SymbolicGenerated` means Jacobian comes from the symbolic pipeline (Lambdify or AOT).  
`AnalyticClosure` means user-provided analytical callbacks.  
`FiniteDifference` means built-in finite-difference Jacobian.

Important current constraint: `AnalyticClosure` and `FiniteDifference` are valid on native solve paths; bridge mode is not designed for those routes.

### `Lsode2LinearSolverBackend`

`Dense` uses dense LU,  
`SparseFaer` uses sparse LU via `faer`,  
`BandedFaithful` uses faithful LAPACK-style banded LU.

At API level, it is often better to specify structure plus policy (`Auto`/`Force`) rather than pinning this backend directly.

## Linear-solver policy: Auto vs Force

`Lsode2LinearSolverPolicy::Auto` is deterministic, not heuristic guessing. It maps from `Lsode2LinearSystemStructure`: dense to `DenseLu`, sparse to `FaerSparseLu`, banded to `LapackFaithfulBandedLu`.

`Force(...)` is useful for targeted experiments, parity diagnostics, and backend-specific performance investigations.

In production, “set structure first, use Auto second” is usually the most robust and reproducible pattern.

## Symbolic assembly backends: `ExprLegacy` and `AtomView`

`Lsode2SymbolicAssemblyBackend` has two options: `ExprLegacy` and `AtomView`.

`ExprLegacy` is the conservative baseline.  
`AtomView` is a newer packed symbolic representation (an internal IR choice) that can improve preparation and/or runtime in some workloads.

IR should be stated plainly. IR (intermediate representation) is the internal format between mathematical expressions and executable code. In LSODE2, IR separation lets you change execution strategy without rewriting the model equations.

## Lambdify and AOT in practical terms

In LSODE2, Lambdify builds executable Rust closures from symbolic equations during solver preparation. It uses no external compiler, starts fast, and is easy to keep portable.

AOT (ahead-of-time) makes a different trade. You pay an upfront cost for code generation, compilation, and linking, then reduce per-call runtime cost for residual/Jacobian. That is why AOT shines for repeated solves of the same model or long heavy integrations.

In RustedSciThe, AOT supports several toolchains: C with `gcc`, C with `tcc`, `zig`, and Rust toolchain. The corresponding compilers must be available in your environment.

`Debug` vs `Release` profile keeps the standard meaning. Debug usually compiles faster but runs slower; Release compiles longer but lowers runtime cost inside the integration loop.

## AOT parallelism/chunking and performance

Generated backend settings include `aot_options`, where residual/Jacobian chunking strategies are defined. This influences function size, compile behavior, and runtime dispatch plan. Residual strategies include `Whole`, `ByTargetChunkCount`, `ByOutputCount`; dense Jacobian strategies include `Whole`, `ByTargetChunkCount`, `ByRowCount`.

At LSODE2 level, this is configured through `SymbolicIvpGeneratedBackendConfig`:

```rust
use RustedSciThe::numerical::LSODE2::{Lsode2BackendConfig, Lsode2ProblemConfig};
use RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};

let aot_options = SymbolicIvpAotOptions {
    residual_strategy: ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 },
    jacobian_strategy: DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 32 },
};

let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-aot")
    .with_c_gcc()
    .with_aot_options(aot_options);

let cfg = Lsode2ProblemConfig::new(/* ... */)
    .with_backend(
        Lsode2BackendConfig::native_sparse_faer_with_generated_backend(generated)
    );
```

This is usually advanced tuning. A good workflow is to establish a correct baseline first, then optimize chunking.

## Why there are three `with_stop_condition_*` variants

Early-stop control is valuable not only for convenience but for physical realism in models with a natural completion state.

`with_stop_condition(...)` is shorthand for `variable >= target`.  
`with_stop_condition_ge(...)` is explicit `>=`.  
`with_stop_condition_le(...)` is explicit `<=`.  
`with_stop_condition_abs(...)` stops when `|variable - target| <= tolerance`.

In combustion-like IVPs this is especially useful: you can stop at conversion 0.999 instead of integrating deep into a formally allowed but physically irrelevant tail.

## What a task document is, and how LSODE2 runs through the parser

RustedSciThe includes a command interpreter that parses a human-readable task document into a typed `IvpTaskSpec`, then builds a `UniversalODESolver`.

That is the task-document path: you describe the model in sections (`task`, `equations`, `initial_conditions`, `solver_options`, `postprocessing`), and the parser normalizes and validates fields. LSODE2-specific keys include `lsode2_symbolic_assembly`, `lsode2_symbolic_execution`, `lsode2_aot_toolchain`, `lsode2_aot_profile`, `lsode2_linear_structure`, `lsode2_linear_solver_policy`, `lsode2_native_execution`, and related options.

For outputs, parser-side postprocessing is intentionally conservative today: CSV export is built-in; `plot` is parsed and preserved for higher-level wrappers/frontends, but plotting is not auto-triggered in parser core.

## Complete Rust example (Lambdify, sparse, faithful BDF)

Below is a full minimal scenario from equation definition to status/statistics output.

```rust
use nalgebra::DVector;
use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-10.0*y1 + 9.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();

    let status = solver.get_status().unwrap_or_else(|| "unknown".to_string());
    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|tv| tv[tv.len() - 1]).unwrap_or(f64::NAN);
    let final_y1 = y.as_ref().map(|m| m[(m.nrows() - 1, 0)]).unwrap_or(f64::NAN);
    let final_y2 = y.as_ref().map(|m| m[(m.nrows() - 1, 1)]).unwrap_or(f64::NAN);

    println!("status  = {status}");
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");

    if let Some(stats) = solver.get_statistics() {
        println!("{}", stats.table_report());
    }
}
```

## Complete task document example

This format is useful when the model is provided from CLI, scripts, or external orchestration.

```text
task
solver: IVP
method: LSODE2

equations
arg: t
y1: -10.0*y1 + 9.0*y2
y2: y1 - y2

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0, 0.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: ExprLegacy
lsode2_symbolic_execution: AOT
lsode2_aot_toolchain: c_gcc
lsode2_aot_profile: release
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
lsode2_native_execution: faithful_bdf_solve

postprocessing
save_csv: true
csv_path: lsode2_result.csv
plot: false
```

## Practical strategy for choosing a configuration

For small systems and early debugging, start with Dense + Lambdify. For large sparse systems, start with Sparse + Lambdify, then move to AOT and compare stage metrics (`prepare` vs `solve`) in multi-run story tests. For truly banded systems, use Banded + faithful backend with carefully validated `kl/ku`.

The sequence is deliberate: first lock mathematical correctness on the most transparent route, then transfer the same math to a more aggressive backend and check equivalence, then optimize compile/runtime balance.

## Parallelism knobs for AOT and Lambdify routes

LSODE2 exposes backend-level chunking controls at config level, so you can tune runtime work splitting without manually assembling low-level generated-backend objects.

For AOT, the highest-level entry point is:

```rust
let config = Lsode2ProblemConfig::new(/* ... */)
    // choose symbolic AOT route
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    })
    // choose matrix structure and solver policy
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // split generated residual/Jacobian runtime work by available parallelism
    .with_aot_parallel_chunking(2);
```

If you need deterministic explicit chunk counts, use:

```rust
let config = config
    .with_aot_target_chunks(8, 8);
```

If you need sparse-Jacobian-specific control, use:

```rust
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
let config = config
    .with_aot_sparse_chunking_strategy(SparseChunkingStrategy::ByTargetChunkCount {
        target_chunks: 8,
    });
```

On Lambdify route, generated runtime chunking can also be tuned through backend config (for heavy symbolic kernels this can matter on multi-core machines):

```rust
let config = Lsode2ProblemConfig::new(/* ... */)
    .with_backend(
        Lsode2BackendConfig::native_sparse_faer()
            .with_generated_backend_target_chunks(4, 4),
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    });
```

The same logic applies to Dense and Banded. Structure selection changes linear algebra backend, while symbolic execution and chunking control callback-evaluation behavior.

## Where to continue in this repository

Hands-on examples are in `examples/lsode2_numerical_guide.rs`, `examples/lsode2_lambdify_guide.rs`, `examples/lsode2_aot_guide.rs`, `examples/lsode2_manual_bdf_guide.rs`, `examples/lsode2_manual_adams_guide.rs`, and `examples/lsode2_task_shell_guide.rs`.

For scenario-level quality/performance behavior, use `story_tests.rs` and `story_tests2.rs`. For math parity against ODEPACK-style control logic, use `parity_micro.rs`, `stiff_parity_tests.rs`, `nonstiff_parity_tests.rs`, plus `MIRRORING_CHECKLIST.md`.

When these layers stay separated (guide for usage, story tests for end-to-end behavior, parity tests for mathematical equivalence), LSODE2 stops feeling complicated and starts behaving like a predictable engineering tool.

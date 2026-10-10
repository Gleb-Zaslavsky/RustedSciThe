# RustedSciThe
RustedSciThe is a Rust framework for symbolic and numerical computing.

Build models from equations, prepare analytical derivatives, and solve ODEs,
boundary-value problems, nonlinear systems, and fitting tasks. The maintained
solver families combine configurable numerical backends with parameter reuse,
typed diagnostics, and reproducible test/benchmark workflows.

PROJECT NEWS - October 2026 development highlights:

- The maintained LSODE2, Radau, BDF, and BE routes now have native solver APIs,
  parameter-continuation workflows, detailed diagnostics, and expanded validation.
- The rebuilt BVP_sci solver adds a modern collocation workflow alongside
  BVP_Damp, with adaptive meshes, symbolic frontends, and generated callbacks.
- Nonlinear systems now include a canonical rectangular least-squares LM core;
  fitting APIs build on the shared numerical infrastructure.
- Task-document execution, commented examples, EN/RU guides, and compact
  story/benchmark campaign reports have expanded across solver families.

See [Choose a Solver](#choose-a-solver) and [Guides and Examples](#guides-and-examples)
for current entry points.

##  The RustedSciThe Zen
"
1. Symbolic expressions are documentation that compiles.

2. If users write equations as strings,
they will eventually write a language.

3. DSLs and symbolic algebra belong together.

4. Lambdify early, optimize later.

5. Differentiate analytically.
Finite differences are for debugging and despair.

6. A symbolic Jacobian today saves ten Newton failures tomorrow.

7. The Jacobian decides
whether your method is mathematics
or optimism.

8. Stiff systems forgive nothing.
Analytical Jacobians forgive more.

9. Exact derivatives beat approximate confidence.

10. If an expression can be compiled ahead-of-time,
it probably should be.

11. A solver without statistics is a black box.
Black boxes breed superstition.

12. Measure iterations.
Measure allocations.
Measure regret.

13. Logs are cheaper than debugging.

14. Every numerical method deserves a postprocessor.

15. Plots reveal bugs.
Tables confirm them.

16. The eleventh solver should reuse the first ten.

17. Infrastructure is an algorithm.

18. FFI is a sin.
Try to keep it Rusty. 

19. Borrow from Fortran.
Return with Jacobians and AOT.

20. Ancient libraries deserve reincarnation.
A good numerical method outlives its language.

21. Dense, sparse, and banded:
serious problems need all three.

22. Bandwidth ignored becomes memory wasted.

23. Most nonlinear solvers
are secretly linear solvers.

24. Benchmark before rewriting.
Profile before parallelizing.
Think before GPU-izing.

25. Numerical stability is a feature.

26. Trust convergence criteria.
Distrust convergence.

27. Warnings ignored become papers retracted.

" 

## Features

- **Symbolic modeling:** parse and transform expressions; differentiate and
  simplify them; substitute aliases; construct Jacobians, vectors, and matrices.
- **Two symbolic representations:** the expression-tree `Expr` route and the
  packed `Atom` / borrowed `AtomView` route, with prepared numerical evaluators.
- **Code generation:** lower supported symbolic models through a shared IR to
  Rust, C, or Zig callbacks; select Lambdify or AOT where the solver supports it.
- **Initial-value ODE solvers:** non-stiff explicit methods, LSODE2, Radau,
  variable-order BDF, and Backward Euler, each with its own documented scope.
- **Boundary-value solvers:** Damped/Frozen Newton and BVP_sci collocation,
  including supported Dense, Sparse, and Banded matrix routes.
- **Nonlinear computation:** square-system root solvers, rectangular
  Levenberg-Marquardt least squares, scalar root finding, curve/kinetic fitting,
  and variable projection for separable models.
- **Reusable workflows:** prepared symbolic models, parameter rebinding,
  continuation and restart in solver APIs that expose those contracts.
- **Numerical infrastructure:** dense, sparse, and banded linear algebra;
  sequential or parallel callback evaluation where supported; typed status and
  error reporting; opt-in telemetry and result postprocessing.
- **Task documents and tooling:** text-based IVP/BVP tasks, recursive symbolic
  aliases, validation, batch execution, plotting, and trajectory visualization.

## Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Choose a Solver](#choose-a-solver)
- [Symbolic Engine and Backend Choices](#symbolic-engine-and-backend-choices)
- [AOT and Parameter Continuation](#aot-and-parameter-continuation)
- [Task Documents and CLI](#task-documents-and-cli)
- [Guides and Examples](#guides-and-examples)
- [Diagnostics, Testing, and Benchmarks](#diagnostics-testing-and-benchmarks)
- [Additional Tools and Dependencies](#additional-tools-and-dependencies)
- [Executable Mode](#executable-mode)
- [Contributing](#contributing)

## Quick Start

The crate's core use case is solving nonlinear initial- and boundary-value
problems. This compact example uses Radau and BVP_sci with their default
symbolic frontend, matrix layout, and tolerances.

```rust
use RustedSciThe::numerical::BVP_sci::BvpSciSolver;
use RustedSciThe::numerical::Radau::{RadauConfig, RadauProblem, RadauSolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Nonlinear IVP: y' = -y^2, y(0) = 1. Radau defaults to t in [0, 1].
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-y*y")],
        vec!["y".into()],
        "t",
    );
    let mut ivp = RadauSolver::prepare(problem, RadauConfig::default())?;
    let ivp_solution = ivp.solve(&[1.0])?;
    println!("IVP: y(1) = {:.6}", ivp_solution.y[0]);

    // Nonlinear BVP: y' = y^2, y(0) = 0; the zero function is an exact solution.
    let bvp = BvpSciSolver::builder(
        vec![Expr::parse_expression("y*y")],
        vec!["y".into()],
    )
    .with_mesh_and_initial_state(vec![0.0, 0.5, 1.0], vec![0.0, 0.0, 0.0])
    .with_boundary_callback(|ya, _yb, _parameters, residual| {
        residual[0] = ya[0];
        Ok(())
    })
    .build()?;
    let solution = bvp.solve()?;
    println!("BVP: converged on {} mesh nodes", solution.x.len());
    Ok(())
}
```

Save it as `src/main.rs` in a Cargo project that depends on `RustedSciThe`,
then run it with `cargo run`. For fuller solver configuration, frontend/layout
choices, and continuation, see the solver guides linked below.

## Choose a Solver

The maintained first-tier families are LSODE2, Radau, BDF, BE, BVP_Damp,
BVP_sci, and the modern nonlinear-system infrastructure. Here, **first-tier**
means a supported primary API with guides, typed diagnostics, correctness
coverage, and story/benchmark infrastructure for its supported routes.
It does not imply identical capabilities, unlimited problem sizes, or a universal
performance winner. Legacy/reference implementations are not the recommended
entry points.

| Problem / solver | Method and intended scope | Start here |
|:--|:--|:--|
| IVP: **LSODE2** | LSODE/LSODA-inspired Adams/BDF families; explicit family selection or automatic switching; Dense, Sparse, Banded | [Lambdify guide](examples/lsode2_lambdify_guide.rs) |
| IVP: **Radau** | SciPy-inspired fifth-order Radau IIA; adaptive integration; Dense, Sparse, Banded | [Public API guide](examples/radau_public_api_guide.rs) |
| IVP: **BDF** | SciPy-inspired variable-order BDF/NDF with adaptive step control; deliberately Dense | [Symbolic continuation](examples/bdf_symbolic_continuation_guide.rs) |
| IVP: **Backward Euler** | First-order implicit method with dense Newton solves; explicit fixed-step use for small/medium systems | [Native callbacks](examples/be_native_callbacks_guide.rs) |
| IVP: **non-stiff methods** | Explicit methods including RK45 and Dormand-Prince | [Universal ODE example](examples/universal_ode_example.rs) |
| BVP: **Damped / Frozen Newton** | Nonlinear boundary-value systems with damping or modified-Newton strategies and configurable linear algebra | [BVP backend guide](examples/bvp_backends_guide.rs) |
| BVP: **BVP_sci** | SciPy-like fourth-order collocation, residual control, adaptive mesh, unknown parameters, singular terms, and spline output; Dense, Sparse, Banded | [Lambdify builder guide](examples/bvp_sci_lambdify_guide.rs) |
| **Nonlinear systems** | Newton, damped Newton, classical/backtracking/MINPACK-style/Nielsen LM variants, trust-region methods, Powell dogleg; dense linear algebra | [Modern nonlinear guide](examples/nonlinear_systems_modern_guide.rs) |
| **Nonlinear least squares** | Separate f64 rectangular LM core with QR/trust-region machinery, typed termination, and symbolic or numerical residuals | [Least-squares API](src/numerical/Nonlinear_systems/NONLINEAR_SYSTEMS_USER_GUIDE_EN.md#rectangular-least-squares-problems) |
| **Fitting and parameter estimation** | Symbolic curve fitting, kinetic fitting, and variable projection (VarPro) for separable models | [Universal fitting](examples/universal_fitting_guide.rs), [VarPro](examples/varpro_fitting_guide.rs) |
| **Scalar roots** | Brent, bisection, secant, and Newton root finding | [Scalar root API](src/numerical/Nonlinear_systems/scalar_root.rs) |

Use the native solver API for detailed numerical and backend configuration.
[UniversalODESolver](src/numerical/ODE_api2.rs) and the
[shared BVP API](src/numerical/BVP_api.rs) provide convenient access for
prototyping across solver families.

Standalone BDF, BE, and nonlinear-system solvers intentionally use dense linear
algebra. For sparse/banded BDF integration, select the BDF family in LSODE2.
BE's optional legacy step heuristic is not adaptive error control. Radau's
maintained implementation has order five; archived order variants are not its
public numerical core.

Root finding seeks a zero of a residual vector; least squares minimizes its
norm and can handle rectangular systems. Fitting APIs build on this distinction.
The symbolic least-squares API also supports user-declared positive variables
for models whose domain requires them, including logarithmic equations.
The older Gavin implementations remain legacy work under review and are not
the recommended fitting path.

## Symbolic Engine and Backend Choices

RustedSciThe started with analytical Jacobians for combustion, chemical kinetics,
and heat/mass transfer. Its symbolic infrastructure now supports IVP, BVP,
nonlinear-system, and fitting workflows.

The engine includes:

- Expression parsing and construction, symbolic differentiation, simplification,
  substitution, and derivative checks against numerical differences.
- Indexed variables, symbolic vectors/matrices, Jacobian construction, and
  structural sparsity analysis.
- **Expr** expression trees and **Atom/AtomView** packed representations.
  AtomView provides borrowed access to expressions, normalization,
  differentiation, reusable workspaces, and prepared numerical evaluators.
- Conversion of symbolic models into callable numerical functions (Lambdify),
  including parameterized prepared models in the supported solver APIs.
- A shared code-generation IR and Rust/C/Zig source generation with optimization
  passes such as common-subexpression elimination.
- Taylor expansions, numerical quadrature, and symbolic integration for
  supported elementary expression classes.

Start with the [symbolic construction guide](examples/symbolic/symbolic_construction_guide.rs)
and [derivatives guide](examples/symbolic/symbolic_derivatives_guide.rs).
The engine is a scientific modeling layer, not a claim of unrestricted
computer-algebra coverage.

Solver configuration separates several choices:

| Axis | Choices | What changes |
|:--|:--|:--|
| Model input | Symbolic equations or numerical callbacks | Who supplies the model and derivatives |
| Symbolic frontend | ExprLegacy / AtomViewNative (called AtomView in some APIs) | Representation used for symbolic preparation |
| Callback execution | Lambdify / AOT | In-process evaluation or generated compiled callbacks |
| Linear layout/backend | Dense / Sparse / Banded, where supported | Matrix storage and factorization |
| Evaluation policy | Sequential / Parallel / Auto, where supported | Callback dispatch and chunking |
| Lifecycle | Fresh preparation / prepared reuse / continuation / restart | Which model and numerical state are retained |

These axes are independent concepts, but not every solver supports their full
Cartesian product. In particular, symbolic Sparse/Banded support does not imply
that the same solver's native callback API accepts sparse/banded Jacobians.
Consult its guide for the supported contract. Several native callback routes
accept an optional analytic Jacobian and use finite differences when it is absent.

Dense can be appropriate for small fully coupled systems; Sparse and Banded
exploit different structures. AtomView, AOT, and parallelism can each reduce
cost for suitable workloads, but preparation overhead and linear solves matter.
Compare the same equations, tolerances, layout, and lifecycle.

## AOT and Parameter Continuation

Ahead-of-time compilation turns a prepared symbolic model into compiled
residual/Jacobian callbacks:

```text
equations -> symbolic derivatives -> codegen IR -> Rust/C/Zig source
          -> materialization -> compilation -> loading -> numerical evaluation
```

BVP discretization is an additional stage where the selected method needs it.
Both Lambdify and AOT prepare symbolic work before numerical evaluation; AOT
additionally requires a suitable toolchain to build new artifacts.

The generated-backend infrastructure includes artifact caching, provenance,
and explicit policies such as `BuildIfMissing`, `RequirePrebuilt`, and
`RebuildAlways`. Supported handoff routes let another process consume a
published artifact without recompilation. Compiler and loader requirements
depend on the selected route; see the solver's AOT guide.

Parameter continuation is useful when solving related models repeatedly.
Value-only parameter rebinding can reuse prepared callbacks and compiled
artifacts. Numerical factorization/history reuse depends on the solver and
what changed. A restart with a new initial state is distinct from continuing
an existing trajectory; structural changes may require rebuilding preparation.

For performance decisions, distinguish **cold preparation**, **warm residual and
Jacobian evaluation**, **full solve**, and **the complete continuation series**.
Faster callbacks alone do not guarantee a faster full solve. AOT setup pays off
only when its subsequent savings outweigh compilation/loading costs.

Examples: [LSODE2 AOT continuation](examples/lsode2_parameter_continuation_aot.rs),
[BE symbolic continuation](examples/be_symbolic_continuation_guide.rs),
[BDF AOT](examples/bdf_aot_guide.rs),
[BVP_sci AOT](examples/bvp_sci_aot_guide.rs), and
[nonlinear AOT lifecycle](examples/nonlinear_aot_lifecycle_guide.rs).

## Task Documents and CLI

Text tasks describe equations, conditions, solver options, and outputs without
a custom Rust program. The DSL includes recursive `where` / `substitute`
aliases, declared parameters, continuation plans, and validation with source
diagnostics. Stiff IVP tasks use their solver's native API; non-stiff tasks use
the universal ODE route.

A minimal parameterized IVP document:

```text
task
solver: IVP
method: LSODA

equations
arg: t
parameters: rate
parameter_values: 1.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 2.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
```

From the repository root, these commands work through Cargo without
platform-specific executable paths:

```sh
cargo run -- check examples/task_docs/reference/ivp_lsoda_continuation.txt
cargo run -- run examples/task_docs/reference/ivp_lsoda_continuation.txt
cargo run -- check examples/task_docs/reference/bvp_damped_lambdify.txt
cargo run -- template ivp
cargo run -- template bvp
```

Additional CLI commands include `convert`, `convert-check` (DOCX conversion
requires Pandoc), `check-ffi-dependencies`, and `showcase`. The library's
`run_task_batch` continues after individual task failures and provides a
structured summary.

See the [task-document guide](src/command_interpreter/TASK_DOCS_GUIDE_EN.md),
[commented reference tasks](examples/task_docs/reference/), and
[recursive symbolic aliases example](examples/task_document_symbolic_aliases_guide.rs)
for the complete syntax and per-solver options.

## Guides and Examples

| Family | English | Russian |
|:--|:--|:--|
| LSODE2 | [Guide](src/numerical/LSODE2/LSODE2_USER_GUIDE_EN.md) | [Guide](src/numerical/LSODE2/LSODE2_USER_GUIDE_RU.md) |
| Radau | [Guide](src/numerical/Radau/RADAU_USER_GUIDE_EN.md) | [Guide](src/numerical/Radau/RADAU_USER_GUIDE_RU.md) |
| BDF | [Guide](src/numerical/BDF/BDF_USER_GUIDE_EN.md) | [Guide](src/numerical/BDF/BDF_USER_GUIDE_RU.md) |
| BE | [Guide](src/numerical/BE/BE_USER_GUIDE_EN.md) | [Guide](src/numerical/BE/BE_USER_GUIDE_RU.md) |
| BVP_Damp | [Guide](src/numerical/BVP_Damp/BVP_DAMP_USER_GUIDE_EN.md) | [Guide](src/numerical/BVP_Damp/BVP_DAMP_USER_GUIDE_RU.md) |
| BVP_sci | [Guide](src/numerical/BVP_sci/BVP_SCI_USER_GUIDE_EN.md) | [Guide](src/numerical/BVP_sci/BVP_SCI_USER_GUIDE_RU.md) |
| Nonlinear systems / least squares | [Guide](src/numerical/Nonlinear_systems/NONLINEAR_SYSTEMS_USER_GUIDE_EN.md) | [Guide](src/numerical/Nonlinear_systems/NONLINEAR_SYSTEMS_USER_GUIDE_RU.md) |
| Shared IVP API | [Guide](src/numerical/IVP_USER_GUIDE_EN.md) | [Guide](src/numerical/IVP_USER_GUIDE_RU.md) |

The [examples](examples/) directory contains complete Rust example-guides;
[examples/rus](examples/rus/) contains Russian-language counterparts.
Fitting examples include [LM](examples/lm_fitting_guide.rs) and
[VarPro](examples/varpro_fitting_guide.rs). For internal design and validation
conventions, see the [architectural guideline](src/RST_ARCHITECTURAL_GUIDELINE.md).

## Diagnostics, Testing, and Benchmarks

The maintained solver APIs provide typed failures/statuses and opt-in telemetry.
Depending on the solver, counters and stage timings expose symbolic preparation,
callback evaluations, Newton iterations, Jacobian refreshes, factorization,
linear solves, and AOT build/load activity. Logging and detailed instrumentation
are configurable; keep their settings matched when comparing performance.

The test corpus includes exact-solution checks, backend/frontend parity,
independent-solver comparisons, continuation/restart contracts, failure paths,
and AOT lifecycle tests. Story tests and benchmarks produce compact Tabled
reports. The release runners preserve technical logs separately and continue
through individual failures while recording them in a final summary.

A focused debug check:

```sh
cargo test --lib --no-default-features numerical::BDF:: -- --test-threads=1
```

Use `cargo test --lib --no-default-features` for the ordinary library suite.
Ignored stories can require external compilers or substantial time; select the
relevant solver campaign instead of treating them as a quick smoke test.

Campaign runners are available for
[LSODE2](scripts/lsode2_release_matrix.ps1),
[Radau](scripts/radau_release_matrix.ps1),
[BDF](scripts/bdf_release_matrix.ps1),
[BE](scripts/be_release_matrix.ps1),
[BVP_Damp](scripts/bvp_damp_release_matrix.ps1),
[BVP_sci](scripts/bvp_sci_release_matrix.ps1), and
[nonlinear systems](scripts/nonlinear_systems_release_matrix.ps1).
They are PowerShell scripts with solver-specific parameters; inspect the script
before selecting workloads. For example, Radau can print its plan first:

```powershell
./scripts/radau_release_matrix.ps1 -PlanOnly
```

Timing tables are workload-specific evidence. Statistical regression decisions
require repeated, matched measurements; a single story run is not a universal
performance threshold. The [Radau statistical runner](scripts/radau_statistical_baseline.ps1)
supports that separate workflow. Raw run archives and generated artifacts are
not part of the published crate.

## Additional Tools and Dependencies

- [Interpolation and extrapolation](src/numerical/interpolation/) include
  polynomial and piecewise-polynomial utilities shared by numerical solvers.
- [Linear algebra](src/somelinalg/) includes banded LU
  (`LapackStyleBandedLuFaithful`), structured solvers, iterative methods, and
  preconditioners. Banded versus general sparse performance depends on bandwidth,
  conditioning, and workload.
- [Plotting utilities](src/Utils/) provide Plotters/Gnuplot output and terminal
  visualization. [Animation examples](examples/plotting/) use Bevy for 2D/3D
  trajectories.
- Default Cargo features are empty. Optional `arrayfire` and `cuda` features
  require external GPU dependencies; `cuda` also enables `arrayfire`.
  ArrayFire needs its native library, and the custom CUDA path needs its compiled
  native components.
- Optional `lapack` / `openblas-system` features select native linear-algebra
  integrations. Generated AOT toolchains, Gnuplot, and Pandoc are needed only
  for the corresponding workflows.

```sh
cargo build --features arrayfire
cargo build --features cuda
```

## Executable Mode

The crate can also be built as a command-line executable. This lets you provide
a human-readable task document instead of writing a Rust program for each
problem. Build the binary, then pass it a task file:

```sh
cargo build --release
target/release/RustedSciThe run model.txt
```

On Windows, run `target\release\RustedSciThe.exe run model.txt`. The task file
contains the equations, solver and options, plus optional `postprocessing`
settings. Those settings determine whether the run writes numerical data, a
solution report, a plot, or several outputs. See [Task Documents and CLI](#task-documents-and-cli)
for a runnable task example and the supported output formats.

For example, save this nonlinear IVP as `model.txt`:

```text
task
solver: IVP
method: LSODE2

equations
arg: t
y: -y*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

postprocessing
write_report: true
report_path: output/solution.md
plotters_png: true
plotters_dir: output/plots
```

Then run `target/release/RustedSciThe run model.txt` (or
`target\release\RustedSciThe.exe run model.txt` on Windows). The task runner
solves the model and writes the Markdown report and Plotters image requested
in `postprocessing`; omit or change those options to select different outputs.

## Contributing

Questions, reproducible bug reports, and contributions are welcome in the
[project repository](https://github.com/Gleb-Zaslavsky/RustedSciThe).
For numerical issues, include the model, initial/boundary conditions, tolerances,
frontend/execution/layout, and a minimal reproducer. Performance reports should
also identify the toolchain, hardware, lifecycle, and instrumentation settings.

RustedSciThe is distributed under the [MIT license](LICENSE.txt).

# External AI-Agent Integration: Architecture Cheat-Sheet

Audit snapshot: 2026-11-15. This document records the current checkout, not a
promise that every solver family has identical semantics. Verify source-linked
details before turning them into a stable external contract.

## Executive Map

| Family | Shared/native Rust entry | Python entry | Task-document runner | Important distinction |
| --- | --- | --- | --- | --- |
| IVP | `numerical::StiffAPI` plus direct BDF, Radau, BE, LSODE2 APIs | Direct solver classes and `StiffSolverRequest` / `PreparedStiffSolver` | Yes: full IVP document and settings-only hybrid path | Prepared lifecycle and continuation/restart semantics vary by native solver. |
| BVP | `numerical::BVP_api` over CorrectionNewton, Damped, Frozen/Naive, BVP_sci | Direct BVP solvers and `BvpSolverRequest` / `PreparedBvpSolver` | Yes: full BVP document and settings-only hybrid path | Mesh, guess layout, refinement and parameter-loop support are solver-specific. |
| Nonlinear systems | `numerical::Nonlinear_systems` symbolic/closure problems, root and rectangular least-squares APIs | `NonlinearSolver`, `FittingSolver` | Yes: typed nonlinear task parser and common runner/batch dispatch | Root-finding and residual minimization are different contracts, even where the same facade selects them. |
| Optimization / fitting | `numerical::optimization`: classic LM fitting, Gavin LM, VarPro, `UniversalFitting` | `FittingSolver`, `GavinFittingSolver`, `VarProSolver`, `UniversalFitting` | Yes: typed fitting task parser and common runner/batch dispatch | Fitting consumes observations/models; it is not interchangeable with solving a free-standing equation system. |

The common text-task runner and versioned Rust JSON adapter dispatch all four
families while preserving family-specific typed specs and native result/error
contracts. `task_contract::inspect_task_json()` validates without preparing or
solving; `run_task_json_envelope()` returns a bounded, versioned response.
Python `TaskDocument.from_json()` is an optional direct client wrapper over the
same adapter, not a required hop in the worker path. This is a versioned task
protocol. A separate versioned outer job wrapper now has a strict schema,
typed preflight parser, and generic `solver_converged` / `finite_result`
acceptance checks. Resource budgets remain worker-owned metadata (not enforced
by RST), the wrapper is not directly executable by the task runner, and durable
result persistence/references are worker/client-owned rather than implemented
by the numerical crate.

## IVP

### Inputs and Settings

- A symbolic IVP consists of ordered RHS expressions, state names, independent
  variable, interval, initial state, and optional ordered symbolic parameters.
- Solver routes include BDF, Radau, Backward Euler, and LSODE2. Solver-native
  settings remain distinct: method/controller, tolerances, step limits,
  Jacobian/callback source, stop conditions, and restart behavior are not
  flattened into one lossless universal option struct.
- Symbolic frontend, callback execution, matrix layout, callback policy,
  telemetry, and logging are selectable only where the concrete route supports
  them. Capability axis lists are not a guarantee that every combination is
  valid.

### Lifecycle, Results, Errors, Telemetry

- `StiffSolverSettings::prepare` / `StiffSolverRequest` construct a typed
  `PreparedStiffSolver`; direct native APIs remain available. The facade has
  solve, parameter-value solve, ordered parameter-loop, initial-state solve,
  interval solve, restart, continuation, trajectory view and postprocessing
  operations. Continuation is not uniformly stateful across solver families.
- Results intentionally retain native ownership/shape: `StiffSolveOutcome`
  distinguishes native outcomes; BDF/BE retain results on solver state, Radau
  returns `RadauSolution`, and LSODE2 returns `Lsode2SolveSummary`. Trajectory
  and telemetry snapshots use typed facade enums, not a common dense result
  copy.
- Rust failures preserve native errors under `StiffApiError`, `StiffSolveError`,
  `StiffRestartError`, loop errors, telemetry errors, and postprocess errors.
  Python exceptions expose machine-readable `stage`, `solver`, `route`,
  `category`, and, for mapped native variants, `native_kind`.
- Telemetry snapshots and supported modes are route-specific. Missing snapshots
  are represented as absent/unsupported where the native API says so; do not
  assume all solvers report identical callback counters.

### Entry Points

- Rust facade: [StiffAPI](numerical/StiffAPI/mod.rs); native modules: `BDF`,
  `Radau::api`, `BE`, and `LSODE2`.
- Python: [stiff_api.py](../bindings/python/src/stiff_api.rs), direct modules
  `bdf.py` (Rust source `bdf.rs`), `radau.rs`, `be.rs`, `lsode2.rs`, and
  `task.py`.
- Parser: [task_runner.rs](command_interpreter/task_runner.rs),
  `task_parser_ivp.rs`, and `task_parser_ivp_hybrid.rs`.

## BVP

### Inputs and Settings

- `BvpProblem` takes ordered symbolic equations, state names, independent
  variable, explicit strictly increasing mesh, and one scalar endpoint
  constraint per state. It can declare fixed external parameter values.
- `BvpSolverConfig` retains native option types and distinct guess conventions:
  CorrectionNewton, Damped, Frozen, and BVP_sci; selected routes support
  reduced or full node-aligned state-major guesses. Do not infer guess layout
  from a generic `initial_guess` field without the selected route.
- Solver-specific settings include derivative scheme, linear layout/backend,
  symbolic frontend/execution, tolerances, damping/Jacobian policy, refinement,
  bounds and output controls. Support differs by solver. BVP_sci also has
  distinct free Newton parameters with extra boundary residual/Jacobian
  callbacks; these are not ordinary external parameters.

### Lifecycle, Results, Errors, Telemetry

- `BvpSolver::new` validates and routes the problem. The facade supports prepare,
  solve, external parameter binding/rebinding, boundary-value rebinding,
  parameter guess operations for applicable routes, and solver-supported
  parameter loops. Mesh refinement is a native Damped lifecycle, not a shared
  property of all BVP routes.
- `BvpSolution` preserves route-specific results; the common view exposes mesh,
  ordered state values and indexed access without forcing native result types
  into one internal representation. Parameter-loop reports retain per-point
  success/failure and telemetry.
- Errors are `BvpApiError` / `BvpSettingsError` with native route errors
  preserved in variants. Python maps these into `BvpError` and related typed
  exceptions with machine-readable context.
- `BvpApiTelemetryMode` has Off/Counters/Timings/Detailed, but a selected native
  route may reject unsupported modes. Snapshots are typed by solver; failure
  telemetry availability follows each native solver's lifecycle.

### Entry Points

- Rust facade: [BVP_api](numerical/BVP_api/mod.rs); native routes:
  `BVP_CorrectionNewton`, `BVP_Damp` (Damped and Frozen/Naive), and `BVP_sci`.
- Python: [bvp_api.py](../bindings/python/src/bvp_api.rs), direct solver
  bindings in `bvp.rs`, `bvp_damp.rs`, and `shooting_bvp.rs`, plus
  `TaskDocument`.
- Parser: `task_parser_bvp.rs` and `task_parser_bvp_hybrid.rs`; the hybrid
  parser can provide solver settings independently of the client's physical
  equation construction.

## Nonlinear Systems

### Inputs and Settings

- Root finding accepts a square system and an initial vector; problem adapters
  can use Rust residual/Jacobian traits or prepared symbolic expressions with
  named variables and ordered parameters. Symbolic routes can declare bounds.
- Rectangular least-squares accepts residual and Jacobian providers plus an
  initial parameter vector. This minimizes a residual norm; it is not the same
  semantic contract as requiring a zero of a square system.
- Root method families include Newton, damped Newton and trust-region variants;
  the rectangular least-squares core is a separate LM family. Symbolic frontend,
  Lambdify/AOT backend, parallel callback policy, bounds, tolerances, telemetry
  and logging vary by route. AOT support is not assumed to be universal.

### Lifecycle, Results, Errors, Telemetry

- `SymbolicNonlinearProblem` can be prepared and bound to parameter values;
  prepared/bound APIs expose residual/Jacobian evaluation, including `*_into`
  forms. Solve-time initial guesses and value-only parameter rebinding avoid
  rebuilding the symbolic problem where supported. The higher-level
  `NonlinearSolver` dispatches root versus least-squares contracts.
- Root `SolveResult` includes final iterate, residual, residual norm,
  iterations, termination reason, optional history, statistics and optional
  memory diagnostics. Least-squares returns a `MinimizationReport` with
  termination, evaluation/rejection counts, statistics and optional typed error.
- Rust errors include `SolveError` and `LeastSquaresError`, with typed variants
  for dimensions, invalid schemas/configuration, bounds, evaluation stages,
  AOT lifecycle, singularity and numerical breakdown. Termination can be a
  non-converged status in a report rather than a thrown error; consumers must
  inspect it.
- Telemetry includes solver counters/timings and symbolic-preparation reports;
  availability and history collection depend on mode and route.

### Entry Points

- Rust: [Nonlinear_systems](numerical/Nonlinear_systems.rs), especially
  `symbolic.rs`, `symbolic_generated.rs`, `solver.rs`, `engine.rs`, and
  `least_squares/`.
- Python: `NonlinearSolver` and `FittingSolver` in
  [nonlinear.rs](../bindings/python/src/nonlinear.rs); shared route metadata in
  [contract.rs](../bindings/python/src/contract.rs).
- Rust task-document parsing is available in `task_parser_nonlinear.rs`, with
  normalized JSON dispatch through `task_runner.rs`; Python `TaskDocument`
  exposes nonlinear inspection/run and batch dispatch. This protocol describes
  numerical tasks but does not yet enforce worker-level deadlines or resource
  budgets.

## Optimization and Fitting

### Inputs and Settings

- Observation-based fitting takes x/data and y/observations, symbolic model
  equations, unknown names and initial guesses; optional controls include
  weights, positivity constraints, frontend, callback policy, tolerances,
  scaling, iteration limits, telemetry and logging.
- The main symbolic families differ: classic LM fitting, Gavin's separately
  controlled LM, and VarPro with linear basis coefficients plus nonlinear
  parameters/invariants. `UniversalFitting` selects among these but rejects
  unsupported method-specific options instead of pretending they are shared.
- Other fitting APIs (`Fitting`, `GavinSymbolicFitting`, `SymbolicVarProBuilder`)
  retain their own input and controller semantics. This family should not be
  mistaken for general-purpose unconstrained optimization of arbitrary scalar
  objectives without checking the concrete API.

### Lifecycle, Results, Errors, Telemetry

- Most public fitting flows are configure/build-or-solve operations. The
  resulting `FittedModel` is reusable for evaluation, extrapolation and explicit
  missing-interval reconstruction; that evaluator reuse is not equivalent to a
  prepared fitting session or implicit re-fit with new observations.
- Results contain method-specific parameter maps, fit quality and, where
  exposed, reusable evaluators and uncertainty/statistics. Telemetry shape and
  controls differ among classic LM, Gavin and VarPro.
- Errors are method-specific (`FittingError`, `GavinLmError`, VarPro errors,
  wrapped by `UniversalFittingError`). The Rust JSON envelope maps their typed
  variants to stable family/backend codes and preserves available structured
  details and source chains; opaque callback sources remain opaque.

### Entry Points

- Rust: [optimization module](numerical/optimization.rs),
  `universal_fitting.rs`, `sym_fitting.rs`, `gavin_fitting.rs`, `varpro.rs`,
  and `fitted_model.rs`.
- Python: `FittingSolver` / `NonlinearSolver` in `nonlinear.rs`,
  `GavinFittingSolver` in [gavin.rs](../bindings/python/src/gavin.rs),
  `VarProSolver` in [varpro.rs](../bindings/python/src/varpro.rs), and
  `UniversalFitting` in [universal_fitting.rs](../bindings/python/src/universal_fitting.rs).
- Rust task-document parsing is available in `task_parser_optimization.rs`,
  with family validation and execution in the common runner. Python
  `TaskDocument` exposes optimization inspection/run and batch dispatch. The
  JSON response envelope is shared with the other families while preserving
  fitting-specific typed results.

## Cross-Cutting Findings for the Agent Contract

### Existing Building Blocks

- Python `capabilities(solver)` / `SolverCapabilities` describes supported
  axes, methods, telemetry modes and selected feature flags. Its documented
  contract explicitly warns that axis lists do not imply all combinations are
  valid; `SolverOptions.validate_for()` and concrete constructors perform
  further validation.
- Python exceptions carry `stage`, `solver`, `route`, `category`, and where
  mapped `native_kind`. Rust APIs generally preserve solver-specific typed
  errors through facade variants.
- Telemetry is opt-in in many routes and has several modes; the exact fields and
  support are not uniform. Logging is a separate channel from telemetry.
- Task documents and `run_task_batch(paths)` support human-readable jobs for
  all four families, compact table/JSON batch reports, per-entry error
  isolation, and solver non-convergence classification. Python `TaskDocument`
  exposes explicit family-specific run methods and detached results for all
  four families.

### Agreed JSON Ingestion Direction

- JSON should enter through a Rust task/orchestration adapter, not travel
  through Python bindings and not be parsed inside numerical solver modules.
- All four task families now expose native parsers from normalized
  `DocumentMap` values, and the common runner dispatches to those validators.
  Implemented JSON path: root `format_version: 1` -> normalized `DocumentMap`
  -> family-specific validators -> existing typed specs and solver routes.
  Rust exposes JSON string parse/run functions and `.json` file/batch routing;
  Python `TaskDocument.from_json()` delegates to the same core path.
- Nonlinear and fitting use separate typed specs/parsers and retain distinct
  domain payloads.
- JSON conversion preserves array order. Numeric arrays map to numeric vectors,
  homogeneous scalar arrays to repeated values, and `null` to explicit optional
  none; missing fields and empty numeric vectors remain distinct.
- JSON ingestion rejects duplicate object keys, nested values, heterogeneous
  arrays, unknown task sections, and unknown fixed fields with typed path
  diagnostics. IVP/BVP retain declaration-driven dynamic state/parameter keys;
  nonlinear/optimization use their existing strict validators.
- A Draft 2020-12 JSON Schema is checked in at
  `command_interpreter/task_document.schema.json` and included in the crate
  source. Fixed IVP/BVP solver-option fields are tested against runtime
  allowlists; full accepted field/type and conditional-method parity remains
  stabilization work. The root crate directly depends on `serde` and
  `serde_json` for this adapter.
  Python bindings remain optional direct clients, not a transport requirement.
- Rust `command_interpreter::task_contract::inspect_task_json()` is a
  non-executing typed preflight: it parses expressions and checks family
  semantics/cross-field rules, but does not construct solvers, prepare
  callbacks, build AOT artifacts, or solve. A successful inspection therefore
  means "valid for the task parser", not "guaranteed to prepare or converge".
- `task_contract::task_capabilities_json()` provides a versioned initial
  method/route catalog. Its axes are descriptive and intentionally not a
  Cartesian-product promise; per-request validation remains authoritative.
  The catalog includes method-level rules and defaults; full conditional
  compatibility parity for all BVP/native settings remains open. Concrete
  advertised defaults are checked against inspection/native defaults, including
  the input-dependent BVP_sci node-limit formula.
- `TaskInspection::effective_defaults` reports adapter-resolved values after
  applying the task route. `native_default` means the native solver still owns
  the value; unbounded numeric limits use the explicit string `unbounded`,
  avoiding accidental JSON `null` for infinity.

### Gaps or Items Requiring a Dedicated Follow-Up Audit

- The versioned JSON task format, typed preflight, and compact result envelope
  are wrapped by a separate outer job contract. Its strict schema and parser
  support generic convergence/finiteness acceptance and worker-owned budget
  metadata; the budgets are not enforced by RST. Durable artifact references
  are owned by the worker/client; consistent telemetry output remains open. See
  `AI_AGENT_INTEROP_JOB_CONTRACT.md` for the v1 contract. Typed BVP, IVP,
  nonlinear and fitting
  failures have stable variant-oriented codes; opaque callback/compiler sources
  retain their typed category and available source chain.
- The current capability description is useful but partly axis-oriented; an
  agent contract needs validation of valid combinations, conditional required
  fields, defaults, and route-specific exclusions before execution.
- A single rich `TaskDocument` facade is not yet the common interface for
  nonlinear/optimization. Any future facade should preserve the distinction
  between equation solving and fitting inputs rather than conflating them.
- Solve-level cancellation is opt-in and route-specific: nonlinear root/LM,
  fitting LM, standalone BDF/Radau/BE/DOP853/RK45/DOPRI/AB4, and BVP_CorrectionNewton expose a shared
  `CancellationToken` with typed partial outcomes. BVP_api forwards it to
  fixed-mesh CorrectionNewton and adaptive BVP_sci; Damped and Frozen reject
  it. Existing
  AOT build/file-lock timeouts remain preparation controls, not solve
  cancellation. Durable pause/resume after unloading a worker remains a
  separate, solver-specific checkpoint question.
- There is no verified common contract here for wall-clock/memory budgets,
  isolated worker execution, or cluster scheduling. Artifact IDs and storage
  are worker/client concerns, not RST guarantees. Treat the other items as
  requirements to design and test, not as existing guarantees.
- Solver callback policies (`Sequential` / `Parallel` / `Auto`, with thresholds
  such as `min_work`) select numerical callback scheduling; they do not expose
  a per-solve CPU-core allocation. Rayon thread-pool sizing is process-level
  (the repository uses `RAYON_NUM_THREADS` in isolated runs), so a worker's
  allocated cores belong to the worker/scheduler contract, not the mathematical
  solver configuration. Configure the worker before initializing RST/Rayon and
  audit any additional native thread pools to avoid oversubscription.
- Success status alone is insufficient for autonomous use. The caller needs
  typed numerical evidence (such as solver-reported residual or boundary
  metrics, when available) to define threshold checks, while interpreting
  domain invariants and physical admissibility itself.
- “Recommended solver” behavior must be explicit and explainable. No route
  switching, tolerance relaxation, bounds changes, or retry with changed
  numerics should happen silently.

## Audit Sources

- Rust routing/facades: [StiffAPI](numerical/StiffAPI/mod.rs),
  [BVP_api](numerical/BVP_api/mod.rs),
  [Nonlinear_systems](numerical/Nonlinear_systems.rs),
  [optimization](numerical/optimization.rs),
  [TaskRunner](command_interpreter/task_runner.rs).
- Python registration and APIs: [module exports](../bindings/python/src/lib.rs),
  [capabilities and options](../bindings/python/src/contract.rs),
  [task document](../bindings/python/src/task.rs),
  [StiffAPI](../bindings/python/src/stiff_api.rs),
  [BVP_api](../bindings/python/src/bvp_api.rs),
  [nonlinear/fitting](../bindings/python/src/nonlinear.rs),
  [Gavin](../bindings/python/src/gavin.rs),
  [VarPro](../bindings/python/src/varpro.rs),
  [UniversalFitting](../bindings/python/src/universal_fitting.rs),
  [Python error mapping](../bindings/python/src/errors.rs).
- Parser documentation: [task document guide](command_interpreter/TASK_DOCS_GUIDE_EN.md).

# Task Documents in RustedSciThe: Practical Guide for IVP/BVP Workflows

This guide explains how to write task documents for RustedSciThe so you can run IVP and BVP jobs from plain text files. The target reader is a Rust developer who wants a reproducible, human-readable input format for numerical experiments, CI scenarios, and solver benchmarking, without building a custom driver for every single model.

The parser stack lives in:

- `src/command_interpreter/task_parser_ivp.rs`
- `src/command_interpreter/task_parser_bvp.rs`
- `src/command_interpreter/task_runner.rs`

Ready-to-run examples are in:

- `examples/task_docs/`

---

## 1. Big Picture

RustedSciThe task docs are section-based text files. You describe:

- what kind of task you want (`solver: IVP` or `solver: BVP`),
- the equations,
- initial/boundary conditions,
- solver options (including backend/AOT knobs),
- optional output preferences.

The runner (`task_runner`) detects task kind automatically and dispatches to the corresponding parser and solver route.

From executable mode:

```bash
RustedSciThe.exe examples/task_docs/ivp_decay_task.txt
RustedSciThe.exe examples/task_docs/bvp_reference_task.txt
```

From Rust API:

```rust
use RustedSciThe::command_interpreter::task_runner::run_task_from_file;

let result = run_task_from_file("examples/task_docs/ivp_decay_task.txt")?;
println!("{:?}", result.kind());
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## 2. Document Layout and Grammar

Each section starts with a header line (for example `task`, `equations`, `solver_options`). Inside a section, fields are written as `key: value`.

Comments and empty lines are allowed and can be used to annotate task files.

### 2.1 Required sections by task kind

For IVP:

- `task`
- `equations`
- `initial_conditions`

The optional `task.schema_version` is currently `1`. Unknown versions are
rejected instead of being silently interpreted with older semantics.

For BVP:

- `task`
- `equations`
- `boundary_conditions`
- `mesh`
- `initial_guess`

Optional for both:

- `solver_options`
- `postprocessing`
- `where` (or `substitute`) for symbolic substitutions

### 2.2 Equation section forms

Both IVP and BVP parsers support two equation styles.

List style:

```text
equations
arg: t
unknowns: y, z
rhs: z, -y
```

Pair style:

```text
equations
arg: t
y: z
z: -y
```

List style is recommended for larger systems because the unknown order is explicit and easier to review.

### 2.3 Symbolic substitutions (`where` / `substitute`)

You can define reusable symbolic aliases and have them substituted before solver construction:

```text
where
k: A*exp(-E/(R*T))
source: k*c
```

Then use `source` in `rhs`. This is symbolic substitution, not numeric parameter assignment.

Numeric parameters remain:

```text
parameters: A, E, R
parameter_values: 1.0e7, 1.2e5, 8.314
```

---

## 3. IVP Task Docs

### 3.1 Minimal LSODE2 example

```text
task
solver: IVP
method: LSODE2

equations
arg: t
parameters: a
parameter_values: 1.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 2.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: faer_sparse_lu
lsode2_native_execution: faithful_bdf_solve
```

### 3.2 `task` section (IVP)

Core fields:

- `solver: IVP`
- `method: ...`

Supported method values:

- `RK45`, `RK4`, `Euler`, `AB4`
- `Radau5`
- `BDF`
- `BackwardEuler`
- `LSODE2`
- `LSODE` (fixed BDF controller)
- `LSODA` (automatic Adams/BDF controller)

`LSODE` and `LSODA` are preserved as distinct typed routes. `LSODE2` keeps
the crate's explicit BDF default, while `lsode2_method_family` may select a
different controller policy.

### 3.3 `initial_conditions` (IVP)

Required:

- `t0`
- `t_end`
- `y0` (vector, same length as unknowns)

### 3.4 General `solver_options` (IVP)

The parser validates these per method. Unsupported combinations fail with a
typed `UnsupportedOption` error; options are never silently ignored.

Common options and their native routes are:

- `step_size`
- `tolerance`
- `max_iterations`
- `rtol`
- `atol`
- `max_step`
- `first_step` (supports `Some(...)` or plain numeric value)
- `vectorized`
- `parallel`
- `neighborhood_check`

`RK45`, `RK4`, `Euler`, and `AB4` accept all common options. BDF accepts
`max_iterations`, `rtol`, `atol`, `max_step`, `first_step`, and `vectorized`.
Backward Euler accepts `step_size`, `tolerance`, `max_iterations`, and
`neighborhood_check`. Radau5 accepts `tolerance`, `max_iterations`, `rtol`,
`atol`, `max_step`, and `first_step`. LSODE/LSODA/LSODE2 accept `rtol`,
`atol`, `max_step`, `first_step`, and `vectorized`.

### 3.5 Parameter continuation

Continuation keeps the symbolic RHS and supports `fresh`, `warm`, and
`prepared` lifecycle modes:

```text
continuation
parameter: rate
values: 0.5, 1.0, 2.0
mode: prepared
restart_each: false
```

`fresh` rebuilds each segment. `warm` and `prepared` reuse the prepared
symbolic callbacks where the native adapter exposes a continuation contract.

Multi-parameter grids and per-segment restart data are also supported:

```text
continuation
parameters: rate, source
rate_values: 1.0, 2.0
source_values: 0.0, 3.0
mode: fresh
restart_policy: restart_with_state
monotonic: allow
y0_values: [1.0], [2.0], [3.0], [4.0]
t0_values: 0.0, 0.1, 0.2, 0.3
t_end_values: 1.0, 1.1, 1.2, 1.3
```

The parameter lists form a Cartesian grid. `restart_policy` may be `continue`,
`restart_each`, or `restart_with_state`; the latter is the explicit form for
segments that provide a new state or interval. `y0_values`, `t0_values`, and
`t_end_values` accept one broadcast value or one value per generated segment.
`monotonic: increasing|decreasing` validates the generated row order, while
`allow` permits a deliberately non-monotone experiment.
For `warm`/`prepared` execution, explicit `y0`/`t0` segments require a native
restart-capable adapter; otherwise the runner returns a typed continuation
error rather than silently ignoring the override. BDF, Radau5, Backward Euler,
and LSODE2 provide this contract. LSODE2 prepares its BDF bridge on demand for
the restart path. Use `fresh` when a solver does not expose that contract.
With `restart_each: false`, BDF continues from the previous accepted state;
the task runner advances the segment bound. With `restart_each: true`, each
segment starts from the document's initial state and interval. Unsupported
native lifecycle combinations return a typed continuation error.

The runner exposes a typed `status_code` and a typed `trajectory` alongside
the legacy textual status and matrix projections. Grid-based solvers return a
sampled grid; Radau returns its native solution object. Missing task files,
invalid numerical configuration, native exhaustion, and postprocessing I/O
are reported as separate typed error categories.

### 3.5.1 BVP routes

The solver-specific guides contain complete annotated task examples:
[`BVP_DAMP_USER_GUIDE_EN.md`](../numerical/BVP_Damp/BVP_DAMP_USER_GUIDE_EN.md)
and
[`BVP_SOLVER_USER_GUIDE.md`](../numerical/BVP_sci/BVP_SOLVER_USER_GUIDE.md).
Use them when a task needs more detail than this shared grammar reference.

`solver: BVP` selects the compatibility task route backed by `BVP_Damp`.
Its default task-document logging is quiet; set `solver_options.loglevel`
explicitly when diagnostic solver output is required. `fresh` continuation
rebuilds each segment. `warm` and `prepared` continuation reuse the prepared
symbolic callbacks and parameter schema; the runner also supports typed
mesh/initial-state restarts. The task result reports segment counts,
fresh/prepared preparation counts, restart counts, and a bounded result-size
retention proxy. Native byte-level allocation retention still requires a
separate release measurement.

`solver: BVP_sci` selects the new SciPy-like collocation route and never falls
back to the historical `BVP` facade. `frontend: ExprLegacy` is the default;
`frontend: AtomViewNative` selects the native symbolic path. `method: Dense`,
`Sparse`, or `Banded` selects the corresponding linear backend. The current
task-document contract requires exactly one scalar boundary condition per
state. `fresh` continuation substitutes parameter values and rebuilds the
numeric segment; `warm`/`prepared` continuation requires declared symbolic
parameters and reuses the prepared BVP_sci plan, including optional per-segment
mesh and initial-state restart. Parameter rows are pinned through the extra
boundary residual block, so they are not silently treated as free unknowns.
Both BVP routes support the shared CSV/TXT/Markdown/plot postprocessing plan.
For new task documents, prefer the typed `output_policy` field:
`none`, `plotters`, `gnuplot`, or `terminal`. The older `plot: true/false`
field remains a compatibility alias and should not be used in new documents.

### 3.6 LSODE2-specific options (IVP)

Symbolic assembly backend:

- `lsode2_symbolic_assembly: ExprLegacy | AtomView`

Symbolic execution mode:

- `lsode2_symbolic_execution: LambdifyExpr`
- `lsode2_symbolic_execution: AOT`

If `AOT`, you can additionally set:

- `lsode2_aot_toolchain: c_tcc | c_gcc | zig | rust`
- `lsode2_aot_profile: debug | release`
- `lsode2_aot_output_dir: <path>`

Linear structure:

- `lsode2_linear_structure: dense`
- `lsode2_linear_structure: sparse`
- `lsode2_linear_structure: banded`
- for banded: `lsode2_banded_kl`, `lsode2_banded_ku`

Linear solver policy:

- `lsode2_linear_solver_policy: auto`
- `lsode2_linear_solver_policy: dense_lu`
- `lsode2_linear_solver_policy: faer_sparse_lu`
- `lsode2_linear_solver_policy: lapack_faithful_banded_lu`

Native execution mode:

- `lsode2_native_execution: faithful_bdf_solve`
- `lsode2_native_execution: probe_before_bridge`
- `lsode2_native_execution: bridge_solve`

Native limits:

- `lsode2_native_max_step_attempts`
- `lsode2_native_max_accepted_steps`

### 3.7 Batch execution

The library runner `run_task_batch(paths)` executes documents independently and
continues after parser, I/O, or solver failures. Its `BatchTaskReport` exposes
pass/fail counts and renders a compact Tabled report with the path, task kind,
status, typed error category, and short error text. Use `write_table(path)` for
the human-facing report; keep Cargo/compiler output in a separate technical
log rather than mixing it into the table.

---

## 4. BVP Task Docs

### 4.1 Minimal BVP example

```text
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse
frontend: AtomViewNative
execution: Lambdify

equations
arg: x
unknowns: z, y
rhs: y-z, -z^3

boundary_conditions
z_left: 1.0
y_right: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 20

initial_guess
z: 0.0
y: 0.0

solver_options
tolerance: 1e-5
max_iterations: 20
generated_backend: sparse_lambdify
```

### 4.2 `task` section (BVP)

Fields:

- `solver: BVP`
- `strategy: Damped | Frozen | Naive` (default `Damped`)
- `scheme` (currently usually `forward`)
- `method: Dense | Sparse | Banded` (default `Sparse`)
- `frontend: ExprLegacy | AtomViewNative` (optional symbolic representation override)
- `execution: Lambdify | AOT` (callback execution route; defaults to `Lambdify`)

### 4.3 Boundary conditions and mesh

Boundary keys use suffixes:

- `<unknown>_left`
- `<unknown>_right`

Mesh:

- `t0`, `t_end`, `n_steps`

### 4.4 Generated backend controls (BVP)

In `solver_options`, parser supports:

- `generated_backend` presets (for example `banded_lambdify`, `banded_aot_tcc`, `sparse_aot_gcc`, etc.)
- `matrix_backend: dense | sparse | banded`
- `backend_policy: lambdify_only | aot_only | prefer_aot_then_lambdify`
- `symbolic_backend: ExprLegacy | AtomView`

`symbolic_backend` and `frontend` select the symbolic representation; they do
not select Lambdify versus AOT. `backend_policy` and the generated-backend
presets are the compatibility controls for the mature `BVP_Damp` route. The
omitted `frontend` preserves the preset default; for `BVP_sci`, omission means
`ExprLegacy`. The explicit task-level `execution` field overrides that policy
for `BVP_Damp` and `BVP_sci`, when present. For `BVP_sci`, task-level AOT uses
the same shared generated lifecycle as the native Rust API; provide
`aot_output_dir` (and optionally `aot_handoff_path`) when the policy may build
or publish an artifact. Omitting the output directory produces a typed AOT
preparation error rather than silently falling back to Lambdify.
- `aot_codegen_backend: rust | c | zig`
- `aot_c_compiler` (for C routes, e.g. `tcc`/`gcc`)
- `aot_build_policy: use_if_available | build_if_missing | require_prebuilt | rebuild_always`
- `aot_build_profile: debug | release`
- `aot_output_dir` (required for BVP_sci task-level AOT build/publication)
- `aot_handoff_path` (optional durable process-isolated resolver handoff)
- `aot_compile_preset: production | fast_build | dev_fastest`
- `aot_execution_policy: auto | sequential`
- `banded_linear_solver` (faithful/block-tridiagonal/faer-sparse variants)
- `refinement_steps`

Note: parser currently rejects `aot_execution_policy: parallel` for BVP task docs because exposing full parallel executor config in task docs is not yet finished.

Also note that BVP task documents are symbolic inputs. They contain equations as text, not Rust closures, so they intentionally do not support `backend_policy: numeric_only`, `backend_policy: prefer_aot_then_numeric`, or `backend_policy: prefer_lambdify_then_numeric`. The pure numerical BVP route exists in the Rust API of the damped Newton solver: call `NRBVP::set_numeric_rhs(...)` or `NRBVP::with_numeric_rhs(...)`, set `BackendSelectionPolicy::NumericOnly`, and the solver will discretize that closure and build the Newton Jacobian by finite differences. The frozen BVP solver remains a symbolic Lambdify/AOT route by design; there is no frozen pure-numeric closure path.

---

## 5. IVP Method Keyword Matrix

This table is intentionally practical: it shows what to put in task docs depending on the method family.

| IVP method | Required task fields | Strongly recommended solver options | LSODE2-only options |
|---|---|---|---|
| `RK45`, `RK4`, `Euler`, `AB4` | `solver: IVP`, `method`, `equations`, `initial_conditions` | `step_size` (where relevant), `rtol`, `atol`, `max_step` | not used |
| `Radau5` | native maintained Radau implementation | `rtol`, `atol`, `max_step`, optionally `first_step` | order selection is not exposed; `Radau3` is rejected |
| `BDF` | same | `rtol`, `atol`, `max_step`, `first_step`, optionally `max_iterations` | not used |
| `BackwardEuler` | same | `step_size` and/or `max_step`, `tolerance` | not used |
| `LSODE2` / `LSODE` / `LSODA` | same | `rtol`, `atol`, `max_step`, `first_step` | `lsode2_symbolic_assembly`, `lsode2_symbolic_execution`, `lsode2_linear_structure`, `lsode2_linear_solver_policy`, `lsode2_native_execution`, optional AOT fields and native limits; `LSODE` is fixed BDF and `LSODA` is automatic Adams/BDF by default |

---

## 6. AOT, Lambdify, Numerical: How to choose in task docs

For LSODE2 from task docs, the practical choices are:

- Numerical callback route: use method/backends from Rust API directly when you already own residual/Jacobian closures in code. Task docs are mainly symbolic-driven.
- Lambdify route: fastest to set up, excellent baseline for correctness and many production runs.
- AOT route: adds build/prepare overhead but is useful when solve-loop throughput matters across repeated runs.

For AOT in IVP task docs:

```text
lsode2_symbolic_execution: AOT
lsode2_aot_toolchain: c_gcc
lsode2_aot_profile: release
```

Toolchain prerequisites: install the corresponding compiler/toolchain on the host machine (`tcc`, `gcc`, `zig`, Rust toolchain for Rust AOT mode).

---

## 7. Parallelism Notes

The task-doc layer currently exposes parallelism in two ways:

- generic IVP option `parallel: true/false`,
- backend policy/execution selections where available.

Low-level chunk-size and advanced executor tuning are still richer in direct Rust API than in task-doc format. This is intentional for now: task docs stay stable and human-readable, while advanced orchestration can evolve in typed APIs.

---

## 8. Postprocessing

Both IVP and BVP task docs support:

- `save_csv: true/false`
- `csv_path: ...`
- `save_txt: true/false`
- `txt_path: ...`
- `write_report: true/false`
- `report_path: ...`
- `plotters_png: true/false`
- `plotters_dir: ...`
- `gnuplot_png: true/false`
- `gnuplot_dir: ...`
- `terminal_plot: true/false`
- `plot: true/false`

CSV, TXT and markdown report export are routed through the unified
`PostprocessPlan` facade. `plotters_png`, `gnuplot_png` and `terminal_plot`
are explicit modern plotting actions; `plot` remains a legacy compatibility flag
and its behavior depends on the specific solver path and available plotting
setup.

---

## 9. Header/Field Aliases (Pseudonyms)

Parsers include tolerant aliases so older or alternative naming can still work. For example:

- section aliases like `problem` for `task`,
- `system` for `equations`,
- `solver_settings` or `options` for `solver_options`,
- `substitute` as alias for `where`.

Still, for new files prefer canonical names from templates to keep docs and CI consistent.

---

## 10. Common Failure Modes and Fast Checks

If parsing fails, check these first:

- `solver` value is exactly `IVP` or `BVP`,
- section names are correct and separated cleanly,
- equation unknown count equals RHS count,
- `parameter_values` count matches `parameters`,
- `y0` length matches number of unknowns,
- BVP boundary keys follow `<unknown>_left` / `<unknown>_right`,
- LSODE2 AOT options are coherent (`lsode2_symbolic_execution: AOT` plus toolchain/profile if needed).

If execution fails in AOT mode, verify compiler availability and output directory permissions first.

---

## 11. Recommended Workflow

Start from templates:

```bash
RustedSciThe.exe --template ivp
RustedSciThe.exe --template bvp
```

Then adapt step by step:

1. make equations/conditions run with baseline options,
2. lock correctness (compare against known solution or baseline route),
3. introduce backend-specific tuning (`Lambdify` -> `AOT`, sparse/banded policy, etc.),
4. only then run multi-run performance stories.

This sequence keeps debugging local and avoids mixing model errors with backend orchestration noise.

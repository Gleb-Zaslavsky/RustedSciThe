# Task Documents in RustedSciThe: Practical Guide for IVP/BVP Workflows

This guide explains how to write task documents for RustedSciThe so you can run IVP, BVP, nonlinear, and fitting jobs from plain text or JSON. The target reader is a Rust developer who wants a reproducible input format for numerical experiments, CI scenarios, and solver benchmarking, without building a custom driver for every single model.

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

## JSON Input

The core accepts JSON directly through `parse_task_spec_from_json` and
`run_task_from_json`. JSON files are also accepted by the file runner when the
extension is `.json`; the task batch runner uses the same extension rule. JSON
is converted in Rust to the normalized `DocumentMap` and then sent through the
same family-specific parser as the text DSL. The Python binding follows this
same core path with `TaskDocument.from_json(...)`; it does not parse task
schemas independently.

The JSON object requires root `format_version: 1`. The remaining top-level
objects mirror DSL sections, and each field is a scalar or a homogeneous
array. For example:

```json
{
  "format_version": 1,
  "problem": {
    "formulation": "root",
    "variables": "x",
    "equations": "x - 2",
    "initial_guess": 0.0
  },
  "solver": { "method": "newton" }
}
```

Numeric arrays become numeric vectors where a field accepts a numeric vector;
arrays of strings or booleans become repeated field values. `null` is accepted
only by fields whose contract is nullable. Nested objects/arrays and
heterogeneous arrays are rejected with a typed parse error that identifies the
JSON path. Duplicate object keys and unknown task sections or fixed fields are
also rejected with typed path diagnostics. Dynamic IVP/BVP
state and parameter names are checked against declarations where applicable.
All semantic validation remains in the native IVP, BVP, nonlinear, or
optimization parser.

JSON uses canonical field names; the text DSL's header/field aliases are not
implicitly normalized on the JSON path. In particular, declare symbolic
equation parameters with `equations.parameters` and `equations.parameter_values`.

The machine-readable Draft 2020-12 shape is maintained in
[`task_document.schema.json`](task_document.schema.json) and included in the
published Rust crate source. Rust clients can also access the embedded schema
as `command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA`. It documents
the four task families, fixed field types, and expressible method-specific
constraints. Symbol resolution, route capabilities, dimensions, and other
cross-field rules remain authoritative in the native typed parsers; JSON
Schema validation is a preflight aid, not a replacement.

The bounded execution response has its own Draft 2020-12 schema at
[`task_result.schema.json`](task_result.schema.json), exposed as
`command_interpreter::task_contract::TASK_RESULT_JSON_SCHEMA`. It describes
success, numerical non-convergence, execution failures, and family-specific
result shapes. The schema is tested against actual responses from all four
task families; native typed APIs remain the source for full in-process results.
Acceptance evaluation is a separate object with its own schema at
[`acceptance_report.schema.json`](acceptance_report.schema.json), exposed as
`command_interpreter::agent_job::ACCEPTANCE_REPORT_JSON_SCHEMA`; it never
overwrites the native solver outcome.

```rust
use RustedSciThe::command_interpreter::task_runner::run_task_from_json;

let result = run_task_from_json(json_document)?;
println!("{:?}", result.kind());
# Ok::<(), Box<dyn std::error::Error>>(())
```

For an agent or workflow that must validate a request before execution, use the
non-executing typed preflight and capability catalog:

```rust
use RustedSciThe::command_interpreter::task_contract::{
    inspect_task_json,
    task_capabilities_json,
};

let capabilities = task_capabilities_json();
let inspection = inspect_task_json(json_document)?;
println!("{}", inspection.to_json());
# let _ = capabilities;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Inspection validates JSON structure, symbolic expression syntax, and the
family parser's semantic and cross-field rules. It does not construct a solver,
prepare callbacks, compile AOT artifacts, or run a numerical solve. A valid
inspection is not a guarantee that native preparation or numerical convergence
will succeed. The capability catalog is versioned, but listed route axes are
not a promise that every combination is supported; the typed parser is
authoritative for each specific request. The inspection's `effective_defaults`
contains adapter-resolved settings; a string value `native_default` means that
the selected native solver owns the default. Unbounded limits are represented
as `unbounded`, not as JSON `null`.

The capability response includes a versioned `preset_catalog`. Presets are
fixed route templates, not model-aware solver recommendations. Expand one with
`command_interpreter::task_presets::expand_task_preset_json`; route selectors
must agree with the preset, while explicit numeric tuning values override
preset defaults. Always inspect the expanded task to see the effective
settings and validate compatibility before execution:

```rust
use RustedSciThe::command_interpreter::task_contract::inspect_task_json;
use RustedSciThe::command_interpreter::task_presets::expand_task_preset_json;

let expanded = expand_task_preset_json(task_json, "ivp_bdf_sparse")?;
let expanded_json = serde_json::to_string(&expanded)?;
let inspection = inspect_task_json(&expanded_json)?;
println!("{}", inspection.to_json());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The preset catalog is an initial route-level set, not yet an exhaustive list
of every solver control. Use the task JSON Schema and typed inspection as the
authoritative sources for custom settings.

To run JSON directly and receive a compact worker-friendly response, Rust
clients can call `task_contract::run_task_json_envelope`. The response uses
`contract_version: 1` and distinguishes `succeeded`, `not_converged`, and
`failed`. A numerical non-convergence contains a result and termination
message, but no execution error. A failure contains a stable variant-oriented
error code and category, stage, optional field path and structured details,
typed task-error origin, task family and solver when identifiable, message,
source chain, and process exit code. Native BVP, IVP, nonlinear root and
least-squares, and fitting errors are mapped from typed variants to stable
machine codes. When a native variant contains only an opaque callback/compiler
source, its code identifies that typed failure category and the source chain
retains the available diagnostic text; no code is derived by parsing
`Display` output.
`retryable: null` means RST makes no generic retry claim; the client must not
infer that retrying is safe.

Some validation failures additionally include optional `expected`, `allowed`,
and `hint` fields. They are emitted only where the parser or capability catalog
provides reliable guidance; absence means RST has no safe generic suggestion.
Hints never rewrite the submitted task or relax numerical settings.

This is the executable numerical task contract, not a worker-job envelope. The
separate versioned wrapper, its strict schema, typed preflight, generic
acceptance checks, and worker ownership boundary are described in
[`AI_AGENT_INTEROP_JOB_CONTRACT.md`](../AI_AGENT_INTEROP_JOB_CONTRACT.md).
`parse_agent_job_json()` validates that wrapper and its embedded task, but does
not execute it or enforce the worker's wall-time, memory, or CPU budget. The
worker passes the validated embedded task to `run_task_json_envelope()` after
applying its own resource policy. Oversized results are still omitted rather
than persisted to durable artifacts.

Canonical small JSON examples for all four task families and a complete outer
job are maintained under [`agent_interop/examples`](../agent_interop/examples/)
and exercised by Rust tests. The request schema does not define physical units;
the client must document units and model-specific admissibility alongside its
job rather than expecting RST to infer them.

IVP/BVP numeric arrays and nonlinear solution vectors are included inline only
when the total numeric payload is at most 4096 values. Otherwise the response
sets the corresponding data field to `null`, reports its size, and marks it as
omitted. The existing typed Rust `TaskRunResult` API still exposes full results
in-process. Durable artifact storage/reference transport for large worker
results is not part of this envelope yet. IVP `states` are indexed as
`[sample][state]`, aligned with the `time` array. BVP `values` use the same
`[mesh_sample][state]` orientation, with a parallel `mesh` array. Non-finite
floating-point values are encoded as JSON `null` and counted explicitly in
`non_finite_values_encoded_as_null`.

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

## Hybrid Settings-Only JSON

Some clients own the physical model and use RST only to interpret solver
settings. The JSON settings adapters mirror the text hybrid parsers and do not
parse or convert client-owned domain fields. They accept the usual
`format_version: 1`, extract only solver-owned sections, and pass those through
the same native settings parser used by the text route. Nested domain objects
therefore remain valid and opaque to RST.

```rust
use RustedSciThe::command_interpreter::task_parser_ivp_hybrid::
    parse_hybrid_stiff_settings_from_json;

let settings = parse_hybrid_stiff_settings_from_json(json_document)?;
// The caller builds the equations and initial state, then supplies them via
// the ordinary StiffAPI request before calling `settings.prepare(...)`.
# Ok::<(), Box<dyn std::error::Error>>(())
```

The IVP JSON settings shape uses `task` and `solver_options`; the BVP shape uses
`solver_settings` and supported companion sections such as `bounds` or
`grid_refinement`. The BVP entry point is
`parse_hybrid_bvp_solver_settings_from_json`. Example inputs live at
[`ivp_hybrid_settings.json`](../agent_interop/examples/ivp_hybrid_settings.json)
and [`bvp_hybrid_settings.json`](../agent_interop/examples/bvp_hybrid_settings.json).
Top-level domain sections are intentionally ignored by these settings-only
functions. Unknown or incompatible fields inside recognized solver-settings
sections remain typed errors. These JSON objects configure a solver; they do
not carry equations, boundary/initial conditions, or physical semantics.

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

For BVP solvers that support component bounds, use a root-level `bounds`
section. Each entry is an inclusive `lower, upper` pair. Damped and Frozen
retain their wide `[-1e6, 1e6]` limits for states omitted from this section;
explicit entries override those defaults. BVP_sci currently rejects bounds
with a typed unsupported-option error rather than ignoring them.

```text
bounds
temperature: 250.0, 5000.0
fuel: 0.0, 1.0
```

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

bounds
z: -50.0, 50.0
y: -35.0, 35.0

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

## Cooperative Cancellation

`run_task_from_json_with_cancellation_token` forwards cancellation for single
IVP tasks using BDF, Backward Euler, Radau, LSODE2, DOP853, RK45, DOPRI, or
AB4; BVP routes; nonlinear root and symbolic least-squares tasks; and classic
LM/VarPro fitting tasks. Supported native routes return typed partial
snapshots; the JSON error envelope keeps `result: null` and places available
diagnostics under `error.details`. IVP parameter-continuation task documents
and Gavin fitting return an explicit unsupported-route error; they do not
silently ignore the token. For
nonlinear root, the symbolic problem object is constructed before the native
engine observes a pre-cancelled token, but no residual/Jacobian evaluation is
performed and the snapshot retains the initial guess with zero completed
iterations.

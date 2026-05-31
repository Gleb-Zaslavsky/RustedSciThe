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
- `Radau3`, `Radau5`
- `BDF`
- `BackwardEuler`
- `LSODE2` (aliases `LSODE`, `LSODA` are accepted)

### 3.3 `initial_conditions` (IVP)

Required:

- `t0`
- `t_end`
- `y0` (vector, same length as unknowns)

### 3.4 General `solver_options` (IVP)

These are parsed for all IVP methods; each solver uses what is relevant:

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

### 3.5 LSODE2-specific options (IVP)

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

---

## 4. BVP Task Docs

### 4.1 Minimal BVP example

```text
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

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
- `aot_codegen_backend: rust | c | zig`
- `aot_c_compiler` (for C routes, e.g. `tcc`/`gcc`)
- `aot_build_policy: use_if_available | build_if_missing | require_prebuilt | rebuild_always`
- `aot_build_profile: debug | release`
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
| `Radau3`, `Radau5` | same | `rtol`, `atol`, `max_step`, optionally `first_step` | not used |
| `BDF` | same | `rtol`, `atol`, `max_step`, `first_step`, optionally `max_iterations` | not used |
| `BackwardEuler` | same | `step_size` and/or `max_step`, `tolerance` | not used |
| `LSODE2` / `LSODE` / `LSODA` | same | `rtol`, `atol`, `max_step`, `first_step` | `lsode2_symbolic_assembly`, `lsode2_symbolic_execution`, `lsode2_linear_structure`, `lsode2_linear_solver_policy`, `lsode2_native_execution`, optional AOT fields and native limits |

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
- `plot: true/false`

CSV export is already wired. Plot behavior depends on the specific solver path and available plotting setup.

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

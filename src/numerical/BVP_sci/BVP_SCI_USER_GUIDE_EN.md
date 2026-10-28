# BVP_sci User Guide

`BVP_sci` is the maintained SciPy-like collocation BVP solver. It supports
the symbolic Lambdify route through `ExprLegacy` and `AtomViewNative`, and the
Dense, Sparse and Banded linear layouts. AOT is an explicit generated-callback
route, not an implicit fallback.

## Task parser route

Use `solver: BVP_sci` to select this solver. Comments beginning with `#` are
ignored, so reference files can document the DSL inline.

```text
# The solver family is explicit; `BVP` selects the compatibility BVP_Damp route.
task
solver: BVP_sci

# ExprLegacy and AtomViewNative are symbolic frontends.
frontend: ExprLegacy
# Lambdify and AOT are execution routes, independent of the frontend.
execution: Lambdify
# Dense/Sparse/Banded selects the linear backend/layout.
method: Sparse

equations
arg: x
unknowns: y
rhs: -rate*y
# BVP_sci parameters are free Newton unknowns, not fixed inputs. This is their
# initial guess; one extra boundary residual is required for this unknown.
parameters: rate
parameter_values: 0.7

boundary_conditions
# A BVP_sci task needs a complete scalar boundary-condition set: normally one
# condition per state, distributed between left and right endpoints.
y_left: 1.0
y_right: 0.36787944117144233

mesh
t0: 0.0
t_end: 1.0
n_steps: 16

initial_guess
y: 0.0

solver_options
tolerance: 1e-6
max_iterations: 40

postprocessing
# Typed output policy; `plot` is only a legacy compatibility alias.
output_policy: none
```

The parser also accepts `matrix_backend` and generated-backend presets in
`solver_options`. `method` is the readable task-level spelling. Exact
boundary-condition validation happens before solver construction.

There is no universal fastest matrix layout. Dense is a natural choice for
small fully coupled systems, while Sparse or Banded can be better when the
global collocation Jacobian has exploitable structure. Compare matched
workloads rather than comparing different equations. The complete solver
reference, including the direct API and numerical configuration, is
[`BVP_SOLVER_USER_GUIDE.md`](BVP_SOLVER_USER_GUIDE.md).

## Frontend versus execution

Do not confuse these axes:

- `frontend: ExprLegacy` follows the compatibility expression graph;
- `frontend: AtomViewNative` prepares and differentiates on the Atom view;
- `execution: Lambdify` creates in-process callbacks;
- `execution: AOT` materializes generated callbacks and follows the AOT
  lifecycle.

An AtomView Lambdify task is therefore:

```text
solver: BVP_sci
frontend: AtomViewNative
execution: Lambdify
method: Sparse
```

The same frontend can be used with AOT by changing `execution` and providing
the generated-backend settings below.

## AOT and process handoff

```text
solver_options
generated_backend: sparse_aot_tcc
symbolic_backend: atomview
aot_build_policy: build_if_missing
aot_build_profile: release
aot_output_dir: test_reports/bvp_sci/aot
aot_handoff_path: test_reports/bvp_sci/aot/handoff.txt
aot_execution_policy: sequential
```

The lifecycle policies are:

- `use_if_available`: use an available linked artifact when possible;
- `build_if_missing`: build/publish only if the artifact is missing;
- `require_prebuilt`: consume only a published artifact, never compile;
- `rebuild_always`: force fresh materialization, build and link.

The producer writes manifest/provenance information to the handoff. A
consumer in another process reads that handoff with `require_prebuilt`; its
telemetry must report zero build attempts while retaining the artifact key and
link/reconnect provenance. Missing, stale, malformed, compiler, linker and
timeout failures remain typed errors. The Dense AOT route is intentionally not
the production generated route; use the supported generated Sparse/Banded
presets or the direct Lambdify Dense route.

## Continuation and restart

```text
continuation
# This name must be declared in equations.parameters.
parameter: rate
values: 0.5, 1.0, 2.0
# fresh: independent preparation; warm: reuse with previous state;
# prepared: explicit prepared-model lifecycle.
mode: prepared
# false means continuation from the previous accepted solution.
restart_each: false
```

For a restart, add per-segment `y0_values`, `t0_values` or `t_end_values`
where supported by the task schema. A mesh/state change invalidates the
affected numeric state but does not silently reuse an incompatible callback or
factor. The result exposes segment matrices and continuation telemetry so a
long series can distinguish fresh preparation, prepared reuse and restart
costs.

## Output and postprocessing

Prefer:

```text
postprocessing
output_policy: none       # none | plotters | gnuplot | terminal
write_report: true
report_path: reports/bvp_sci.md
```

CSV, TXT and Markdown reports are independent actions. `plot: true/false`
remains accepted for old files but should not be used in new task documents.

Use the direct typed BVP_sci Rust API when you need custom callbacks, detailed
solver configuration, dense output control or programmatic error recovery.

## Telemetry

Select one collection level with `BvpSciTelemetryMode`: `Off` disables runtime
collection, `Counters` records discrete work without reading the clock, and
`Timings` records durations without collecting discrete counts. `Detailed`
combines both and retains the per-mesh/Newton trace. Timing scopes may overlap;
do not add parent and child durations together.

## Ordered fixed-parameter solves

`parameter_names` declares unknowns that BVP_sci solves together with the
state profile. Fixed model inputs use the separate
`with_external_parameter_names` / `with_external_parameter_values` schema.
They are callback inputs, not additional Newton columns. The prepared
ExprLegacy, AtomViewNative or AOT evaluator is reused while values change.

Call `solve_parameter_loop(&[vec![...], ...])` to process points in order. A
successful point becomes the next initial mesh/profile and free-parameter
guess. A failed point retains its typed error and telemetry snapshot; the
solver restores the last successful checkpoint (or its original state if the
first point fails) and continues. Each telemetry snapshot is cumulative for
the solver at that point, not a per-point delta. Named external inputs are not
currently supported for numerical callbacks because those callback APIs do
not yet define a separate fixed-input channel.


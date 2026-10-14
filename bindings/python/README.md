# RustedSciThe Python Package

`rustedscithe` is the Python interface to RustedSciThe. It is a separate
package built with PyO3 and is intentionally kept outside the published core
Rust crate: the core crate remains usable without Python, while Python users
get a small, documented Python API.

The numerical and symbolic work is performed by compiled Rust code under the
hood, not by a second Python implementation. This gives Python users the same
native symbolic engine, prepared derivatives, solver lifecycles, telemetry and
typed diagnostics as the Rust API, while keeping the calling style idiomatic
for Python. Rust acceleration is an implementation advantage, not a promise
that every solver/layout/workload is universally fastest; matched workloads
and telemetry should guide frontend, execution and matrix-layout choices.

## Quick start

```python
from rustedscithe import Expr, RadauSolver

y = Expr.variable("y")
solver = RadauSolver(
    [-y], ["y"], "t", 0.0, 1.0, 0.1,
    rtol=1e-6,
    atol=1e-8,
)
solution = solver.solve([1.0])
print(solution.final_state)
```

The public Python spelling is deliberately not Rust syntax: use Python
keyword arguments, lists/tuples or NumPy arrays for data, and Python exception
classes for failures. The full builder-style options, prepared continuation,
telemetry and task-document APIs are described below and in `guides/`.

## Current scope

The first public slice contains:

- `Expr`, an immutable handle to RustedSciThe's native symbolic expression;
- typed Python exceptions for expression and configuration failures;
- shared solver vocabulary: `SolverOptions`, `SolverCapabilities`, symbolic
  frontend, execution route, matrix layout, execution policy, and telemetry;
- `capabilities("radau5")`, `capabilities("bdf")`, and
  `capabilities("backward_euler")`, plus `capabilities("lsode2")`,
  `capabilities("bvp_damp")` and `capabilities("bvp_sci")` route metadata.
- direct `BvpSciSolver`/`BvpSciSolution` for the symbolic Lambdify lifecycle;
  parsed BVP documents remain available through `TaskDocument.run_bvp()`.
- `NonlinearSolver` for the prepared square nonlinear-system methods and
  `FittingSolver` for canonical symbolic least-squares fitting, both with
  ExprLegacy/AtomNative Lambdify selection and detached telemetry.

Concrete numerical solver wrappers are added only after their Rust API has a
stable Python contract. `capabilities(...)` describes the routes that are
actually wired for each wrapper; a common option name is never treated as an
implicit implementation promise.

## Build locally

From the repository root, with a Python virtual environment active:

```text
python -m pip install maturin
python -m maturin develop --manifest-path bindings/python/Cargo.toml
```

The extension uses the stable Python ABI (`abi3`, Python 3.10 or newer), so a
single build can be used by compatible Python versions.

## Symbolic expressions

```python
from rustedscithe import Expr, InvalidExpressionError

x = Expr.variable("x")
y = Expr.variable("y")
model = x.pow(Expr.constant(2.0)) + y

assert model.variables() == ["x", "y"]
assert model.evaluate([2.0, 3.0]) == 7.0
assert model.diff("x").evaluate([2.0]) == 4.0
assert model.substitute("y", 5.0).evaluate([2.0]) == 9.0

compiled = model.lambdify(["y", "x"])
assert compiled.evaluate([3.0, 2.0]) == 7.0

try:
    Expr.parse("x +")
except InvalidExpressionError:
    pass
```

`evaluate(values)` follows the exact order returned by `variables()`. The
function methods include `exp`, `ln`, `sin`, `cos`, `tan`, `asin`, `acos`,
`atan` and `acot`; native aliases such as `arcsin` and `arctg` are also
available. No SymPy objects are involved: parsing, differentiation,
substitution and simplification use RustedSciThe's own symbolic engine.
`Expr.lambdify(...)` compiles a reusable native closure once; subsequent
`LambdifiedExpr.evaluate(...)` calls do not repeat symbolic preparation.

NumPy is the standard array interchange for the Python package. IVP/BVP and
nonlinear result objects expose owned `*_array()` methods, for example
`solution.times_array()`, `solution.states_array()`, `solution.mesh_array()`
and `solution.values_array()`. The older list-returning methods remain
available. The arrays are copies owned by NumPy; they are safe after solver
reuse and never borrow a mutable Rust workspace. The thin facade type hints
these inputs as NumPy `ArrayLike`, so ordinary lists, tuples and NumPy
vectors/matrices are accepted for initial states, meshes, guesses and
parameter values. Result arrays are always `float64`.

For a vector residual and its Jacobian, use `SymbolicSystem`:

```python
from rustedscithe import SymbolicSystem

system = SymbolicSystem([x.pow(Expr.constant(2.0)) + y, x * y], ["x", "y"])
assert system.evaluate([2.0, 3.0]) == [7.0, 6.0]
jacobian = system.jacobian()
compiled_system = system.lambdify()
assert compiled_system.evaluate([2.0, 3.0]) == [7.0, 6.0]
assert compiled_system.evaluate_jacobian([2.0, 3.0]) == [[4.0, 1.0], [3.0, 2.0]]
```

The Jacobian is differentiated by the Rust symbolic engine and remains a
vector of native `Expr` objects. `LambdifiedSystem` compiles each residual
once and reuses the closures for later evaluations.
The Jacobian closures are prepared alongside the residual closures, so
repeated `evaluate_jacobian(...)` calls do not differentiate or allocate a new
symbolic system on the hot path.

Task documents can express reusable symbolic aliases through a `where`-style
mapping. The binding resolves the dependency graph before applying it, so the
mapping order is irrelevant and cycles are reported as typed errors:

```python
rate = Expr.variable("rate")
scaled = Expr.variable("scaled")
expanded = (scaled * y).where_substitute({
    "scaled": rate + Expr.constant(1.0),
    "rate": Expr.constant(2.0),
})
assert expanded.evaluate([3.0, 4.0]) == 12.0
```

`SymbolicSystem.where_substitute(...)` applies the same operation to every
residual and rebuilds the variable schema after expansion. Scalar and
Lambdify evaluation reject non-finite inputs and outputs with
`InvalidExpressionError`; positivity assumptions are deliberately not
inferred from expression spelling and must remain an explicit solver policy.

## Nonlinear systems and fitting

`NonlinearSolver` is the direct Python entry point for square nonlinear
systems. It prepares the symbolic residual and analytic Jacobian once, then
uses the selected native nonlinear method:

```python
from rustedscithe import Expr, NonlinearSolver, SymbolicSystem

system = SymbolicSystem([Expr.parse("x^2 + y^2 - 25"), Expr.parse("x - y")], ["x", "y"])
solver = NonlinearSolver(
    system,
    method="levenberg_marquardt",
    frontend="atom_native",          # or "expr_legacy"
    execution_policy="sequential",   # or "parallel" with min_work
    telemetry="detailed",
    tolerance=1e-10,
)
assert not solver.is_prepared()
solver.prepare()                         # explicit cold preparation boundary
assert solver.is_prepared()
solution = solver.solve([3.0, 3.0])
assert solution.residual_norm < 1e-8
print(solution.solution(), solution.telemetry_counters())
```

Supported method names include `newton`, `damped_newton`,
`levenberg_marquardt`, `levenberg_marquardt_minpack`,
`nielsen_levenberg_marquardt`, `trust_region`, `powell_dogleg` and
`trust_region_lm`. Optional box bounds are supplied as `(lower, upper)` pairs.
`parameters` and `parameter_values` support repeated solves without repeating
symbolic preparation. `solve()` also prepares lazily when convenient, so the
explicit lifecycle is opt-in rather than a compatibility trap. Method-specific
numeric settings go in `method_options`, and boolean settings go in
`method_flags`; unknown keys raise `UnsupportedOptionError` instead of being
silently ignored. For example:

```python
solver = NonlinearSolver(
    system,
    method="levenberg_marquardt",
    method_options={"lambda_init": 1e-2, "max_lambda": 1e4},
    method_flags={"diag_scaling": False},
)
```

Logging is off unless `loglevel` is explicitly set.

For rectangular residuals and data fitting use `FittingSolver`. It is the
canonical symbolic least-squares route and supports the same Expr/Atom frontend
choice, parameter rebinding, positive-variable constraints and telemetry:

```python
from rustedscithe import Expr, FittingSolver

fit = FittingSolver(
    [Expr.parse("a*x - y"), Expr.parse("a*z - w")],
    [1.0],
    variables=["a"],
    parameters=["x", "y", "z", "w"],
    frontend="atom_native",
    telemetry="detailed",
    positive_variables=["a"],
)
fit_result = fit.solve_with_parameters([2.0, 4.0, 3.0, 6.0])
assert fit_result.solution()[0] == 2.0
print(fit_result.objective_function, fit_result.termination)
print(fit_result.diagnostics())
```

`FittingSolution.diagnostics()` preserves the native termination name,
objective value, iteration count and residual-evaluation count. Covariance and
R2 are intentionally not invented for a generic residual-equation API: those
statistics require an observation/weight contract and remain a separate fitting
adapter task.

The observation-based Gavin controller and symbolic VarPro fitter are exposed
as separate explicit classes rather than hidden behind `FittingSolver`:

```python
from rustedscithe import Expr, GavinFittingSolver, VarProSolver

x = [0.0, 1.0, 2.0, 3.0]
gavin = GavinFittingSolver(
    [Expr.parse("a*x+b")], x, [0.4, 2.1, 3.8, 5.5],
    "x", ["a", "b"], [1.0, 0.0],
    frontend="expr_legacy", telemetry="detailed",
)
gavin_result = gavin.solve()
print(gavin_result.parameters(), gavin_result.r_squared)

varpro = VarProSolver("x")
varpro.with_parameters(["tau"])
varpro.with_initial_parameters([1.0])
varpro.with_data(x, [2.0, 1.0, 0.5, 0.25])
varpro.with_basis("tau", Expr.parse("exp(-x/tau)"))
print(varpro.solve().solution())
```

Both classes preserve native typed failures and detach fit statistics such as
R2, covariance/correlation or fitted output where the underlying algorithm
defines them. Their current native builders are consumed by `solve()`, so they
are intentionally one-shot; use `FittingSolver` for prepared parameter
rebinds. Gavin supports ExprLegacy/AtomNative Lambdify preparation, opt-in
telemetry and positive-parameter constraints. VarPro's basis/invariant model
is built with symbolic differentiation before the numerical solve.

These wrappers currently expose the prepared Lambdify route. AOT is not
advertised for nonlinear/fitting until the Python boundary can execute and
report the selected compiled provider; an unsupported AOT route is rejected
rather than silently falling back.

### Python convenience facade

The same wheel also exports a thin Python-native facade. It does not replace
the native classes or reinterpret solver options:

```python
from rustedscithe import Expr, solve_ivp

y = Expr.variable("y")
solution = solve_ivp(
    "radau5", [-y], ["y"], "t", [1.0], 0.0, 1.0, 0.1,
    rtol=1e-5, atol=1e-7,
)
```

Use `create_ivp(...)` when the prepared solver object is needed, `task(...)`
for text/file task documents, and `run_task_batch(...)` for independent jobs.
The batch helper continues after failures and preserves `error_type` in its
compact result rows. `run_task_batch_report(...)` additionally emits a
scalar-only routing/status table with no trajectory objects, compiler output or
repeated warm-up messages. `render_table(...)` and `write_table(...)` produce
clean reports without redirecting compiler or process output.

The same lifecycle is available for canonical nonlinear systems and fitting:
`create_nonlinear(...)` and `create_fit(...)` return native solver objects so
callers can explicitly prepare, inspect telemetry and reuse a prepared model;
`solve_nonlinear(...)` and `fit(...)` are the concise one-shot forms.

The same pattern is available for the two direct BVP families. The boundary
argument is intentionally family-specific: BVP_sci receives a list of
boundary expressions, while BVP_Damp receives its state-to-boundary mapping.
All remaining keyword options are forwarded to the selected native adapter,
so layout, frontend, AOT/Lambdify, continuation and telemetry settings remain
visible:

```python
from rustedscithe import Expr, SolverOptions, solve_bvp

y = Expr.variable("y")
solution = solve_bvp(
    "bvp_sci", [-y], ["y"], [Expr.parse("ya_y - 1")],
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    options=SolverOptions(frontend="expr_legacy", matrix_layout="dense"),
    output="dense",
)
print(solution.mesh(), solution.values())
```

Use `create_bvp(...)` instead when `prepare()`, parameter rebinding or
restart is needed. For `"bvp_damp"`, pass for example
`{"y": [(0, 1.0)]}` as the boundary argument and provide the required
`bounds` option. The facade does not turn BVP_sci and BVP_Damp into one
lowest-common-denominator solver.

## First Numerical Adapter: BDF

The first concrete solver slice is intentionally narrow and honest. It uses
the native BDF implementation with `ExprLegacy` or `AtomNative`, dense linear
algebra, configurable sequential/parallel/auto evaluation, and either in-process `Lambdify` or the
generated AOT route:

```python
from rustedscithe import BdfSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BdfSolver(
    [-y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
    rtol=1e-5,
    atol=1e-7,
    options=SolverOptions(telemetry="counters"),
)
solver.prepare()                 # cold symbolic/runtime preparation
solution = solver.solve()       # numerical integration
assert solution.status == "finished"
assert solution.times()[0] == 0.0
print(solution.states()[-1][0])
print(solution.telemetry_counters())
print(solution.telemetry_timings_ms())
print(solver.telemetry_report())

# Reuse the prepared symbolic model for another initial state/interval.
restarted = solver.restart(0.5, [1.0], 1.0)
assert restarted.status == "finished"
```

Parameter continuation keeps the prepared symbolic callbacks and only
reinitializes the numerical history and factorization:

```python
a = Expr.variable("a")
solver = BdfSolver(
    [-a * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
    parameters=["a"],
    parameter_values=[1.0],
    options=SolverOptions(frontend="atom_native"),
)
solver.prepare()
solver.solve()
continued = solver.continue_with_parameters([2.0], 1.0)
```

For AOT, use the same builder with `execution="aot"` and the shared lifecycle
options. `build_if_missing` is the default policy; `require_prebuilt` and
`rebuild_always` are also accepted for explicit artifact workflows. The BDF
Python adapter currently exposes the Dense layout; its callback evaluation
policy may be Sequential, Parallel or Auto. Sparse/Banded BDF selection is
not silently mapped to Dense and remains a future adapter task.

For already prepared Python numerical callbacks, use `BdfCallbackSolver` rather
than constructing symbolic expressions:

```python
from rustedscithe import BdfCallbackSolver, SolverOptions

def residual(t, state):
    return [-state[0]]

def jacobian(t, state):
    return [[-1.0]]

callback_solver = BdfCallbackSolver(
    1, 0.0, 0.5, 0.1, [1.0], residual, jacobian,
    options=SolverOptions(telemetry="counters"),
)
callback_solution = callback_solver.solve()
```

The Jacobian argument is optional. Omitting it selects the native finite-
difference Jacobian policy. This callback entry point is intentionally Dense,
sequential and Lambdify-only; Python callables are not embedded into AOT
artifacts, and structured or parallel routes are rejected explicitly. Callback
shape, finiteness and raised Python exceptions are reported as `CallbackError`.

`BdfSolution` owns detached Python lists, so a later solver operation cannot
mutate a result that has already been returned. `BdfSolver` is intentionally
not shareable between threads because the native runtime owns mutable callback
and factorization state. Sparse/banded layouts are rejected explicitly until
their adapters are implemented; sequential/parallel/auto policies are
forwarded to the native callback scheduler, including for dense AOT.

The BDF-specific controls `first_step`, `max_bdf_order` and `max_steps` are
available as named constructor arguments. `first_step=None` keeps native
automatic startup selection, `max_bdf_order` must be in `1..=5`, and
`max_steps` bounds high-level integration attempts. Invalid values are
rejected before symbolic preparation and the selected values are retained in
`provenance()`.

All initial IVP wrappers expose stable `solver.provenance()` metadata.
It records the solver family, symbolic frontend, execution route, matrix
layout, execution policy and telemetry mode. This is intentionally a plain
detached mapping so reports can preserve the route without exposing mutable
Rust state. All detached IVP solutions expose `telemetry_mode`, structured
counters/timings and a compact, deterministic `telemetry_report()` for batch
logs. The BDF telemetry mappings distinguish preparation stages from numerical
stages. Runtime scopes such as
`linear_factorization_ms` and
`linear_matrix_assembly_ms` are nested diagnostics, so they must not be added
together; the compact `telemetry_report()` remains available for logs.
`solver.provenance()` returns the stable route metadata (`solver`, `frontend`,
`execution`, `matrix_layout`, `execution_policy` and telemetry mode). BDF
preparation failures, invalid continuation/configuration, numerical step
failures and exhausted work budgets use distinct Python exception classes.

Every initial IVP solution also exposes `solution.diagnostics()`. It returns a
detached common mapping with `solver`, `status`, `telemetry`, `sample_count`,
`state_dimension` and `final_time`, which is useful for batch runners that
handle several solver families without depending on solver-specific result
types. LSODE2 additionally reports its controller mode, active method family,
switch reason and whether a fallback was used. Detailed counters and stage
timings remain available through the solver-specific `telemetry_*` methods;
the common diagnostics mapping is intentionally compact and non-owning.

Each detached IVP solution also provides `write_telemetry_report(path)`. It
writes only the deterministic compact telemetry text returned by
`telemetry_report()` and does not redirect process stdout/stderr, so compiler
progress and other technical noise cannot enter the report file. Empty or
unwritable paths raise `TelemetryError`.

Invalid solver setup raises `InvalidConfigurationError`; native preparation
failures use `PreparationError`, exhausted work budgets use `ExhaustionError`,
Python residual/Jacobian failures use `CallbackError`, and numerical
Newton/linear failures use `SolverError`. All are rooted at
`RustedSciTheError`, so applications can catch the stable base class while
still distinguishing configuration from numerical failure. Every mapped
exception also carries machine-readable `category`, `stage`, `solver` and
`route` attributes. High-level native task execution, Gavin, VarPro and
callback-BDF operations use a typed `PanicError` boundary; PyO3's generated
trampoline remains the final ABI safety net for unexpected lower-level panics.

## First Numerical Adapter: Radau

Radau follows the same explicit prepare/solve lifecycle. The Python adapter
supports both symbolic frontend choices, Lambdify or AOT execution,
Dense/Sparse/Banded symbolic linear layouts, Sequential/Parallel/Auto callback
policies, an optional analytic state Jacobian, dense-output sampling and
parameter continuation:

```python
from rustedscithe import Expr, RadauSolver, SolverOptions

y = Expr.variable("y")
a = Expr.variable("a")
solver = RadauSolver(
    [-a * y], ["y"], "t", 0.0, 1.0, 0.25,
    rtol=1e-5,
    atol=1e-7,
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="timings",
    ),
    parameters=["a"],
    jacobian=[-a],
    output="dense",
)

solver.prepare()                         # symbolic preparation once
solution = solver.solve([1.0], [1.0])    # first parameter segment
print(solution.final_state)
print(solution.sample(0.5))               # dense output, when requested
print(solution.telemetry_counters())
print(solution.telemetry_timings_ms())

next_solution = solver.continue_with_parameters([1.0], [2.0])
print(next_solution.final_state)
```

`expr_legacy` is available by selecting `frontend="expr_legacy"`. The
frontend choice is independent from the execution route: `atom_native` and
`expr_legacy` describe symbolic preparation, while `lambdify` and `aot`
describe execution. For a structured route, provide the analytic Jacobian in
the symbolic state order and select the storage layout explicitly:

```python
structured = SolverOptions(
    frontend="expr_legacy",
    execution="lambdify",
    matrix_layout="banded",
    execution_policy="auto",
    telemetry="counters",
)
structured.set_banded_width(1, 1)
structured_solver = RadauSolver(
    [-y], ["y"], "t", 0.0, 1.0, 0.1,
    options=structured,
    jacobian=[Expr.constant(-1.0)],
)
structured_solution = structured_solver.solve([1.0])
```

Radau's native step controls are available as named constructor arguments:
`first_step=None` keeps automatic startup selection, while `max_steps`,
`max_newton_iterations` and `max_retries` bound the corresponding numerical
budgets. Values are validated before symbolic preparation and are retained in
`provenance()`, so an exhausted solve can be reproduced without inspecting
private Rust state.

For Python-owned numerical callbacks, use `RadauCallbackSolver`. It accepts a
`residual_callback(time, state) -> sequence` and an optional
`jacobian_callback(time, state) -> matrix`. Omitting the Jacobian selects the
native finite-difference fallback. This route is deliberately Dense,
sequential and Lambdify-only: Python callables cannot be embedded in AOT
artifacts or promise a Sparse/Banded structure. It has the same typed callback
errors and Radau step-budget controls as the symbolic wrapper.

The entry-point decision is intentional: use `RadauSolver` for symbolic
ExprLegacy/AtomNative preparation and Lambdify/AOT lifecycle, and use
`RadauCallbackSolver` when the residual is already a Python callable. The
callback class does not pretend to provide symbolic preparation, AOT embedding,
or structured sparsity metadata.

Sparse uses the prepared nonzero Jacobian pattern; Banded writes compact
`(lower + upper + 1) x n` storage. Neither route converts through a dense
matrix. The symbolic finite-difference fallback is intentionally Dense-only:
requesting Sparse or Banded without `jacobian=...` raises
`UnsupportedOptionError` rather than silently changing the algorithm.
Callback-only Radau entry points remain a separate contract because generic
Python closures cannot promise a structural Jacobian layout.

For a dense AOT run, change only the execution route and provide an optional
artifact directory or lifecycle policy. The same symbolic expressions and
continuation API are used:

```python
aot_solver = RadauSolver(
    [-a * y], ["y"], "t", 0.0, 1.0, 0.25,
    options=SolverOptions(
        frontend="expr_legacy",
        execution="aot",
        matrix_layout="dense",
        execution_policy="sequential",
        aot_policy="build_if_missing",
        aot_toolchain="gcc",
        aot_profile="release",
        telemetry="counters",
    ),
    parameters=["a"],
)
aot_solution = aot_solver.solve([1.0], [1.0])
print(aot_solution.telemetry_counters())
```

## First Numerical Adapter: Backward Euler

Backward Euler is available as a dense, sequential Lambdify or AOT route. Its native
core prepares symbolic callbacks lazily on the first solve, so the Python
contract exposes `validate()` for preflight validation and does not pretend
that validation is numerical preparation:

```python
from rustedscithe import BackwardEulerSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BackwardEulerSolver(
    [-y], ["y"], "t", [1.0], 0.0, 1.0,
    h=0.1,
    tolerance=1e-8,
    options=SolverOptions(frontend="atom_native", telemetry="counters"),
)
solver.validate()
solution = solver.solve()
print(solution.times())
print(solution.states()[-1])
print(solution.telemetry_counters())

# Continue from the last accepted state without rebuilding the expression.
continued = solver.continue_to(2.0)
```

The adapter supports dense sequential Lambdify and AOT. Sparse/banded layouts
and parallel execution are rejected with `UnsupportedOptionError` because the
native Backward Euler contract currently exposes only dense sequential state.
`restart(t0, y0, t_end)` is available for an explicit new initial state;
`continue_to(...)` is the prepared continuation operation and can optionally
rebind declared symbolic parameter values.

## Direct BVP_sci Adapter

The direct `BvpSciSolver` wrapper is the builder-style symbolic Lambdify
route. RHS expressions use the independent variable, state names and declared
parameter names. Boundary expressions use `ya_<state>` and `yb_<state>` for
the left and right endpoint values; they must contain exactly
`state_dimension + parameter_dimension` residuals. Boundary derivatives are
prepared symbolically by default, while `boundary_jacobian="finite_difference"`
selects the native finite-difference fallback.

```python
from rustedscithe import BvpSciSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BvpSciSolver(
    [-y], ["y"], [Expr.parse("ya_y - 1")],
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    options=SolverOptions(frontend="expr_legacy", matrix_layout="dense"),
    output="dense",
)
solution = solver.solve()
print(solution.mesh(), solution.values(), solution.sample(1.0))

# Reuse the prepared symbolic model for a new mesh/guess or parameter values.
solver.restart([0.0, 1.0], [1.0, 0.37])
solution = solver.solve()
```

SciPy's optional singular term `S*y/(x-a)` is available on the same direct
Lambdify route. Pass `singular_term_a` equal to the left mesh endpoint and a
row-major square `singular_term_matrix` with one row and column per state
variable. The two arguments must be supplied together; the adapter rejects
non-finite values, wrong matrix sizes and endpoint mismatches before building
the numerical solver:

```python
singular_solver = BvpSciSolver(
    [-y], ["y"], [Expr.parse("ya_y - 1")],
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    singular_term_a=0.0,
    singular_term_matrix=[0.0],
    options=SolverOptions(telemetry="counters"),
)
singular_solution = singular_solver.solve()
assert singular_solution.telemetry_counters()["singular_term_applications"] > 0
```

Sparse factorization is a separate policy from matrix layout. Select it on
`SolverOptions` only for a Sparse solve:

```python
options = SolverOptions(matrix_layout="sparse", telemetry="timings")
options.set_sparse_factorization("superlu")
solver = BvpSciSolver(
    [-y], ["y"], [Expr.parse("ya_y - 1")],
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37], options=options,
)
print(solver.sparse_factorization)
```

`superlu` is an opt-in binding feature and requires the target-native bridge
and `SUPERLU_LIBRARY` at runtime. If the bridge is unavailable, the native
BVP_sci core reports the condition and falls back to StandardFaer Sparse;
`superlu_fallbacks` remains visible in telemetry. Build the binding with
`--features extension-module,abi3,superlu` only when that optional route is
intended to be available.

The direct wrapper currently supports Lambdify with `ExprLegacy` or
`AtomNative` preparation and Dense/Sparse/Banded storage. AOT uses the shared
native generated-backend lifecycle and preserves its typed artifact/compiler/
link failures; configure it with `SolverOptions(execution="aot", ...)`.
Python-callable boundary callbacks are also supported as an explicit route.
Pass an empty `boundary_expressions` list and provide
`boundary_callback(ya, yb, parameters) -> list[float]`. An optional
`boundary_jacobian_callback(...)` must return
`(dya, dyb, dp)` as row-major nested lists. Omitting that callback uses the
native finite-difference boundary-Jacobian fallback. Python callback failures,
wrong shapes and non-finite values become typed `CallbackError` failures.
`BvpSciSolution` owns its mesh and values, exposes SciPy-style status codes,
dense sampling when requested, and provides compact counters/timings reports.

## Direct BVP_Damp Adapter

`BvpDampSolver` exposes the established damped Newton BVP route without
reimplementing its discretization. The initial guess is supplied in the same
node-major flat layout as `BvpSciSolver`; boundary conditions use the native
`{variable: [(0, value), (1, value)]}` representation, where `0` is the left
endpoint and `1` is the right endpoint:

```python
from rustedscithe import BvpDampSolver, Expr, SolverOptions

y = Expr.variable("y")
solver = BvpDampSolver(
    [-y], ["y"], {"y": [(0, 1.0)]},
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    options=SolverOptions(
        frontend="expr_legacy",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="timings",
    ),
    bounds={"y": (-10.0, 10.0)},
)
solution = solver.solve()
print(solution.mesh(), solution.values())
print(solution.telemetry_report())
```

`frontend` accepts `expr_legacy` or `atom_native`; `matrix_layout` accepts
`dense`, `sparse` or `banded`; and `execution` selects `lambdify` or the
native generated AOT lifecycle. The wrapper keeps the BVP_Damp strategy and
its native continuation semantics intact. The native solver may refine the
mesh, so callers should treat the returned mesh as the authoritative grid.
The solver-specific controls are forwarded directly rather than silently
defaulted. For example:

```python
solver = BvpDampSolver(
    [-y], ["y"], {"y": [(0, 1.0)]},
    [0.0, 0.5, 1.0], [1.0, 0.6, 0.37],
    options=SolverOptions(matrix_layout="dense", telemetry="timings"),
    max_jacobian_refreshes=3,
    max_damping_iterations=8,
    damping_factor=0.5,
    grid_method="pearson",
    grid_parameters=[1.0e-3, 1.4],
    max_refinements=4,
    logging_mode="detailed",
    logging_max_events=256,
    bounds={"y": (-10.0, 10.0)},
)
```

The available grid methods are `double_points`, `easy`, `pearson`,
`grcar_smooke`, `sci` and `two_point`; each validates its required parameter
count. For a Banded route, call `options.set_banded_width(lower, upper)` and
optionally select `banded_linear_solver` (`auto`, `banded`,
`general_banded_partial_pivot`, `block_tridiagonal` or `faer_sparse`) and
`banded_fallback` (`never` or `to_faer_sparse`). `iterative_refinement_steps`
is available for the native guarded refinement budget. `loglevel="off"` and
`logging_mode="off"` are the quiet default; structured logging is opt-in.

`Frozen` is intentionally not advertised by the direct Python class yet: it
is a native/task route stub and requesting `strategy="frozen"` raises a typed
`UnsupportedOptionError`. Likewise, a BVP_Damp `aot_output_dir` is rejected
until the native `GeneratedBackendConfig` has a corresponding output-directory
field; AOT build policy, toolchain, profile and process handoff are forwarded.
The direct result provides `diagnostics()`, `provenance()`,
`write_solution_csv(path)` and `write_solution_txt(path)` in addition to the
telemetry methods. Logging is disabled by this wrapper by default; telemetry
is selected through `SolverOptions`.

For a deliberately slower native combustion story, kept separate from the
fast contract tests, run:

```powershell
$env:RST_PYTHON_REPORT_DIR = "test_reports/python_bvp_damp_story"
& .\.venv-gavin-win\Scripts\python.exe -m pytest -q `
  bindings/python/tests/test_bvp_damp_story.py
```

The test writes one compact Markdown table for ExprLegacy and AtomNative. Its
timing columns use the native BVP_Damp scopes (`total_ms`,
`symbolic_operations_ms`, `residual_ms`, `jacobian_ms`, and
`factorization_ms`); they are not an artificial `prepare_ms`/`solve_ms`
split.

For a parameter sweep, declare symbolic parameters once and separate the cold
preparation from prepared solves:

```python
solver = BvpDampSolver(
    [Expr.parse("alpha * y - z"), Expr.parse("-z")],
    ["y", "z"],
    {"y": [(0, 0.0)], "z": [(0, 1.0)]},
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.25] * 10,
    parameters=["alpha"],
    parameter_values=[2.0],
    bounds={"y": (-10.0, 10.0), "z": (-10.0, 10.0)},
)
solver.prepare()
first = solver.solve_prepared()
solver.set_parameters([5.0])
second = solver.solve_prepared()
```

`set_parameters` rebinds numeric values and invalidates only numeric Jacobian
and factor state; it does not repeat symbolic preparation. `set_prepared_iterate`
installs a warm Newton iterate without invalidating the prepared plan, while
`restart` replaces the numeric initial guess, invalidates numeric factor state,
and keeps the prepared symbolic/runtime plan reusable.
The scalar `tolerance` is applied as both absolute and per-state relative
tolerance in this compact adapter.

## Task Documents

The Python package can parse the same IVP/BVP task documents as the executable
without duplicating the Rust DSL or silently selecting a solver route. The
initial binding is deliberately a parse/preview/check surface:

```python
from rustedscithe import TaskDocument

document = TaskDocument.parse("""
task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
""")

print(document.kind)       # "ivp"
print(document.route())    # normalized dispatch/frontend/layout metadata
print(document.preview())  # compact task table
print(document.check())    # semantic validation report

# IVP documents can also use the shared native task runner.
result = document.run()
print(result.times(), result.states(), result.status_code)
```

`TaskDocument.from_file(path)` provides the file-based equivalent. Invalid
documents and file-loading failures raise `TaskDocumentError`. `run()` is
available for IVP documents and returns a detached `IvpTaskResult`, including
continuation segment timing when the document requests continuation.
`route()` is available before execution and reports the normalized family,
method, stiff/nonstiff dispatch, symbolic frontend, execution route, matrix
layout and continuation presence. Stiff methods therefore remain visibly
native rather than being silently sent through the universal nonstiff API.
`options()` returns only options explicitly present in the document, using
normalized names and values; it does not invent native defaults. This is an
inspection method, while parsing, validation and execution remain owned by the
Rust task runner.

BVP documents use the explicit `run_bvp()` route and return a detached
`BvpTaskResult`. This keeps IVP and BVP result shapes distinct while reusing the
same core task runner:

```python
bvp = TaskDocument.parse("""
task
solver: BVP_sci
frontend: ExprLegacy
execution: Lambdify
method: Dense

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0
""")
solution = bvp.run_bvp()
print(solution.mesh(), solution.values(), solution.diagnostics())
```

`BvpTaskResult.mesh()` is the actual fixed or adaptive mesh of the final
segment, not a reconstructed placeholder. `continuation_segments()` preserves
the mesh and state matrix for every segment. `telemetry_counters()` and
`telemetry_report()` expose the native task-level lifecycle counters; an
unmeasured counter is omitted rather than reported as zero. BVP execution
failures raise `BvpError`, while parser and file-loading failures remain
`TaskDocumentError`. The core executable continues to own batch failure
handling.

The Python package is library-only: it does not spawn the Rust executable or
depend on a checkout-relative binary. Use the core executable for interactive
wizard, shell and process-level batch workflows; use `TaskDocument` and
`run_task_batch_report` when the workflow is already inside Python.

## First LSODE2 Adapter

The initial LSODE2 Python route exposes the native LSODE2 configuration
dimensions without collapsing them into one string option: `ExprLegacy` or
`AtomNative` preparation, `lambdify` execution, `dense`, `sparse` or `banded`
linear layout, and `sequential`, `parallel` or `auto` callback policy. Banded
systems require explicit `band_lower` and `band_upper` values:

```python
from rustedscithe import Expr, Lsode2Solver, SolverOptions

y = Expr.variable("y")
solver = Lsode2Solver(
    [-y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
    options=SolverOptions(
        frontend="atom_native",
        execution="lambdify",
        matrix_layout="dense",
        execution_policy="sequential",
        telemetry="timings",
    ),
)
solver.prepare()
solution = solver.solve()
print(solution.final_state)
print(solution.telemetry_report())
```

`Lsode2Solution` returns detached time/state lists and stage-aware counters
for symbolic preparation, integration, residual/Jacobian callbacks and linear
solves. The first Python controller is explicitly BDF. A pair of Python-owned
analytical residual/Jacobian callbacks is supported for the native Dense,
Sparse and Banded routes. Callback exceptions, wrong shapes and non-finite
values become `CallbackError` rather than being silently ignored. Python
callbacks require `execution="lambdify"`; AOT uses the
symbolic expression route. Adams and automatic controller selection are still
rejected with typed exceptions rather than silently mapped to BDF. The native LSODE2 API
already supports broader controller and lifecycle policies; those routes will
be added after their Python ownership and artifact contracts are specified.

The callback route keeps the expression list as the state-schema input, but
uses the callbacks for numerical residual/Jacobian evaluation. This makes the
boundary explicit and avoids pretending that a Python callback is an AOT or
Lambdify artifact:

```python
def residual(t, y):
    return [-y[0]]

def jacobian(t, y):
    return [[-1.0]]

solver = Lsode2Solver(
    [Expr.constant(0.0)], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
    options=SolverOptions(matrix_layout="sparse"),
    residual_callback=residual,
    jacobian_callback=jacobian,
)
solution = solver.solve()
```

Both callbacks must be supplied together. The callback executes at the
Python/Rust boundary on every requested evaluation, so this route is intended
for API compatibility and analytical-callback experiments, not as a claim of
parallel Python speedup. Use the symbolic route when the residual and
Jacobian can be expressed in the Rust symbolic layer.

The initial AOT route is available for all three LSODE2 layouts. By default it
uses the native `BuildIfMissing` lifecycle with `C + gcc` and a Release
profile. The lifecycle can be selected explicitly without reimplementing
code generation in Python:

```python
aot_solver = Lsode2Solver(
    [-y], ["y"], "t", [1.0], 0.0, 0.2, 0.05,
    options=SolverOptions(
        frontend="atom_native",
        execution="aot",
        matrix_layout="sparse",
        aot_policy="build_if_missing",  # require_prebuilt or rebuild_always
        aot_toolchain="gcc",             # gcc, tcc, zig or rust
        aot_profile="release",           # debug or release
        aot_output_dir="target/python-ivp-aot",
        telemetry="timings",
    ),
)
aot_solver.prepare()
aot_solution = aot_solver.solve()
```

`solver.provenance()` records the selected AOT policy, toolchain, profile and
artifact directory. AOT-only fields are rejected on Lambdify routes instead of
being silently ignored. The explicit handoff path below is the opt-in bridge
for a process-isolated producer/consumer workflow. The subprocess contract is
covered by `test_lsode2_aot_process_isolated_producer_consumer_handoff`.

AOT preparation failures retain their lifecycle category across the Python
boundary. Catch `AotArtifactError` for missing or not-yet-built artifacts,
`AotCompilerError` for compiler failures, `AotLinkError` for link or runtime
library failures, and `AotTimeoutError` for toolchain deadlines. Other
lifecycle classes are reported as `AotLifecycleError`; callers do not need to
match human-readable error strings.

For a process-isolated producer/consumer workflow, set the same explicit
registry path on both `SolverOptions` objects with
`options.set_aot_handoff_path("target/python-ivp-aot/registry.txt")`. The
producer uses `BuildIfMissing` or `RebuildAlways`; the consumer uses
`RequirePrebuilt`. The path and selected policy are retained in
`solver.provenance()`. After `solve()`, `solution.aot_artifact_keys()` and the
`aot_build_attempts`/`aot_reconnects` counters provide a detached check that
the consumer reconnected to the producer's artifact rather than rebuilding.

Parameter continuation reuses the prepared symbolic route and starts at the
last successful endpoint. It does not silently restart from the original
`y0`; use `restart(t0, y0, t_end)` when a new initial state is intended:

```python
a = Expr.variable("a")
solver = Lsode2Solver(
    [-a * y], ["y"], "t", [1.0], 0.0, 0.5, 0.1,
    parameters=["a"], parameter_values=[1.0],
)
first = solver.solve()
continued = solver.continue_with_parameters([2.0], 1.0)
```

Calling `continue_with_parameters` before one successful solve raises
`InvalidConfigurationError`; this prevents an ambiguous "prepared but no
current state" lifecycle.

## Shared solver contract

```python
from rustedscithe import SolverOptions, capabilities

options = SolverOptions(
    frontend="expr_legacy",
    execution="lambdify",
    matrix_layout="dense",
    execution_policy="sequential",
    telemetry="timings",
)
options.validate_for("bdf")
print(capabilities("bdf").executions)
```

The vocabulary is deliberately independent: `atom_native`/`expr_legacy` are
symbolic preparation routes, `lambdify`/`aot` are execution routes, and
`dense`/`sparse`/`banded` are matrix layouts. Unsupported combinations raise a
typed exception instead of being silently ignored.
`capabilities()` reports the routes currently exposed by this Python package;
native Rust routes that are not yet wrapped are not advertised as Python
support. For layout-specific callback contracts, inspect
`capabilities("lsode2").analytical_callback_layouts`; the current value is
`["dense", "sparse", "banded"]` for native Lambdify execution.

Native-mapped exceptions retain stable diagnostic attributes in addition to
their Python class and message: `stage`, `solver` and `route`. These fields
are intended for programmatic routing and logging; applications should not
parse the human-readable message.

LSODE2 accepts `method="bdf"`, `method="adams"`, or
`method="auto"` (the `"lsoda"` alias selects the same automatic
Adams/BDF controller). The returned solution reports both the requested
method and the actual controller family. Python analytical callbacks are
supported for all three native layouts; they remain incompatible with AOT,
where callbacks must be represented by symbolic expressions before code
generation.

The current IVP capability matrix is intentionally explicit:

| Python solver | methods | execution | layouts | analytical callbacks |
| --- | --- | --- | --- | --- |
| `RadauSolver` / `RadauCallbackSolver` | `radau5` | Lambdify, AOT / callback Lambdify | Dense, Sparse, Banded / callback Dense | callback Dense only |
| `BdfSolver` | `bdf` | Lambdify, AOT | Dense | symbolic route only |
| `BackwardEulerSolver` | `backward_euler` | Lambdify, AOT | Dense | symbolic route only |
| `Lsode2Solver` | `bdf`, `adams`, `auto`, `lsoda` | Lambdify, AOT | Dense, Sparse, Banded | native Lambdify |
| `TaskDocument.run_bvp()` | `damped`, `frozen`, `scipy_like` task routes | Lambdify and parsed AOT routes | Dense, Sparse, Banded (route-dependent) | task-document symbolic route |

Use `capabilities(name)` rather than inferring support from a solver name.
This table describes the Python package contract; native Rust APIs may expose
additional controls that are intentionally not flattened into one wrapper.

## Tests

Rust contract tests run with:

```text
cargo test --manifest-path bindings/python/Cargo.toml
```

On Windows, this Rust test command links the PyO3 test executable and needs a
Python development import library (`python3.lib` for the configured ABI), not
only a virtual environment. If that SDK artifact is unavailable, use
`cargo check` plus the wheel-installed Python suite below; do not interpret a
linker failure as a numerical or binding-contract failure.

The Python smoke tests in `tests/` are intended to run after `maturin develop`
with `python -m pytest bindings/python/tests`.

The Rust binding test suite also contains native-adapter parity gates for the
dense ExprLegacy/Lambdify routes of BDF, Radau5 and Backward Euler, plus a
dense sequential AOT smoke gate for all three. The Python suite additionally
covers Radau Sparse/Banded symbolic routing, structured-Jacobian rejection,
process-isolated AOT handoff and BDF/Radau execution policies. These are
adapter-contract checks, not claims of SciPy parity or universal performance;
the full callback and benchmark matrices remain separate release work.

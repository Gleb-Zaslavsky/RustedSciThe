# Optimization and Data-Fitting Guide

This module provides nonlinear least-squares fitting APIs. It is intended for
estimating model parameters from observations, not as a replacement for the
ODE/BVP solvers or a general-purpose constrained optimizer.

## Choose a fitting method

| Method | Choose it when | Main distinction |
|---|---|---|
| `Method::LM` | You need a general symbolic nonlinear least-squares fit. | Uses the crate's canonical least-squares LM implementation. |
| `Method::GAVIN` | You specifically want the Gavin damping/Broyden controller. | Separate algorithm with Marquardt, Quadratic, and Nielsen update policies; supports positive parameters and observation weights. |
| `Method::VARPRO` | The model is separable into nonlinear basis parameters and linear coefficients. | Eliminates/solves linear coefficients inside the nonlinear search. |

There is no universally fastest method. Compare methods on the same data,
initial guess, and error criteria. VarPro is specialized and is not a drop-in
replacement for a general LM fit.

## Basic fit with `UniversalFitting`

The builder collects data, model, unknown names, and initial values, then
`build()` solves the selected route. `ExprLegacy` and sequential evaluation
are the defaults.

```rust,ignore
use RustedSciThe::numerical::optimization::{
    Method, UniversalFitting, UniversalFittingResult,
};

let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.4 * (-x / 1.7_f64).exp() + 0.35).collect();

let result = UniversalFitting::new()
    .with_method(Method::LM)
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_max_iterations(500)
    .build()?;

let UniversalFittingResult::LM(fit) = result else {
    unreachable!("the selected method is LM");
};
let parameters = fit.solution_map().expect("fit has fitted parameters");
```

`with_equation_str` accepts a model expression. The listed unknowns are fitted
parameters; the `with_arg` name identifies the independent variable associated
with each `x` sample. Supply vectors with matching lengths and an initial guess
in the same order as the unknown names.

The generic builder accepts `&str` and string iterators. Existing owned
`String` and `Vec<String>` calls remain valid.

## Select the symbolic frontend

For supported symbolic Lambdify routes, choose between:

- `SymbolicLambdifyFrontend::ExprLegacy`: the default Expr-based path.
- `SymbolicLambdifyFrontend::AtomViewNative`: the packed AtomView-native path.

LM and Gavin support both. VarPro currently supports `ExprLegacy` and
sequential execution only; asking UniversalFitting for an unsupported VarPro
combination returns `UniversalFittingError::UnsupportedOption` rather than
silently ignoring the setting.

```rust,ignore
use RustedSciThe::numerical::optimization::{Method, UniversalFitting};
use RustedSciThe::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend,
};

let x = vec![0.0, 0.5, 1.0];
let y = vec![1.0, 0.8, 0.6];
let result = UniversalFitting::new()
    .with_method(Method::GAVIN)
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_gavin_positive_parameters(["tau"])
    .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
    .build()?;
```

Parallel callback evaluation is opt-in. The `min_work` threshold avoids
dispatching small callback workloads to Rayon; parallelism is not guaranteed
to improve total fit time. Benchmark the full fit, not only preparation or one
callback stage.

The fitting APIs described here use Lambdify. AOT is not currently a fitting
backend.

## Gavin-specific controls

Gavin is a distinct LM-style controller, not an alias for the canonical LM
method. Through `UniversalFitting`, use `Method::GAVIN` or `.with_gavin()`.
Its focused options include:

```rust,ignore
use RustedSciThe::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;
use RustedSciThe::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;

use RustedSciThe::numerical::optimization::{UniversalFitting};
let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.4 * (-x / 1.7_f64).exp() + 0.35).collect();
let weights = vec![1.0; x.len()];
let result = UniversalFitting::new()
    .with_gavin()
    .with_data(x, y)
    .with_equation_str("a * exp(-x / tau) + b")
    .with_arg("x")
    .with_unknowns(["a", "tau", "b"])
    .with_initial_guess(vec![1.8, 1.2, 0.2])
    .with_gavin_positive_parameters(["tau"])
    .with_gavin_update_policy(GavinLmUpdatePolicy::Marquardt)
    .with_gavin_solver_controls(1000, 1e-6, 1e-6, 1e-1)
    .with_gavin_weights(weights)
    .with_telemetry(LeastSquaresTelemetryMode::Detailed)
    .with_logging(false)
    .build()?;
```

`with_gavin_solver_controls` takes the iteration budget, gradient tolerance,
parameter-step tolerance, and gain-ratio threshold, in that order. The update
policy can be Marquardt, Quadratic, or Nielsen. Weights must be nonnegative and
match the observation layout. Positive-parameter constraints are explicit
user choices; they are useful whenever the model domain requires positivity,
not only for chemical models.

`with_tolerance` maps to the parameter-step tolerance on Gavin, and
`with_g_tolerance` maps to its gradient tolerance. `f_tolerance` and
`scale_diag` have no equivalent in Gavin and are rejected as typed unsupported
options. Prefer the Gavin-specific controls when configuring Gavin so the
controller semantics remain explicit.

For direct access to preparation telemetry and Gavin result provenance, use
`GavinSymbolicFitting` from `optimization::gavin_fitting`; see the dedicated
example `rus_gavin_fitting_guide`.

## Variable projection

Use VarPro when the model has a separable form such as
`y(x) = a * exp(-x / tau) + b`: `tau` is nonlinear, while `a` and `b` are
linear coefficients. The generic wrapper receives nonlinear parameters via
`with_parameters`, basis terms via `with_basis_str`, and invariant/background
terms via `with_equations`.

```rust,ignore
use RustedSciThe::numerical::optimization::{UniversalFitting, UniversalFittingResult};

let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
let y: Vec<f64> = x.iter().map(|&x| 2.5 * (-x / 1.7_f64).exp() + 0.8).collect();
let result = UniversalFitting::new()
    .with_varpro()
    .with_data(x, y)
    .with_arg("x")
    .with_parameters(["tau"])
    .with_initial_guess(vec![1.2])
    .with_basis_str("tau", "exp(-x/tau)")
    .with_equations_str(["1"])
    .build()?;
let UniversalFittingResult::VARPRO(fit) = result else {
    unreachable!("the selected method is VarPro");
};
let nonlinear_parameters = fit.nonlinear_parameters();
let linear_coefficients = fit
    .linear_coefficients()
    .expect("the fitted model has linear coefficients");
```

Use the dedicated `SymbolicVarProBuilder` API when you need its lower-level
model construction or solver configuration.

## Telemetry, logging, and errors

Telemetry and logging are disabled by default. Enable telemetry explicitly:

```rust,ignore
use RustedSciThe::numerical::optimization::UniversalFitting;
use RustedSciThe::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;

let configured = UniversalFitting::new()
    .with_telemetry(LeastSquaresTelemetryMode::Detailed);
```

The canonical least-squares solver does not collect runtime counters or read
the clock in `Off` mode. Gavin currently exposes one complete solve snapshot,
so either enabled telemetry level requests that snapshot. Symbolic preparation
is a separate lifecycle; the direct Gavin builder additionally exposes
`with_preparation_telemetry`.

Logging is opt-in with `.with_logging(true)` and uses the application's
installed `log` facade; the library does not install a logger. Universal
settings are shared where semantics match. Unsupported options return a typed
error rather than being silently reinterpreted.

`UniversalFitting::build()` returns `Result<UniversalFittingResult,
UniversalFittingError>`. Gavin's direct builder returns `GavinLmError`, which
classifies configuration, symbolic preparation, callback, numerical, and
statistical failures. Inspect the selected result variant and solver
termination/report as well as the fitted parameter values.

## Examples and evidence

Run the maintained examples from the repository root:

```powershell
cargo run --no-default-features --example rus_universal_fitting_guide
cargo run --no-default-features --example rus_gavin_fitting_guide
```

The first demonstrates the classic LM and VarPro facade routes; the second
shows the dedicated Gavin builder. Story and performance evidence, including
matched ExprLegacy/AtomView measurements, is maintained in
[`GAVIN_STORY_TESTS.md`](GAVIN_STORY_TESTS.md). Current evidence shows that
frontend speed depends on sample count and controller policy; do not infer a
universal winner from preparation time alone.

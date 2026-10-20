# Python Examples

These scripts use the installed `rustedscithe` facade and the native Rust
solvers. Run them from `bindings/python/` after installing the package or
building the extension with Maturin. NumPy is required for array-oriented
examples; the package declares it as a runtime dependency.

Run an individual guide from this directory with `python examples/<name>.py`.
Each script begins with the mathematical problem and the result to expect.
The printed timings and counters help explain the selected route; they are not
controlled performance measurements. For correctness and performance claims,
use the core crate's dedicated tests and benchmark reports.

Start with the [English overview](../guides/PYTHON_SOLVERS_GUIDE_EN.md) or
[Russian overview](../guides/PYTHON_SOLVERS_GUIDE_RU.md). Solver-specific
English/Russian guides are paired below; each explains the Python API and its
native solver's route and numerical policies.

## Choosing a route

The common options describe separate decisions, not one backend name:

| Choice | What it controls | Typical values and cautions |
| --- | --- | --- |
| `frontend` | How symbolic expressions are prepared | `expr_legacy` or `atom_native`; `numerical` is a direct-callback route only on selected solvers. |
| `execution` | How prepared equations run | `lambdify` is the usual no-compiler route; `aot` generates/compiles native callbacks; `numerical` means user callbacks where supported. |
| `matrix_layout` | Storage/solver for the numerical linear system | `dense`, `sparse` or `banded`; availability and bandwidth requirements are solver-specific. There is no universally fastest layout. |
| `execution_policy` | Scheduling of eligible callback entries inside the Rust core | `sequential`, `parallel` or `auto`; this does not make Python callbacks parallel or release the Python GIL contract. |
| `telemetry` | Native solve measurements | `off`, `counters`, `timings` or `detailed`; common `SolverOptions` defaults to `timings`, but standalone facades can have their own defaults. |

Not every Cartesian combination is supported. Check
`rustedscithe.capabilities("bvp_sci")` for a solver's advertised axes, then
read its route guide for cross-axis restrictions and concrete examples. The
capability axes are not a promise that every combination is valid. For
method-specific fitting dispatch, query `universal_fitting_lm`,
`universal_fitting_gavin` or `universal_fitting_varpro`; the aggregate
`universal_fitting` record describes the union across its methods. The
constructor rejects unsupported combinations instead of silently switching
to another frontend, execution route or matrix layout.

| Script | Demonstrates | Guidance |
| --- | --- | --- |
| [`radau_basic.py`](radau_basic.py) | Minimal stiff IVP with Radau5 and telemetry | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`radau_robertson_routes.py`](radau_robertson_routes.py) | Correctness-gated table for one stiff kinetic system over both frontends and all layouts (not a benchmark) | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`radau_robertson_aot.py`](radau_robertson_aot.py) | Optional compiler-backed Radau AOT preparation and provenance | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`bdf_continuation.py`](bdf_continuation.py) | Prepared BDF solves and parameter rebinding | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`bdf_robertson.py`](bdf_robertson.py) | Robertson stiff chemistry and mass conservation | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`bdf_aot_build.py`](bdf_aot_build.py) | Optional BDF AOT build lifecycle with local compiler | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`backward_euler_basic.py`](backward_euler_basic.py) | Fixed-step implicit Euler and first-order accuracy | [EN](../guides/BACKWARD_EULER_GUIDE_EN.md) / [RU](../guides/BACKWARD_EULER_GUIDE_RU.md) |
| [`backward_euler_routes.py`](backward_euler_routes.py) | Step refinement over ExprLegacy and AtomNative | [EN](../guides/BACKWARD_EULER_GUIDE_EN.md) / [RU](../guides/BACKWARD_EULER_GUIDE_RU.md) |
| [`lsode2_controller_matrix.py`](lsode2_controller_matrix.py) | Exact-solution-gated table comparing requested and active Adams, BDF and Auto policies (not a benchmark) | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`lsode2_aot_continuation.py`](lsode2_aot_continuation.py) | Compile one AtomNative LSODE2 route and reuse it through parameter continuation | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`ode_api2_nonstiff.py`](ode_api2_nonstiff.py) | ODE_api2 RK45/DOPRI/AB4 selection and its fixed-step boundary semantics | [EN overview](../guides/PYTHON_SOLVERS_GUIDE_EN.md#ivp) / [RU overview](../guides/PYTHON_SOLVERS_GUIDE_RU.md#ivp-и-continuation) |
| [`lsode2_lorenz.py`](lsode2_lorenz.py) | Chaotic Lorenz IVP and automatic controller selection | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`lsode2_three_body.py`](lsode2_three_body.py) | Twelve-state gravitational three-body problem | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`callback_ivp.py`](callback_ivp.py) | Native callback residual with analytic or FD Jacobian | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`bvp_sci_sparse.py`](bvp_sci_sparse.py) | BVP_sci with sparse layout and symbolic residuals | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_sci_frontend_layouts.py`](bvp_sci_frontend_layouts.py) | Correctness-gated table for both frontends and all layouts on an exact profile (not a benchmark) | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_sci_aot.py`](bvp_sci_aot.py) | Same Sparse/AtomNative BVP through Lambdify and compiler-backed AOT | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_sci_numerical_callbacks.py`](bvp_sci_numerical_callbacks.py) | Dense BVP_sci numerical RHS with an analytical state Jacobian | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`shooting_bvp_harmonic.py`](shooting_bvp_harmonic.py) | Small exact harmonic BVP, AtomNative preparation, solve telemetry and optional phase portrait | [EN](../guides/SHOOTING_BVP_GUIDE_EN.md) / [RU](../guides/SHOOTING_BVP_GUIDE_RU.md) |
| [`bvp_damp_observations.py`](bvp_damp_observations.py) | Boundary observations, bounds and telemetry | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`bvp_damp_layout_matrix.py`](bvp_damp_layout_matrix.py) | Correctness-gated table for both frontends and all layouts on an exact profile (not a benchmark) | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`bvp_damp_naive_continuation.py`](bvp_damp_naive_continuation.py) | Native fixed-mesh Naive Newton with prepared parameter rebinding | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`bvp_damp_frozen_policy.py`](bvp_damp_frozen_policy.py) | Frozen-Jacobian Newton with an explicit reuse policy and prepared continuation | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`nonlinear_and_fitting.py`](nonlinear_and_fitting.py) | Nonlinear root solve and least-squares fitting | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`nonlinear_parameter_sweep.py`](nonlinear_parameter_sweep.py) | Reuse prepared AtomNative nonlinear equations while rebinding numeric parameters | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`nonlinear_aot.py`](nonlinear_aot.py) | AtomNative AOT preparation and repeated parameterized nonlinear solves | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`lm_observation_fit.py`](lm_observation_fit.py) | Symbolic LM residual system, repeated data rebinds and solve telemetry | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`gavin_exponential_fit.py`](gavin_exponential_fit.py) | Gavin fit of exponential decay with a positive time-scale constraint | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`varpro_exponential_fit.py`](varpro_exponential_fit.py) | Variable projection separating nonlinear decay shape from linear coefficients | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`universal_fitting_routes.py`](universal_fitting_routes.py) | Native method selection across LM, Gavin and VarPro | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`symbolic_calculus.py`](symbolic_calculus.py) | Differentiation, recursive substitutions, integration, AtomNative preparation and NumPy batches | [EN](../guides/SYMBOLIC_MATH_GUIDE_EN.md) / [RU](../guides/SYMBOLIC_MATH_GUIDE_RU.md) |
| [`symbolic_arrays.py`](symbolic_arrays.py) | Symbolic vector/matrix algebra and owned NumPy outputs | [EN](../guides/SYMBOLIC_MATH_GUIDE_EN.md) / [RU](../guides/SYMBOLIC_MATH_GUIDE_RU.md) |
| [`data_processing_signal.py`](data_processing_signal.py) | Raw/Savitzky-Golay/LOWESS quality comparison, spline resampling and NumPy shape checks | Data-processing API is documented in the script; native APIs are listed in `src/numerical/data_processing/` |
| [`data_processing_bspline_interpolator.py`](data_processing_bspline_interpolator.py) | Clamped B-spline interpolation from control values, owned batch output and endpoint checks | Data-processing API is documented in the script; native APIs are listed in `src/numerical/data_processing/` |
| [`data_processing_lsq_spline.py`](data_processing_lsq_spline.py) | Correctness-gated DenseQR/Banded weighted B-spline fit table and batch evaluation (not a benchmark) | Data-processing API is documented in the script; native APIs are listed in `src/numerical/data_processing/` |
| [`somelinalg_banded.py`](somelinalg_banded.py) | Compact faithful LAPACK-style banded LU, factor reuse, multiple RHS and explicit refinement | [EN](../guides/SOMELINALG_BANDED_GUIDE_EN.md) / [RU](../guides/SOMELINALG_BANDED_GUIDE_RU.md) |
| [`somelinalg_tridiagonal.py`](somelinalg_tridiagonal.py) | Pivoted tridiagonal LU, pivot-required correctness case, factor reuse and padded multiple RHS | [EN](../guides/SOMELINALG_TRIDIAGONAL_GUIDE_EN.md) / [RU](../guides/SOMELINALG_TRIDIAGONAL_GUIDE_RU.md) |
| [`somelinalg_block_tridiagonal.py`](somelinalg_block_tridiagonal.py) | Compact block-tridiagonal LU, multiple RHS, diagnostics and iterative refinement | [EN](../guides/SOMELINALG_BLOCK_TRIDIAGONAL_GUIDE_EN.md) / [RU](../guides/SOMELINALG_BLOCK_TRIDIAGONAL_GUIDE_RU.md) |
| [`lsode2_lorenz_animation.py`](lsode2_lorenz_animation.py) | Lorenz solve and default 3D animation; optional 2D/both or `--no-animate` | [EN](../guides/TRAJECTORY_ANIMATION_GUIDE_EN.md) / [RU](../guides/TRAJECTORY_ANIMATION_GUIDE_RU.md) |
| [`lsode2_three_body_animation.py`](lsode2_three_body_animation.py) | Three-body solve and default selected-body XY animation (use `--no-animate` for headless mode) | [EN](../guides/TRAJECTORY_ANIMATION_GUIDE_EN.md) / [RU](../guides/TRAJECTORY_ANIMATION_GUIDE_RU.md) |
| [`task_batch.py`](task_batch.py) | Failure-tolerant task batch and compact status table | [Overview EN](../guides/PYTHON_SOLVERS_GUIDE_EN.md#text-tasks-and-examples) / [Overview RU](../guides/PYTHON_SOLVERS_GUIDE_RU.md) |

The examples are intentionally small and deterministic. They demonstrate
configuration and result inspection; use the Rust crate's story tests and
benchmarks for broad correctness and performance evidence. A timing printed
by an example is not a benchmark claim.

## Wrapper boundary diagnostic

`python ../benchmarks/python_boundary_bvp.py --repeats 7 --nodes 64` prints a
compact Markdown comparison of constructor time, Python solve wall time and
native telemetry for the same exact BVP through ExprLegacy and numerical
callbacks. It is diagnostic evidence only: the callback path includes
Python/GIL crossings, so it is not a matched frontend-performance claim.
Pass `--output path/to/report.md` to save the table separately.

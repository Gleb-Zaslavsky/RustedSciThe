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

| Script | Demonstrates | Guidance |
| --- | --- | --- |
| [`radau_basic.py`](radau_basic.py) | Minimal stiff IVP with Radau5 and telemetry | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`radau_robertson_routes.py`](radau_robertson_routes.py) | Same stiff kinetic system over both frontends and all layouts | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`radau_robertson_aot.py`](radau_robertson_aot.py) | Optional compiler-backed Radau AOT preparation and provenance | [EN](../guides/RADAU_GUIDE_EN.md) / [RU](../guides/RADAU_GUIDE_RU.md) |
| [`bdf_continuation.py`](bdf_continuation.py) | Prepared BDF solves and parameter rebinding | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`bdf_robertson.py`](bdf_robertson.py) | Robertson stiff chemistry and mass conservation | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`bdf_aot_build.py`](bdf_aot_build.py) | Optional BDF AOT build lifecycle with local compiler | [EN](../guides/BDF_GUIDE_EN.md) / [RU](../guides/BDF_GUIDE_RU.md) |
| [`backward_euler_basic.py`](backward_euler_basic.py) | Fixed-step implicit Euler and first-order accuracy | [EN](../guides/BACKWARD_EULER_GUIDE_EN.md) / [RU](../guides/BACKWARD_EULER_GUIDE_RU.md) |
| [`backward_euler_routes.py`](backward_euler_routes.py) | Step refinement over ExprLegacy and AtomNative | [EN](../guides/BACKWARD_EULER_GUIDE_EN.md) / [RU](../guides/BACKWARD_EULER_GUIDE_RU.md) |
| [`lsode2_controller_matrix.py`](lsode2_controller_matrix.py) | Matched Adams, BDF and Auto controller routes | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`lsode2_lorenz.py`](lsode2_lorenz.py) | Chaotic Lorenz IVP and automatic controller selection | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`lsode2_three_body.py`](lsode2_three_body.py) | Twelve-state gravitational three-body problem | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`callback_ivp.py`](callback_ivp.py) | Native callback residual with analytic or FD Jacobian | [EN](../guides/LSODE2_GUIDE_EN.md) / [RU](../guides/LSODE2_GUIDE_RU.md) |
| [`bvp_sci_sparse.py`](bvp_sci_sparse.py) | BVP_sci with sparse layout and symbolic residuals | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_sci_frontend_layouts.py`](bvp_sci_frontend_layouts.py) | Matched frontend/layout matrix on an exact profile | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_sci_aot.py`](bvp_sci_aot.py) | Same Sparse/AtomNative BVP through Lambdify and compiler-backed AOT | [EN](../guides/BVP_SCI_GUIDE_EN.md) / [RU](../guides/BVP_SCI_GUIDE_RU.md) |
| [`bvp_damp_observations.py`](bvp_damp_observations.py) | Boundary observations, bounds and telemetry | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`bvp_damp_layout_matrix.py`](bvp_damp_layout_matrix.py) | Exact profile over both frontends and all layouts | [EN](../guides/BVP_DAMP_GUIDE_EN.md) / [RU](../guides/BVP_DAMP_GUIDE_RU.md) |
| [`nonlinear_and_fitting.py`](nonlinear_and_fitting.py) | Nonlinear root solve and least-squares fitting | [EN](../guides/NONLINEAR_FITTING_GUIDE_EN.md) / [RU](../guides/NONLINEAR_FITTING_GUIDE_RU.md) |
| [`symbolic_calculus.py`](symbolic_calculus.py) | Differentiation, recursive substitutions, integration, AtomNative preparation and NumPy batches | [EN](../guides/SYMBOLIC_MATH_GUIDE_EN.md) / [RU](../guides/SYMBOLIC_MATH_GUIDE_RU.md) |
| [`symbolic_arrays.py`](symbolic_arrays.py) | Symbolic vector/matrix algebra and owned NumPy outputs | [EN](../guides/SYMBOLIC_MATH_GUIDE_EN.md) / [RU](../guides/SYMBOLIC_MATH_GUIDE_RU.md) |
| [`lsode2_lorenz_animation.py`](lsode2_lorenz_animation.py) | Lorenz solve with compute-only, 2D projection and 3D playback modes | [EN](../guides/TRAJECTORY_ANIMATION_GUIDE_EN.md) / [RU](../guides/TRAJECTORY_ANIMATION_GUIDE_RU.md) |
| [`lsode2_three_body_animation.py`](lsode2_three_body_animation.py) | Three-body solve and optional selected-body XY animation | [EN](../guides/TRAJECTORY_ANIMATION_GUIDE_EN.md) / [RU](../guides/TRAJECTORY_ANIMATION_GUIDE_RU.md) |
| [`task_batch.py`](task_batch.py) | Failure-tolerant task batch and compact status table | [Overview EN](../guides/PYTHON_SOLVERS_GUIDE_EN.md#text-tasks-and-examples) / [Overview RU](../guides/PYTHON_SOLVERS_GUIDE_RU.md) |

The examples are intentionally small and deterministic. They demonstrate
configuration and result inspection; use the Rust crate's story tests and
benchmarks for broad correctness and performance evidence. A timing printed
by an example is not a benchmark claim.

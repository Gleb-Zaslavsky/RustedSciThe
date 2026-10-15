# RustedSciThe for Python

RustedSciThe brings a symbolic engine and a family of stiff ODE, boundary-value,
nonlinear-system and fitting solvers to Python. The Python package is a thin,
idiomatic interface over a compiled Rust numerical core: expressions are parsed
and differentiated in Rust, solver workspaces and algorithms stay in Rust, and
Python receives ordinary typed objects and detached NumPy results.

This project is for problems where a trustworthy derivative, a stiff or
strongly nonlinear method, and visibility into numerical behavior matter more
than hiding every solver choice behind one universal interface.

## Why RustedSciThe

- **Symbolic derivatives without a Python CAS dependency.** Build expressions,
  simplify and substitute them, and derive exact Jacobians with RustedSciThe's
  own symbolic engine. Prepared residual and Jacobian callbacks are reused
  instead of being rebuilt during every numerical iteration.
- **Native stiff IVP and BVP solvers.** Use Rust implementations inspired by
  SciPy's Radau, variable-order BDF and collocation methods, alongside LSODE2,
  Backward Euler and the damped BVP solver. The numerical solve runs in Rust,
  not in a Python loop around a native linear algebra call.
- **More than ODE integration.** Solve square nonlinear systems, rectangular
  least-squares problems, observation-based fitting and separable models with
  VarPro. These are first-class numerical workflows, not just helper routines
  attached to an IVP API.
- **Algorithms beyond the SciPy-like IVP/BVP set.** Native nonlinear methods,
  a dedicated rectangular Levenberg-Marquardt solver, Gavin fitting and
  variable projection cover workflows that are not merely wrappers around
  SciPy's ODE interfaces.
- **Choose the linear algebra layout when the solver supports it.** Dense,
  Sparse and Banded routes are available across most solver families. Their
  performance depends on the equation structure and problem size; there is no
  universally fastest layout. BVP_sci additionally offers opt-in SuperLU.
- **Lambdify and AOT execution.** Use prepared in-process callbacks for a fast
  development cycle, or generated compiled callbacks where supported. The
  symbolic frontend (`expr_legacy` or `atom_native`) is a separate choice from
  the execution route (`lambdify` or `aot`).
- **Prepared models and parameter series.** Reuse symbolic preparation across
  continuation and restart workflows where the solver exposes that lifecycle.
- **Diagnostics designed for numerical work.** Typed exceptions, route
  provenance, optional counters and stage timings help explain what happened
  without parsing console output or error messages.

Performance is workload-dependent. A Rust implementation is not automatically
the fastest choice for every equation, frontend, layout or lifecycle. Compare
the same workload and inspect preparation, callback and full-solve timings
before choosing a route.

## Architecture

```text
Python model and NumPy inputs
             |
       PyO3 boundary
             |
Rust symbolic preparation -> exact derivatives -> Lambdify or AOT callbacks
             |
Rust solver -> Dense / Sparse / Banded linear algebra where supported
             |
typed result, telemetry and owned NumPy arrays
```

Python is the user-facing layer, not a second implementation of the algorithms.
NumPy arrays returned by result objects own their data; they do not borrow a
mutable solver workspace. The package does not require SymPy.

## Quick Start

This small stiff IVP uses Radau with its default symbolic and numerical
configuration:

```python
from rustedscithe import Expr, RadauSolver

y = Expr.variable("y")
solver = RadauSolver([-y], ["y"], "t", 0.0, 1.0, 0.1)
solution = solver.solve([1.0])
print(f"y(1) = {solution.final_state[0]:.8f}")
```

Install the published package with `python -m pip install rustedscithe`. For a
checkout-based editable build, install `maturin` and run:

```powershell
python -m maturin develop --manifest-path bindings/python/Cargo.toml
```

The extension uses the stable `abi3` Python 3.10+ ABI and NumPy 2.x. Import the
public API from `rustedscithe`; `_native` is an implementation module.

## Solver Families

| Problem | Python entry points | Useful distinction |
| --- | --- | --- |
| Stiff initial-value problems | `RadauSolver`, `BdfSolver`, `BackwardEulerSolver`, `Lsode2Solver` | Radau and LSODE2 expose structured layouts where supported; standalone BDF and Backward Euler are Dense. LSODE2 separates its solver family from its Adams/BDF/Auto controller. |
| Boundary-value problems | `BvpSciSolver`, `BvpDampSolver` | BVP_sci uses symbolic boundary residuals and collocation; BVP_Damp uses observed state/boundary data and damped Newton. Their boundary contracts intentionally differ. |
| Nonlinear systems | `NonlinearSolver` | Prepared square systems with Newton, LM and trust-region method families. |
| Least squares and fitting | `FittingSolver`, `GavinFittingSolver`, `VarProSolver` | Rectangular symbolic least squares, observation-based fitting and separable-model fitting have distinct contracts. |
| Text task documents | `TaskDocument`, `task`, `run_task_batch` | Parse and execute reusable IVP/BVP documents through the native task infrastructure. |

Use `capabilities("solver_name")` to inspect the Python-supported routes.
Native features that are not exposed by this package are not implied by a
shared option name; unsupported combinations raise a typed error instead of
silently selecting a different backend.

## Configuration Choices

Several independent axes shape a solve, and a given solver supports only a
documented subset of their combinations:

| Axis | Examples | What it selects |
| --- | --- | --- |
| Symbolic frontend | `expr_legacy`, `atom_native` | Representation used for symbolic preparation and differentiation. |
| Callback execution | `lambdify`, `aot` | In-process prepared closures or generated compiled artifacts. |
| Matrix layout | `dense`, `sparse`, `banded` | Storage and linear-system route. |
| Callback policy | `sequential`, `parallel`, `auto` | Scheduling where the native solver supports it. |
| Lifecycle | fresh solve, `prepare`, continuation, restart | Whether prepared model or previous state is reused. |

Check the relevant guide and `capabilities()` before combining options. A
structured layout is useful only when the problem has exploitable structure;
parallel callback evaluation and AOT compilation also have workload-dependent
break-even points. For continuation, distinguish cold preparation from warm
solve time.

## Guides and Examples

The overview guides explain shared Python conventions. Solver-specific guides
preserve each native solver's option breadth, supported routes, limitations and
recommended workflows.

| Family | English | Русский |
| --- | --- | --- |
| Python API overview | [Python solvers](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/PYTHON_SOLVERS_GUIDE_EN.md) | [Обзор Python API](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/PYTHON_SOLVERS_GUIDE_RU.md) |
| Radau | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/RADAU_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/RADAU_GUIDE_RU.md) |
| BDF | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BDF_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BDF_GUIDE_RU.md) |
| Backward Euler | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BACKWARD_EULER_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BACKWARD_EULER_GUIDE_RU.md) |
| LSODE2 | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/LSODE2_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/LSODE2_GUIDE_RU.md) |
| BVP_sci | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BVP_SCI_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BVP_SCI_GUIDE_RU.md) |
| BVP_Damp | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BVP_DAMP_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/BVP_DAMP_GUIDE_RU.md) |
| Nonlinear systems and fitting | [Guide](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/NONLINEAR_FITTING_GUIDE_EN.md) | [Руководство](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/guides/NONLINEAR_FITTING_GUIDE_RU.md) |

Runnable, commented Python examples include Robertson kinetics, Lorenz and
three-body systems, frontend/layout comparisons, continuation, AOT and task
documents. See the [Python example index](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/examples/README.md);
the repository's Rust examples are collected separately in the root
[examples directory](https://github.com/Gleb-Zaslavsky/RustedSciThe/tree/master/examples/).

## Text Tasks and Executable

Python users can parse and run text task documents with `TaskDocument` or
`task(...)`, and batch independent documents with `run_task_batch(...)`. The
batch API preserves typed per-task failures and can write a compact table. The
Windows console executable is part of the main Rust project, not this Python
package; see the [task parser documentation](https://github.com/Gleb-Zaslavsky/RustedSciThe/tree/master/src/command_interpreter/).

## Diagnostics and Development

Native exceptions preserve their Python class and expose machine-readable
`stage`, `solver` and `route` attributes. Telemetry is opt-in and solutions
provide detached counters, timings and report writers where supported. Prefer
these fields over parsing exception text or redirecting build output into a
performance report.

For development and contribution checks, see the [Python contract tests](https://github.com/Gleb-Zaslavsky/RustedSciThe/tree/master/bindings/python/tests/),
the [strict typing consumer](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/tests/typing_consumer.py),
and the [Python binding roadmap](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/bindings/python/TODO.md).
The core crate's broader algorithm documentation and release evidence remain
in the [main README](https://github.com/Gleb-Zaslavsky/RustedSciThe/blob/master/README.md)
and solver guides.

[TOC]

# RustedSciThe
RustedSciThe is a Rust framework for symbolic and numerical computing.

PROJECT NEWS:

## Content
- [Motivation](#motivation)
- [Features](#features)
- [Project Documentation and Navigation](#project-documentation-and-navigation)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [To do](#to-do)

## Motivation
RustedSciThe started as part of the KiThe crate and was originally focused on analytical Jacobians for combustion, chemical kinetics, and heat/mass-transfer equation systems. It soon became clear that the same architecture could serve a much broader scientific audience.

## Features
* symbolic engine:
    * parsing string expressions to symbolic expressions/functions
    * symbolic/analytical differentiation of symbolic expressions/functions
    * comparison of analytical and numerical derivatives
    * calculation of vectors of partial derivatives
    * transformation of symbolic expressions/functions (including derivatives) into regular Rust functions
    * symbolic/analytical Jacobian construction for multiple numerical methods with functional conversion
    * analytical Taylor expansion
    * matrices and vectors of symbolic expressions/functions
    * symbolic/analytical integration
    * numerical integration of symbolic expressions/functions
* IVP for stiff and non-stiff problems with analytical Jacobian:
    * Backward Euler method
    * Backward Differentiation Formula (BDF)
    * Radau method
    * classical non-stiff methods RK45 and DP
    * LSODE2 solver (LSODE/LSODA-inspired Adams/BDF family with automatic and manual family control)
* Boundary Value Problem for ODE:
    * damped/modified Newton-Raphson methods (several versions available)
    * symbolic, Lambdify and AOT-generated residual/Jacobian backends for sparse and banded BVP systems
    * sequential and parallel execution of generated residual/Jacobian modules
    * Newton-Raphson 4th-order collocation algorithm with residual control
    * shooting method for solving BVP
* Optimization with (if needed) analytical Jacobian:
    * curve fitting
    * Levenberg-Marquardt method with trust region
    * H. P. Gavin Levenberg-Marquardt method
    * scalar optimization (Brent, bisection, secant, Newton-Raphson)
    * interpolation/extrapolation (Lagrangian, Newtonian, polynomial; SciPy-inspired)
* utilities:
    * command interpreter to parse task files
    * wrappers around plotters and GNUplot
    * Bevy-based 2D and 3D curve animation module to visualize solutions and phase portraits
* solving systems of nonlinear equations with analytical Jacobian:
    * Newton-Raphson method
    * damped Newton-Raphson
    * Levenberg-Marquardt
    * Levenberg-Marquardt-Nielsen
    * trust region
    * dogleg
* linear algebra backends:
    * solving large banded linear systems with BiCGSTAB and GMRES and several preconditioners
    * faithful LAPACK-style LU for general banded matrices via `LapackStyleBandedLuFaithful`
      (`src/somelinalg/banded/lapack_style_banded.rs`)
  (ArrayFire C++ library is needed for the GPU feature set; see below)

### AOT pipeline
AOT means ahead-of-time. In simple words, RustedSciThe can take symbolic equations, differentiate them, simplify the result, and generate a separate Rust module with ready numerical code before the solver starts the main calculation.

Why this is useful:

- the solver does not need to carry symbolic work during the expensive runtime part;
- the generated residual and Jacobian are close to what a very careful human would write by hand after differentiating a huge system and simplifying it;
- for large sparse and banded problems, generated code can be executed sequentially or in parallel.

This is the main idea: imagine that a person manually differentiated one thousand functions by one thousand variables, removed zero terms, reused repeated subexpressions, and then wrote all final formulas into an optimized Rust file. That would be possible in theory, but painfully slow, error-prone, and almost impossible to maintain. The AOT pipeline automates exactly this kind of work.

In practice, the pipeline does the following:

- takes equations, variables, optional parameters, boundary conditions, and backend settings;
- discretizes the problem and constructs symbolic residual and symbolic Jacobian;
- lowers them into internal code generation IR;
- generates Rust/C/Zig source code for residual and Jacobian functions;
- lets the solver use generated modules through the same high-level solver interface.

That is why AOT is powerful here: it combines the convenience of symbolic mathematics with runtime behavior much closer to carefully hand-written optimized numerical code.

### Symbolic engine
The symbolic engine covers the core operations expected from a scientific symbolic layer: parsing, differentiation, simplification-oriented workflows, integration-related transformations, and conversion into executable numerical functions. A practical highlight is Jacobian construction for multiple solver families, so the same model can feed ODE, BVP, nonlinear-equation, and fitting pipelines without rewriting derivatives manually for each case.

### LSODA/LSODE-class IVP solver
One of the central solver capabilities in RustedSciThe is an LSODA/LSODE-class IVP workflow. Algorithmically, it mirrors the ODEPACK philosophy: Adams/BDF families, stiffness-aware method choreography, and robust step control in difficult regions. The intent is not merely to solve toy equations, but to reproduce the behavior users expect from mature Fortran-class integrators on real mixed-regime problems.

On top of this algorithmic core, the solver is integrated with RustedSciThe backend architecture: pure numerical callbacks, symbolic Lambdify execution, and AOT-generated residual/Jacobian routes are all available under a unified configuration model with Dense/Sparse/Banded linear structures. In practice this makes it a Swiss-army knife for IVP workloads that move between non-stiff and stiff phases.

### Native BDF and Radau
The native BDF and Radau solvers are direct rewrites of the corresponding SciPy solver lines, adapted to RustedSciThe's backend stack. This means the numerical method logic follows the proven SciPy/SUNDIALS-era style, while backend execution can use project-native symbolic/Lambdify/AOT and linear algebra routes. The result is a useful combination: familiar algorithmic behavior with modern backend flexibility for performance and deployment trade-offs.

### Banded Matrix LU (LAPACK-style)
RustedSciThe includes a faithful LAPACK-style LU factorization path for general banded matrices (implemented in `src/somelinalg/banded/lapack_style_banded.rs`). The implementation is intentionally close to classical banded storage/factorization semantics, which helps preserve predictable numerical behavior on true banded systems.

For many genuinely banded tasks, this route outperforms generic sparse linear algebra because it exploits narrow bandwidth directly instead of paying sparse-indirection overhead. In practical terms, when Jacobian structure is truly banded and bandwidth is known, this is often the best first choice for linear solves.

### BVP Newton-Raphson family (Sparse/Banded + Lambdify/AOT)
The BVP stack is built around modified Newton and damped Newton methods for nonlinear boundary-value systems, with additional collocation-based residual-control workflows. This is not a legacy demonstration API; it is a production-oriented family designed for large nonlinear systems where Jacobian strategy and linear backend choice are first-class concerns.

The same mathematical BVP workflow can be paired with sparse or banded linear algebra and with symbolic Lambdify or AOT-generated residual/Jacobian evaluation. This separation between numerical method and backend route lets users tune robustness/performance without rewriting model equations or changing solver family.

### Nonlinear equations, optimization, and fitting
Beyond IVP/BVP, RustedSciThe includes a coherent stack for nonlinear systems, optimization, and curve fitting. Newton-class and trust-region methods, Levenberg-Marquardt variants, and scalar optimization tools are designed to reuse the same symbolic/Jacobian infrastructure where possible. This unification is important in practice: users can move from model derivation to root finding, parameter estimation, and calibration workflows within one consistent ecosystem.

### ArrayFire and CUDA features
To enable GPU features, you need to have the ArrayFire C++ library installed on your machine. Installation instructions: [ArrayFire website](https://arrayfire.org/docs/installing.htm).

```bash
cargo build --features arrayfire
```

- Enables GPU-accelerated linear algebra via ArrayFire.
- BiCGStab and GMRES solvers with GPU acceleration.
- Vanilla (no preconditioning), Jacobi and ILU0 preconditioners on GPU.
- Requires ArrayFire C++ library installation.

```bash
cargo build --features cuda
```

- Enables all ArrayFire features.
- GPU-native Gauss-Seidel preconditioner via custom CUDA kernels.
- Requires both ArrayFire and compiled CUDA library.

## Project Documentation and Navigation

| solver/feature | folder |
|:--|--:|
| ODE solvers for stiff problems | `src/numerical/` |
| BDF (Backward Differentiation Formula) | `src/numerical/BDF/` |
| Radau | `src/numerical/Radau/` |
| Backward Euler method | `src/numerical/BE.rs` |
| LSODE2 (LSODE/LSODA-inspired) | `src/numerical/LSODE2/` |
| ODE solver for non-stiff problems (RK45, DP) | `src/numerical/NonStiff_api.rs` |
| Boundary Value Problem (BVP), damped Newton family | `src/numerical/BVP_Damp/` |
| BVP 4th-order collocation with residual control | `src/numerical/BVP_sci/` |
| Optimization | `src/numerical/optimization/` |
| Symbolic expression parsing | `src/symbolic/parse_expr.rs` |
| Core symbolic engine | `src/symbolic/symbolic_engine.rs` |
| Symbolic vectors and matrices | `src/symbolic/` |
| Task parsers and command interpreter | `src/command_interpreter/` |
| Linear algebra collection | `src/somelinalg/` |
| Faithful banded LU backend | `src/somelinalg/banded/lapack_style_banded.rs` |
| GPU iterative solvers | `src/somelinalg/iterative_solvers_gpu/` |
| LSODE2 user guide (EN) | `src/numerical/LSODE2/LSODE2_USER_GUIDE_EN.md` |
| LSODE2 user guide (RU) | `src/numerical/LSODE2/LSODE2_USER_GUIDE_RU.md` |

## Project Documentation
In the `Book` folder of the project (on GitHub), there is an in-depth scientific manual as well as developer and user manuals in English and Russian. The documentation is actively evolving.

## Examples
Practical usage scenarios are kept in the `examples` folder and in `examples/task_docs`. This includes complete workflows for IVP/BVP solvers, LSODE2 numerical/lambdify/AOT routes, backend comparison stories, and task-document driven execution.

## Testing
Our project is covered by tests and you can run them by standard command:

```sh
cargo test
```

## Contributing
If you have any questions, comments or want to contribute, please feel free to contact us at https://github.com/

## To do
- [x] Write basic functionality
- [x] Write Jacobians
- [x] Write Newton-Raphson
- [x] Write BDF
- [x] Write Backward Euler
- [x] Write non-stiff methods
- [x] Add indexed variables and matrices
- [x] Add BVP methods for stiff ODEs
- [ ] GPU-accelerated computations (extended coverage)
- [ ] More methods for stiff ODEs

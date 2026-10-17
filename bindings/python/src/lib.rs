//! Python-facing entry point for the optional `rustedscithe` extension.
//!
//! The binding starts with a stable symbolic layer, a common solver vocabulary
//! and thin vertical slices for the first numerical solvers. Concrete wrappers
//! can be added without changing expression semantics or option spelling.

mod animation;
mod aot;
mod bdf;
mod be;
pub(crate) mod bvp;
mod bvp_damp;
mod contract;
mod data_processing;
mod errors;
mod expr;
mod gavin;
mod lambdify;
mod lsode2;
mod nonlinear;
mod numpy_support;
mod ode_api2;
#[cfg(test)]
mod parity;
mod radau;
mod shooting_bvp;
mod symbolic_arrays;
mod system;
mod task;
mod telemetry;
mod universal_fitting;
mod varpro;

use pyo3::prelude::*;
use pyo3::PyTypeInfo;

use crate::bdf::{BdfCallbackSolver, BdfSolution, BdfSolver};
use crate::be::{BackwardEulerSolution, BackwardEulerSolver};
use crate::bvp::{BvpSciSolution, BvpSciSolver};
use crate::bvp_damp::{BvpDampFrozenSolver, BvpDampNaiveSolver, BvpDampSolution, BvpDampSolver};
use crate::contract::{
    capabilities, ExecutionPolicy, ExecutionRoute, MatrixLayout, SolverCapabilities, SolverOptions,
    SymbolicFrontend, TelemetryMode,
};
use crate::errors::{
    AotArtifactError, AotCompilerError, AotLifecycleError, AotLinkError, AotTimeoutError, BvpError,
    CallbackError, ExhaustionError, InvalidConfigurationError, InvalidExpressionError, PanicError,
    PoisonedSolverError, PreparationError, RustedSciTheError, SolverError, TaskDocumentError,
    TelemetryError, UnsupportedOptionError, UnsupportedSymbolicOperationError,
};
use crate::expr::{constant, indexed_symbols, parse_expr, symbol, symbols, PyExpr};
use crate::gavin::{GavinFittingSolution, GavinFittingSolver};
use crate::lambdify::LambdifiedExpr;
use crate::lsode2::{Lsode2PythonSolver, Lsode2Solution};
use crate::nonlinear::{FittingSolution, FittingSolver, NonlinearSolution, NonlinearSolver};
use crate::ode_api2::{UniversalNonStiffSolution, UniversalNonStiffSolver};
use crate::radau::{RadauCallbackSolver, RadauSolution, RadauSolver};
use crate::shooting_bvp::{PyShootingBvpSolution, PyShootingBvpSolver};
use crate::symbolic_arrays::{PyExprMatrix, PyExprVector};
use crate::system::{LambdifiedSystem, SymbolicSystem};
use crate::task::{BvpTaskResult, IvpTaskResult, TaskDocument};
use crate::universal_fitting::{PyUniversalFitting, PyUniversalFittingSolution};
use crate::varpro::{VarProSolution, VarProSolver};

/// Initialize the optional Python extension module.
///
/// The exception classes are exported explicitly so Python callers can catch
/// configuration, expression, and AOT lifecycle failures without matching
/// human-readable error strings.
#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExpr>()?;
    module.add_class::<PyExprVector>()?;
    module.add_class::<PyExprMatrix>()?;
    module.add_class::<BdfSolver>()?;
    module.add_class::<BdfCallbackSolver>()?;
    module.add_class::<BdfSolution>()?;
    module.add_class::<BackwardEulerSolver>()?;
    module.add_class::<BackwardEulerSolution>()?;
    module.add_class::<BvpSciSolver>()?;
    module.add_class::<BvpSciSolution>()?;
    module.add_class::<BvpDampSolver>()?;
    module.add_class::<BvpDampNaiveSolver>()?;
    module.add_class::<BvpDampFrozenSolver>()?;
    module.add_class::<BvpDampSolution>()?;
    module.add_class::<RadauSolver>()?;
    module.add_class::<RadauCallbackSolver>()?;
    module.add_class::<RadauSolution>()?;
    module.add_class::<PyShootingBvpSolver>()?;
    module.add_class::<PyShootingBvpSolution>()?;
    module.add_class::<LambdifiedExpr>()?;
    module.add_class::<Lsode2PythonSolver>()?;
    module.add_class::<Lsode2Solution>()?;
    module.add_class::<UniversalNonStiffSolver>()?;
    module.add_class::<UniversalNonStiffSolution>()?;
    module.add_class::<SymbolicSystem>()?;
    module.add_class::<LambdifiedSystem>()?;
    module.add_class::<NonlinearSolver>()?;
    module.add_class::<NonlinearSolution>()?;
    module.add_class::<FittingSolver>()?;
    module.add_class::<FittingSolution>()?;
    module.add_class::<GavinFittingSolver>()?;
    module.add_class::<GavinFittingSolution>()?;
    module.add_class::<VarProSolver>()?;
    module.add_class::<VarProSolution>()?;
    module.add_class::<PyUniversalFitting>()?;
    module.add_class::<PyUniversalFittingSolution>()?;
    module.add_class::<SymbolicFrontend>()?;
    module.add_class::<ExecutionRoute>()?;
    module.add_class::<MatrixLayout>()?;
    module.add_class::<ExecutionPolicy>()?;
    module.add_class::<TelemetryMode>()?;
    module.add_class::<SolverOptions>()?;
    module.add_class::<SolverCapabilities>()?;
    module.add_class::<TaskDocument>()?;
    module.add_class::<IvpTaskResult>()?;
    module.add_class::<BvpTaskResult>()?;
    #[cfg(feature = "test-panic-probe")]
    module.add_class::<crate::errors::PanicProbeSolver>()?;

    module.add_function(wrap_pyfunction!(parse_expr, module)?)?;
    module.add_function(wrap_pyfunction!(symbol, module)?)?;
    module.add_function(wrap_pyfunction!(symbols, module)?)?;
    module.add_function(wrap_pyfunction!(indexed_symbols, module)?)?;
    module.add_function(wrap_pyfunction!(constant, module)?)?;
    crate::data_processing::register(module)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(superlu_feature_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(
        crate::animation::animate_trajectory_2d,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        crate::animation::animate_trajectory_3d,
        module
    )?)?;

    module.add("RustedSciTheError", RustedSciTheError::type_object(py))?;
    module.add(
        "InvalidExpressionError",
        InvalidExpressionError::type_object(py),
    )?;
    module.add(
        "UnsupportedSymbolicOperationError",
        UnsupportedSymbolicOperationError::type_object(py),
    )?;
    module.add(
        "UnsupportedOptionError",
        UnsupportedOptionError::type_object(py),
    )?;
    module.add(
        "InvalidConfigurationError",
        InvalidConfigurationError::type_object(py),
    )?;
    module.add("SolverError", SolverError::type_object(py))?;
    module.add("PreparationError", PreparationError::type_object(py))?;
    module.add("AotArtifactError", AotArtifactError::type_object(py))?;
    module.add("AotCompilerError", AotCompilerError::type_object(py))?;
    module.add("AotLinkError", AotLinkError::type_object(py))?;
    module.add("AotTimeoutError", AotTimeoutError::type_object(py))?;
    module.add("AotLifecycleError", AotLifecycleError::type_object(py))?;
    module.add("CallbackError", CallbackError::type_object(py))?;
    module.add("TaskDocumentError", TaskDocumentError::type_object(py))?;
    module.add("BvpError", BvpError::type_object(py))?;
    module.add("TelemetryError", TelemetryError::type_object(py))?;
    module.add("ExhaustionError", ExhaustionError::type_object(py))?;
    module.add("PanicError", PanicError::type_object(py))?;
    module.add("PoisonedSolverError", PoisonedSolverError::type_object(py))?;
    Ok(())
}

/// Report whether this extension was compiled with the optional SuperLU bridge.
#[pyfunction]
fn superlu_feature_enabled() -> bool {
    cfg!(feature = "superlu")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialize_python() {
        Python::initialize();
    }

    #[test]
    fn symbolic_expression_contract_is_usable_without_python_glue() {
        initialize_python();

        let expression = PyExpr::parse("x^2 + y").expect("valid expression");
        assert_eq!(expression.variables(), vec!["x", "y"]);
        assert_eq!(expression.evaluate(vec![2.0, 3.0]).unwrap(), 7.0);
        assert_eq!(
            expression.diff("x").unwrap().evaluate(vec![2.0]).unwrap(),
            4.0
        );
        assert_eq!(
            expression
                .substitute("y", 5.0)
                .unwrap()
                .evaluate(vec![2.0])
                .unwrap(),
            9.0
        );

        let parsed = parse_expr("sin(x) + 1").expect("parser helper");
        assert!(parsed.evaluate(vec![0.0]).unwrap() > 0.99);
    }

    #[test]
    fn expression_validation_returns_typed_errors() {
        initialize_python();

        assert!(PyExpr::parse("x +").is_err());
        assert!(PyExpr::variable("").is_err());
        assert!(PyExpr::constant(f64::NAN).is_err());
        assert!(PyExpr::parse("x").unwrap().evaluate(vec![]).is_err());
        assert!(PyExpr::parse("x")
            .unwrap()
            .evaluate(vec![f64::INFINITY])
            .is_err());
    }

    #[test]
    fn common_solver_contract_validates_capabilities() {
        initialize_python();

        let defaults =
            SolverOptions::new(None, None, None, None, None, None, None, None, None).unwrap();
        defaults.validate_for("radau5").unwrap();
        assert_eq!(defaults.frontend(), "expr_legacy");
        assert_eq!(defaults.execution(), "lambdify");
        assert_eq!(defaults.logging_level(), "off");
        let mut logged =
            SolverOptions::new(None, None, None, None, None, None, None, None, None).unwrap();
        logged.set_logging_level("INFO").unwrap();
        assert_eq!(logged.logging_level(), "info");
        assert_eq!(
            crate::contract::native_logging_level(&logged, "bdf").unwrap(),
            rustedscithe_core::numerical::logging::SolverLogLevel::Info
        );
        assert!(logged.validate_for("radau5").is_err());

        let native_unwired = SolverOptions::new(
            Some("atom-view"),
            Some("AOT"),
            Some("sparse"),
            Some("parallel"),
            Some("counters"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(native_unwired.validate_for("bdf").is_err());
        let bdf = capabilities("bdf").unwrap();
        assert_eq!(
            bdf.frontends(),
            vec!["expr_legacy".to_owned(), "atom_native".to_owned()]
        );
        assert_eq!(bdf.executions(), vec!["lambdify", "aot"]);
        assert_eq!(bdf.layouts(), vec!["dense"]);
        assert_eq!(
            bdf.telemetry_modes(),
            vec!["off", "counters", "timings", "detailed"]
        );
        assert!(SolverOptions::new(
            None,
            None,
            Some("sparse"),
            None,
            None,
            None,
            None,
            None,
            None
        )
        .unwrap()
        .validate_for("be")
        .is_err());

        let be = capabilities("backward_euler").unwrap();
        assert_eq!(be.solver(), "backward_euler");
        assert_eq!(be.layouts(), vec!["dense"]);
        assert_eq!(
            capabilities("nonstiff").unwrap().telemetry_modes(),
            vec!["off"]
        );
        assert!(capabilities("unknown").is_err());

        let lsode2 = capabilities("lsode2").unwrap();
        assert_eq!(lsode2.executions(), vec!["lambdify", "aot"]);
        assert_eq!(lsode2.methods(), vec!["bdf", "adams", "auto", "lsoda"]);
        assert_eq!(lsode2.layouts(), vec!["dense", "sparse", "banded"]);
        assert_eq!(lsode2.policies(), vec!["sequential", "parallel", "auto"]);
        assert!(lsode2.supports_analytical_callbacks());
        assert_eq!(
            lsode2.analytical_callback_layouts(),
            vec!["dense", "sparse", "banded"]
        );
        SolverOptions::new(
            Some("atom_native"),
            Some("lambdify"),
            Some("sparse"),
            Some("auto"),
            Some("counters"),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .validate_for("lsode2")
        .unwrap();

        let shooting = capabilities("shooting_bvp").unwrap();
        assert_eq!(shooting.frontends(), vec!["expr_legacy", "atom_native"]);
        assert_eq!(shooting.executions(), vec!["lambdify"]);
        assert_eq!(shooting.layouts(), vec!["dense"]);
    }
}

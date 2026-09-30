//! Release-only Lambdify callback corpus for large prepared systems.
//!
//! The story intentionally excludes controller and linear-system time. It is
//! a direct comparison of the same prepared callback workload, with cold
//! preparation and repeated residual/Jacobian execution reported separately.

use super::story_support::{chain_state, max_matrix_diff, max_vector_diff, prepare_chain};
use super::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot, IvpWarmStage,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Instant;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|item| *item > 0)
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn combustion_chain_equations(dimension: usize) -> Vec<Expr> {
    (0..dimension)
        .map(|index| {
            let left = (index > 0)
                .then(|| format!("y{}", index - 1))
                .unwrap_or_else(|| "0".to_string());
            let right = (index + 1 < dimension)
                .then(|| format!("y{}", index + 1))
                .unwrap_or_else(|| "0".to_string());
            Expr::parse_expression(&format!(
                "-k*exp(-E/(R*(T0 + beta*t)))*y{index}*y{index} + d*({left} - 2*y{index} + {right}) - loss*y{index} + q*exp(-t)"
            ))
        })
        .collect()
}

fn combustion_state(dimension: usize) -> DVector<f64> {
    DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 0.15 + 0.005 * (index % 17) as f64),
    )
}

fn prepare_combustion(
    dimension: usize,
    frontend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem {
    prepare_symbolic_ivp_problem(
        combustion_chain_equations(dimension),
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_equation_parameters(vec![
                "k".to_string(),
                "E".to_string(),
                "R".to_string(),
                "T0".to_string(),
                "beta".to_string(),
                "d".to_string(),
                "loss".to_string(),
                "q".to_string(),
            ])
            .with_equation_parameter_values(DVector::from_vec(vec![
                2.0e3, 5.0e3, 8.314, 300.0, 0.8, 0.25, 0.05, 0.01,
            ]))
            .with_symbolic_assembly_backend(frontend)
            .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
            .with_telemetry(telemetry),
    )
    .expect("large combustion-like Lambdify preparation should succeed")
}

fn cold_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

fn warm_ms(snapshot: &IvpTelemetrySnapshot, stage: IvpWarmStage) -> f64 {
    snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

#[test]
#[ignore = "release large Lambdify callback corpus; run explicitly with --ignored"]
fn lsode2_lambdify_large_callback_corpus_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_large_scale_story_tests::lsode2_lambdify_large_callback_corpus_story",
    );
    let dimensions = env_usize_list("LSODE2_LAMBDIFY_LARGE_CALLBACK_DIMENSIONS", &[1024, 2048]);
    let combustion_dimensions = env_usize_list("LSODE2_LAMBDIFY_COMBUSTION_DIMENSIONS", &[32, 64]);
    let repetitions = std::env::var("LSODE2_LAMBDIFY_LARGE_CALLBACK_REPETITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3);

    reportln!(
        "[LSODE2 Lambdify large callback corpus] dimensions={dimensions:?}; combustion_dimensions={combustion_dimensions:?}; repetitions={repetitions}; callback-only; controller/linear solve excluded"
    );
    reportln!(
        "workload | frontend | dimension | prepare_ms | expr_to_atom_ms | diff_ms | pattern_ms | binding_ms | residual_eval_ms | residual_output_ms | jacobian_eval_ms | jacobian_output_ms | residual_ms/call | jacobian_ms/call | copies | allocated_bytes | residual_diff | jacobian_diff"
    );
    reportln!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    reportln!(
        "cold-stages | workload | frontend | dimension | validation_ms | atom_residual_prepare_ms | atom_jacobian_prepare_ms | symbolic_jacobian_ms | differentiation_ms | simplify_ms | pattern_ms | layout_ms | residual_compile_ms | residual_lambdify_ms | jacobian_compile_ms | jacobian_lambdify_ms"
    );
    reportln!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for (workload, workload_dimensions) in [
        ("diffusion-chain", dimensions),
        ("combustion-like", combustion_dimensions),
    ] {
        for dimension in workload_dimensions {
            let state = if workload == "diffusion-chain" {
                chain_state(dimension)
            } else {
                combustion_state(dimension)
            };
            let mut reference_residual = None;
            let mut reference_jacobian = None;

            for frontend in [
                IvpSymbolicAssemblyBackend::ExprLegacy,
                IvpSymbolicAssemblyBackend::AtomView,
            ] {
                let telemetry = IvpTelemetry::detailed();
                let started = Instant::now();
                let problem = if workload == "diffusion-chain" {
                    prepare_chain(
                        dimension,
                        frontend,
                        IvpLambdifyExecutionPolicy::Sequential,
                        telemetry.clone(),
                    )
                } else {
                    prepare_combustion(dimension, frontend, telemetry.clone())
                };
                let prepare_ms = started.elapsed().as_secs_f64() * 1.0e3;
                let snapshot = telemetry.snapshot();

                let mut residual = None;
                let mut jacobian = None;
                let residual_started = Instant::now();
                for _ in 0..repetitions {
                    residual = Some(
                        problem
                            .try_evaluate_residual(0.5, &state)
                            .expect("large Lambdify residual should evaluate"),
                    );
                }
                let residual_ms =
                    residual_started.elapsed().as_secs_f64() * 1.0e3 / repetitions as f64;
                let jacobian_started = Instant::now();
                for _ in 0..repetitions {
                    jacobian = Some(
                        problem
                            .try_evaluate_jacobian(0.5, &state)
                            .expect("large Lambdify Jacobian should evaluate"),
                    );
                }
                let jacobian_ms =
                    jacobian_started.elapsed().as_secs_f64() * 1.0e3 / repetitions as f64;
                let residual = black_box(residual.expect("residual sample should exist"));
                let jacobian = black_box(jacobian.expect("Jacobian sample should exist"));

                let residual_diff = reference_residual
                    .as_ref()
                    .map(|reference| max_vector_diff(reference, &residual))
                    .unwrap_or(0.0);
                let jacobian_diff = reference_jacobian
                    .as_ref()
                    .map(|reference| max_matrix_diff(reference, &jacobian))
                    .unwrap_or(0.0);
                if reference_residual.is_none() {
                    reference_residual = Some(residual);
                    reference_jacobian = Some(jacobian);
                }
                let frontend_label = match frontend {
                    IvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                    IvpSymbolicAssemblyBackend::AtomView => "AtomViewNative",
                    IvpSymbolicAssemblyBackend::AtomViewExprCompat => "AtomViewExprCompat",
                };
                reportln!(
                    "{workload} | {frontend_label} | {dimension:>9} | {prepare_ms:>11.3} | {expr_to_atom:>15.3} | {diff:>7.3} | {pattern:>10.3} | {binding:>10.3} | {res_eval:>16.3} | {res_output:>18.3} | {jac_eval:>15.3} | {jac_output:>17.3} | {residual_ms:>17.6} | {jacobian_ms:>17.6} | {copies:>6} | {allocated:>15} | {residual_diff:.3e} | {jacobian_diff:.3e}",
                    expr_to_atom = cold_ms(&snapshot, IvpColdStage::ExprToAtom),
                    diff = cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
                    pattern = cold_ms(&snapshot, IvpColdStage::SparsePattern),
                    binding = warm_ms(&snapshot, IvpWarmStage::ArgumentBinding),
                    res_eval = warm_ms(&snapshot, IvpWarmStage::ResidualEvaluation),
                    res_output = warm_ms(&snapshot, IvpWarmStage::ResidualOutputAssembly),
                    jac_eval = warm_ms(&snapshot, IvpWarmStage::JacobianEvaluation),
                    jac_output = warm_ms(&snapshot, IvpWarmStage::JacobianOutputAssembly),
                    copies = snapshot.copies,
                    allocated = snapshot.allocated_bytes,
                );
                reportln!(
                    "cold-stages | {workload} | {frontend_label} | {dimension:>9} | {validation:>13.3} | {atom_residual:>24.3} | {atom_jacobian:>24.3} | {symbolic_jacobian:>19.3} | {differentiation:>18.3} | {simplify:>11.3} | {pattern:>10.3} | {layout:>9.3} | {residual_compile:>20.3} | {residual_lambdify:>22.3} | {jacobian_compile:>20.3} | {jacobian_lambdify:>22.3}",
                    validation = cold_ms(&snapshot, IvpColdStage::Validation),
                    atom_residual = cold_ms(&snapshot, IvpColdStage::AtomResidualPreparation),
                    atom_jacobian = cold_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
                    symbolic_jacobian = cold_ms(&snapshot, IvpColdStage::SymbolicJacobian),
                    differentiation = cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
                    simplify = cold_ms(&snapshot, IvpColdStage::Simplification),
                    pattern = cold_ms(&snapshot, IvpColdStage::SparsePattern),
                    layout = cold_ms(&snapshot, IvpColdStage::LayoutPlanning),
                    residual_compile = cold_ms(&snapshot, IvpColdStage::ResidualCompilation),
                    residual_lambdify = cold_ms(&snapshot, IvpColdStage::ResidualLambdification),
                    jacobian_compile = cold_ms(&snapshot, IvpColdStage::JacobianCompilation),
                    jacobian_lambdify = cold_ms(&snapshot, IvpColdStage::JacobianLambdification),
                );
                reportln!(
                    "native-preparation-leaves | {workload} | {frontend_label} | {dimension} | dependency_ms={:.6} | differentiation_ms={:.6} | evaluator_ms={:.6}",
                    cold_ms(&snapshot, IvpColdStage::AtomDependencyAnalysis),
                    cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
                    cold_ms(&snapshot, IvpColdStage::NativeJacobianEvaluatorPreparation),
                );
                assert!(
                    residual_diff <= 1.0e-9,
                    "{workload} residual drift at {dimension}"
                );
                assert!(
                    jacobian_diff <= 1.0e-8,
                    "{workload} Jacobian drift at {dimension}"
                );
            }
        }
    }
}

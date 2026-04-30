#![cfg(test)]

//! Solver-facing IVP backend comparisons.
//!
//! These tests answer a different question than the lower-level IR/codegen
//! checks:
//! - `ivp_be_generated_backend_end_to_end_compare_table`
//!   measures practical end-to-end latency for Backward Euler on a small and a
//!   medium stiff IVP scenario. It tests the hypothesis that a compiled dense
//!   backend can amortize its setup cost even on a relatively small state size.
//! - `ivp_bdf_generated_backend_end_to_end_compare_table`
//!   does the same for the BDF solver, where dense Jacobians are reused across
//!   many implicit steps and Newton iterations. It tests the hypothesis that
//!   IVP defaults should bias towards better runtime throughput rather than the
//!   cheapest possible build.
//!
//! The tables are intentionally solver-facing and practical:
//! - baseline `Lambdify`,
//! - generated dense backends through `C + gcc`, `C + tcc`, and `Zig`,
//! - total wall-clock time,
//! - final-solution agreement with the lambdify baseline,
//! - and an automatic summary of whether the scenario is currently
//!   `residual-dominated`, `jacobian-dominated`, or `mixed`.

use crate::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver};
use crate::numerical::BE::{BE, BeSolverOptions};
use crate::symbolic::codegen::tests::codegen_test_support::command_exists;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::DVector;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct IvpScenario {
    label: &'static str,
    equations: Vec<Expr>,
    values: Vec<String>,
    y0: DVector<f64>,
    t0: f64,
    t_bound: f64,
    be_step: f64,
    bdf_max_step: f64,
    tolerance: f64,
    max_iterations: usize,
    rtol: f64,
    atol: f64,
}

#[derive(Clone, Copy)]
enum IvpBackendVariant {
    Lambdify,
    CGcc,
    CTcc,
    Zig,
}

impl IvpBackendVariant {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "Lambdify",
            Self::CGcc => "C-gcc",
            Self::CTcc => "C-tcc",
            Self::Zig => "Zig",
        }
    }
}

struct IvpRunSummary {
    variant: IvpBackendVariant,
    total: Duration,
    statistics: IvpBackendStatistics,
    max_abs_solution: f64,
    solution_diff: f64,
    status: String,
}

fn small_stiff_ivp_case() -> IvpScenario {
    IvpScenario {
        label: "small-stiff-2",
        equations: vec![
            Expr::parse_expression("-15.0*y1 + 14.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        values: vec!["y1".to_string(), "y2".to_string()],
        y0: DVector::from_vec(vec![1.0, 0.0]),
        t0: 0.0,
        t_bound: 500.0,
        be_step: 0.02,
        bdf_max_step: 0.05,
        tolerance: 1e-9,
        max_iterations: 25,
        rtol: 1e-7,
        atol: 1e-9,
    }
}

fn robertson_ivp_case() -> IvpScenario {
    IvpScenario {
        label: "robertson-3",
        equations: vec![
            Expr::parse_expression("-0.04*y1 + 1.0e4*y2*y3"),
            Expr::parse_expression("0.04*y1 - 1.0e4*y2*y3 - 3.0e7*y2^2"),
            Expr::parse_expression("3.0e7*y2^2"),
        ],
        values: vec!["y1".to_string(), "y2".to_string(), "y3".to_string()],
        y0: DVector::from_vec(vec![1.0, 0.0, 0.0]),
        t0: 0.0,
        t_bound: 500.0,
        be_step: 0.0025,
        bdf_max_step: 0.02,
        tolerance: 1e-10,
        max_iterations: 30,
        rtol: 1e-8,
        atol: 1e-10,
    }
}

fn hires_8_ivp_case() -> IvpScenario {
    IvpScenario {
        label: "hires-8",
        equations: vec![
            Expr::parse_expression("-1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007"),
            Expr::parse_expression("1.71*y1 - 8.75*y2"),
            Expr::parse_expression("-10.03*y3 + 0.43*y4 + 0.035*y5"),
            Expr::parse_expression("8.32*y2 + 1.71*y3 - 1.12*y4"),
            Expr::parse_expression("-1.745*y5 + 0.43*y6 + 0.43*y7"),
            Expr::parse_expression("-280.0*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7"),
            Expr::parse_expression("280.0*y6*y8 - 1.81*y7"),
            Expr::parse_expression("-280.0*y6*y8 + 1.81*y7"),
        ],
        values: vec![
            "y1".to_string(),
            "y2".to_string(),
            "y3".to_string(),
            "y4".to_string(),
            "y5".to_string(),
            "y6".to_string(),
            "y7".to_string(),
            "y8".to_string(),
        ],
        y0: DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]),
        t0: 0.0,
        t_bound: 20.0,
        be_step: 0.005,
        bdf_max_step: 0.02,
        tolerance: 1e-9,
        max_iterations: 30,
        rtol: 1e-7,
        atol: 1e-10,
    }
}

fn vanderpol_1000_ivp_case() -> IvpScenario {
    IvpScenario {
        label: "vanderpol-1000",
        equations: vec![
            Expr::parse_expression("y2"),
            Expr::parse_expression("1000.0*(1.0-y1*y1)*y2 - y1"),
        ],
        values: vec!["y1".to_string(), "y2".to_string()],
        y0: DVector::from_vec(vec![2.0, 0.0]),
        t0: 0.0,
        t_bound: 50.0,
        be_step: 0.001,
        bdf_max_step: 0.01,
        tolerance: 1e-9,
        max_iterations: 40,
        rtol: 1e-7,
        atol: 1e-10,
    }
}

fn available_ivp_variants() -> Vec<IvpBackendVariant> {
    let mut variants = vec![IvpBackendVariant::Lambdify];
    if command_exists("gcc") {
        variants.push(IvpBackendVariant::CGcc);
    }
    if command_exists("tcc") {
        variants.push(IvpBackendVariant::CTcc);
    }
    if command_exists("zig") {
        variants.push(IvpBackendVariant::Zig);
    }
    variants
}

fn compare_output_dir(solver: &str, scenario: &IvpScenario, variant: IvpBackendVariant) -> PathBuf {
    PathBuf::from("target")
        .join("test-artifacts")
        .join("ivp-backend-compare")
        .join(format!(
            "{}-{}-{}",
            solver,
            scenario.label,
            variant.label().to_ascii_lowercase().replace('+', "_")
        ))
}

fn apply_variant_to_be_options(
    options: BeSolverOptions,
    scenario: &IvpScenario,
    variant: IvpBackendVariant,
) -> BeSolverOptions {
    match variant {
        IvpBackendVariant::Lambdify => options,
        IvpBackendVariant::CGcc => {
            options.with_dense_generated_backend_c_gcc(compare_output_dir("be", scenario, variant))
        }
        IvpBackendVariant::CTcc => {
            options.with_dense_generated_backend_c_tcc(compare_output_dir("be", scenario, variant))
        }
        IvpBackendVariant::Zig => {
            options.with_dense_generated_backend_zig(compare_output_dir("be", scenario, variant))
        }
    }
}

fn apply_variant_to_bdf_options(
    options: BdfSolverOptions,
    scenario: &IvpScenario,
    variant: IvpBackendVariant,
) -> BdfSolverOptions {
    match variant {
        IvpBackendVariant::Lambdify => options,
        IvpBackendVariant::CGcc => options
            .with_dense_generated_backend_c_gcc(compare_output_dir("bdf", scenario, variant)),
        IvpBackendVariant::CTcc => options
            .with_dense_generated_backend_c_tcc(compare_output_dir("bdf", scenario, variant)),
        IvpBackendVariant::Zig => {
            options.with_dense_generated_backend_zig(compare_output_dir("bdf", scenario, variant))
        }
    }
}

fn max_abs_vector(v: &DVector<f64>) -> f64 {
    v.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn max_abs_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()))
}

fn run_be_variant(
    scenario: &IvpScenario,
    variant: IvpBackendVariant,
) -> Result<(Duration, IvpBackendStatistics, DVector<f64>, String), String> {
    let options = BeSolverOptions::new(
        scenario.equations.clone(),
        scenario.values.clone(),
        "t".to_string(),
        scenario.tolerance,
        scenario.max_iterations,
        Some(scenario.be_step),
        scenario.t0,
        scenario.t_bound,
        scenario.y0.clone(),
    );
    let options = apply_variant_to_be_options(options, scenario, variant);
    let mut solver = BE::new_with_options(options);
    let start = Instant::now();
    solver.solve();
    let total = start.elapsed();
    let status = solver.get_status().clone();
    let statistics = solver.get_statistics();
    let (_, maybe_y) = solver.get_result();
    let y = maybe_y.ok_or_else(|| "BE solver did not produce result matrix".to_string())?;
    if y.nrows() == 0 {
        return Err("BE solver returned empty result matrix".to_string());
    }
    let final_solution = y.row(y.nrows() - 1).transpose().into_owned();
    Ok((total, statistics, final_solution, status))
}

fn run_bdf_variant(
    scenario: &IvpScenario,
    variant: IvpBackendVariant,
) -> Result<(Duration, IvpBackendStatistics, DVector<f64>, String), String> {
    let options = BdfSolverOptions::new(
        scenario.equations.clone(),
        scenario.values.clone(),
        "t".to_string(),
        "BDF".to_string(),
        scenario.t0,
        scenario.y0.clone(),
        scenario.t_bound,
        scenario.bdf_max_step,
        scenario.rtol,
        scenario.atol,
        None,
        false,
        None,
    );
    let options = apply_variant_to_bdf_options(options, scenario, variant);
    let mut solver = ODEsolver::new_with_options(options);
    let start = Instant::now();
    solver.solve();
    let total = start.elapsed();
    let status = solver.get_status().clone();
    let statistics = solver.get_statistics();
    let (_, y) = solver.get_result();
    if y.nrows() == 0 {
        return Err("BDF solver returned empty result matrix".to_string());
    }
    let final_solution = y.row(y.nrows() - 1).transpose().into_owned();
    Ok((total, statistics, final_solution, status))
}

fn print_ivp_compare_table(
    solver_label: &str,
    scenario: &IvpScenario,
    summaries: &[IvpRunSummary],
) {
    println!(
        "[IVP backend compare] solver={}, scenario={}, vars={}",
        solver_label,
        scenario.label,
        scenario.values.len()
    );
    println!(
        "{:<12} | {:>12} | {:>14} | {:>24} | {:<6}",
        "variant", "total_ms", "max_abs_solution", "solution_diff_vs_lambdify", "status"
    );
    println!("{}", "-".repeat(84));
    for row in summaries {
        println!(
            "{:<12} | {:>12.3} | {:>14.6e} | {:>24.6e} | {:<6}",
            row.variant.label(),
            row.total.as_secs_f64() * 1e3,
            row.max_abs_solution,
            row.solution_diff,
            row.status
        );
    }
    println!(
        "{:<12} | {:>10} | {:>10} | {:>14} | {:>14} | {:>12}",
        "variant",
        "setup_ms",
        "solve_ms",
        "residual_ms(avg)",
        "jacobian_ms(avg)",
        "steps"
    );
    println!("{}", "-".repeat(84));
    for row in summaries {
        println!(
            "{:<12} | {:>10.3} | {:>10.3} | {:>14.6} | {:>14.6} | {:>12}",
            row.variant.label(),
            row.statistics.backend_prepare_ms_total,
            row.statistics.solve_ms_total,
            row.statistics.avg_residual_ms().unwrap_or(0.0),
            row.statistics.avg_jacobian_ms().unwrap_or(0.0),
            row.statistics.step_calls
        );
    }
    println!(
        "{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "variant", "res_calls", "jac_calls", "nl_solves", "nl_iters", "bdf nlu"
    );
    println!("{}", "-".repeat(84));
    for row in summaries {
        println!(
            "{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
            row.variant.label(),
            row.statistics.residual_calls,
            row.statistics.jacobian_calls,
            row.statistics.nonlinear_solve_calls,
            row.statistics.nonlinear_iterations_total,
            row.statistics.bdf_nlu_total
        );
    }
    if let Some(best_total) = summaries.iter().min_by(|lhs, rhs| lhs.total.cmp(&rhs.total)) {
        let best_solve = summaries
            .iter()
            .min_by(|lhs, rhs| lhs.statistics.solve_ms_total.total_cmp(&rhs.statistics.solve_ms_total))
            .expect("summary rows should be non-empty");
        let baseline = summaries
            .iter()
            .find(|row| matches!(row.variant, IvpBackendVariant::Lambdify))
            .expect("lambdify baseline should exist");
        let residual_total = baseline.statistics.residual_ms_total;
        let jacobian_total = baseline.statistics.jacobian_ms_total;
        let dominant = if residual_total > 2.0 * jacobian_total {
            "residual-dominated"
        } else if jacobian_total > 2.0 * residual_total {
            "jacobian-dominated"
        } else {
            "mixed"
        };
        println!(
            "[IVP backend compare] summary: dominant_hot_path={}, best_total={}, best_solve={}, baseline_residual_ms_total={:.3}, baseline_jacobian_ms_total={:.3}",
            dominant,
            best_total.variant.label(),
            best_solve.variant.label(),
            residual_total,
            jacobian_total
        );
    }
    println!(
        "[IVP backend compare] finished solver={} scenario `{}`",
        solver_label, scenario.label
    );
}

fn run_be_compare_for_scenario(scenario: &IvpScenario) {
    let variants = available_ivp_variants();
    let mut rows = Vec::new();
    let mut baseline_solution: Option<DVector<f64>> = None;

    for variant in variants {
        let (total, statistics, solution, status) = run_be_variant(scenario, variant)
            .unwrap_or_else(|err| panic!("BE {} failed for {}: {err}", variant.label(), scenario.label));
        let diff = if let Some(reference) = baseline_solution.as_ref() {
            max_abs_diff(&solution, reference)
        } else {
            baseline_solution = Some(solution.clone());
            0.0
        };
        rows.push(IvpRunSummary {
            variant,
            total,
            statistics,
            max_abs_solution: max_abs_vector(&solution),
            solution_diff: diff,
            status,
        });
    }

    print_ivp_compare_table("BE", scenario, rows.as_slice());
}

fn run_bdf_compare_for_scenario(scenario: &IvpScenario) {
    let variants = available_ivp_variants();
    let mut rows = Vec::new();
    let mut baseline_solution: Option<DVector<f64>> = None;

    for variant in variants {
        let (total, statistics, solution, status) = run_bdf_variant(scenario, variant)
            .unwrap_or_else(|err| panic!("BDF {} failed for {}: {err}", variant.label(), scenario.label));
        let diff = if let Some(reference) = baseline_solution.as_ref() {
            max_abs_diff(&solution, reference)
        } else {
            baseline_solution = Some(solution.clone());
            0.0
        };
        rows.push(IvpRunSummary {
            variant,
            total,
            statistics,
            max_abs_solution: max_abs_vector(&solution),
            solution_diff: diff,
            status,
        });
    }

    print_ivp_compare_table("BDF", scenario, rows.as_slice());
}

#[test]
#[ignore]
fn ivp_be_generated_backend_end_to_end_compare_table() {
    for scenario in [
        small_stiff_ivp_case(),
        robertson_ivp_case(),
        hires_8_ivp_case(),
        vanderpol_1000_ivp_case(),
    ] {
        run_be_compare_for_scenario(&scenario);
    }
}

#[test]
#[ignore]
fn ivp_bdf_generated_backend_end_to_end_compare_table() {
    for scenario in [
        small_stiff_ivp_case(),
        robertson_ivp_case(),
        hires_8_ivp_case(),
        vanderpol_1000_ivp_case(),
    ] {
        run_bdf_compare_for_scenario(&scenario);
    }
}

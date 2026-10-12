use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
use RustedSciThe::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;
use RustedSciThe::numerical::optimization::universal_fitting::{
    Method, UniversalFitting, UniversalFittingResult,
};
use std::time::Instant;
use tabled::{Table, Tabled};

const REPEATS: usize = 5;

#[derive(Tabled)]
struct BenchRow {
    frontend: &'static str,
    update_policy: &'static str,
    samples: usize,
    repeats: usize,
    prepare_median_ms: String,
    solve_median_ms: String,
    full_median_ms: String,
    objective_calls: usize,
    jacobian_calls: usize,
    objective_median_ms: String,
    jacobian_median_ms: String,
    linearization_median_ms: String,
    linear_solve_median_ms: String,
    fd_jacobians: usize,
    iterations: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    max_parameter_error: String,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(
    frontend: SymbolicLambdifyFrontend,
    update_policy: GavinLmUpdatePolicy,
    samples: usize,
) -> BenchRow {
    let x = (0..samples)
        .map(|i| 20.0 * i as f64 / (samples - 1) as f64)
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|&x| 2.0 * (-x / 0.5_f64).exp() + (x / 2.0_f64).sin())
        .collect::<Vec<_>>();
    let mut prepare = Vec::with_capacity(REPEATS);
    let mut solve = Vec::with_capacity(REPEATS);
    let mut full = Vec::with_capacity(REPEATS);
    let mut objective = Vec::with_capacity(REPEATS);
    let mut jacobian = Vec::with_capacity(REPEATS);
    let mut linearization = Vec::with_capacity(REPEATS);
    let mut linear_solve = Vec::with_capacity(REPEATS);
    let mut objective_calls = 0;
    let mut jacobian_calls = 0;
    let mut fd_jacobians = 0;
    let mut iterations = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut max_error = 0.0_f64;

    for _ in 0..REPEATS {
        let started = Instant::now();
        let result = UniversalFitting::new()
            .with_method(Method::GAVIN)
            .with_data(x.clone(), y.clone())
            .with_equation_str("a*exp(-x/tau)+c*sin(x/omega)".to_owned())
            .with_arg("x".to_owned())
            .with_unknowns(vec![
                "a".to_owned(),
                "tau".to_owned(),
                "c".to_owned(),
                "omega".to_owned(),
            ])
            .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
            .with_gavin_frontend(frontend)
            .with_gavin_update_policy(update_policy)
            .with_gavin_telemetry(true)
            .build()
            .expect("Gavin benchmark fit should complete");
        full.push(started.elapsed().as_secs_f64() * 1e3);
        let UniversalFittingResult::GAVIN(fit) = result else {
            panic!("wrong fitting method result");
        };
        assert!(fit.fit.termination.is_converged());
        for (actual, expected) in fit.fit.parameters.iter().zip([2.0, 0.5, 1.0, 2.0]) {
            max_error = max_error.max((actual - expected).abs());
        }
        let stats = fit.telemetry.expect("benchmark enables solver telemetry");
        prepare.push(fit.preparation.preparation_duration.as_secs_f64() * 1e3);
        solve.push(stats.total_time.as_secs_f64() * 1e3);
        objective.push(stats.objective_time.as_secs_f64() * 1e3);
        jacobian.push(stats.analytic_jacobian_time.as_secs_f64() * 1e3);
        linearization.push(stats.linearization_time.as_secs_f64() * 1e3);
        linear_solve.push(stats.linear_solve_time.as_secs_f64() * 1e3);
        objective_calls = stats.objective_evaluations;
        jacobian_calls = stats.analytic_jacobian_evaluations;
        fd_jacobians = stats.finite_difference_jacobians;
        iterations = stats.iterations;
        accepted_steps = stats.accepted_steps;
        rejected_steps = stats.rejected_steps;
    }
    assert!(max_error < 2e-2, "fit error={max_error}");
    BenchRow {
        frontend: match frontend {
            SymbolicLambdifyFrontend::ExprLegacy => "ExprLegacy",
            SymbolicLambdifyFrontend::AtomViewNative => "AtomViewNative",
        },
        update_policy: match update_policy {
            GavinLmUpdatePolicy::Marquardt => "Marquardt",
            GavinLmUpdatePolicy::Quadratic => "Quadratic",
            GavinLmUpdatePolicy::Nielsen => "Nielsen",
        },
        samples,
        repeats: REPEATS,
        prepare_median_ms: format!("{:.3}", median(&mut prepare)),
        solve_median_ms: format!("{:.3}", median(&mut solve)),
        full_median_ms: format!("{:.3}", median(&mut full)),
        objective_calls,
        jacobian_calls,
        objective_median_ms: format!("{:.3}", median(&mut objective)),
        jacobian_median_ms: format!("{:.3}", median(&mut jacobian)),
        linearization_median_ms: format!("{:.3}", median(&mut linearization)),
        linear_solve_median_ms: format!("{:.3}", median(&mut linear_solve)),
        fd_jacobians,
        iterations,
        accepted_steps,
        rejected_steps,
        max_parameter_error: format!("{max_error:.3e}"),
    }
}

fn main() {
    let mut rows = Vec::new();
    for samples in [201, 1001] {
        for policy in [
            GavinLmUpdatePolicy::Marquardt,
            GavinLmUpdatePolicy::Quadratic,
            GavinLmUpdatePolicy::Nielsen,
        ] {
            for frontend in [
                SymbolicLambdifyFrontend::ExprLegacy,
                SymbolicLambdifyFrontend::AtomViewNative,
            ] {
                rows.push(measure(frontend, policy, samples));
            }
        }
    }
    let table = Table::new(&rows).to_string();
    let report = format!(
        "# Gavin symbolic fitting performance matrix\n\n- repeats_per_row: {REPEATS}\n- controller policies: Marquardt, Quadratic, Nielsen\n- model: `a*exp(-x/tau)+c*sin(x/omega)`\n- lifecycle: fresh symbolic preparation and solve per sample\n- prepare_median_ms is symbolic preparation; solve_median_ms is solver-internal; full_median_ms is external end-to-end time including parsing/adapter overhead. Do not require these medians to sum exactly\n- inner solver timing scopes overlap (objective and Jacobian evaluations occur during linearization)\n\n{table}\n"
    );
    println!("{report}");
    let path = write_test_report(
        "Optimization_Gavin",
        "gavin_fitting_performance_matrix",
        &report,
    )
    .expect("write Gavin fitting benchmark report");
    println!("[Gavin benchmark] report={}", path.display());
}

//! Independent correctness gates for the staged rectangular LM controller.
//!
//! The old `optimization` implementation was intentionally removed after the
//! production users migrated. These tests therefore exercise only the staged
//! core and can remain as long-term regression gates.

use super::lm::{LeastSquaresTelemetryMode, LevenbergMarquardt};
use super::problem::{ClosureLeastSquaresProblem, LeastSquaresProblem};
use levenberg_marquardt::LeastSquaresProblem as UpstreamLeastSquaresProblem;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Copy)]
enum UpstreamParityCase {
    Rosenbrock,
    Overdetermined,
    Underdetermined,
    Wide,
    RankDeficient,
    BadlyScaled,
    NearlySingular,
}

impl UpstreamParityCase {
    fn name(self) -> &'static str {
        match self {
            Self::Rosenbrock => "rosenbrock",
            Self::Overdetermined => "overdetermined",
            Self::Underdetermined => "underdetermined",
            Self::Wide => "wide",
            Self::RankDeficient => "rank-deficient",
            Self::BadlyScaled => "badly-scaled",
            Self::NearlySingular => "nearly-singular",
        }
    }

    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::Rosenbrock => (2, 2),
            Self::Overdetermined => (4, 2),
            Self::Underdetermined => (1, 2),
            Self::Wide => (2, 4),
            Self::RankDeficient => (2, 2),
            Self::BadlyScaled => (2, 2),
            Self::NearlySingular => (2, 2),
        }
    }

    fn initial(self) -> Vec<f64> {
        match self {
            Self::Rosenbrock => vec![-1.2, 1.0],
            Self::Overdetermined => vec![0.0, 0.0],
            Self::Underdetermined | Self::RankDeficient | Self::NearlySingular => {
                vec![0.0, 0.0]
            }
            Self::Wide => vec![0.0, 0.0, 0.0, 0.0],
            Self::BadlyScaled => vec![0.0, 0.0],
        }
    }

    fn residuals(self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Rosenbrock => vec![1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])],
            Self::Overdetermined => {
                vec![x[0] - 1.0, 2.0 * x[0] - 2.0, x[1] + 1.0, 3.0 * x[1] + 3.0]
            }
            Self::Underdetermined => vec![x[0] + x[1] - 1.0],
            Self::Wide => vec![
                x[0] + x[1] + x[2] + x[3] - 1.0,
                x[0] - x[1] + x[2] - x[3] - 0.5,
            ],
            Self::RankDeficient => vec![x[0] + x[1] - 1.0, 2.0 * x[0] + 2.0 * x[1] - 2.0],
            Self::BadlyScaled => vec![1.0e3 * x[0] - 1.0, 1.0e-3 * x[1] - 1.0],
            Self::NearlySingular => {
                let eps = 1.0e-8;
                vec![x[0] + x[1] - 2.0, x[0] + (1.0 + eps) * x[1] - (2.0 + eps)]
            }
        }
    }

    fn jacobian(self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Rosenbrock => vec![-1.0, 0.0, -20.0 * x[0], 10.0],
            Self::Overdetermined => vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0],
            Self::Underdetermined => vec![1.0, 1.0],
            Self::Wide => vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
            Self::RankDeficient => vec![1.0, 1.0, 2.0, 2.0],
            Self::BadlyScaled => vec![1.0e3, 0.0, 0.0, 1.0e-3],
            Self::NearlySingular => vec![1.0, 1.0, 1.0, 1.0 + 1.0e-8],
        }
    }
}

struct UpstreamLmProblem {
    case: UpstreamParityCase,
    params: nalgebra_lm::DVector<f64>,
    residual_trace: Rc<RefCell<Vec<Vec<f64>>>>,
}

impl levenberg_marquardt::LeastSquaresProblem<f64, nalgebra_lm::Dyn, nalgebra_lm::Dyn>
    for UpstreamLmProblem
{
    type ParameterStorage = nalgebra_lm::storage::Owned<f64, nalgebra_lm::Dyn>;
    type ResidualStorage = nalgebra_lm::storage::Owned<f64, nalgebra_lm::Dyn>;
    type JacobianStorage = nalgebra_lm::storage::Owned<f64, nalgebra_lm::Dyn, nalgebra_lm::Dyn>;

    fn set_params(&mut self, x: &nalgebra_lm::DVector<f64>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> nalgebra_lm::DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<nalgebra_lm::DVector<f64>> {
        let x = self.params.as_slice();
        self.residual_trace.borrow_mut().push(x.to_vec());
        Some(nalgebra_lm::DVector::from_vec(self.case.residuals(x)))
    }

    fn jacobian(&self) -> Option<nalgebra_lm::DMatrix<f64>> {
        let (rows, cols) = self.case.dimensions();
        Some(nalgebra_lm::DMatrix::from_row_slice(
            rows,
            cols,
            &self.case.jacobian(self.params.as_slice()),
        ))
    }
}

fn run_upstream_parity_case(case: UpstreamParityCase) -> (usize, usize, f64, f64, f64, String) {
    const TOL: f64 = 1.0e-12;
    const PATIENCE: usize = 200;

    let local_trace = Rc::new(RefCell::new(Vec::new()));
    let local_residual_trace = Rc::clone(&local_trace);
    let local_problem = ClosureLeastSquaresProblem::new(
        DVector::from_vec(case.initial()),
        move |x: &DVector<f64>| {
            local_residual_trace
                .borrow_mut()
                .push(x.as_slice().to_vec());
            DVector::from_vec(case.residuals(x.as_slice()))
        },
        move |x: &DVector<f64>| {
            let (rows, cols) = case.dimensions();
            DMatrix::from_row_slice(rows, cols, &case.jacobian(x.as_slice()))
        },
    );
    let (local_problem, local_report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .with_patience(PATIENCE)
        .with_telemetry(LeastSquaresTelemetryMode::Counters)
        .minimize(local_problem);

    let upstream_trace = Rc::new(RefCell::new(Vec::new()));
    let upstream_problem = UpstreamLmProblem {
        case,
        params: nalgebra_lm::DVector::from_vec(case.initial()),
        residual_trace: Rc::clone(&upstream_trace),
    };
    let (upstream_problem, upstream_report) = levenberg_marquardt::LevenbergMarquardt::new()
        .with_tol(TOL)
        .with_patience(PATIENCE)
        .minimize(upstream_problem);

    let local_trace = local_trace.borrow();
    let upstream_trace = upstream_trace.borrow();
    assert_eq!(
        local_trace.len(),
        upstream_trace.len(),
        "residual callback count differs for case: {} vs {}",
        local_trace.len(),
        upstream_trace.len()
    );
    let mut trajectory_max_abs_diff: f64 = 0.0;
    for (step, (local, upstream)) in local_trace.iter().zip(upstream_trace.iter()).enumerate() {
        assert_eq!(local.len(), upstream.len());
        for (index, (local, upstream)) in local.iter().zip(upstream.iter()).enumerate() {
            trajectory_max_abs_diff = trajectory_max_abs_diff.max((local - upstream).abs());
            assert!(
                (local - upstream).abs() <= 1.0e-10 * (1.0 + upstream.abs()),
                "trajectory differs at step {step}, parameter {index}: local={local:e}, upstream={upstream:e}"
            );
        }
    }

    assert_eq!(
        format!("{:?}", local_report.termination),
        format!("{:?}", upstream_report.termination),
        "termination differs for parity case"
    );
    assert_eq!(
        local_report.number_of_evaluations,
        upstream_report.number_of_evaluations
    );
    let objective_abs_diff =
        (local_report.objective_function - upstream_report.objective_function).abs();
    assert!(
        objective_abs_diff <= 1.0e-12 * (1.0 + upstream_report.objective_function.abs()),
        "objective differs: local={}, upstream={}",
        local_report.objective_function,
        upstream_report.objective_function
    );
    let mut final_parameter_max_abs_diff: f64 = 0.0;
    for (index, (local, upstream)) in local_problem
        .params()
        .iter()
        .zip(upstream_problem.params().iter())
        .enumerate()
    {
        final_parameter_max_abs_diff = final_parameter_max_abs_diff.max((local - upstream).abs());
        assert!(
            (local - upstream).abs() <= 1.0e-10 * (1.0 + upstream.abs()),
            "final parameter {index} differs: local={local:e}, upstream={upstream:e}"
        );
    }

    (
        local_report.number_of_evaluations,
        local_report.statistics.rejected_steps,
        trajectory_max_abs_diff,
        final_parameter_max_abs_diff,
        objective_abs_diff,
        format!("{:?}", local_report.termination),
    )
}

#[test]
fn canonical_lm_matches_upstream_0150_trajectory_and_report() {
    println!(
        "[LM upstream 0.15.0 parity] case | residual_calls | local_rejected_steps | trajectory_max_abs_diff | final_parameter_max_abs_diff | objective_abs_diff | termination"
    );
    for case in [
        UpstreamParityCase::Rosenbrock,
        UpstreamParityCase::Overdetermined,
        UpstreamParityCase::Underdetermined,
        UpstreamParityCase::Wide,
        UpstreamParityCase::RankDeficient,
        UpstreamParityCase::BadlyScaled,
        UpstreamParityCase::NearlySingular,
    ] {
        let (
            residual_calls,
            rejected_steps,
            trajectory_diff,
            parameter_diff,
            objective_diff,
            termination,
        ) = run_upstream_parity_case(case);
        if matches!(case, UpstreamParityCase::Rosenbrock) {
            assert!(
                rejected_steps > 0,
                "Rosenbrock parity case must exercise rejected trust-region trials"
            );
        }
        println!(
            "[LM upstream 0.15.0 parity] {} | {} | {} | {:.3e} | {:.3e} | {:.3e} | {}",
            case.name(),
            residual_calls,
            rejected_steps,
            trajectory_diff,
            parameter_diff,
            objective_diff,
            termination
        );
    }
}

#[test]
fn staged_lm_converges_on_rosenbrock() {
    let problem = ClosureLeastSquaresProblem::new(
        DVector::from_vec(vec![-1.2, 1.0]),
        |x: &DVector<f64>| DVector::from_vec(vec![1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])]),
        |x: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -20.0 * x[0], 10.0]),
    );
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(1e-12)
        .with_patience(200)
        .minimize(problem);

    assert!(report.termination.was_successful());
    assert!(report.objective_function <= 1e-18);
    assert!((problem.params() - DVector::from_vec(vec![1.0, 1.0])).norm() <= 1e-8);
}

#[test]
fn staged_lm_handles_underdetermined_problem() {
    let problem = ClosureLeastSquaresProblem::new(
        DVector::from_vec(vec![0.0, 0.0, 0.0]),
        |x: &DVector<f64>| DVector::from_vec(vec![x[0] + x[1] + x[2] - 1.0]),
        |_x: &DVector<f64>| DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]),
    );
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(1e-12)
        .with_patience(100)
        .minimize(problem);

    assert!(report.termination.was_successful());
    assert!(report.objective_function <= 1e-18);
    assert!((problem.params().sum() - 1.0).abs() <= 1e-8);
}

#[test]
fn staged_lm_preserves_exact_initial_solution() {
    let problem = ClosureLeastSquaresProblem::new(
        DVector::from_vec(vec![2.0, -1.0]),
        |x: &DVector<f64>| DVector::from_vec(vec![x[0] - 2.0, x[1] + 1.0]),
        |_x: &DVector<f64>| DMatrix::identity(2, 2),
    );
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);

    assert!(report.termination.was_successful());
    assert_eq!(problem.params(), DVector::from_vec(vec![2.0, -1.0]));
    assert_eq!(report.number_of_evaluations, 1);
}

#[test]
fn staged_lm_rejects_domain_trial_and_recovers() {
    struct PositiveTrialProblem {
        params: DVector<f64>,
    }

    impl LeastSquaresProblem for PositiveTrialProblem {
        fn set_params(&mut self, x: &DVector<f64>) {
            self.params.copy_from(x);
        }

        fn params(&self) -> DVector<f64> {
            self.params.clone()
        }

        fn residuals(&self) -> Option<DVector<f64>> {
            Some(DVector::from_vec(vec![self.params[0] - 1.0]))
        }

        fn jacobian(&self) -> Option<DMatrix<f64>> {
            // Deliberately conservative Jacobian: the unconstrained LM trial
            // overshoots below zero, while a reduced trust region can recover.
            Some(DMatrix::from_row_slice(1, 1, &[0.1]))
        }

        fn validate_trial(&self, x: &DVector<f64>) -> bool {
            x.len() == 1 && x[0].is_finite() && x[0] > 0.0
        }
    }

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(1e-10)
        .with_patience(200)
        .with_telemetry(LeastSquaresTelemetryMode::Detailed)
        .minimize(PositiveTrialProblem {
            params: DVector::from_vec(vec![1.2]),
        });

    assert!(report.termination.was_successful());
    assert!(report.rejected_domain_trials > 0);
    assert!(report.statistics.domain_rejections > 0);
    assert!((problem.params()[0] - 1.0).abs() <= 1e-8);
}

#[test]
fn staged_lm_telemetry_is_disabled_without_runtime_cost_contract() {
    let problem = ClosureLeastSquaresProblem::new(
        DVector::from_vec(vec![0.0]),
        |x: &DVector<f64>| DVector::from_vec(vec![x[0] - 1.0]),
        |_x: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[1.0]),
    );
    let (_problem, report) = LevenbergMarquardt::new().minimize(problem);

    assert_eq!(
        report.statistics.availability,
        crate::numerical::Nonlinear_systems::engine::StatisticsAvailability::NotCollected
    );
    assert!(!report.statistics.timings_collected);
    assert_eq!(report.statistics.residual_evaluations, 0);
    assert_eq!(report.statistics.linear_solves, 0);
    assert_eq!(report.statistics.total_duration, std::time::Duration::ZERO);
    assert_eq!(
        report.statistics.trust_region_subproblem_duration,
        std::time::Duration::ZERO
    );
}

#[test]
fn staged_lm_counter_and_detailed_telemetry_have_distinct_semantics() {
    let make_problem = || {
        ClosureLeastSquaresProblem::new(
            DVector::from_vec(vec![0.0]),
            |x: &DVector<f64>| DVector::from_vec(vec![x[0] - 1.0]),
            |_x: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[1.0]),
        )
    };

    let (_problem, counters) = LevenbergMarquardt::new()
        .with_telemetry(LeastSquaresTelemetryMode::Counters)
        .minimize(make_problem());
    assert!(counters.statistics.availability.is_collected());
    assert!(!counters.statistics.timings_collected);
    assert!(counters.statistics.residual_evaluations > 0);
    assert!(counters.statistics.jacobian_evaluations > 0);
    assert!(counters.statistics.linear_factorizations > 0);
    assert!(counters.statistics.trust_region_trials > 0);
    assert_eq!(
        counters.statistics.total_duration,
        std::time::Duration::ZERO
    );

    let (_problem, detailed) = LevenbergMarquardt::new()
        .with_telemetry(LeastSquaresTelemetryMode::Detailed)
        .minimize(make_problem());
    assert!(detailed.statistics.availability.is_collected());
    assert!(detailed.statistics.timings_collected);
    assert!(detailed.statistics.total_duration > std::time::Duration::ZERO);
    assert!(detailed.statistics.trust_region_subproblem_duration > std::time::Duration::ZERO);
    assert!(detailed.statistics.linear_solves >= detailed.statistics.trust_region_trials);
}

#[test]
fn staged_lm_try_api_preserves_typed_callback_errors() {
    #[derive(Debug)]
    struct FailingProblem;

    impl LeastSquaresProblem for FailingProblem {
        fn set_params(&mut self, _: &DVector<f64>) {}
        fn params(&self) -> DVector<f64> {
            DVector::from_element(1, 1.0)
        }
        fn residuals(&self) -> Option<DVector<f64>> {
            None
        }
        fn jacobian(&self) -> Option<DMatrix<f64>> {
            None
        }
        fn try_residuals(&self) -> Result<DVector<f64>, super::errors::LeastSquaresError> {
            Err(
                crate::numerical::Nonlinear_systems::error::SolveError::ResidualEvaluation(
                    "synthetic callback failure".to_string(),
                )
                .into(),
            )
        }
    }

    let error = LevenbergMarquardt::new()
        .try_minimize(FailingProblem)
        .expect_err("fallible API should retain callback errors");
    assert!(matches!(
        error,
        super::errors::LeastSquaresError::Problem(_)
    ));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn staged_lm_try_api_reports_invalid_configuration_without_panicking() {
    let error = LevenbergMarquardt::new()
        .try_with_ftol(f64::NAN)
        .expect_err("NaN tolerance must be rejected as a typed error");
    assert!(matches!(
        error,
        super::errors::LeastSquaresError::InvalidConfiguration { field: "ftol", .. }
    ));
}

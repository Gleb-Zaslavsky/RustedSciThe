//! Callback-only execution-policy matrix for the AtomView Lambdify route.
//!
//! ```text
//! cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_callback_matrix -- --nocapture --test-threads=1
//! cargo test --release --lib --no-default-features numerical::BVP_Damp::test_lambdify_callback_matrix::tests::atomview_callback_only_auto_policy_matrix_release_story -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The first command is a debug correctness and diagnostic gate. The ignored
//! test is the stable release repetition story; it uses the same fixture and
//! protocol, but records callback-only timing rather than solver performance.
//! It evaluates one prepared Atom system through the production faer Sparse
//! and native Banded callback boundaries with Sequential, Parallel and Auto.
//! Cold preparation, Newton iterations and linear solves are intentionally
//! outside this test. The worker pool is warmed before any timed callback, and
//! both Banded chunking strategies are reported independently.

#[cfg(test)]
mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::symbolic::View::atom::Atom;
    use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
    use crate::symbolic::View::parser;
    use crate::symbolic::bvp::atom_lambdify::{
        sparse_jacobian_with_binding_and_policy, sparse_residual_with_binding_and_policy,
    };
    use crate::symbolic::bvp::direct::{
        BandedJacobianChunking, BandedLambdifyConfig, DirectBandedProblem,
    };
    use crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle;
    use crate::symbolic::bvp::telemetry::{
        BvpLambdifyExecutionPolicy, BvpLambdifyTelemetry, BvpLambdifyTelemetryMode,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use faer::col::Col;
    use nalgebra::DMatrix;
    use rayon::prelude::*;
    use std::time::Instant;

    const CALLBACK_REPETITIONS: usize = 3;
    const DIMENSIONS: &[usize] = &[16, 64, 256];

    #[derive(Clone, Copy, Debug)]
    enum MatrixRoute {
        Sparse,
        Banded,
    }

    impl MatrixRoute {
        const fn label(self) -> &'static str {
            match self {
                Self::Sparse => "Sparse-faer",
                Self::Banded => "Banded-native",
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct PolicyCase {
        label: &'static str,
        policy: BvpLambdifyExecutionPolicy,
    }

    fn policy_cases() -> [PolicyCase; 3] {
        [
            PolicyCase {
                label: "Sequential",
                policy: BvpLambdifyExecutionPolicy::Sequential,
            },
            PolicyCase {
                label: "Parallel",
                policy: BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
            },
            PolicyCase {
                label: "Auto",
                policy: BvpLambdifyExecutionPolicy::Auto { min_work: 0 },
            },
        ]
    }

    #[derive(Clone, Copy, Debug)]
    struct ChunkCase {
        label: &'static str,
        chunking: BandedJacobianChunking,
    }

    fn chunk_cases() -> [ChunkCase; 2] {
        [
            ChunkCase {
                label: "Diagonal",
                chunking: BandedJacobianChunking::Diagonal,
            },
            ChunkCase {
                label: "EntryChunks",
                chunking: BandedJacobianChunking::EntryChunks,
            },
        ]
    }

    /// Initialize Rayon outside the callback timing window.
    ///
    /// The production callbacks use the process-global pool. Without this
    /// explicit warm-up, the first `Parallel`/`Auto` observation can include
    /// pool construction and make a callback-only comparison misleading.
    fn warm_worker_pool() -> usize {
        let workers = rayon::current_num_threads().max(1);
        let work_items = workers.saturating_mul(64).max(64);
        let checksum = (0..work_items)
            .into_par_iter()
            .map(|value| value.wrapping_mul(3).wrapping_add(1))
            .sum::<usize>();
        assert!(checksum > 0);
        workers
    }

    struct PreparedAtomSystem {
        residuals: Vec<Atom>,
        jacobian: Vec<SparseAtomJacobianEntry>,
        variables: Vec<String>,
        state: Vec<f64>,
        expr_residuals: Vec<Expr>,
    }

    fn parse_atom(expression: &str) -> Atom {
        parser::parse(expression).unwrap_or_else(|error| {
            panic!("callback matrix fixture expression should parse: {expression}: {error:?}")
        })
    }

    fn parse_expr(expression: &str) -> Expr {
        Expr::parse_expression(expression)
    }

    fn prepared_system(dimension: usize) -> PreparedAtomSystem {
        let variables = (0..dimension)
            .map(|index| format!("y_{index}"))
            .collect::<Vec<_>>();
        let mut residuals = Vec::with_capacity(dimension);
        let mut expr_residuals = Vec::with_capacity(dimension);
        let mut jacobian = Vec::with_capacity(dimension * 3);

        for row in 0..dimension {
            let mut terms = vec![format!("y_{row}")];
            jacobian.push(SparseAtomJacobianEntry {
                row,
                col: row,
                value: parse_atom("1.0"),
            });
            if row > 0 {
                terms.push(format!("-0.25*y_{}", row - 1));
                jacobian.push(SparseAtomJacobianEntry {
                    row,
                    col: row - 1,
                    value: parse_atom("-0.25"),
                });
            }
            if row + 1 < dimension {
                terms.push(format!("0.5*y_{}", row + 1));
                jacobian.push(SparseAtomJacobianEntry {
                    row,
                    col: row + 1,
                    value: parse_atom("0.5"),
                });
            }
            let expression = terms.join("+");
            residuals.push(parse_atom(&expression));
            expr_residuals.push(parse_expr(&expression));
        }

        let state = (0..dimension)
            .map(|index| 1.0 + (index as f64) * 1e-3)
            .collect();
        PreparedAtomSystem {
            residuals,
            jacobian,
            variables,
            state,
            expr_residuals,
        }
    }

    fn matrix_max_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        assert_eq!(lhs.shape(), rhs.shape());
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max)
    }

    fn vector_max_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max)
    }

    fn banded_to_dense(
        assembly: &crate::somelinalg::banded::banded_assembly::BandedAssembly,
    ) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(assembly.n(), assembly.n());
        for row in 0..assembly.n() {
            for col in 0..assembly.n() {
                let offset = col as isize - row as isize;
                if (-(assembly.kl() as isize)..=(assembly.ku() as isize)).contains(&offset) {
                    matrix[(row, col)] = assembly
                        .get(row, col)
                        .expect("in-band fixture coordinates must be valid");
                }
            }
        }
        matrix
    }

    #[derive(Debug)]
    struct Observation {
        route: &'static str,
        dimension: usize,
        policy: &'static str,
        chunking: &'static str,
        residual_us: f64,
        jacobian_us: f64,
        residual_calls: u64,
        jacobian_calls: u64,
        sequential_dispatches: u64,
        parallel_dispatches: u64,
        effective_tasks: u64,
        evaluator_calls: u64,
        storage_writes: u64,
        max_residual_diff: f64,
        max_jacobian_diff: f64,
    }

    fn timed_repetitions<T>(repetitions: usize, mut callback: impl FnMut() -> T) -> (f64, T) {
        let mut elapsed_us = 0.0;
        let mut last = None;
        for _ in 0..repetitions {
            let started = Instant::now();
            last = Some(callback());
            elapsed_us += started.elapsed().as_secs_f64() * 1_000_000.0;
        }
        (
            elapsed_us / repetitions as f64,
            last.expect("callback repetitions are non-zero"),
        )
    }

    fn run_sparse(
        system: &PreparedAtomSystem,
        case: PolicyCase,
        repetitions: usize,
        expected_residual: Option<&[f64]>,
        expected_jacobian: Option<&DMatrix<f64>>,
    ) -> Observation {
        let telemetry = BvpLambdifyTelemetry::counters();
        let binding = BvpParameterBindingHandle::new(None);
        let residual = sparse_residual_with_binding_and_policy(
            system.residuals.clone(),
            system.variables.clone(),
            binding.clone(),
            0,
            telemetry.clone(),
            case.policy,
        );
        let mut jacobian = sparse_jacobian_with_binding_and_policy(
            system.jacobian.clone(),
            system.state.len(),
            system.state.len(),
            system.variables.clone(),
            binding,
            0,
            telemetry.clone(),
            case.policy,
        );
        let state = Col::from_fn(system.state.len(), |index| system.state[index]);

        let (residual_us, residual_result) = timed_repetitions(repetitions, || {
            residual
                .try_call(0.0, &state)
                .expect("Sparse residual callback should succeed")
                .to_DVectorType()
        });
        let (jacobian_us, jacobian_result) = timed_repetitions(repetitions, || {
            jacobian
                .try_call(0.0, &state)
                .expect("Sparse Jacobian callback should succeed")
                .to_DMatrixType()
        });
        let snapshot = telemetry.snapshot();
        let residual_values = residual_result.as_slice();
        Observation {
            route: MatrixRoute::Sparse.label(),
            dimension: system.state.len(),
            policy: case.label,
            chunking: "n/a",
            residual_us,
            jacobian_us,
            residual_calls: snapshot.residual_calls,
            jacobian_calls: snapshot.jacobian_calls,
            sequential_dispatches: snapshot.sequential_dispatches,
            parallel_dispatches: snapshot.parallel_dispatches,
            effective_tasks: 0,
            evaluator_calls: 0,
            storage_writes: 0,
            max_residual_diff: expected_residual
                .map(|expected| vector_max_diff(expected, residual_values))
                .unwrap_or(0.0),
            max_jacobian_diff: expected_jacobian
                .map(|expected| matrix_max_diff(expected, &jacobian_result))
                .unwrap_or(0.0),
        }
    }

    fn run_banded(
        system: &PreparedAtomSystem,
        case: PolicyCase,
        chunk_case: ChunkCase,
        repetitions: usize,
        expected_residual: Option<&[f64]>,
        expected_jacobian: Option<&DMatrix<f64>>,
    ) -> Observation {
        let mut problem = DirectBandedProblem::default()
            .with_atom_view_data(system.residuals.clone(), system.jacobian.clone());
        problem.vector_of_functions = system.expr_residuals.clone();
        problem.vector_of_variables = system
            .variables
            .iter()
            .map(|name| Expr::parse_expression(name))
            .collect();
        problem.variable_string = system.variables.clone();
        problem.bandwidth = Some((1, 1));

        let config = BandedLambdifyConfig {
            execution_policy: case.policy,
            jacobian_chunking: chunk_case.chunking,
            telemetry_mode: BvpLambdifyTelemetryMode::Counters,
            ..BandedLambdifyConfig::default()
        };
        let residual = problem
            .generate_banded_residual_with_config(&config)
            .expect("Banded residual preparation should succeed");
        let jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&config)
            .expect("Banded Jacobian preparation should succeed");

        let (residual_us, residual_result) = timed_repetitions(repetitions, || {
            residual(&system.state).expect("Banded residual callback should succeed")
        });
        let (jacobian_us, jacobian_result) = timed_repetitions(repetitions, || {
            jacobian.callback()(&system.state).expect("Banded Jacobian callback should succeed")
        });
        let snapshot = jacobian.telemetry_snapshot();
        let residual_values = residual_result.as_slice();
        let jacobian_values = banded_to_dense(&jacobian_result);
        Observation {
            route: MatrixRoute::Banded.label(),
            dimension: system.state.len(),
            policy: case.label,
            chunking: chunk_case.label,
            residual_us,
            jacobian_us,
            residual_calls: repetitions as u64,
            jacobian_calls: snapshot.calls,
            sequential_dispatches: snapshot.sequential_dispatches,
            parallel_dispatches: snapshot.parallel_dispatches,
            effective_tasks: snapshot.effective_task_count,
            evaluator_calls: snapshot.evaluator_calls,
            storage_writes: snapshot.storage_writes,
            max_residual_diff: expected_residual
                .map(|expected| vector_max_diff(expected, residual_values))
                .unwrap_or(0.0),
            max_jacobian_diff: expected_jacobian
                .map(|expected| matrix_max_diff(expected, &jacobian_values))
                .unwrap_or(0.0),
        }
    }

    fn assert_observation(observation: &Observation, repetitions: usize) {
        assert_eq!(observation.residual_calls, repetitions as u64);
        assert_eq!(observation.jacobian_calls, repetitions as u64);
        let expected_dispatches = if observation.route == MatrixRoute::Sparse.label() {
            repetitions as u64 * 2
        } else {
            // The direct Banded telemetry stream is Jacobian-specific. The
            // residual callback still has an external timing row, but it does
            // not publish a dispatch counter yet.
            repetitions as u64
        };
        assert_eq!(
            observation.sequential_dispatches + observation.parallel_dispatches,
            expected_dispatches,
            "each instrumented callback must publish one dispatch decision"
        );
        assert!(observation.max_residual_diff <= 1e-12);
        assert!(observation.max_jacobian_diff <= 1e-12);
    }

    fn run_matrix(report_name: &str, repetitions: usize) {
        assert!(repetitions > 0);
        let rayon_threads = warm_worker_pool();
        let build_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let mut report = format!(
            "status: passed\n\nbuild_profile: {build_profile}\nkind: callback-only policy matrix; timing is diagnostic and not a solver benchmark\nfrontend: AtomView\nbackends: faer Sparse, native Banded\npolicies: Sequential, Parallel, Auto\nbanded_chunking: Diagonal, EntryChunks\ncallback_repetitions: {repetitions}\nrayon_threads: {rayon_threads}\nworker_pool_warmed: yes; pool construction is outside timed callbacks\nallocation_counters: not instrumented in this gate\ncopy_accounting: output diagnostic conversions are excluded from callback timers\n\n"
        );
        report.push_str("dimension | route | policy | chunking | residual_us | jacobian_us | residual_calls | jacobian_calls | seq_dispatch | par_dispatch | effective_tasks | evaluator_calls | storage_writes | max_residual_diff | max_jacobian_diff\n");
        report.push_str("--- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:\n");

        for &dimension in DIMENSIONS {
            let system = prepared_system(dimension);
            let mut expected_residual = None;
            let mut expected_jacobian = None;
            for case in policy_cases() {
                let sparse = run_sparse(
                    &system,
                    case,
                    repetitions,
                    expected_residual.as_deref(),
                    expected_jacobian.as_ref(),
                );
                assert_observation(&sparse, repetitions);
                if expected_residual.is_none() {
                    let telemetry =
                        BvpLambdifyTelemetry::with_mode(BvpLambdifyTelemetryMode::Counters);
                    let binding = BvpParameterBindingHandle::new(None);
                    let residual = sparse_residual_with_binding_and_policy(
                        system.residuals.clone(),
                        system.variables.clone(),
                        binding.clone(),
                        0,
                        telemetry.clone(),
                        BvpLambdifyExecutionPolicy::Sequential,
                    );
                    let mut jacobian = sparse_jacobian_with_binding_and_policy(
                        system.jacobian.clone(),
                        dimension,
                        dimension,
                        system.variables.clone(),
                        binding,
                        0,
                        telemetry,
                        BvpLambdifyExecutionPolicy::Sequential,
                    );
                    let state = Col::from_fn(dimension, |index| system.state[index]);
                    expected_residual = Some(
                        residual
                            .try_call(0.0, &state)
                            .expect("reference residual should succeed")
                            .to_DVectorType()
                            .as_slice()
                            .to_vec(),
                    );
                    expected_jacobian = Some(
                        jacobian
                            .try_call(0.0, &state)
                            .expect("reference Jacobian should succeed")
                            .to_DMatrixType(),
                    );
                }
                let observations =
                    std::iter::once(sparse).chain(chunk_cases().into_iter().map(|chunk_case| {
                        let banded = run_banded(
                            &system,
                            case,
                            chunk_case,
                            repetitions,
                            expected_residual.as_deref(),
                            expected_jacobian.as_ref(),
                        );
                        assert_observation(&banded, repetitions);
                        assert!(banded.effective_tasks > 0);
                        assert_eq!(banded.evaluator_calls, banded.storage_writes);
                        banded
                    }));
                for observation in observations {
                    println!(
                        "[BVP Lambdify callback matrix] dimension={} route={} policy={} chunking={} residual_us={:.3} jacobian_us={:.3} dispatch={}/{} tasks={} evals={} writes={} parity={:.3e}/{:.3e}",
                        observation.dimension,
                        observation.route,
                        observation.policy,
                        observation.chunking,
                        observation.residual_us,
                        observation.jacobian_us,
                        observation.sequential_dispatches,
                        observation.parallel_dispatches,
                        observation.effective_tasks,
                        observation.evaluator_calls,
                        observation.storage_writes,
                        observation.max_residual_diff,
                        observation.max_jacobian_diff,
                    );
                    report.push_str(&format!(
                        "{} | {} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {:.3e} | {:.3e}\n",
                        observation.dimension,
                        observation.route,
                        observation.policy,
                        observation.chunking,
                        observation.residual_us,
                        observation.jacobian_us,
                        observation.residual_calls,
                        observation.jacobian_calls,
                        observation.sequential_dispatches,
                        observation.parallel_dispatches,
                        observation.effective_tasks,
                        observation.evaluator_calls,
                        observation.storage_writes,
                        observation.max_residual_diff,
                        observation.max_jacobian_diff,
                    ));
                }
            }
        }

        report.push_str("\ninterpretation: all policies, both Banded chunking strategies and both production callback boundaries preserve the same residual/Jacobian values. The release story is suitable for repeated callback-only timing, but it does not claim end-to-end solver break-even. Allocation and copy counters remain a separate telemetry slice.\n");
        write_test_report("BVP_Damp_Lambdify_Callback", report_name, &report)
            .expect("callback-only matrix report should be written");
    }

    #[test]
    fn atomview_callback_only_auto_policy_matrix_preserves_values_and_reports_dispatch() {
        run_matrix(
            "atomview_callback_only_auto_policy_matrix_preserves_values_and_reports_dispatch",
            CALLBACK_REPETITIONS,
        );
    }

    #[test]
    #[ignore = "release-only repeated callback timing story"]
    fn atomview_callback_only_auto_policy_matrix_release_story() {
        let repetitions = std::env::var("BVP_LAMBDIFY_CALLBACK_REPETITIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(7);
        run_matrix(
            "atomview_callback_only_auto_policy_matrix_release_story",
            repetitions,
        );
    }
}

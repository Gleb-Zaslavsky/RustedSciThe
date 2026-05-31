#![cfg(test)]

mod tests {
    use crate::numerical::BVP_Damp::BVP_traits::{BandedMatrixType, Vectors_type_casting};
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, BuildDampedSolverRequest,
        DampedSolverBuildRequest,
    };
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::somelinalg::banded::LinearSystemRef;
    use crate::somelinalg::banded::banded_assembly::BandedAssembly;
    use crate::somelinalg::banded::block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent;
    use crate::somelinalg::banded::block_tridiagonal_lu_consistent::IterativeRefinementReport;
    use crate::somelinalg::banded::lapack_style_banded::LapackStyleBandedLuFaithful;
    use crate::somelinalg::banded::linear_solver::build_solver_for_system;
    use crate::somelinalg::banded::node_major_layout::NodeMajorLayout;
    use crate::somelinalg::banded::solver_policy::{
        FallbackPolicy, LinearSolverConfig, LinearSolverPolicy,
    };
    use crate::somelinalg::banded::solver_traits::DirectLinearSolver;
    use crate::somelinalg::banded::storage::Banded;
    use crate::somelinalg::banded::superblock_layout::SuperBlockLayout;
    use crate::symbolic::View::bvp::{discretization_system_bvp_par_atom, eq_step_atom};
    use crate::symbolic::View::conversions::atom_to_expr;
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        register_generated_sparse_cdylib_backend, resolve_linked_sparse_backend,
        unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen::codegen_backend_selection::{
        BackendSelectionPolicy, SelectedBackendKind,
    };
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
        AotBuildProfile, AotBuildRequest,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{
        BvpBackendIntegrationError, BvpSparseSolverBundle, BvpSymbolicAssemblyBackend, Jacobian,
    };
    use faer::linalg::solvers::Solve;
    use faer::sparse::SparseColMat;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::fs;
    use std::hint::black_box;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::Duration;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const ISOLATED_TUNING_CHILD_INDEX_ENV: &str = "BVP_DAMP_ISOLATED_TUNING_CHILD_INDEX";
    const ISOLATED_TUNING_CHILD_STEPS_ENV: &str = "BVP_DAMP_ISOLATED_TUNING_CHILD_STEPS";
    const ISOLATED_TUNING_TIME_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_TIME]";
    const ISOLATED_TUNING_SOLUTION_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_SOLUTION]";
    const ISOLATED_TUNING_METRICS_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_METRICS]";
    const ISOLATED_TUNING_PID_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_PID]";
    const COLD_COOLDOWN_MS_ENV: &str = "BVP_AOT_COLD_COOLDOWN_MS";
    const COLD_CLEAN_ARTIFACTS_ENV: &str = "BVP_AOT_COLD_CLEAN_ARTIFACTS";

    #[derive(Clone, Debug)]
    struct IsolatedColdMetrics {
        total_timer_ms: f64,
        symbolic_ms: f64,
        linear_ms: f64,
        jac_ms: f64,
        fun_ms: f64,
        cb_residual_values_ms: f64,
        cb_jacobian_values_ms: f64,
        cb_jacobian_assembly_ms: f64,
        residual_actual_jobs: f64,
        sparse_jacobian_actual_jobs: f64,
        initial_symbolic_jacobian_ms: f64,
        post_build_rebind_ms: f64,
        aot_artifact_ms: f64,
        aot_materialize_ms: f64,
        aot_compile_link_ms: f64,
        aot_register_link_ms: f64,
    }

    #[derive(Clone, Debug)]
    struct IsolatedColdObservation {
        elapsed_ms: f64,
        solution: Vec<f64>,
        metrics: IsolatedColdMetrics,
    }

    #[derive(Clone, Debug)]
    struct RuntimeTuningSample {
        label: String,
        n_steps: usize,
        honest_user_e2e_ms: f64,
        honest_speedup_vs_seq: f64,
        bootstrap_ms: f64,
        solve_ms: f64,
        speedup_vs_seq: f64,
        max_diff_vs_ref: f64,
        total_timer_ms: f64,
        symbolic_ms: f64,
        linear_ms: f64,
        jac_ms: f64,
        fun_ms: f64,
        cb_residual_values_ms: f64,
        cb_jacobian_values_ms: f64,
        cb_jacobian_assembly_ms: f64,
        iterations: usize,
        linear_solves: usize,
        jac_rebuilds: usize,
        cold: IsolatedColdMetrics,
    }

    #[derive(Clone, Copy, Debug)]
    struct RuntimeTuningAggregate {
        mean: f64,
        stddev: f64,
        min: f64,
        max: f64,
    }

    #[derive(Debug)]
    struct RuntimeTuningSummary {
        label: String,
        n_steps: usize,
        runs: usize,
        honest_user_e2e_ms: RuntimeTuningAggregate,
        honest_speedup_vs_seq: RuntimeTuningAggregate,
        bootstrap_ms: RuntimeTuningAggregate,
        solve_ms: RuntimeTuningAggregate,
        speedup_vs_seq: RuntimeTuningAggregate,
        max_diff_vs_ref: RuntimeTuningAggregate,
        total_timer_ms: RuntimeTuningAggregate,
        symbolic_ms: RuntimeTuningAggregate,
        linear_ms: RuntimeTuningAggregate,
        jac_ms: RuntimeTuningAggregate,
        fun_ms: RuntimeTuningAggregate,
        cb_residual_values_ms: RuntimeTuningAggregate,
        cb_jacobian_values_ms: RuntimeTuningAggregate,
        cb_jacobian_assembly_ms: RuntimeTuningAggregate,
        iterations: RuntimeTuningAggregate,
        linear_solves: RuntimeTuningAggregate,
        jac_rebuilds: RuntimeTuningAggregate,
        cold_total_timer_ms: RuntimeTuningAggregate,
        cold_symbolic_ms: RuntimeTuningAggregate,
        cold_linear_ms: RuntimeTuningAggregate,
        cold_jac_ms: RuntimeTuningAggregate,
        cold_fun_ms: RuntimeTuningAggregate,
        cold_cb_residual_values_ms: RuntimeTuningAggregate,
        cold_cb_jacobian_values_ms: RuntimeTuningAggregate,
        cold_cb_jacobian_assembly_ms: RuntimeTuningAggregate,
        cold_residual_actual_jobs: RuntimeTuningAggregate,
        cold_sparse_jacobian_actual_jobs: RuntimeTuningAggregate,
        cold_initial_symbolic_jacobian_ms: RuntimeTuningAggregate,
        cold_post_build_rebind_ms: RuntimeTuningAggregate,
        cold_aot_artifact_ms: RuntimeTuningAggregate,
        cold_aot_materialize_ms: RuntimeTuningAggregate,
        cold_aot_compile_link_ms: RuntimeTuningAggregate,
        cold_aot_register_link_ms: RuntimeTuningAggregate,
    }

    #[derive(Clone, Copy, Debug)]
    enum RuntimeTuningToolchain {
        Rust,
        Gcc,
        Tcc,
        Zig,
    }

    impl RuntimeTuningToolchain {
        fn variants() -> [Self; 4] {
            [Self::Rust, Self::Gcc, Self::Tcc, Self::Zig]
        }

        fn label(self) -> &'static str {
            match self {
                Self::Rust => "rust",
                Self::Gcc => "gcc",
                Self::Tcc => "tcc",
                Self::Zig => "zig",
            }
        }

        fn apply_to(
            self,
            config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
        ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
            match self {
                Self::Rust => config.with_aot_codegen_backend(AotCodegenBackend::Rust),
                Self::Gcc => config
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("gcc"),
                Self::Tcc => config
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("tcc"),
                Self::Zig => config.with_aot_codegen_backend(AotCodegenBackend::Zig),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct RuntimeTuningChunkCase {
        label: &'static str,
        execution_policy: AotExecutionPolicy,
        chunking_policy: AotChunkingPolicy,
    }

    struct LinkedBackendGuard {
        problem_key: String,
    }

    #[derive(Clone, Copy, Debug)]
    enum BandedStorySolver {
        LegacyAuto,
        ConsistentSuperblock {
            nodes_per_superblock: usize,
            refinement_steps: usize,
        },
        LapackStyle {
            refinement_steps: usize,
        },
    }

    impl BandedStorySolver {
        fn variants() -> [Self; 4] {
            [
                Self::LegacyAuto,
                Self::LapackStyle {
                    refinement_steps: 0,
                },
                Self::LapackStyle {
                    refinement_steps: 1,
                },
                Self::ConsistentSuperblock {
                    nodes_per_superblock: 2,
                    refinement_steps: 1,
                },
            ]
        }

        fn label(self) -> String {
            match self {
                Self::LegacyAuto => "legacy_auto".to_string(),
                Self::LapackStyle { refinement_steps } => {
                    if refinement_steps == 0 {
                        "lapack_style_banded_lu".to_string()
                    } else {
                        format!("lapack_style_banded_lu+refine{refinement_steps}")
                    }
                }
                Self::ConsistentSuperblock {
                    nodes_per_superblock,
                    refinement_steps,
                } => format!(
                    "block_tridiagonal_lu_consistent[g={nodes_per_superblock},refine={refinement_steps}]"
                ),
            }
        }
    }

    #[derive(Debug)]
    struct BandedStorySolveMetrics {
        linear_solver: String,
        solution: Option<Vec<f64>>,
        report: Option<IterativeRefinementReport>,
        layout: String,
        status: String,
    }

    #[derive(Debug)]
    struct AotCrateBuildRow {
        backend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
        jacobian_prepare_ms: Option<f64>,
        atom_sparse_lookup_prepare_ms: Option<f64>,
        atom_sparse_jacobian_build_ms: Option<f64>,
        atom_finalize_codegen_plan_ms: Option<f64>,
        atom_sparse_nnz: Option<usize>,
        atom_residual_view_collect_ms: Option<f64>,
        atom_residual_lower_many_ms: Option<f64>,
        atom_residual_peephole_ms: Option<f64>,
        atom_residual_reuse_temps_ms: Option<f64>,
        atom_sparse_view_collect_ms: Option<f64>,
        atom_sparse_lower_many_ms: Option<f64>,
        atom_sparse_peephole_ms: Option<f64>,
        atom_sparse_reuse_temps_ms: Option<f64>,
        module_build_ms: Option<f64>,
        source_emit_ms: Option<f64>,
        materialize_ms: Option<f64>,
        build_ms: Option<f64>,
        source_kb: Option<f64>,
        module_blocks: Option<usize>,
        total_block_instructions: Option<usize>,
        total_block_temps: Option<usize>,
        max_block_instructions: Option<usize>,
        total_block_outputs: Option<usize>,
        status: String,
    }

    #[derive(Debug)]
    struct ChunkIrCompareRow {
        fn_name: String,
        outputs: usize,
        legacy_instr: usize,
        atom_instr: usize,
        legacy_temps: usize,
        atom_temps: usize,
    }

    impl Drop for LinkedBackendGuard {
        fn drop(&mut self) {
            let _ = unregister_linked_sparse_backend(&self.problem_key);
        }
    }

    fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(
            variable_count,
            n_steps,
            DVector::from_element(variable_count * n_steps, value).as_slice(),
        )
    }

    fn sparse_parallel_policy() -> AotExecutionPolicy {
        AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(8),
            max_sparse_jobs: Some(8),
            fallback_policy: ParallelFallbackPolicy::Never,
        })
    }

    fn unique_test_artifact_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-artifacts")
            .join("bvp-damp-tests3")
            .join(format!("{label}-{}-{nonce}", std::process::id()))
    }

    fn sparse_bundle_from_solver_request(
        solver: &mut NRBVP,
    ) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
        let request = solver.build_solver_request(None, None);
        let mut jacobian = Jacobian::new();
        jacobian.set_symbolic_assembly_backend(request.symbolic_assembly_backend);
        if let Some(param_names) = request.param_names.as_ref() {
            let param_refs: Vec<&str> = param_names.iter().map(|name| name.as_str()).collect();
            jacobian.set_params(Some(param_refs.as_slice()));
        } else {
            jacobian.set_params(None);
        }
        jacobian.set_param_values(request.param_values.clone());
        jacobian.set_banded_linear_solver_config(request.banded_linear_solver_config.clone());
        match (
            request.aot_chunking_policy.residual,
            request.aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian.try_generate_sparse_solver_bundle_with_backend_selection(
                request.eq_system,
                request.values,
                request.arg,
                None,
                request.t0,
                None,
                request.n_steps,
                request.h,
                request.mesh,
                request.border_conditions,
                request.bounds,
                request.rel_tolerance,
                request.scheme,
                request.method,
                request.bandwidth,
                request.backend_policy,
                request.resolver.as_ref(),
            ),
            (residual, sparse_jacobian) => jacobian
                .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                    request.eq_system,
                    request.values,
                    request.arg,
                    None,
                    request.t0,
                    None,
                    request.n_steps,
                    request.h,
                    request.mesh,
                    request.border_conditions,
                    request.bounds,
                    request.rel_tolerance,
                    request.scheme,
                    request.method,
                    request.bandwidth,
                    request.backend_policy,
                    request.resolver.as_ref(),
                    residual.unwrap_or(ResidualChunkingStrategy::Whole),
                    sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
                ),
        }
    }

    fn sparse_bundle_from_request_with_symbolic_backend(
        request: DampedSolverBuildRequest,
        symbolic_backend: BvpSymbolicAssemblyBackend,
    ) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
        let mut jacobian = Jacobian::new();
        jacobian.set_symbolic_assembly_backend(symbolic_backend);
        if let Some(param_names) = request.param_names.as_ref() {
            let param_refs: Vec<&str> = param_names.iter().map(|name| name.as_str()).collect();
            jacobian.set_params(Some(param_refs.as_slice()));
        } else {
            jacobian.set_params(None);
        }
        jacobian.set_param_values(request.param_values.clone());
        jacobian.set_banded_linear_solver_config(request.banded_linear_solver_config.clone());
        match (
            request.aot_chunking_policy.residual,
            request.aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian.try_generate_sparse_solver_bundle_with_backend_selection(
                request.eq_system,
                request.values,
                request.arg,
                None,
                request.t0,
                None,
                request.n_steps,
                request.h,
                request.mesh,
                request.border_conditions,
                request.bounds,
                request.rel_tolerance,
                request.scheme,
                request.method,
                request.bandwidth,
                request.backend_policy,
                request.resolver.as_ref(),
            ),
            (residual, sparse_jacobian) => jacobian
                .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                    request.eq_system,
                    request.values,
                    request.arg,
                    None,
                    request.t0,
                    None,
                    request.n_steps,
                    request.h,
                    request.mesh,
                    request.border_conditions,
                    request.bounds,
                    request.rel_tolerance,
                    request.scheme,
                    request.method,
                    request.bandwidth,
                    request.backend_policy,
                    request.resolver.as_ref(),
                    residual.unwrap_or(ResidualChunkingStrategy::Whole),
                    sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
                ),
        }
    }

    fn measure_sparse_bundle_build_with_symbolic_backend(
        request: DampedSolverBuildRequest,
        symbolic_backend: BvpSymbolicAssemblyBackend,
    ) -> Result<(BvpSparseSolverBundle, f64), BvpBackendIntegrationError> {
        let begin = Instant::now();
        let bundle = sparse_bundle_from_request_with_symbolic_backend(request, symbolic_backend)?;
        Ok((bundle, begin.elapsed().as_secs_f64() * 1_000.0))
    }

    fn measure_generated_crate_build_with_symbolic_backend(
        request: DampedSolverBuildRequest,
        symbolic_backend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
        label: &str,
    ) -> Result<AotCrateBuildRow, BvpBackendIntegrationError> {
        let bundle = sparse_bundle_from_request_with_symbolic_backend(request, symbolic_backend)?;
        let prepared = bundle.execution.selected().prepared_problem.clone();
        let crate_name = format!(
            "generated_bvp_compare_{}_{}",
            match symbolic_backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "expr",
                BvpSymbolicAssemblyBackend::AtomView => "atom",
            },
            prepared.problem_key()
        );
        let module_name = format!("generated_bvp_compare_module_{}", prepared.problem_key());

        let attempt = catch_unwind(AssertUnwindSafe(|| {
            let (crate_spec, breakdown) =
                prepared.generated_aot_crate_with_breakdown(crate_name, &module_name);

            let dir = unique_test_artifact_dir(label);
            fs::create_dir_all(&dir).expect("test artifact directory should be creatable");

            let materialize_begin = Instant::now();
            let build = AotBuildRequest::new(crate_spec, dir.as_path(), AotBuildProfile::Release)
                .materialize()
                .expect("AOT crate compare materialization should succeed");
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;

            let execute_begin = Instant::now();
            let executed = build
                .execute()
                .expect("AOT crate compare cargo build should execute");
            let build_ms = execute_begin.elapsed().as_secs_f64() * 1_000.0;
            let status = if executed.succeeded() {
                "ok".to_string()
            } else {
                format!(
                    "cargo-build-failed({})",
                    executed.status_code.unwrap_or_default()
                )
            };
            (
                breakdown.jacobian_prepare_ms,
                breakdown.atom_sparse_lookup_prepare_ms,
                breakdown.atom_sparse_jacobian_build_ms,
                breakdown.atom_finalize_codegen_plan_ms,
                breakdown.atom_sparse_nnz,
                breakdown.atom_residual_view_collect_ms,
                breakdown.atom_residual_lower_many_ms,
                breakdown.atom_residual_peephole_ms,
                breakdown.atom_residual_reuse_temps_ms,
                breakdown.atom_sparse_view_collect_ms,
                breakdown.atom_sparse_lower_many_ms,
                breakdown.atom_sparse_peephole_ms,
                breakdown.atom_sparse_reuse_temps_ms,
                breakdown.module_build_ms,
                breakdown.source_emit_ms,
                materialize_ms,
                build_ms,
                breakdown.source_kb,
                breakdown.module_blocks,
                breakdown.total_block_instructions,
                breakdown.total_block_temps,
                breakdown.max_block_instructions,
                breakdown.total_block_outputs,
                status,
            )
        }));

        let row = match attempt {
            Ok((
                jacobian_prepare_ms,
                atom_sparse_lookup_prepare_ms,
                atom_sparse_jacobian_build_ms,
                atom_finalize_codegen_plan_ms,
                atom_sparse_nnz,
                atom_residual_view_collect_ms,
                atom_residual_lower_many_ms,
                atom_residual_peephole_ms,
                atom_residual_reuse_temps_ms,
                atom_sparse_view_collect_ms,
                atom_sparse_lower_many_ms,
                atom_sparse_peephole_ms,
                atom_sparse_reuse_temps_ms,
                module_build_ms,
                source_emit_ms,
                materialize_ms,
                build_ms,
                source_kb,
                module_blocks,
                total_block_instructions,
                total_block_temps,
                max_block_instructions,
                total_block_outputs,
                status,
            )) => AotCrateBuildRow {
                backend: symbolic_backend,
                n_steps,
                jacobian_prepare_ms: Some(jacobian_prepare_ms),
                atom_sparse_lookup_prepare_ms: Some(atom_sparse_lookup_prepare_ms),
                atom_sparse_jacobian_build_ms: Some(atom_sparse_jacobian_build_ms),
                atom_finalize_codegen_plan_ms: Some(atom_finalize_codegen_plan_ms),
                atom_sparse_nnz: Some(atom_sparse_nnz),
                atom_residual_view_collect_ms: Some(atom_residual_view_collect_ms),
                atom_residual_lower_many_ms: Some(atom_residual_lower_many_ms),
                atom_residual_peephole_ms: Some(atom_residual_peephole_ms),
                atom_residual_reuse_temps_ms: Some(atom_residual_reuse_temps_ms),
                atom_sparse_view_collect_ms: Some(atom_sparse_view_collect_ms),
                atom_sparse_lower_many_ms: Some(atom_sparse_lower_many_ms),
                atom_sparse_peephole_ms: Some(atom_sparse_peephole_ms),
                atom_sparse_reuse_temps_ms: Some(atom_sparse_reuse_temps_ms),
                module_build_ms: Some(module_build_ms),
                source_emit_ms: Some(source_emit_ms),
                materialize_ms: Some(materialize_ms),
                build_ms: Some(build_ms),
                source_kb: Some(source_kb),
                module_blocks: Some(module_blocks),
                total_block_instructions: Some(total_block_instructions),
                total_block_temps: Some(total_block_temps),
                max_block_instructions: Some(max_block_instructions),
                total_block_outputs: Some(total_block_outputs),
                status,
            },
            Err(panic_payload) => {
                let status = if let Some(message) = panic_payload.downcast_ref::<String>() {
                    format!("panic({message})")
                } else if let Some(message) = panic_payload.downcast_ref::<&str>() {
                    format!("panic({message})")
                } else {
                    "panic(non-string payload)".to_string()
                };
                AotCrateBuildRow {
                    backend: symbolic_backend,
                    n_steps,
                    jacobian_prepare_ms: None,
                    atom_sparse_lookup_prepare_ms: None,
                    atom_sparse_jacobian_build_ms: None,
                    atom_finalize_codegen_plan_ms: None,
                    atom_sparse_nnz: None,
                    atom_residual_view_collect_ms: None,
                    atom_residual_lower_many_ms: None,
                    atom_residual_peephole_ms: None,
                    atom_residual_reuse_temps_ms: None,
                    atom_sparse_view_collect_ms: None,
                    atom_sparse_lower_many_ms: None,
                    atom_sparse_peephole_ms: None,
                    atom_sparse_reuse_temps_ms: None,
                    module_build_ms: None,
                    source_emit_ms: None,
                    materialize_ms: None,
                    build_ms: None,
                    source_kb: None,
                    module_blocks: None,
                    total_block_instructions: None,
                    total_block_temps: None,
                    max_block_instructions: None,
                    total_block_outputs: None,
                    status,
                }
            }
        };

        Ok(row)
    }

    fn measure_codegen_module_with_symbolic_backend(
        request: DampedSolverBuildRequest,
        symbolic_backend: BvpSymbolicAssemblyBackend,
    ) -> Result<
        (
            crate::symbolic::codegen::CodegenIR::CodegenModule,
            crate::symbolic::symbolic_functions_BVP::BvpGeneratedAotCrateBreakdown,
        ),
        BvpBackendIntegrationError,
    > {
        let bundle = sparse_bundle_from_request_with_symbolic_backend(request, symbolic_backend)?;
        let prepared = bundle.execution.selected().prepared_problem.clone();
        Ok(prepared.codegen_module_with_breakdown("generated_bvp_chunk_compare"))
    }

    fn measure_symbolic_generation_breakdown_with_symbolic_backend(
        request: DampedSolverBuildRequest,
        symbolic_backend: BvpSymbolicAssemblyBackend,
    ) -> Result<HashMap<String, f64>, BvpBackendIntegrationError> {
        let mut jacobian = Jacobian::new();
        jacobian.set_symbolic_assembly_backend(symbolic_backend);
        let _execution = match (
            request.aot_chunking_policy.residual,
            request.aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian.generate_BVP_with_backend_selection(
                request.eq_system,
                request.values,
                request.arg,
                None,
                request.t0,
                None,
                request.n_steps,
                request.h,
                request.mesh,
                request.border_conditions,
                request.bounds,
                request.rel_tolerance,
                request.scheme,
                request.method,
                request.bandwidth,
                request.backend_policy,
                request.resolver.as_ref(),
            ),
            (residual, sparse_jacobian) => jacobian
                .generate_BVP_with_backend_selection_and_chunking(
                    request.eq_system,
                    request.values,
                    request.arg,
                    None,
                    request.t0,
                    None,
                    request.n_steps,
                    request.h,
                    request.mesh,
                    request.border_conditions,
                    request.bounds,
                    request.rel_tolerance,
                    request.scheme,
                    request.method,
                    request.bandwidth,
                    request.backend_policy,
                    request.resolver.as_ref(),
                    residual.unwrap_or(ResidualChunkingStrategy::Whole),
                    sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
                ),
        };
        Ok(jacobian
            .last_generate_timer_snapshot()
            .cloned()
            .expect("symbolic generation should leave a timing snapshot"))
    }

    fn compare_sparse_bundles_numerically(
        lhs: &mut BvpSparseSolverBundle,
        rhs: &mut BvpSparseSolverBundle,
        args: &DVector<f64>,
        label: &str,
    ) -> (f64, f64) {
        let typed = &*Vectors_type_casting(args, "Sparse".to_string());
        let lhs_residual = lhs
            .residual_call(1.0, typed)
            .expect("lhs sparse bundle should expose residual callback")
            .to_DVectorType();
        let rhs_residual = rhs
            .residual_call(1.0, typed)
            .expect("rhs sparse bundle should expose residual callback")
            .to_DVectorType();
        assert_eq!(
            lhs_residual.len(),
            rhs_residual.len(),
            "{label}: residual lengths should match"
        );
        let mut residual_max_diff: f64 = 0.0;
        for index in 0..lhs_residual.len() {
            let lhs_value = lhs_residual[index];
            let rhs_value = rhs_residual[index];
            residual_max_diff = residual_max_diff.max((lhs_value - rhs_value).abs());
        }

        let lhs_jacobian = lhs
            .jacobian_call(1.0, typed)
            .expect("lhs sparse bundle should expose jacobian callback")
            .to_DMatrixType();
        let rhs_jacobian = rhs
            .jacobian_call(1.0, typed)
            .expect("rhs sparse bundle should expose jacobian callback")
            .to_DMatrixType();
        assert_eq!(
            lhs_jacobian.shape(),
            rhs_jacobian.shape(),
            "{label}: jacobian shapes should match"
        );
        let mut jacobian_max_diff: f64 = 0.0;
        for row in 0..lhs_jacobian.nrows() {
            for col in 0..lhs_jacobian.ncols() {
                let lhs_value = lhs_jacobian[(row, col)];
                let rhs_value = rhs_jacobian[(row, col)];
                jacobian_max_diff = jacobian_max_diff.max((lhs_value - rhs_value).abs());
            }
        }
        println!(
            "[BVP symbolic assembly diff] label={label}, residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}"
        );
        (residual_max_diff, jacobian_max_diff)
    }

    fn bootstrap_callable_aot_backend(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<LinkedBackendGuard, BvpBackendIntegrationError> {
        let build_begin = Instant::now();
        solver.try_eq_generate(None, None)?;
        println!(
            "[AOT bootstrap] {label}: build/materialize stage took {:?}",
            build_begin.elapsed()
        );

        let bundle = sparse_bundle_from_solver_request(solver)?;
        assert_eq!(
            bundle.effective_backend(),
            SelectedBackendKind::AotCompiled,
            "{label}: sparse bundle should resolve to compiled AOT after bootstrap"
        );
        assert!(
            bundle.resolved_aot_artifact().is_some(),
            "{label}: compiled AOT artifact metadata should be present"
        );

        let resolved = bundle
            .resolved_aot_artifact()
            .expect("compiled AOT artifact metadata should be present for runtime linking");
        register_generated_sparse_cdylib_backend(&resolved.registered).map_err(|_| {
            BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                problem_key: resolved.registered.problem_key.clone(),
            }
        })?;
        let linked_guard = LinkedBackendGuard {
            problem_key: resolved.registered.problem_key.clone(),
        };
        let updated_config = solver
            .generated_backend_config()
            .clone()
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver.set_generated_backend_config(updated_config);
        solver.try_eq_generate(None, None)?;
        Ok(linked_guard)
    }

    fn max_abs_error_against_exact<F>(solver: &NRBVP, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result = solver
            .get_result()
            .expect("AOT acceptance check requires a computed solution matrix");
        let y = result.column(0);
        solver
            .x_mesh
            .iter()
            .zip(y.iter())
            .map(|(&x, &y_num)| (y_num - exact(x)).abs())
            .fold(0.0, f64::max)
    }

    fn l2_error_against_exact<F>(solver: &NRBVP, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result = solver
            .get_result()
            .expect("AOT acceptance check requires a computed solution matrix");
        let y = result.column(0);
        let mse = solver
            .x_mesh
            .iter()
            .zip(y.iter())
            .map(|(&x, &y_num)| {
                let diff = y_num - exact(x);
                diff * diff
            })
            .sum::<f64>()
            / solver.x_mesh.len() as f64;
        mse.sqrt()
    }

    fn solve_with_aot_and_report(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<LinkedBackendGuard, BvpBackendIntegrationError> {
        let guard = bootstrap_callable_aot_backend(solver, label)?;
        let solve_begin = Instant::now();
        solver.try_solve()?;
        println!(
            "[AOT solve] {label}: solve took {:?}",
            solve_begin.elapsed()
        );
        Ok(guard)
    }

    fn solve_with_aot_and_measure(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<(LinkedBackendGuard, f64, f64), BvpBackendIntegrationError> {
        let build_begin = Instant::now();
        let guard = bootstrap_callable_aot_backend(solver, label)?;
        let bootstrap_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
        let solve_begin = Instant::now();
        solver.try_solve()?;
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        println!("[AOT measure] {label}: bootstrap={bootstrap_ms:.3} ms, solve={solve_ms:.3} ms");
        Ok((guard, bootstrap_ms, solve_ms))
    }

    fn solve_honest_user_e2e_and_measure(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<f64, BvpBackendIntegrationError> {
        let begin = Instant::now();
        solver.try_solve()?;
        let elapsed_ms = begin.elapsed().as_secs_f64() * 1_000.0;
        println!("[BVP honest e2e] {label}: full solve took {elapsed_ms:.3} ms");
        Ok(elapsed_ms)
    }

    fn solve_with_lambdify_and_measure(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<(f64, f64), BvpBackendIntegrationError> {
        let prepare_begin = Instant::now();
        solver.try_eq_generate(None, None)?;
        let prepare_ms = prepare_begin.elapsed().as_secs_f64() * 1_000.0;
        let solve_begin = Instant::now();
        solver.try_solve()?;
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        println!("[Lambdify measure] {label}: prepare={prepare_ms:.3} ms, solve={solve_ms:.3} ms");
        Ok((prepare_ms, solve_ms))
    }

    fn is_aot_environment_issue(err: &BvpBackendIntegrationError) -> bool {
        match err {
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
            | BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. }
            | BvpBackendIntegrationError::AutomaticAotBuildRequested { .. }
            | BvpBackendIntegrationError::AutomaticAotRebuildRequested { .. } => true,
            BvpBackendIntegrationError::AutomaticAotBuildFailed { message, .. } => {
                let msg = message.to_ascii_lowercase();
                msg.contains("permission denied")
                    || msg.contains("not found")
                    || msg.contains("failed to spawn")
                    || msg.contains("status=some(1)")
                    || msg.contains("toolchain")
            }
            BvpBackendIntegrationError::PipelinePanicked(message) => {
                let msg = message.to_ascii_lowercase();
                msg.contains("generatedbackendfailure")
                    || msg.contains("permission denied")
                    || msg.contains("failed to spawn")
                    || msg.contains("status=some(1)")
                    || msg.contains("toolchain")
            }
            _ => false,
        }
    }

    fn measure_sparse_runtime_callback_throughput(
        bundle: &mut BvpSparseSolverBundle,
        typed: &dyn crate::numerical::BVP_Damp::BVP_traits::VectorType,
        iters: usize,
    ) -> (f64, f64) {
        assert!(
            bundle.is_runtime_callable(),
            "callback throughput benchmark requires runtime-callable sparse bundle"
        );

        let residual_begin = Instant::now();
        for _ in 0..iters {
            let residual = bundle
                .residual_call(1.0, typed)
                .expect("runtime-callable sparse bundle must expose residual callback");
            black_box(residual);
        }
        let residual_ms = residual_begin.elapsed().as_secs_f64() * 1_000.0;

        let jacobian_begin = Instant::now();
        for _ in 0..iters {
            let jacobian = bundle
                .jacobian_call(1.0, typed)
                .expect("runtime-callable sparse bundle must expose jacobian callback");
            black_box(jacobian);
        }
        let jacobian_ms = jacobian_begin.elapsed().as_secs_f64() * 1_000.0;

        (residual_ms, jacobian_ms)
    }

    fn max_abs_vector_diff(lhs: &DVector<f64>, rhs: &DVector<f64>) -> f64 {
        assert_eq!(lhs.len(), rhs.len(), "vector lengths should match");
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs_matrix_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        assert_eq!(lhs.shape(), rhs.shape(), "matrix shapes should match");
        let mut max_diff: f64 = 0.0;
        for row in 0..lhs.nrows() {
            for col in 0..lhs.ncols() {
                max_diff = max_diff.max((lhs[(row, col)] - rhs[(row, col)]).abs());
            }
        }
        max_diff
    }

    fn infer_bandwidth_from_dense_matrix(matrix: &DMatrix<f64>) -> (usize, usize) {
        let mut kl = 0usize;
        let mut ku = 0usize;
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                let value = matrix[(row, col)];
                if value == 0.0 {
                    continue;
                }
                if row >= col {
                    kl = kl.max(row - col);
                } else {
                    ku = ku.max(col - row);
                }
            }
        }
        (kl, ku)
    }

    fn banded_assembly_from_dense_matrix(dense: &DMatrix<f64>) -> BandedAssembly {
        let (kl, ku) = infer_bandwidth_from_dense_matrix(dense);
        let mut assembly = BandedAssembly::zeros(dense.nrows(), kl, ku)
            .expect("dense callback matrix should define a valid banded allocation");
        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let value = dense[(row, col)];
                if value == 0.0 {
                    continue;
                }
                assembly
                    .set(row, col, value)
                    .expect("dense callback entry should fit inside inferred band");
            }
        }
        assembly
    }

    fn dense_from_compact_banded(a: &Banded<f64>) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(a.n(), a.n());
        for col in 0..a.n() {
            let row_lo = col.saturating_sub(a.ku());
            let row_hi = (col + a.kl() + 1).min(a.n());
            for row in row_lo..row_hi {
                dense[(row, col)] = a[(row, col)];
            }
        }
        dense
    }

    fn dense_from_vecvec(matrix: &[Vec<f64>]) -> DMatrix<f64> {
        let n = matrix.len();
        let mut dense = DMatrix::zeros(n, n);
        for row in 0..n {
            for col in 0..n {
                dense[(row, col)] = matrix[row][col];
            }
        }
        dense
    }

    fn dense_to_sparse_col_mat(dense: &DMatrix<f64>) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let value = dense[(row, col)];
                if value != 0.0 {
                    triplets.push(faer::sparse::Triplet::new(row, col, value));
                }
            }
        }
        SparseColMat::<usize, f64>::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("dense matrix should convert to sparse triplets")
    }

    fn relative_dense_residual(a: &DMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let mut rmax = 0.0_f64;
        let mut bmax = 0.0_f64;
        for row in 0..a.nrows() {
            let mut ax = 0.0;
            for col in 0..a.ncols() {
                ax += a[(row, col)] * x[col];
            }
            rmax = rmax.max((ax - b[row]).abs());
            bmax = bmax.max(b[row].abs());
        }
        rmax / bmax.max(1.0)
    }

    fn vector_linf_norm(x: &[f64]) -> f64 {
        x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    fn relative_x_diff(x: &[f64], x_ref: &[f64]) -> f64 {
        let abs = x
            .iter()
            .zip(x_ref.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        abs / vector_linf_norm(x_ref).max(1.0)
    }

    fn fmt_metric(value: f64) -> String {
        if value.is_finite() {
            format!("{value:.3e}")
        } else {
            "-".to_string()
        }
    }

    fn stats_count(stats: &DampedBvpStatistics, key: &str) -> usize {
        stats.counters.get(key).copied().unwrap_or(0)
    }

    fn stats_timer(stats: &DampedBvpStatistics, prefix: &str) -> String {
        stats
            .timers
            .iter()
            .find(|(key, _)| key.starts_with(prefix))
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| "-".to_string())
    }

    fn stats_timer_ms(stats: &DampedBvpStatistics, prefix: &str) -> f64 {
        stats
            .timers
            .iter()
            .find(|(key, _)| key.starts_with(prefix))
            .and_then(|(key, value)| timer_value_to_ms(key, value))
            .unwrap_or(f64::NAN)
    }

    fn stats_diagnostic_ms(stats: &DampedBvpStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    }

    fn stats_diagnostic_usize(stats: &DampedBvpStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value as f64)
            .unwrap_or(f64::NAN)
    }

    fn isolated_cold_metrics_from_solver(solver: &NRBVP) -> IsolatedColdMetrics {
        let stats = solver.get_statistics();
        IsolatedColdMetrics {
            total_timer_ms: stats_timer_ms(&stats, "time elapsed"),
            symbolic_ms: stats_timer_ms(&stats, "Symbolic Operations"),
            linear_ms: stats_timer_ms(&stats, "Linear System"),
            jac_ms: stats_timer_ms(&stats, "Jacobian"),
            fun_ms: stats_timer_ms(&stats, "Function"),
            cb_residual_values_ms: stats_timer_ms(&stats, "Callback Residual Values"),
            cb_jacobian_values_ms: stats_timer_ms(&stats, "Callback Jacobian Values"),
            cb_jacobian_assembly_ms: stats_timer_ms(&stats, "Callback Jacobian Matrix Assembly"),
            residual_actual_jobs: stats_diagnostic_usize(
                &stats,
                "aot.runtime.residual.actual_jobs",
            ),
            sparse_jacobian_actual_jobs: stats_diagnostic_usize(
                &stats,
                "aot.runtime.sparse_jacobian.actual_jobs",
            ),
            initial_symbolic_jacobian_ms: stats_diagnostic_ms(
                &stats,
                "generated.handoff.initial.symbolic_jacobian_time_ms",
            ),
            post_build_rebind_ms: stats_diagnostic_ms(
                &stats,
                "generated.handoff.post_build_rebind_wall_ms",
            ),
            aot_artifact_ms: stats_diagnostic_ms(&stats, "generated.aot.artifact_wall_ms"),
            aot_materialize_ms: stats_diagnostic_ms(&stats, "generated.aot.materialize_ms"),
            aot_compile_link_ms: stats_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
            aot_register_link_ms: stats_diagnostic_ms(&stats, "generated.aot.register_link_ms"),
        }
    }

    fn timer_value_to_ms(key: &str, value: &str) -> Option<f64> {
        let duration = value
            .split(',')
            .next_back()
            .and_then(|part| part.trim().parse::<f64>().ok())?;
        let multiplier = if key.contains("ms") {
            1.0
        } else if key.contains("min") {
            60_000.0
        } else if key.contains('h') {
            3_600_000.0
        } else {
            1_000.0
        };
        Some(duration * multiplier)
    }

    fn runtime_tuning_aggregate(values: impl IntoIterator<Item = f64>) -> RuntimeTuningAggregate {
        let values = values
            .into_iter()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if values.is_empty() {
            return RuntimeTuningAggregate {
                mean: f64::NAN,
                stddev: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }

        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / count;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        RuntimeTuningAggregate {
            mean,
            stddev: variance.sqrt(),
            min,
            max,
        }
    }

    fn fmt_tuning_agg(value: RuntimeTuningAggregate) -> String {
        if value.mean.is_finite() {
            format!(
                "{:.3} +/- {:.3} [{:.3}, {:.3}]",
                value.mean, value.stddev, value.min, value.max
            )
        } else {
            "-".to_string()
        }
    }

    fn fmt_tuning_short(value: RuntimeTuningAggregate) -> String {
        if value.mean.is_finite() {
            format!("{:.3} +/- {:.3}", value.mean, value.stddev)
        } else {
            "-".to_string()
        }
    }

    fn fmt_tuning_exp(value: RuntimeTuningAggregate) -> String {
        if value.mean.is_finite() {
            format!("{:.3e} +/- {:.1e}", value.mean, value.stddev)
        } else {
            "-".to_string()
        }
    }

    fn solve_sparse_lu_for_rhs(matrix: &SparseColMat<usize, f64>, rhs: &[f64]) -> Vec<f64> {
        let lu = matrix
            .sp_lu()
            .expect("sparse LU factorization should succeed");
        let rhs_col = faer::Col::from_fn(rhs.len(), |index| rhs[index]);
        let solution = lu.solve(&rhs_col);
        solution.iter().copied().collect()
    }

    fn solve_banded_story_for_rhs(
        assembly: &BandedAssembly,
        n_steps: usize,
        rhs: &[f64],
        solver_choice: BandedStorySolver,
    ) -> BandedStorySolveMetrics {
        match solver_choice {
            BandedStorySolver::LegacyAuto => {
                let layout =
                    NodeMajorLayout::new(n_steps, 6).expect("combustion node-major layout valid");
                let layout_label = format!("{}x{}", layout.n_blocks(), layout.block_size());
                match build_solver_for_system(
                    LinearSystemRef::NodeMajorAssembly { assembly, layout },
                    LinearSolverConfig {
                        policy: LinearSolverPolicy::Auto,
                        fallback: FallbackPolicy::ToFaerSparse,
                        iterative_refinement_steps: 0,
                    },
                ) {
                    Ok(solver) => {
                        let linear_solver = solver.backend_name().to_string();
                        let mut rhs_owned = rhs.to_vec();
                        match solver.solve_in_place(rhs_owned.as_mut_slice()) {
                            Ok(()) => BandedStorySolveMetrics {
                                linear_solver,
                                solution: Some(rhs_owned),
                                report: None,
                                layout: layout_label,
                                status: "ok".to_string(),
                            },
                            Err(err) => BandedStorySolveMetrics {
                                linear_solver,
                                solution: None,
                                report: None,
                                layout: layout_label,
                                status: format!("solve_failed({err:?})"),
                            },
                        }
                    }
                    Err(err) => BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: None,
                        report: None,
                        layout: layout_label,
                        status: format!("factorization_failed({err:?})"),
                    },
                }
            }
            BandedStorySolver::ConsistentSuperblock {
                nodes_per_superblock,
                refinement_steps,
            } => {
                let layout = SuperBlockLayout::new(n_steps, 6, nodes_per_superblock)
                    .expect("combustion superblock layout should be valid");
                assert!(
                    layout.is_evenly_divisible(),
                    "combustion superblock diagnostic requires an even node grouping"
                );
                let block = assembly.finalize_superblock_tridiagonal(&layout).expect(
                    "combustion banded assembly should finalize into a uniform superblock chain",
                );
                let mut solver =
                    match BlockTridiagonalLuConsistent::new(block.n_blocks(), block.block_size()) {
                        Ok(solver) => solver,
                        Err(err) => {
                            return BandedStorySolveMetrics {
                                linear_solver: solver_choice.label(),
                                solution: None,
                                report: None,
                                layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                                status: format!("factorization_failed({err:?})"),
                            };
                        }
                    };
                if let Err(err) = solver.factor_from(&block) {
                    return BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: None,
                        report: None,
                        layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                        status: format!("factorization_failed({err:?})"),
                    };
                }
                let mut rhs_owned = rhs.to_vec();
                match solver.solve_in_place_with_iterative_refinement_report(
                    &block,
                    rhs_owned.as_mut_slice(),
                    refinement_steps,
                ) {
                    Ok(report) => BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: Some(rhs_owned),
                        report: Some(report),
                        layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                        status: "ok".to_string(),
                    },
                    Err(err) => BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: None,
                        report: None,
                        layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                        status: format!("solve_failed({err:?})"),
                    },
                }
            }
            BandedStorySolver::LapackStyle { refinement_steps } => {
                match build_solver_for_system(
                    LinearSystemRef::BandedAssembly(assembly),
                    LinearSolverConfig {
                        policy: LinearSolverPolicy::ForceBanded,
                        fallback: FallbackPolicy::Never,
                        iterative_refinement_steps: refinement_steps,
                    },
                ) {
                    Ok(solver) => {
                        let linear_solver = solver.backend_name().to_string();
                        let mut rhs_owned = rhs.to_vec();
                        match solver.solve_in_place(rhs_owned.as_mut_slice()) {
                            Ok(()) => BandedStorySolveMetrics {
                                linear_solver,
                                solution: Some(rhs_owned),
                                report: None,
                                layout: format!(
                                    "n{} kl{} ku{}",
                                    assembly.n(),
                                    assembly.kl(),
                                    assembly.ku()
                                ),
                                status: "ok".to_string(),
                            },
                            Err(err) => BandedStorySolveMetrics {
                                linear_solver,
                                solution: None,
                                report: None,
                                layout: format!(
                                    "n{} kl{} ku{}",
                                    assembly.n(),
                                    assembly.kl(),
                                    assembly.ku()
                                ),
                                status: format!("solve_failed({err:?})"),
                            },
                        }
                    }
                    Err(err) => BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: None,
                        report: None,
                        layout: format!(
                            "n{} kl{} ku{}",
                            assembly.n(),
                            assembly.kl(),
                            assembly.ku()
                        ),
                        status: format!("factorization_failed({err:?})"),
                    },
                }
            }
        }
    }

    fn flattened_initial_guess_state(
        solver: &NRBVP,
        expected_len: usize,
        label: &str,
    ) -> DVector<f64> {
        let dense = DVector::from_vec(solver.initial_guess.iter().cloned().collect());
        assert_eq!(
            dense.len(),
            expected_len,
            "{label}: flattened initial_guess length must match bundle variable count"
        );
        dense
    }

    fn runtime_vector_backend_for_matrix_backend(matrix_backend: MatrixBackend) -> &'static str {
        match matrix_backend {
            MatrixBackend::Banded => "Dense",
            _ => "Sparse",
        }
    }

    fn eval_solver_callback_state(
        solver: &mut NRBVP,
        args: &DVector<f64>,
        matrix_backend: MatrixBackend,
        label: &str,
    ) -> (
        DVector<f64>,
        DMatrix<f64>,
        Option<SparseColMat<usize, f64>>,
        Option<BandedMatrixType>,
    ) {
        let backend = runtime_vector_backend_for_matrix_backend(matrix_backend);
        let typed = &*Vectors_type_casting(args, backend.to_string());
        let residual = solver.fun.call(1.0, typed).to_DVectorType();
        let jacobian = solver
            .jac
            .as_mut()
            .unwrap_or_else(|| panic!("{label}: jacobian callback should be available"))
            .call(1.0, typed);
        let dense = jacobian.to_DMatrixType();

        match matrix_backend {
            MatrixBackend::Banded => {
                let banded = jacobian
                    .as_any()
                    .downcast_ref::<BandedMatrixType>()
                    .unwrap_or_else(|| {
                        panic!("{label}: banded callback should produce BandedMatrixType")
                    })
                    .clone();
                (residual, dense, None, Some(banded))
            }
            _ => {
                let sparse = jacobian
                    .as_any()
                    .downcast_ref::<SparseColMat<usize, f64>>()
                    .unwrap_or_else(|| {
                        panic!("{label}: sparse callback should produce SparseColMat")
                    })
                    .to_owned();
                (residual, dense, Some(sparse), None)
            }
        }
    }

    fn runtime_tuning_sample_from_solver(
        label: impl Into<String>,
        n_steps: usize,
        honest_user_e2e_ms: f64,
        honest_speedup_vs_seq: f64,
        bootstrap_ms: f64,
        solve_ms: f64,
        speedup_vs_seq: f64,
        max_diff_vs_ref: f64,
        solver: &NRBVP,
        cold: IsolatedColdMetrics,
    ) -> RuntimeTuningSample {
        let statistics = solver.get_statistics();
        RuntimeTuningSample {
            label: label.into(),
            n_steps,
            honest_user_e2e_ms,
            honest_speedup_vs_seq,
            bootstrap_ms,
            solve_ms,
            speedup_vs_seq,
            max_diff_vs_ref,
            total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
            symbolic_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
            linear_ms: stats_timer_ms(&statistics, "Linear System"),
            jac_ms: stats_timer_ms(&statistics, "Jacobian"),
            fun_ms: stats_timer_ms(&statistics, "Function"),
            cb_residual_values_ms: stats_timer_ms(&statistics, "Callback Residual Values"),
            cb_jacobian_values_ms: stats_timer_ms(&statistics, "Callback Jacobian Values"),
            cb_jacobian_assembly_ms: stats_timer_ms(
                &statistics,
                "Callback Jacobian Matrix Assembly",
            ),
            iterations: stats_count(&statistics, "number of iterations"),
            linear_solves: stats_count(&statistics, "number of solving linear systems"),
            jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
            cold,
        }
    }

    fn summarize_runtime_tuning_samples(
        labels: &[String],
        samples: &[RuntimeTuningSample],
    ) -> Vec<RuntimeTuningSummary> {
        labels
            .iter()
            .map(|label| {
                let rows = samples
                    .iter()
                    .filter(|sample| sample.label.as_str() == label.as_str())
                    .collect::<Vec<_>>();
                RuntimeTuningSummary {
                    label: label.clone(),
                    n_steps: rows.first().map(|sample| sample.n_steps).unwrap_or(0),
                    runs: rows.len(),
                    honest_user_e2e_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.honest_user_e2e_ms),
                    ),
                    honest_speedup_vs_seq: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.honest_speedup_vs_seq),
                    ),
                    bootstrap_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.bootstrap_ms),
                    ),
                    solve_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.solve_ms)),
                    speedup_vs_seq: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.speedup_vs_seq),
                    ),
                    max_diff_vs_ref: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.max_diff_vs_ref),
                    ),
                    total_timer_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.total_timer_ms),
                    ),
                    symbolic_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.symbolic_ms),
                    ),
                    linear_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.linear_ms)),
                    jac_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.jac_ms)),
                    fun_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.fun_ms)),
                    cb_residual_values_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cb_residual_values_ms),
                    ),
                    cb_jacobian_values_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cb_jacobian_values_ms),
                    ),
                    cb_jacobian_assembly_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cb_jacobian_assembly_ms),
                    ),
                    iterations: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.iterations as f64),
                    ),
                    linear_solves: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.linear_solves as f64),
                    ),
                    jac_rebuilds: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.jac_rebuilds as f64),
                    ),
                    cold_total_timer_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.total_timer_ms),
                    ),
                    cold_symbolic_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.symbolic_ms),
                    ),
                    cold_linear_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.linear_ms),
                    ),
                    cold_jac_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.jac_ms),
                    ),
                    cold_fun_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.fun_ms),
                    ),
                    cold_cb_residual_values_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.cb_residual_values_ms),
                    ),
                    cold_cb_jacobian_values_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.cb_jacobian_values_ms),
                    ),
                    cold_cb_jacobian_assembly_ms: runtime_tuning_aggregate(
                        rows.iter()
                            .map(|sample| sample.cold.cb_jacobian_assembly_ms),
                    ),
                    cold_residual_actual_jobs: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.residual_actual_jobs),
                    ),
                    cold_sparse_jacobian_actual_jobs: runtime_tuning_aggregate(
                        rows.iter()
                            .map(|sample| sample.cold.sparse_jacobian_actual_jobs),
                    ),
                    cold_initial_symbolic_jacobian_ms: runtime_tuning_aggregate(
                        rows.iter()
                            .map(|sample| sample.cold.initial_symbolic_jacobian_ms),
                    ),
                    cold_post_build_rebind_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.post_build_rebind_ms),
                    ),
                    cold_aot_artifact_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.aot_artifact_ms),
                    ),
                    cold_aot_materialize_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.aot_materialize_ms),
                    ),
                    cold_aot_compile_link_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.aot_compile_link_ms),
                    ),
                    cold_aot_register_link_ms: runtime_tuning_aggregate(
                        rows.iter().map(|sample| sample.cold.aot_register_link_ms),
                    ),
                }
            })
            .collect()
    }

    fn print_runtime_tuning_summary_table(
        scenario_label: &str,
        n_steps: usize,
        repetitions: usize,
        rows: &[RuntimeTuningSummary],
    ) {
        println!();
        println!(
            "[AOT combustion tuning map] scenario={scenario_label}, n_steps={n_steps}, repetitions={repetitions}"
        );
        println!(
            "[AOT combustion tuning map] runner=manual_prelinked_runtime_tuning; solve_ms is the runtime Newton solve after callbacks are prepared."
        );
        println!(
            "[AOT combustion tuning map] manual_bootstrap_ms is diagnostic setup for this test only; do not read it as normal solver end-to-end time."
        );
        println!(
            "[AOT combustion tuning map] honest_user_e2e_ms is measured in a fresh child process around solver.try_solve(); AOT rows force RebuildAlways Release so codegen/build/link/Newton are included without registry/DLL carry-over."
        );
        println!("[AOT combustion tuning map] correctness summary");
        println!(
            "{:<30} | {:>7} | {:>4} | {:<18} | {:<18} | {:<18} | {:<18}",
            "config",
            "n_steps",
            "runs",
            "honest_user_e2e_ms",
            "solve_ms",
            "speedup_vs_seq",
            "max_diff_vs_ref"
        );
        println!("{}", "-".repeat(143));
        for row in rows {
            println!(
                "{:<30} | {:>7} | {:>4} | {:<18} | {:<18} | {:<18} | {:<18}",
                row.label,
                row.n_steps,
                row.runs,
                fmt_tuning_short(row.honest_user_e2e_ms),
                fmt_tuning_short(row.solve_ms),
                fmt_tuning_short(row.speedup_vs_seq),
                fmt_tuning_exp(row.max_diff_vs_ref),
            );
        }

        println!();
        println!(
            "[AOT combustion tuning map] honest wall-clock summary; all time columns are milliseconds"
        );
        println!(
            "note: this is the closest table to stopwatch timing from button press to finished result."
        );
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<31}",
            "config",
            "honest_user_e2e_ms",
            "honest_speedup",
            "runtime_solve_ms",
            "manual_bootstrap_ms mean+/-std [min,max]"
        );
        println!("{}", "-".repeat(132));
        for row in rows {
            println!(
                "{:<30} | {:<18} | {:<18} | {:<18} | {:<31}",
                row.label,
                fmt_tuning_short(row.honest_user_e2e_ms),
                fmt_tuning_short(row.honest_speedup_vs_seq),
                fmt_tuning_short(row.solve_ms),
                fmt_tuning_agg(row.bootstrap_ms),
            );
        }

        println!();
        println!(
            "[AOT combustion tuning map] isolated cold stage breakdown; every column in this table comes from the same child solve as honest_user_e2e_ms."
        );
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}",
            "config",
            "honest_e2e_ms",
            "solver_total_ms",
            "symbolic_ms",
            "initial_sym_jac",
            "artifact_ms",
            "materialize_ms",
            "compile_link_ms"
        );
        println!("{}", "-".repeat(188));
        for row in rows {
            println!(
                "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}",
                row.label,
                fmt_tuning_short(row.honest_user_e2e_ms),
                fmt_tuning_short(row.cold_total_timer_ms),
                fmt_tuning_short(row.cold_symbolic_ms),
                fmt_tuning_short(row.cold_initial_symbolic_jacobian_ms),
                fmt_tuning_short(row.cold_aot_artifact_ms),
                fmt_tuning_short(row.cold_aot_materialize_ms),
                fmt_tuning_short(row.cold_aot_compile_link_ms),
            );
        }

        println!();
        println!(
            "[AOT combustion tuning map] isolated cold numerical/runtime stages; all columns are from the fresh child solve."
        );
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12} | {:<18} | {:<18}",
            "config",
            "linear_ms",
            "jac_ms",
            "fun_ms",
            "residual_values",
            "jacobian_values",
            "jacobian_assembly",
            "res_jobs",
            "jac_jobs",
            "rebind_ms",
            "register_link_ms"
        );
        println!("{}", "-".repeat(222));
        for row in rows {
            println!(
                "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12} | {:<18} | {:<18}",
                row.label,
                fmt_tuning_short(row.cold_linear_ms),
                fmt_tuning_short(row.cold_jac_ms),
                fmt_tuning_short(row.cold_fun_ms),
                fmt_tuning_short(row.cold_cb_residual_values_ms),
                fmt_tuning_short(row.cold_cb_jacobian_values_ms),
                fmt_tuning_short(row.cold_cb_jacobian_assembly_ms),
                fmt_tuning_short(row.cold_residual_actual_jobs),
                fmt_tuning_short(row.cold_sparse_jacobian_actual_jobs),
                fmt_tuning_short(row.cold_post_build_rebind_ms),
                fmt_tuning_short(row.cold_aot_register_link_ms),
            );
        }

        println!();
        println!(
            "[AOT combustion tuning map] runtime timing/counter summary; all time columns are milliseconds"
        );
        println!(
            "note: manual_bootstrap_ms is callback preparation/linking performed by this diagnostic runner. Use solve_ms, callback stages, and counters for chunking decisions."
        );
        println!(
            "{:<30} | {:<18} | {:<31} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<14} | {:<14} | {:<14}",
            "config",
            "solve_ms",
            "manual_bootstrap_ms mean+/-std [min,max]",
            "solver_total_ms",
            "symbolic_ms",
            "linear_ms",
            "jac_ms",
            "fun_ms",
            "iters",
            "linsys",
            "jac_re"
        );
        println!("{}", "-".repeat(258));
        for row in rows {
            println!(
                "{:<30} | {:<18} | {:<31} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<14} | {:<14} | {:<14}",
                row.label,
                fmt_tuning_short(row.solve_ms),
                fmt_tuning_agg(row.bootstrap_ms),
                fmt_tuning_short(row.total_timer_ms),
                fmt_tuning_short(row.symbolic_ms),
                fmt_tuning_short(row.linear_ms),
                fmt_tuning_short(row.jac_ms),
                fmt_tuning_short(row.fun_ms),
                fmt_tuning_short(row.iterations),
                fmt_tuning_short(row.linear_solves),
                fmt_tuning_short(row.jac_rebuilds),
            );
        }

        println!();
        println!(
            "[AOT combustion tuning map] linked callback stage summary; all time columns are milliseconds"
        );
        println!(
            "note: these columns are populated by linked AOT callbacks; Lambdify rows may be blank."
        );
        println!(
            "{:<30} | {:<18} | {:<18} | {:<20}",
            "config", "residual_values", "jacobian_values", "jacobian_assembly"
        );
        println!("{}", "-".repeat(98));
        for row in rows {
            println!(
                "{:<30} | {:<18} | {:<18} | {:<20}",
                row.label,
                fmt_tuning_short(row.cb_residual_values_ms),
                fmt_tuning_short(row.cb_jacobian_values_ms),
                fmt_tuning_short(row.cb_jacobian_assembly_ms),
            );
        }
    }

    fn runtime_tuning_lambdify_config()
    -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_matrix_backend_override(MatrixBackend::SparseCol)
    }

    fn runtime_tuning_aot_base_config(
        toolchain: RuntimeTuningToolchain,
    ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_matrix_backend_override(MatrixBackend::SparseCol)
                .with_aot_compile_dev_fastest();
        toolchain.apply_to(config)
    }

    fn sparse_atomview_rebuild_release_devfastest_config(
        toolchain: RuntimeTuningToolchain,
    ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile:
                        crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });
        toolchain.apply_to(config)
    }

    fn runtime_tuning_sequential_case() -> RuntimeTuningChunkCase {
        RuntimeTuningChunkCase {
            label: "seq",
            execution_policy: AotExecutionPolicy::SequentialOnly,
            chunking_policy: AotChunkingPolicy::default(),
        }
    }

    fn runtime_tuning_parallel_cases() -> Vec<RuntimeTuningChunkCase> {
        vec![
            RuntimeTuningChunkCase {
                label: "par-4x4-jobs4",
                execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(4),
                    max_sparse_jobs: Some(4),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                chunking_policy: AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                ),
            },
            RuntimeTuningChunkCase {
                label: "par-8x8-jobs8",
                execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(8),
                    max_sparse_jobs: Some(8),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                chunking_policy: AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ),
            },
            RuntimeTuningChunkCase {
                label: "par-16x16-jobs16",
                execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(16),
                    max_sparse_jobs: Some(16),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                chunking_policy: AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                ),
            },
            RuntimeTuningChunkCase {
                label: "par-res16-row32-jobs8",
                execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(8),
                    max_sparse_jobs: Some(8),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                chunking_policy: AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                    Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 32 }),
                ),
            },
        ]
    }

    fn runtime_tuning_variant_label(
        toolchain: RuntimeTuningToolchain,
        chunk_case: &RuntimeTuningChunkCase,
    ) -> String {
        format!("{}/{}", toolchain.label(), chunk_case.label)
    }

    fn runtime_tuning_aot_config(
        toolchain: RuntimeTuningToolchain,
        chunk_case: &RuntimeTuningChunkCase,
    ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        runtime_tuning_aot_base_config(toolchain)
            .with_aot_execution_policy(chunk_case.execution_policy.clone())
            .with_aot_chunking_policy(chunk_case.chunking_policy)
    }

    fn runtime_tuning_honest_cold_aot_config(
        toolchain: RuntimeTuningToolchain,
        chunk_case: &RuntimeTuningChunkCase,
    ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        runtime_tuning_aot_config(toolchain, chunk_case).with_aot_build_policy(
            AotBuildPolicy::RebuildAlways {
                profile:
                    crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
            },
        )
    }

    fn runtime_tuning_cold_variants() -> Vec<(
        String,
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    )> {
        let sequential_case = runtime_tuning_sequential_case();
        let parallel_cases = runtime_tuning_parallel_cases();
        let mut variants = vec![(
            "lambdify-baseline".to_string(),
            runtime_tuning_lambdify_config(),
        )];
        for toolchain in RuntimeTuningToolchain::variants() {
            variants.push((
                runtime_tuning_variant_label(toolchain, &sequential_case),
                runtime_tuning_honest_cold_aot_config(toolchain, &sequential_case),
            ));
            for chunk_case in &parallel_cases {
                variants.push((
                    runtime_tuning_variant_label(toolchain, chunk_case),
                    runtime_tuning_honest_cold_aot_config(toolchain, chunk_case),
                ));
            }
        }
        variants
    }

    fn encode_isolated_tuning_solution(solution: &DMatrix<f64>) -> String {
        solution
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("\t")
    }

    fn decode_isolated_tuning_solution(line: &str) -> Vec<f64> {
        line.strip_prefix(ISOLATED_TUNING_SOLUTION_MARKER)
            .expect("isolated tuning solution marker should be present")
            .trim_start_matches('\t')
            .split('\t')
            .filter(|value| !value.is_empty())
            .map(|value| {
                value
                    .parse::<f64>()
                    .unwrap_or_else(|err| panic!("isolated tuning solution value invalid: {err}"))
            })
            .collect()
    }

    fn encode_isolated_cold_metrics(metrics: &IsolatedColdMetrics) -> String {
        [
            metrics.total_timer_ms,
            metrics.symbolic_ms,
            metrics.linear_ms,
            metrics.jac_ms,
            metrics.fun_ms,
            metrics.cb_residual_values_ms,
            metrics.cb_jacobian_values_ms,
            metrics.cb_jacobian_assembly_ms,
            metrics.residual_actual_jobs,
            metrics.sparse_jacobian_actual_jobs,
            metrics.initial_symbolic_jacobian_ms,
            metrics.post_build_rebind_ms,
            metrics.aot_artifact_ms,
            metrics.aot_materialize_ms,
            metrics.aot_compile_link_ms,
            metrics.aot_register_link_ms,
        ]
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\t")
    }

    fn decode_isolated_cold_metrics(line: &str) -> IsolatedColdMetrics {
        let values = line
            .strip_prefix(ISOLATED_TUNING_METRICS_MARKER)
            .expect("isolated tuning metrics marker should be present")
            .trim_start_matches('\t')
            .split('\t')
            .map(|value| {
                value
                    .parse::<f64>()
                    .unwrap_or_else(|err| panic!("isolated cold metric invalid: {err}"))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values.len(),
            16,
            "isolated cold metrics payload should contain every stage field"
        );
        IsolatedColdMetrics {
            total_timer_ms: values[0],
            symbolic_ms: values[1],
            linear_ms: values[2],
            jac_ms: values[3],
            fun_ms: values[4],
            cb_residual_values_ms: values[5],
            cb_jacobian_values_ms: values[6],
            cb_jacobian_assembly_ms: values[7],
            residual_actual_jobs: values[8],
            sparse_jacobian_actual_jobs: values[9],
            initial_symbolic_jacobian_ms: values[10],
            post_build_rebind_ms: values[11],
            aot_artifact_ms: values[12],
            aot_materialize_ms: values[13],
            aot_compile_link_ms: values[14],
            aot_register_link_ms: values[15],
        }
    }

    fn cold_cooldown_ms() -> u64 {
        std::env::var(COLD_COOLDOWN_MS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn clean_cold_artifacts_enabled() -> bool {
        std::env::var(COLD_CLEAN_ARTIFACTS_ENV)
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    }

    fn remove_generated_aot_builds_for_child(child_pid: u32) {
        if !clean_cold_artifacts_enabled() {
            return;
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("generated-aot");
        let prefix = format!("build-{child_pid}-");
        let Ok(problem_dirs) = fs::read_dir(&root) else {
            return;
        };
        for problem_dir in problem_dirs.flatten() {
            let Ok(build_dirs) = fs::read_dir(problem_dir.path()) else {
                continue;
            };
            for build_dir in build_dirs.flatten() {
                let name = build_dir.file_name().to_string_lossy().to_string();
                if name.starts_with(&prefix) {
                    fs::remove_dir_all(build_dir.path()).unwrap_or_else(|err| {
                        panic!(
                            "failed to remove isolated child AOT artifact directory {}: {err}",
                            build_dir.path().display()
                        )
                    });
                }
            }
        }
    }

    fn run_isolated_tuning_child(index: usize, n_steps: usize) {
        let variants = runtime_tuning_cold_variants();
        let (label, config) = variants
            .get(index)
            .unwrap_or_else(|| panic!("isolated tuning child index {index} is invalid"));
        let mut solver = make_combustion_solver(n_steps, config.clone());
        let elapsed_ms = solve_honest_user_e2e_and_measure(
            &mut solver,
            &format!("isolated-cold-{label}-{n_steps}"),
        )
        .expect("isolated cold tuning variant should solve");
        let solution = solver
            .get_result()
            .expect("isolated cold tuning variant should produce a solution");
        let metrics = isolated_cold_metrics_from_solver(&solver);
        println!("{ISOLATED_TUNING_PID_MARKER}\t{}", std::process::id());
        println!("{ISOLATED_TUNING_TIME_MARKER}\t{elapsed_ms}");
        println!(
            "{ISOLATED_TUNING_METRICS_MARKER}\t{}",
            encode_isolated_cold_metrics(&metrics)
        );
        println!(
            "{ISOLATED_TUNING_SOLUTION_MARKER}\t{}",
            encode_isolated_tuning_solution(&solution)
        );
    }

    fn solve_isolated_cold_tuning_variant(index: usize, n_steps: usize) -> IsolatedColdObservation {
        let output = Command::new(std::env::current_exe().expect("test executable should resolve"))
            .arg("--exact")
            .arg("numerical::BVP_Damp::BVP_Damp_tests3::tests::aot_combustion_parallel_tuning_reports_runtime_table")
            .arg("--ignored")
            .arg("--nocapture")
            .env(ISOLATED_TUNING_CHILD_INDEX_ENV, index.to_string())
            .env(ISOLATED_TUNING_CHILD_STEPS_ENV, n_steps.to_string())
            .output()
            .expect("isolated cold tuning child should launch");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "isolated cold tuning child {index} failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        let elapsed_ms = stdout
            .lines()
            .find(|line| line.starts_with(ISOLATED_TUNING_TIME_MARKER))
            .and_then(|line| {
                line.strip_prefix(ISOLATED_TUNING_TIME_MARKER)
                    .map(str::trim)
            })
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no time"));
        let metrics = stdout
            .lines()
            .find(|line| line.starts_with(ISOLATED_TUNING_METRICS_MARKER))
            .map(decode_isolated_cold_metrics)
            .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no metrics"));
        let solution = stdout
            .lines()
            .find(|line| line.starts_with(ISOLATED_TUNING_SOLUTION_MARKER))
            .map(decode_isolated_tuning_solution)
            .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no solution"));
        let child_pid = stdout
            .lines()
            .find(|line| line.starts_with(ISOLATED_TUNING_PID_MARKER))
            .and_then(|line| line.strip_prefix(ISOLATED_TUNING_PID_MARKER).map(str::trim))
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no pid"));
        remove_generated_aot_builds_for_child(child_pid);
        let cooldown_ms = cold_cooldown_ms();
        if cooldown_ms > 0 {
            thread::sleep(Duration::from_millis(cooldown_ms));
        }
        IsolatedColdObservation {
            elapsed_ms,
            solution,
            metrics,
        }
    }

    fn print_isolated_cold_raw_observation(
        repetition: usize,
        label: &str,
        observation: &IsolatedColdObservation,
    ) {
        println!(
            "[AOT isolated cold raw] rep={} config={} total_ms={:.3} symbolic_ms={:.3} initial_sym_jac_ms={:.3} materialize_ms={:.3} compile_link_ms={:.3} res_jobs={:.3} jac_jobs={:.3}",
            repetition + 1,
            label,
            observation.elapsed_ms,
            observation.metrics.symbolic_ms,
            observation.metrics.initial_symbolic_jacobian_ms,
            observation.metrics.aot_materialize_ms,
            observation.metrics.aot_compile_link_ms,
            observation.metrics.residual_actual_jobs,
            observation.metrics.sparse_jacobian_actual_jobs,
        );
    }

    fn run_combustion_tuning_scenario(
        n_steps: usize,
        repetitions: usize,
        scenario_label: &str,
    ) -> Result<(), BvpBackendIntegrationError> {
        let toolchains = RuntimeTuningToolchain::variants();
        let sequential_case = runtime_tuning_sequential_case();
        let parallel_cases = runtime_tuning_parallel_cases();
        let mut labels = Vec::with_capacity(1 + toolchains.len() * (1 + parallel_cases.len()));
        labels.push("lambdify-baseline".to_string());
        for toolchain in toolchains {
            labels.push(runtime_tuning_variant_label(toolchain, &sequential_case));
            for chunk_case in &parallel_cases {
                labels.push(runtime_tuning_variant_label(toolchain, chunk_case));
            }
        }

        let mut samples = Vec::with_capacity(labels.len() * repetitions);
        println!(
            "[AOT combustion tuning map] isolated cold protocol: cooldown_ms={}, cleanup_child_artifacts={}",
            cold_cooldown_ms(),
            clean_cold_artifacts_enabled()
        );
        for repetition in 0..repetitions {
            println!(
                "[AOT combustion tuning map] starting repetition {}/{}",
                repetition + 1,
                repetitions
            );
            let lambdify_config = runtime_tuning_lambdify_config();
            let lambdify_cold = solve_isolated_cold_tuning_variant(0, n_steps);
            print_isolated_cold_raw_observation(repetition, "lambdify-baseline", &lambdify_cold);
            let mut lambdify = make_combustion_solver(n_steps, lambdify_config);
            let (lambdify_prepare_ms, lambdify_solve_ms) = solve_with_lambdify_and_measure(
                &mut lambdify,
                &format!("combustion-lambdify-baseline-{n_steps}-rep{repetition}"),
            )?;
            let lambdify_solution = lambdify
                .get_result()
                .expect("lambdify baseline for AOT combustion tuning should produce a solution");
            assert!(
                lambdify_solution.iter().all(|value| value.is_finite()),
                "lambdify baseline for AOT combustion tuning should remain finite"
            );
            let lambdify_honest_diff = lambdify_solution
                .iter()
                .zip(lambdify_cold.solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                lambdify_honest_diff < 1.0e-4,
                "isolated Lambdify cold result disagrees with prepared reference by {lambdify_honest_diff}"
            );
            samples.push(runtime_tuning_sample_from_solver(
                "lambdify-baseline",
                n_steps,
                lambdify_cold.elapsed_ms,
                f64::NAN,
                lambdify_prepare_ms,
                lambdify_solve_ms,
                f64::NAN,
                0.0,
                &lambdify,
                lambdify_cold.metrics,
            ));

            let mut cold_variant_index = 1usize;
            for toolchain in toolchains {
                let sequential_label = runtime_tuning_variant_label(toolchain, &sequential_case);
                let sequential_config = runtime_tuning_aot_config(toolchain, &sequential_case);
                let mut sequential = make_combustion_solver(n_steps, sequential_config);
                let (_seq_guard, seq_bootstrap_ms, seq_solve_ms) = solve_with_aot_and_measure(
                    &mut sequential,
                    &format!("combustion-{sequential_label}-{n_steps}-rep{repetition}"),
                )?;
                let sequential_solution = sequential.get_result().unwrap_or_else(|| {
                    panic!("{sequential_label}: AOT tuning case should produce a solution")
                });
                assert!(
                    sequential_solution.iter().all(|value| value.is_finite()),
                    "{sequential_label}: AOT tuning solution should remain finite"
                );
                let seq_max_diff_vs_ref = lambdify_solution
                    .iter()
                    .zip(sequential_solution.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    seq_max_diff_vs_ref < 1.0e-4,
                    "{sequential_label}: AOT tuning disagreement with lambdify reference {seq_max_diff_vs_ref} is too large"
                );
                let seq_cold = solve_isolated_cold_tuning_variant(cold_variant_index, n_steps);
                cold_variant_index += 1;
                print_isolated_cold_raw_observation(repetition, &sequential_label, &seq_cold);
                let seq_honest_diff_vs_ref = lambdify_solution
                    .iter()
                    .zip(seq_cold.solution.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    seq_honest_diff_vs_ref < 1.0e-4,
                    "{sequential_label}: honest AOT e2e disagreement with lambdify reference {seq_honest_diff_vs_ref} is too large"
                );
                samples.push(runtime_tuning_sample_from_solver(
                    sequential_label.clone(),
                    n_steps,
                    seq_cold.elapsed_ms,
                    1.0,
                    seq_bootstrap_ms,
                    seq_solve_ms,
                    1.0,
                    seq_max_diff_vs_ref,
                    &sequential,
                    seq_cold.metrics,
                ));

                for chunk_case in &parallel_cases {
                    let label = runtime_tuning_variant_label(toolchain, chunk_case);
                    let generated_backend_config = runtime_tuning_aot_config(toolchain, chunk_case);
                    let mut solver = make_combustion_solver(n_steps, generated_backend_config);
                    let (_guard, bootstrap_ms, solve_ms) = solve_with_aot_and_measure(
                        &mut solver,
                        &format!("combustion-{label}-{n_steps}-rep{repetition}"),
                    )?;
                    let solution = solver.get_result().unwrap_or_else(|| {
                        panic!("{label}: AOT tuning case should produce a solution")
                    });
                    assert!(
                        solution.iter().all(|value| value.is_finite()),
                        "{label}: AOT combustion tuning solution should remain finite"
                    );
                    let max_diff_vs_ref = lambdify_solution
                        .iter()
                        .zip(solution.iter())
                        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        max_diff_vs_ref < 1.0e-4,
                        "{label}: AOT combustion tuning disagreement with lambdify reference {max_diff_vs_ref} is too large"
                    );
                    let cold = solve_isolated_cold_tuning_variant(cold_variant_index, n_steps);
                    cold_variant_index += 1;
                    print_isolated_cold_raw_observation(repetition, &label, &cold);
                    let honest_max_diff_vs_ref = lambdify_solution
                        .iter()
                        .zip(cold.solution.iter())
                        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        honest_max_diff_vs_ref < 1.0e-4,
                        "{label}: honest AOT e2e disagreement with lambdify reference {honest_max_diff_vs_ref} is too large"
                    );
                    samples.push(runtime_tuning_sample_from_solver(
                        label,
                        n_steps,
                        cold.elapsed_ms,
                        seq_cold.elapsed_ms / cold.elapsed_ms.max(f64::EPSILON),
                        bootstrap_ms,
                        solve_ms,
                        seq_solve_ms / solve_ms,
                        max_diff_vs_ref,
                        &solver,
                        cold.metrics,
                    ));
                }
            }
        }

        let rows = summarize_runtime_tuning_samples(&labels, &samples);
        print_runtime_tuning_summary_table(scenario_label, n_steps, repetitions, &rows);

        let winner = rows
            .iter()
            .min_by(|lhs, rhs| lhs.solve_ms.mean.total_cmp(&rhs.solve_ms.mean))
            .expect("runtime tuning table should contain at least one row");
        println!(
            "[AOT runtime tuning winner] scenario={}, config={}, n_steps={}, runs={}, solve_ms_mean={:.3}, speedup_vs_seq_mean={:.3}, manual_bootstrap_ms_mean={:.3}",
            scenario_label,
            winner.label,
            winner.n_steps,
            winner.runs,
            winner.solve_ms.mean,
            winner.speedup_vs_seq.mean,
            winner.bootstrap_ms.mean,
        );
        let cold_winner = rows
            .iter()
            .min_by(|lhs, rhs| {
                lhs.honest_user_e2e_ms
                    .mean
                    .total_cmp(&rhs.honest_user_e2e_ms.mean)
            })
            .expect("cold wall-clock tuning table should contain at least one row");
        println!(
            "[AOT isolated cold wall-clock winner] scenario={}, config={}, n_steps={}, runs={}, honest_user_e2e_ms_mean={:.3}",
            scenario_label,
            cold_winner.label,
            cold_winner.n_steps,
            cold_winner.runs,
            cold_winner.honest_user_e2e_ms.mean,
        );
        Ok(())
    }

    fn make_example_solver(
        equation: &NonlinEquation,
        n_steps: usize,
        strategy_params: Option<SolverParams>,
        generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    ) -> NRBVP {
        let eq_system = equation.setup();
        let values = equation.values();
        let border_conditions = equation.boundary_conditions();
        let bounds = equation.Bounds();
        let rel_tolerance = equation.rel_tolerance();
        let (t0, t_end) = equation.span(None, None);
        let initial_guess = uniform_initial_guess(values.len(), n_steps, 0.7);
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(strategy_params)
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(40)
            .with_bounds(bounds)
            .with_generated_backend_config(generated_backend_config);

        let mut solver = NRBVP::new_with_options(
            eq_system,
            initial_guess,
            values,
            "x".to_string(),
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn make_combustion_solver(
        n_steps: usize,
        generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    ) -> NRBVP {
        let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str.clone());
        let teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let c0 = unknowns[2].clone();
        let j0 = unknowns[3].clone();
        let j1 = unknowns[5].clone();

        let q_heat = 3000.0 * 1e3 * 0.034;
        let dt = 600.0;
        let t_scale = 600.0;
        let l: f64 = 3e-4;
        let m0 = 34.2 / 1000.0;
        let lambda = 0.07;
        let p = 2e6;
        let tm = 1500.0;
        let c1_0 = 1.0;
        let t_initial = 1000.0;
        let pe_q = 0.0090168;
        let d_ro = 2.88e-4;
        let pe_d = 1.50e-3;
        let ro_m_ = m0 * p / (8.314 * tm);

        let dt_sym = Expr::Const(dt);
        let t_scale_sym = Expr::Const(t_scale);
        let lambda_sym = Expr::Const(lambda);
        let q_heat = Expr::Const(q_heat);
        let a = Expr::Const(1.3e5);
        let e = Expr::Const(5000.0 * 4.184);
        let m = Expr::Const(m0);
        let r_g = Expr::Const(8.314);
        let ro_m = Expr::Const(ro_m_);
        let qm = Expr::Const(l.powf(2.0) / t_scale);
        let qs = Expr::Const(l.powf(2.0));
        let pe_q_sym = Expr::Const(pe_q);
        let ro_d = vec![Expr::Const(d_ro), Expr::Const(d_ro)];
        let pe_d = vec![Expr::Const(pe_d), Expr::Const(pe_d)];
        let minus = Expr::Const(-1.0);
        let m_reag = Expr::Const(0.342);

        let rate = a
            * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
            * c0.clone()
            * (ro_m.clone() / m_reag.clone());
        let eq_t = q.clone() / lambda_sym;
        let eq_q = q * pe_q_sym - q_heat * rate.clone() * qm;
        let eq_c0 = j0.clone() / ro_d[0].clone();
        let eq_j0 = j0 * pe_d[0].clone()
            - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone();
        let eq_c1 = j1.clone() / ro_d[1].clone();
        let eq_j1 = j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs;
        let eqs = vec![eq_t, eq_q, eq_c0, eq_j0, eq_c1, eq_j1];

        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, c1_0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);
        let bounds = HashMap::from([
            ("Teta".to_string(), (0.0, 10.0)),
            ("q".to_string(), (-1e20, 1e20)),
            ("C0".to_string(), (0.0, 1.5)),
            ("J0".to_string(), (-1e2, 1e2)),
            ("C1".to_string(), (0.0, 1.5)),
            ("J1".to_string(), (-1e2, 1e2)),
        ]);
        let rel_tolerance = HashMap::from([
            ("Teta".to_string(), 1e-5),
            ("q".to_string(), 1e-5),
            ("C0".to_string(), 1e-5),
            ("J0".to_string(), 1e-5),
            ("C1".to_string(), 1e-5),
            ("J1".to_string(), 1e-5),
        ]);
        let strategy_params = SolverParams {
            max_jac: Some(6),
            max_damp_iter: Some(6),
            damp_factor: Some(0.5),
            adaptive: None,
        };
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(strategy_params))
            .with_abs_tolerance(1e-6)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(100)
            .with_bounds(bounds)
            .with_generated_backend_config(generated_backend_config)
            .with_loglevel(Some("none".to_string()));
        let initial_guess = uniform_initial_guess(unknowns_str.len(), n_steps, 0.99);

        let mut solver = NRBVP::new_with_options(
            eqs,
            initial_guess,
            unknowns_str.iter().map(|value| value.to_string()).collect(),
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn make_oscillator_solver(
        n_steps: usize,
        generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    ) -> NRBVP {
        let eqs = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x.sin();
            guess[i * values.len() + 1] = x.cos();
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]);
        let bounds = HashMap::from([
            ("y".to_string(), (-1.2, 1.2)),
            ("z".to_string(), (-1.2, 1.2)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);

        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(60)
            .with_bounds(bounds)
            .with_generated_backend_config(generated_backend_config)
            .with_loglevel(Some("error".to_string()));

        let mut solver = NRBVP::new_with_options(
            eqs,
            initial_guess,
            values,
            "x".to_string(),
            boundary_conditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    #[test]
    fn symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle() {
        let label = "two-point-24";
        let equation = NonlinEquation::TwoPointBVP;
        let n_steps = 24usize;

        let mut legacy_solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        );
        let legacy_request = legacy_solver.build_solver_request(None, None);
        let atom_request = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        )
        .build_solver_request(None, None);

        let (mut legacy_bundle, legacy_ms) = measure_sparse_bundle_build_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        )
        .expect("legacy sparse bundle should build for exact-case backend comparison");
        let (mut atom_bundle, atom_ms) = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        )
        .expect("atom sparse bundle should build for exact-case backend comparison");

        let args = DVector::from_element(atom_bundle.variable_string.len(), 0.7);
        let (residual_max_diff, jacobian_max_diff) =
            compare_sparse_bundles_numerically(&mut legacy_bundle, &mut atom_bundle, &args, label);
        println!(
            "[BVP symbolic assembly exact-case] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_ms={atom_ms:.3}, speedup={:.3}x, residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}",
            legacy_ms / atom_ms,
        );
        assert!(
            jacobian_max_diff < 1.0e-6,
            "{label}: jacobian disagreement between ExprLegacy and AtomView is too large: {jacobian_max_diff}"
        );
        assert!(
            residual_max_diff < 1.0e-6,
            "{label}: residual disagreement between ExprLegacy and AtomView is too large: {residual_max_diff}"
        );
    }

    #[test]
    #[ignore = "diagnostic timing table for symbolic assembly backends across representative BVP fixtures"]
    fn symbolic_assembly_backends_report_representative_fixture_table() {
        let cases = [
            ("two-point-72", NonlinEquation::TwoPointBVP, 72usize),
            ("clairaut-72", NonlinEquation::Clairaut, 72usize),
            ("parachute-48", NonlinEquation::ParachuteEquation, 48usize),
            ("lane-emden-48", NonlinEquation::LaneEmden5, 48usize),
        ];

        println!();
        println!(
            "{:<18} | {:>7} | {:>14} | {:>12} | {:>12} | {:>14} | {:>14}",
            "fixture",
            "n_steps",
            "status",
            "legacy_ms",
            "atom_ms",
            "residual_diff",
            "jacobian_diff"
        );
        println!("{}", "-".repeat(108));

        for (label, equation, n_steps) in cases {
            let mut legacy_solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
            );
            let legacy_request = legacy_solver.build_solver_request(None, None);
            let atom_request = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
            )
            .build_solver_request(None, None);
            let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
            );
            let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
                atom_request,
                BvpSymbolicAssemblyBackend::AtomView,
            );

            match (legacy_result, atom_result) {
                (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
                    let args = DVector::from_element(atom_bundle.variable_string.len(), 0.7);
                    let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                        &mut legacy_bundle,
                        &mut atom_bundle,
                        &args,
                        label,
                    );
                    println!(
                        "{:<18} | {:>7} | {:>14} | {:>12.3} | {:>12.3} | {:>14.6e} | {:>14.6e}",
                        label,
                        n_steps,
                        if atom_ms < legacy_ms {
                            "atom faster"
                        } else {
                            "legacy faster"
                        },
                        legacy_ms,
                        atom_ms,
                        residual_max_diff,
                        jacobian_max_diff
                    );
                }
                (Ok((_legacy_bundle, legacy_ms)), Err(atom_error)) => {
                    println!(
                        "{:<18} | {:>7} | {:>14} | {:>12.3} | {:>12} | {:>14} | {:>14}",
                        label,
                        n_steps,
                        "atom failed",
                        legacy_ms,
                        "-",
                        "-",
                        format!("{atom_error:?}")
                    );
                }
                (Err(legacy_error), Ok((_atom_bundle, atom_ms))) => {
                    println!(
                        "{:<18} | {:>7} | {:>14} | {:>12} | {:>12.3} | {:>14} | {:>14}",
                        label,
                        n_steps,
                        "legacy failed",
                        "-",
                        atom_ms,
                        format!("{legacy_error:?}"),
                        "-"
                    );
                }
                (Err(legacy_error), Err(atom_error)) => {
                    println!(
                        "{:<18} | {:>7} | {:>14} | {:>12} | {:>12} | {:>14} | {:>14}",
                        label,
                        n_steps,
                        "both failed",
                        "-",
                        "-",
                        format!("{legacy_error:?}"),
                        format!("{atom_error:?}")
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "diagnostic solver-level ExprLegacy vs AtomView compare on representative exact-like BVP fixtures"]
    fn symbolic_assembly_backends_report_representative_solver_table() {
        #[derive(Debug)]
        struct RepresentativeSolverRow {
            label: &'static str,
            backend: BvpSymbolicAssemblyBackend,
            n_steps: usize,
            generate_ms: f64,
            solve_ms: f64,
            max_diff_vs_legacy: f64,
        }

        let cases = [
            ("parachute-96", NonlinEquation::ParachuteEquation, 96usize),
            ("lane-emden-96", NonlinEquation::LaneEmden5, 96usize),
        ];
        let mut rows = Vec::new();
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

        for (label, equation, n_steps) in cases {
            let mut legacy_solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_generate_begin = Instant::now();
            legacy_solver
                .try_eq_generate(None, None)
                .expect("ExprLegacy representative solve-path generate should succeed");
            let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solve_begin = Instant::now();
            legacy_solver
                .try_solve()
                .expect("ExprLegacy representative solve-path solve should succeed");
            let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solution = legacy_solver
                .get_result()
                .expect("ExprLegacy representative solve-path should produce a solution");

            let mut atom_solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_generate_begin = Instant::now();
            atom_solver
                .try_eq_generate(None, None)
                .expect("AtomView representative solve-path generate should succeed");
            let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solve_begin = Instant::now();
            atom_solver
                .try_solve()
                .expect("AtomView representative solve-path solve should succeed");
            let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solution = atom_solver
                .get_result()
                .expect("AtomView representative solve-path should produce a solution");

            let max_diff_vs_legacy = legacy_solution
                .iter()
                .zip(atom_solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);

            rows.push(RepresentativeSolverRow {
                label,
                backend: BvpSymbolicAssemblyBackend::ExprLegacy,
                n_steps,
                generate_ms: legacy_generate_ms,
                solve_ms: legacy_solve_ms,
                max_diff_vs_legacy: 0.0,
            });
            rows.push(RepresentativeSolverRow {
                label,
                backend: BvpSymbolicAssemblyBackend::AtomView,
                n_steps,
                generate_ms: atom_generate_ms,
                solve_ms: atom_solve_ms,
                max_diff_vs_legacy,
            });
        }

        println!("[BVP symbolic assembly representative solver compare] ExprLegacy vs AtomView");
        println!(
            "{:<16} | {:<12} | {:>7} | {:>12} | {:>10} | {:>18}",
            "fixture", "backend", "n_steps", "generate_ms", "solve_ms", "max_diff_vs_legacy"
        );
        println!("{}", "-".repeat(92));
        for row in &rows {
            let backend = match row.backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            };
            println!(
                "{:<16} | {:<12} | {:>7} | {:>12.3} | {:>10.3} | {:>18.6e}",
                row.label,
                backend,
                row.n_steps,
                row.generate_ms,
                row.solve_ms,
                row.max_diff_vs_legacy
            );
        }
    }

    #[test]
    #[ignore = "diagnostic compare for symbolic assembly backends across full BVP fixtures"]
    fn symbolic_assembly_backends_build_sparse_bundles_for_representative_examples() {
        let exact_examples = [
            ("two-point-72", NonlinEquation::TwoPointBVP, 72usize),
            ("clairaut-72", NonlinEquation::Clairaut, 72usize),
            ("parachute-48", NonlinEquation::ParachuteEquation, 48usize),
            ("lane-emden-48", NonlinEquation::LaneEmden5, 48usize),
        ];

        for (label, equation, n_steps) in exact_examples {
            let mut solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
            );
            let legacy_request = solver.build_solver_request(None, None);
            let atom_request = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
            )
            .build_solver_request(None, None);
            let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
            );
            let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
                atom_request,
                BvpSymbolicAssemblyBackend::AtomView,
            );
            match (legacy_result, atom_result) {
                (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
                    assert_eq!(legacy_bundle.variable_string, atom_bundle.variable_string);
                    assert_eq!(legacy_bundle.jacobian_shape(), atom_bundle.jacobian_shape());
                    assert_eq!(legacy_bundle.residual_len(), atom_bundle.residual_len());
                    let args = DVector::from_vec(
                        (0..atom_bundle.variable_string.len())
                            .map(|index| 0.2 + index as f64 * 0.01)
                            .collect(),
                    );
                    let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                        &mut legacy_bundle,
                        &mut atom_bundle,
                        &args,
                        label,
                    );
                    println!(
                        "[BVP symbolic assembly compare] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_ms={atom_ms:.3}, speedup={:.3}x, residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}",
                        legacy_ms / atom_ms,
                    );
                }
                (Ok((_legacy_bundle, legacy_ms)), Err(error)) => {
                    println!(
                        "[BVP symbolic assembly compare] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_status=ERR({error:?})"
                    );
                }
                (Err(error), Ok((_atom_bundle, atom_ms))) => {
                    panic!(
                        "{label}: legacy symbolic assembly unexpectedly failed while atom succeeded in {atom_ms:.3} ms: {error:?}"
                    );
                }
                (Err(legacy_error), Err(atom_error)) => {
                    panic!(
                        "{label}: both symbolic assembly backends failed, legacy={legacy_error:?}, atom={atom_error:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "diagnostic symbolic assembly comparison on real combustion sparse bundle"]
    fn symbolic_assembly_backends_report_combustion_sparse_bundle_timings() {
        let n_steps = 100usize;
        let mut solver = make_combustion_solver(
            n_steps,
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        );
        let legacy_request = solver.build_solver_request(None, None);
        let atom_request = make_combustion_solver(
            n_steps,
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        )
        .build_solver_request(None, None);
        let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        );

        println!();
        println!("[BVP symbolic assembly compare] combustion sparse bundle, n_steps={n_steps}");
        match (legacy_result, atom_result) {
            (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
                let args = DVector::from_vec(
                    (0..atom_bundle.variable_string.len())
                        .map(|index| 0.3 + index as f64 * 0.005)
                        .collect(),
                );
                let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                    &mut legacy_bundle,
                    &mut atom_bundle,
                    &args,
                    "combustion-assembly-compare",
                );
                println!("{:<20} | {:>12}", "backend", "build_ms");
                println!("{}", "-".repeat(37));
                println!("{:<20} | {:>12.3}", "expr-legacy", legacy_ms);
                println!("{:<20} | {:>12.3}", "atom-view", atom_ms);
                println!("{:<20} | {:>12.6e}", "residual_max_diff", residual_max_diff);
                println!("{:<20} | {:>12.6e}", "jacobian_max_diff", jacobian_max_diff);
                println!(
                    "[BVP symbolic assembly winner] backend={}, speedup={:.3}x",
                    if atom_ms < legacy_ms {
                        "atom-view"
                    } else {
                        "expr-legacy"
                    },
                    if atom_ms < legacy_ms {
                        legacy_ms / atom_ms
                    } else {
                        atom_ms / legacy_ms
                    }
                );
            }
            (Ok((_legacy_bundle, legacy_ms)), Err(atom_error)) => {
                println!("{:<20} | {:>12}", "backend", "status");
                println!("{}", "-".repeat(37));
                println!("{:<20} | {:>12.3} ms", "expr-legacy", legacy_ms);
                println!(
                    "{:<20} | {:>12}",
                    "atom-view",
                    format!("ERR({atom_error:?})")
                );
            }
            (Err(legacy_error), Ok((_atom_bundle, atom_ms))) => {
                println!("{:<20} | {:>12}", "backend", "status");
                println!("{}", "-".repeat(37));
                println!(
                    "{:<20} | {:>12}",
                    "expr-legacy",
                    format!("ERR({legacy_error:?})")
                );
                println!("{:<20} | {:>12.3} ms", "atom-view", atom_ms);
            }
            (Err(legacy_error), Err(atom_error)) => {
                println!("{:<20} | {:>12}", "backend", "status");
                println!("{}", "-".repeat(37));
                println!(
                    "{:<20} | {:>12}",
                    "expr-legacy",
                    format!("ERR({legacy_error:?})")
                );
                println!(
                    "{:<20} | {:>12}",
                    "atom-view",
                    format!("ERR({atom_error:?})")
                );
            }
        }
    }

    #[test]
    #[ignore = "diagnostic row-level compare for combustion discretized residual assembly"]
    fn symbolic_assembly_backends_report_combustion_discretized_row_diagnostics() {
        let n_steps = 100usize;
        let mut solver = make_combustion_solver(
            n_steps,
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        );
        let request = solver.build_solver_request(None, None);

        let mut legacy = Jacobian::new();
        let request_eq_system = request.eq_system.clone();
        let request_values = request.values.clone();
        let request_arg = request.arg.clone();
        let request_mesh = request.mesh.clone();
        let request_border_conditions = request.border_conditions.clone();
        let request_scheme = request.scheme.clone();
        let request_t0 = request.t0;
        let request_h = request.h;
        legacy.discretization_system_BVP_par(
            request.eq_system.clone(),
            request.values.clone(),
            request.arg.clone(),
            request.t0,
            request.n_steps,
            request.h,
            request.mesh.clone(),
            request.border_conditions.clone(),
            request.bounds.clone(),
            request.rel_tolerance.clone(),
            request.scheme.clone(),
        );

        let atom_discretized = discretization_system_bvp_par_atom(
            request.eq_system,
            request.values,
            request.arg,
            request.t0,
            request.n_steps,
            request.h,
            request.mesh,
            request.border_conditions,
            request.bounds,
            request.rel_tolerance,
            request.scheme,
        );

        let atom_exprs = atom_discretized
            .vector_of_functions
            .iter()
            .map(atom_to_expr)
            .collect::<Vec<_>>();
        let variable_names = legacy.variable_string.clone();
        let args = (0..variable_names.len())
            .map(|index| 0.25 + index as f64 * 0.0025)
            .collect::<Vec<_>>();
        let variable_refs = variable_names
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>();

        let mut max_diff = 0.0f64;
        let mut max_index = 0usize;
        for index in 0..legacy.vector_of_functions.len() {
            let legacy_value =
                legacy.vector_of_functions[index].eval_expression(&variable_refs, &args);
            let atom_value = atom_exprs[index].eval_expression(&variable_refs, &args);
            let diff = (legacy_value - atom_value).abs();
            if diff > max_diff {
                max_diff = diff;
                max_index = index;
            }
        }

        println!(
            "[combustion discretized row diagnostics] n_steps={n_steps}, max_index={}, max_diff={:.6e}",
            max_index, max_diff
        );
        println!(
            "[combustion discretized row diagnostics] legacy_row={}",
            legacy.vector_of_functions[max_index]
        );
        println!(
            "[combustion discretized row diagnostics] atom_row={}",
            atom_exprs[max_index]
        );

        let row_step = max_index / request_values.len();
        let row_eq = max_index % request_values.len();
        let matrix_of_names = (0..=n_steps)
            .map(|step| {
                request_values
                    .iter()
                    .map(|name| format!("{name}_{step}"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let bc_value_map = request_border_conditions
            .iter()
            .flat_map(|(name, entries)| {
                entries.iter().filter_map(move |(side, value)| match side {
                    0 => Some((format!("{name}_0"), *value)),
                    1 => Some((format!("{name}_{n_steps}"), *value)),
                    _ => None,
                })
            })
            .collect::<HashMap<_, _>>();

        let legacy_eq_step = {
            let rename_map = request_values
                .iter()
                .zip(matrix_of_names[row_step].iter())
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>();
            request_eq_system[row_eq]
                .rename_variables(&rename_map)
                .set_variable(
                    &request_arg,
                    request_mesh
                        .as_ref()
                        .map(|m| m[row_step])
                        .unwrap_or(request_t0 + request_h.unwrap_or(1.0) * row_step as f64),
                )
        };
        let atom_eq_step = eq_step_atom(
            &crate::symbolic::View::conversions::expr_to_atom(&request_eq_system[row_eq]),
            &matrix_of_names,
            &request_values,
            &request_arg,
            row_step,
            request_mesh
                .as_ref()
                .map(|m| m[row_step])
                .unwrap_or(request_t0 + request_h.unwrap_or(1.0) * row_step as f64),
            &request_scheme,
        );
        let legacy_eq_step_bc = legacy_eq_step
            .set_variable_from_map(&bc_value_map)
            .simplify();
        let atom_eq_step_bc =
            atom_to_expr(&crate::symbolic::View::transform::substitute_symbol_values(
                &atom_eq_step,
                &bc_value_map
                    .iter()
                    .map(|(name, value)| {
                        (
                            crate::symbolic::View::state::Symbol::new(crate::wrap_symbol!(
                                name.as_str()
                            )),
                            *value,
                        )
                    })
                    .collect(),
            ));
        println!(
            "[combustion discretized row diagnostics] row_step={}, row_eq={}",
            row_step, row_eq
        );
        println!(
            "[combustion discretized row diagnostics] legacy_eq_step={}",
            legacy_eq_step
        );
        println!(
            "[combustion discretized row diagnostics] atom_eq_step={}",
            atom_to_expr(&atom_eq_step)
        );
        println!(
            "[combustion discretized row diagnostics] legacy_eq_step_bc={}",
            legacy_eq_step_bc
        );
        println!(
            "[combustion discretized row diagnostics] atom_eq_step_bc={}",
            atom_eq_step_bc
        );

        for index in
            max_index.saturating_sub(2)..=(max_index + 2).min(legacy.vector_of_functions.len() - 1)
        {
            let legacy_value =
                legacy.vector_of_functions[index].eval_expression(&variable_refs, &args);
            let atom_value = atom_exprs[index].eval_expression(&variable_refs, &args);
            println!(
                "[combustion discretized row diagnostics] row={}, legacy={:.6e}, atom={:.6e}, diff={:.6e}",
                index,
                legacy_value,
                atom_value,
                (legacy_value - atom_value).abs()
            );
        }
    }

    #[test]
    #[ignore = "diagnostic solver-level ExprLegacy vs AtomView stress comparison on combustion"]
    fn symbolic_assembly_backends_report_combustion_solver_stress_table() {
        #[derive(Debug)]
        struct SymbolicAssemblyStressRow {
            backend: BvpSymbolicAssemblyBackend,
            n_steps: usize,
            bundle_ms: f64,
            generate_ms: f64,
            solve_ms: f64,
            residual_max_diff: f64,
            jacobian_max_diff: f64,
            max_diff_vs_legacy: f64,
        }

        let mut rows = Vec::new();
        let n_steps_list = [200usize, 300usize];
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

        for &n_steps in &n_steps_list {
            let mut legacy_request_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_request = legacy_request_solver.build_solver_request(None, None);
            let (mut legacy_bundle, legacy_bundle_ms) =
                measure_sparse_bundle_build_with_symbolic_backend(
                    legacy_request,
                    BvpSymbolicAssemblyBackend::ExprLegacy,
                )
                .expect("ExprLegacy symbolic backend should build combustion sparse bundle");

            let mut atom_request_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_request = atom_request_solver.build_solver_request(None, None);
            let (mut atom_bundle, atom_bundle_ms) =
                measure_sparse_bundle_build_with_symbolic_backend(
                    atom_request,
                    BvpSymbolicAssemblyBackend::AtomView,
                )
                .expect("AtomView symbolic backend should build combustion sparse bundle");

            let args = DVector::from_vec(
                (0..legacy_bundle.variable_string.len())
                    .map(|index| 0.25 + index as f64 * 0.0025)
                    .collect(),
            );
            let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                &mut legacy_bundle,
                &mut atom_bundle,
                &args,
                &format!("combustion-solver-stress-{n_steps}"),
            );

            let mut legacy_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_generate_begin = Instant::now();
            legacy_solver
                .try_eq_generate(None, None)
                .expect("ExprLegacy combustion generate should succeed");
            let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solve_begin = Instant::now();
            legacy_solver
                .try_solve()
                .expect("ExprLegacy combustion solve should succeed");
            let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solution = legacy_solver
                .get_result()
                .expect("ExprLegacy combustion solve should produce a solution");

            let mut atom_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_generate_begin = Instant::now();
            atom_solver
                .try_eq_generate(None, None)
                .expect("AtomView combustion generate should succeed");
            let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solve_begin = Instant::now();
            atom_solver
                .try_solve()
                .expect("AtomView combustion solve should succeed");
            let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solution = atom_solver
                .get_result()
                .expect("AtomView combustion solve should produce a solution");

            let max_diff_vs_legacy = legacy_solution
                .iter()
                .zip(atom_solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);

            rows.push(SymbolicAssemblyStressRow {
                backend: BvpSymbolicAssemblyBackend::ExprLegacy,
                n_steps,
                bundle_ms: legacy_bundle_ms,
                generate_ms: legacy_generate_ms,
                solve_ms: legacy_solve_ms,
                residual_max_diff: 0.0,
                jacobian_max_diff: 0.0,
                max_diff_vs_legacy: 0.0,
            });
            rows.push(SymbolicAssemblyStressRow {
                backend: BvpSymbolicAssemblyBackend::AtomView,
                n_steps,
                bundle_ms: atom_bundle_ms,
                generate_ms: atom_generate_ms,
                solve_ms: atom_solve_ms,
                residual_max_diff,
                jacobian_max_diff,
                max_diff_vs_legacy,
            });
        }

        println!("[BVP symbolic assembly solver stress] combustion ExprLegacy vs AtomView");
        println!(
            "{:<12} | {:>7} | {:>10} | {:>12} | {:>10} | {:>16} | {:>16} | {:>16}",
            "backend",
            "n_steps",
            "bundle_ms",
            "generate_ms",
            "solve_ms",
            "residual_diff",
            "jacobian_diff",
            "max_diff_vs_legacy"
        );
        println!("{}", "-".repeat(124));
        for row in &rows {
            let backend = match row.backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            };
            println!(
                "{:<12} | {:>7} | {:>10.3} | {:>12.3} | {:>10.3} | {:>16.6e} | {:>16.6e} | {:>16.6e}",
                backend,
                row.n_steps,
                row.bundle_ms,
                row.generate_ms,
                row.solve_ms,
                row.residual_max_diff,
                row.jacobian_max_diff,
                row.max_diff_vs_legacy
            );
        }
    }

    #[test]
    #[ignore = "diagnostic repeated 300-step combustion compare with min/median/max for ExprLegacy vs AtomView"]
    fn symbolic_assembly_backends_report_combustion_solver_stress_stats_1000() {
        #[derive(Debug)]
        struct StressSample {
            bundle_ms: f64,
            generate_ms: f64,
            solve_ms: f64,
            residual_max_diff: f64,
            jacobian_max_diff: f64,
            max_diff_vs_legacy: f64,
        }

        #[derive(Debug)]
        struct StageStats {
            min: f64,
            median: f64,
            max: f64,
        }

        fn compute_stage_stats(mut values: Vec<f64>) -> StageStats {
            values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            let len = values.len();
            let median = if len % 2 == 0 {
                (values[len / 2 - 1] + values[len / 2]) * 0.5
            } else {
                values[len / 2]
            };
            StageStats {
                min: values[0],
                median,
                max: values[len - 1],
            }
        }

        let iterations = 5usize;
        let n_steps = 200usize;
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

        let mut legacy_samples = Vec::with_capacity(iterations);
        let mut atom_samples = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let mut legacy_request_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_request = legacy_request_solver.build_solver_request(None, None);
            let (mut legacy_bundle, legacy_bundle_ms) =
                measure_sparse_bundle_build_with_symbolic_backend(
                    legacy_request,
                    BvpSymbolicAssemblyBackend::ExprLegacy,
                )
                .expect("ExprLegacy symbolic backend should build combustion sparse bundle");

            let mut atom_request_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_request = atom_request_solver.build_solver_request(None, None);
            let (mut atom_bundle, atom_bundle_ms) =
                measure_sparse_bundle_build_with_symbolic_backend(
                    atom_request,
                    BvpSymbolicAssemblyBackend::AtomView,
                )
                .expect("AtomView symbolic backend should build combustion sparse bundle");

            let args = DVector::from_vec(
                (0..legacy_bundle.variable_string.len())
                    .map(|index| 0.25 + index as f64 * 0.0025)
                    .collect(),
            );
            let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                &mut legacy_bundle,
                &mut atom_bundle,
                &args,
                "combustion-solver-stress-stats-1000",
            );

            let mut legacy_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_generate_begin = Instant::now();
            legacy_solver
                .try_eq_generate(None, None)
                .expect("ExprLegacy combustion generate should succeed");
            let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solve_begin = Instant::now();
            legacy_solver
                .try_solve()
                .expect("ExprLegacy combustion solve should succeed");
            let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let legacy_solution = legacy_solver
                .get_result()
                .expect("ExprLegacy combustion solve should produce a solution");

            let mut atom_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_generate_begin = Instant::now();
            atom_solver
                .try_eq_generate(None, None)
                .expect("AtomView combustion generate should succeed");
            let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solve_begin = Instant::now();
            atom_solver
                .try_solve()
                .expect("AtomView combustion solve should succeed");
            let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let atom_solution = atom_solver
                .get_result()
                .expect("AtomView combustion solve should produce a solution");

            let max_diff_vs_legacy = legacy_solution
                .iter()
                .zip(atom_solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);

            legacy_samples.push(StressSample {
                bundle_ms: legacy_bundle_ms,
                generate_ms: legacy_generate_ms,
                solve_ms: legacy_solve_ms,
                residual_max_diff: 0.0,
                jacobian_max_diff: 0.0,
                max_diff_vs_legacy: 0.0,
            });
            atom_samples.push(StressSample {
                bundle_ms: atom_bundle_ms,
                generate_ms: atom_generate_ms,
                solve_ms: atom_solve_ms,
                residual_max_diff,
                jacobian_max_diff,
                max_diff_vs_legacy,
            });
        }

        let legacy_bundle =
            compute_stage_stats(legacy_samples.iter().map(|row| row.bundle_ms).collect());
        let legacy_generate =
            compute_stage_stats(legacy_samples.iter().map(|row| row.generate_ms).collect());
        let legacy_solve =
            compute_stage_stats(legacy_samples.iter().map(|row| row.solve_ms).collect());

        let atom_bundle =
            compute_stage_stats(atom_samples.iter().map(|row| row.bundle_ms).collect());
        let atom_generate =
            compute_stage_stats(atom_samples.iter().map(|row| row.generate_ms).collect());
        let atom_solve = compute_stage_stats(atom_samples.iter().map(|row| row.solve_ms).collect());

        let atom_residual = compute_stage_stats(
            atom_samples
                .iter()
                .map(|row| row.residual_max_diff)
                .collect(),
        );
        let atom_jacobian = compute_stage_stats(
            atom_samples
                .iter()
                .map(|row| row.jacobian_max_diff)
                .collect(),
        );
        let atom_solution_diff = compute_stage_stats(
            atom_samples
                .iter()
                .map(|row| row.max_diff_vs_legacy)
                .collect(),
        );

        println!(
            "[BVP symbolic assembly solver stress stats] combustion ExprLegacy vs AtomView, n_steps=1000, runs={iterations}"
        );
        println!(
            "{:<12} | {:<8} | {:>10} | {:>10} | {:>10}",
            "backend", "stage", "min_ms", "median_ms", "max_ms"
        );
        println!("{}", "-".repeat(63));
        for (backend, stage, stats) in [
            ("ExprLegacy", "bundle", &legacy_bundle),
            ("ExprLegacy", "generate", &legacy_generate),
            ("ExprLegacy", "solve", &legacy_solve),
            ("AtomView", "bundle", &atom_bundle),
            ("AtomView", "generate", &atom_generate),
            ("AtomView", "solve", &atom_solve),
        ] {
            println!(
                "{:<12} | {:<8} | {:>10.3} | {:>10.3} | {:>10.3}",
                backend, stage, stats.min, stats.median, stats.max
            );
        }

        println!("[BVP symbolic assembly solver stress stats] AtomView numeric diffs, n_steps=300");
        println!(
            "residual_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
            atom_residual.min, atom_residual.median, atom_residual.max
        );
        println!(
            "jacobian_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
            atom_jacobian.min, atom_jacobian.median, atom_jacobian.max
        );
        println!(
            "solution_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
            atom_solution_diff.min, atom_solution_diff.median, atom_solution_diff.max
        );
    }

    #[test]
    #[ignore = "diagnostic combustion Rust-AOT-only compile-preset table; compares rustc presets, not C/Zig toolchains"]
    fn symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000() {
        #[derive(Debug)]
        struct Row {
            backend: &'static str,
            preset: &'static str,
            bootstrap_ms: f64,
            solve_ms: f64,
            max_abs_solution: f64,
        }

        let n_steps = 200usize;
        let mut rows = Vec::new();

        for (backend_label, symbolic_backend) in [
            ("ExprLegacy", BvpSymbolicAssemblyBackend::ExprLegacy),
            ("AtomView", BvpSymbolicAssemblyBackend::AtomView),
        ] {
            for (preset_label, config) in [
                (
                    "Production",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_production(),
                ),
                (
                    "FastBuild",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_fast_build(),
                ),
                (
                    "DevFastest",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_dev_fastest(),
                ),
            ] {
                let generated_backend_config = config
                    .with_symbolic_assembly_backend(symbolic_backend)
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                        profile:
                            crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                    });

                let mut solver = make_combustion_solver(n_steps, generated_backend_config);
                let label = format!(
                    "combustion-compile-preset-{}-{}-{}",
                    backend_label, preset_label, n_steps
                );
                let (_guard, bootstrap_ms, solve_ms) = solve_with_aot_and_measure(&mut solver, &label)
                    .expect("compile-preset combustion run should solve");
                let solution = solver
                    .get_result()
                    .expect("compile-preset combustion run should produce a solution");
                let max_abs_solution = solution.iter().copied().map(f64::abs).fold(0.0, f64::max);
                assert!(
                    solution.iter().all(|value| value.is_finite()),
                    "{backend_label} {preset_label} solution should remain finite"
                );

                rows.push(Row {
                    backend: backend_label,
                    preset: preset_label,
                    bootstrap_ms,
                    solve_ms,
                    max_abs_solution,
                });
            }
        }

        println!(
            "[BVP symbolic assembly Rust compile presets] combustion build-vs-runtime, n_steps={n_steps}"
        );
        println!(
            "note: this is intentionally Rust-AOT only; C/Zig toolchain comparisons live in the end-to-end and callback-throughput matrices."
        );
        println!(
            "{:<12} | {:<11} | {:>12} | {:>10} | {:>16}",
            "backend", "preset", "bootstrap_ms", "solve_ms", "max_abs_solution"
        );
        println!("{}", "-".repeat(74));
        for row in rows {
            println!(
                "{:<12} | {:<11} | {:>12.3} | {:>10.3} | {:>16.6e}",
                row.backend, row.preset, row.bootstrap_ms, row.solve_ms, row.max_abs_solution
            );
        }
    }

    #[test]
    #[ignore = "diagnostic combustion Lambdify vs AtomView DevFastest end-to-end matrix across Rust/gcc/tcc/zig"]
    fn combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000() {
        #[derive(Debug)]
        struct Row {
            backend: String,
            setup_ms: f64,
            solve_ms: f64,
            total_ms: f64,
            solution_diff: f64,
            max_abs_solution: f64,
        }

        // Keep this compact: it is an end-to-end toolchain comparison, while large-grid
        // stress belongs to the BVP_Damp_tests4 release matrices. Rust AOT can become a
        // compiler-stress test before the solver path is reached on larger generated crates.
        let n_steps = 200usize;

        let lambdify_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

        let mut lambdify_solver = make_combustion_solver(n_steps, lambdify_config);
        let lambdify_generate_begin = Instant::now();
        lambdify_solver
            .try_eq_generate(None, None)
            .expect("lambdify combustion generate should succeed");
        let lambdify_setup_ms = lambdify_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let lambdify_solve_begin = Instant::now();
        lambdify_solver
            .try_solve()
            .expect("lambdify combustion solve should succeed");
        let lambdify_solve_ms = lambdify_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let lambdify_solution = lambdify_solver
            .get_result()
            .expect("lambdify combustion solve should produce a solution");

        let mut rows = vec![Row {
            backend: "Lambdify".to_string(),
            setup_ms: lambdify_setup_ms,
            solve_ms: lambdify_solve_ms,
            total_ms: lambdify_setup_ms + lambdify_solve_ms,
            solution_diff: 0.0,
            max_abs_solution: lambdify_solution
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max),
        }];

        for toolchain in RuntimeTuningToolchain::variants() {
            let config = sparse_atomview_rebuild_release_devfastest_config(toolchain);
            let label = format!("AtomView+{}", toolchain.label());
            let mut atom_solver = make_combustion_solver(n_steps, config);
            let (_atom_guard, atom_setup_ms, atom_solve_ms) = solve_with_aot_and_measure(
                &mut atom_solver,
                &format!(
                    "combustion-atomview-{}-devfastest-vs-lambdify-{n_steps}",
                    toolchain.label()
                ),
            )
            .unwrap_or_else(|err| {
                panic!("{label} DevFastest combustion solve should succeed: {err:?}")
            });
            let atom_solution = atom_solver.get_result().unwrap_or_else(|| {
                panic!("{label} DevFastest combustion solve should produce a solution")
            });

            let solution_diff = lambdify_solution
                .iter()
                .zip(atom_solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                solution_diff < 1e-6,
                "{label}: Lambdify vs AtomView DevFastest disagreement {solution_diff} is too large"
            );

            rows.push(Row {
                backend: label,
                setup_ms: atom_setup_ms,
                solve_ms: atom_solve_ms,
                total_ms: atom_setup_ms + atom_solve_ms,
                solution_diff,
                max_abs_solution: atom_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
            });
        }

        println!(
            "[BVP end-to-end compare] combustion Lambdify vs AtomView DevFastest toolchain matrix, n_steps={n_steps}"
        );
        println!(
            "{:<16} | {:>12} | {:>10} | {:>10} | {:>14} | {:>16}",
            "backend", "setup_ms", "solve_ms", "total_ms", "diff_vs_base", "max_abs_solution"
        );
        println!("{}", "-".repeat(95));
        for row in rows {
            println!(
                "{:<16} | {:>12.3} | {:>10.3} | {:>10.3} | {:>14.6e} | {:>16.6e}",
                row.backend,
                row.setup_ms,
                row.solve_ms,
                row.total_ms,
                row.solution_diff,
                row.max_abs_solution
            );
        }
    }

    #[test]
    #[ignore = "diagnostic combustion callback-throughput matrix: Lambdify baseline vs linked AtomView Rust/gcc/tcc/zig"]
    fn combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000() {
        #[derive(Debug)]
        struct Row {
            backend: String,
            residual_ms: f64,
            jacobian_ms: f64,
            total_ms: f64,
            speedup_vs_lambdify: f64,
            residual_diff: f64,
            jacobian_diff: f64,
        }

        // This diagnostic is a runtime callback-binding guard, not a Rust compiler stress test.
        // Large Rust AOT artifacts can overflow rustc's stack before the test reaches the linked
        // chunk callbacks we want to validate here.
        let n_steps = 200usize;
        let iters = 20usize;
        // Cross-backend callback equivalence compares Lambdify against generated
        // cdylib code. Tiny libm/codegen ordering differences show up around
        // 1e-6 on this combustion fixture, while actual callback wiring bugs
        // were orders of magnitude larger.
        let cross_backend_callback_tol = 5.0e-6;

        let lambdify_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

        let mut lambdify_solver = make_combustion_solver(n_steps, lambdify_config);
        let mut lambdify_bundle = sparse_bundle_from_solver_request(&mut lambdify_solver)
            .expect("lambdify sparse bundle should build for callback throughput compare");
        assert_eq!(
            lambdify_bundle.effective_backend(),
            SelectedBackendKind::Lambdify,
            "lambdify callback throughput compare should stay on lambdify backend"
        );
        assert!(
            lambdify_bundle.is_runtime_callable(),
            "lambdify sparse bundle should be runtime-callable"
        );

        let args = DVector::from_element(lambdify_bundle.jacobian_shape().1, 0.99);
        let typed = Vectors_type_casting(
            &args,
            crate::symbolic::symbolic_functions_BVP::BvpMatrixBackend::FaerSparseCol
                .legacy_method()
                .to_string(),
        );

        let (lambdify_residual_ms, lambdify_jacobian_ms) =
            measure_sparse_runtime_callback_throughput(&mut lambdify_bundle, &*typed, iters);
        let lambdify_total_ms = lambdify_residual_ms + lambdify_jacobian_ms;

        let mut rows = vec![Row {
            backend: "Lambdify".to_string(),
            residual_ms: lambdify_residual_ms,
            jacobian_ms: lambdify_jacobian_ms,
            total_ms: lambdify_total_ms,
            speedup_vs_lambdify: 1.0,
            residual_diff: 0.0,
            jacobian_diff: 0.0,
        }];

        for toolchain in RuntimeTuningToolchain::variants() {
            let label = format!("AtomView+{}", toolchain.label());
            let config = sparse_atomview_rebuild_release_devfastest_config(toolchain);
            let mut atom_solver = make_combustion_solver(n_steps, config);
            let _atom_guard = bootstrap_callable_aot_backend(
                &mut atom_solver,
                &format!(
                    "combustion-linked-runtime-callback-throughput-{}-{n_steps}",
                    toolchain.label()
                ),
            )
            .unwrap_or_else(|err| {
                panic!("{label} bootstrap should succeed for callback throughput compare: {err:?}")
            });
            let mut atom_bundle = sparse_bundle_from_solver_request(&mut atom_solver)
                .unwrap_or_else(|err| {
                    panic!("{label} sparse bundle should rebuild after linked bootstrap: {err:?}")
                });
            assert_eq!(
                atom_bundle.effective_backend(),
                SelectedBackendKind::AotCompiled,
                "{label}: callback throughput compare should resolve to compiled AOT selection"
            );
            assert!(
                atom_bundle.is_runtime_callable(),
                "{label}: linked sparse bundle should be runtime-callable after bootstrap"
            );

            let (residual_diff, jacobian_diff) = compare_sparse_bundles_numerically(
                &mut lambdify_bundle,
                &mut atom_bundle,
                &args,
                &format!(
                    "combustion-callback-throughput-{}-{n_steps}",
                    toolchain.label()
                ),
            );
            assert!(
                residual_diff <= cross_backend_callback_tol
                    && jacobian_diff <= cross_backend_callback_tol,
                "{label}: runtime callback compare requires numerically close callbacks within {cross_backend_callback_tol:e}, got residual_diff={residual_diff}, jacobian_diff={jacobian_diff}"
            );

            let (atom_residual_ms, atom_jacobian_ms) =
                measure_sparse_runtime_callback_throughput(&mut atom_bundle, &*typed, iters);
            let atom_total_ms = atom_residual_ms + atom_jacobian_ms;

            rows.push(Row {
                backend: label,
                residual_ms: atom_residual_ms,
                jacobian_ms: atom_jacobian_ms,
                total_ms: atom_total_ms,
                speedup_vs_lambdify: lambdify_total_ms / atom_total_ms.max(f64::EPSILON),
                residual_diff,
                jacobian_diff,
            });
        }

        println!(
            "[BVP callback throughput] combustion Lambdify vs AtomView linked-runtime toolchain matrix, n_steps={n_steps}, iters={iters}"
        );
        println!(
            "[BVP callback throughput] note: AtomView+Linked now measures the generated cdylib runtime path loaded into the current process"
        );
        println!(
            "{:<16} | {:>12} | {:>12} | {:>12} | {:>18} | {:>13} | {:>13}",
            "backend",
            "residual_ms",
            "jacobian_ms",
            "total_ms",
            "speedup_vs_lambdify",
            "res_diff",
            "jac_diff"
        );
        println!("{}", "-".repeat(112));
        for row in rows {
            println!(
                "{:<16} | {:>12.3} | {:>12.3} | {:>12.3} | {:>18.3}x | {:>13.6e} | {:>13.6e}",
                row.backend,
                row.residual_ms,
                row.jacobian_ms,
                row.total_ms,
                row.speedup_vs_lambdify,
                row.residual_diff,
                row.jacobian_diff
            );
        }
    }

    #[test]
    #[ignore = "diagnostic sparse AOT whole-vs-chunked callback throughput across Rust/gcc/tcc/zig; isolates runtime parallelism from Newton/bootstrap cost"]
    fn combustion_sparse_aot_callback_chunking_parallelism_diagnostic() {
        #[derive(Debug)]
        struct CallbackChunkRow {
            config: String,
            bootstrap_ms: f64,
            rayon_workers: usize,
            residual_chunks: usize,
            sparse_chunks: usize,
            residual_jobs: usize,
            sparse_jobs: usize,
            residual_ms: RuntimeTuningAggregate,
            jacobian_ms: RuntimeTuningAggregate,
            total_ms: RuntimeTuningAggregate,
            speedup_vs_whole: RuntimeTuningAggregate,
            residual_diff_vs_whole: f64,
            jacobian_diff_vs_whole: f64,
        }

        fn sparse_aot_callback_config(
            toolchain: RuntimeTuningToolchain,
            execution_policy: AotExecutionPolicy,
            chunking_policy: AotChunkingPolicy,
        ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
            let config = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Debug,
                })
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(execution_policy)
                .with_aot_chunking_policy(chunking_policy);
            toolchain.apply_to(config)
        }

        fn linked_job_range_count(chunk_count: usize, max_jobs: usize) -> usize {
            if chunk_count == 0 {
                return 0;
            }
            let target_jobs = chunk_count.min(max_jobs.max(1));
            let items_per_job = chunk_count.div_ceil(target_jobs);
            chunk_count.div_ceil(items_per_job)
        }

        fn policy_max_jobs(policy: &AotExecutionPolicy) -> (usize, usize) {
            match policy {
                AotExecutionPolicy::Parallel(config) => {
                    let workers = rayon::current_num_threads().max(1);
                    let worker_jobs = workers.saturating_mul(config.jobs_per_worker.max(1));
                    (
                        config.max_residual_jobs.unwrap_or(worker_jobs).max(1),
                        config.max_sparse_jobs.unwrap_or(worker_jobs).max(1),
                    )
                }
                _ => (1, 1),
            }
        }

        fn linked_sparse_chunk_counts(
            bundle: &BvpSparseSolverBundle,
            residual_max_jobs: usize,
            sparse_max_jobs: usize,
        ) -> (usize, usize, usize, usize) {
            let Some(problem_key) = bundle
                .resolved_aot_artifact()
                .map(|artifact| artifact.registered.problem_key.as_str())
            else {
                return (0, 0, 0, 0);
            };
            let Some(linked) = resolve_linked_sparse_backend(problem_key) else {
                return (0, 0, 0, 0);
            };
            let residual_chunks = linked.residual_chunks.len();
            let sparse_chunks = linked.jacobian_value_chunks.len();
            (
                residual_chunks,
                sparse_chunks,
                linked_job_range_count(residual_chunks, residual_max_jobs),
                linked_job_range_count(sparse_chunks, sparse_max_jobs),
            )
        }

        // This diagnostic is a runtime callback-binding guard, not a Rust compiler stress test.
        // Large Rust AOT artifacts can overflow rustc's stack before the test reaches the linked
        // chunk callbacks we want to validate here.
        let n_steps = 200usize;
        let callback_iters = 30usize;
        let measurement_repeats = 5usize;
        let variants = [
            (
                "whole-sequential",
                AotExecutionPolicy::SequentialOnly,
                AotChunkingPolicy::default(),
            ),
            (
                "par-4x4-jobs4",
                AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(4),
                    max_sparse_jobs: Some(4),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                ),
            ),
            (
                "par-8x8-jobs8",
                AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(8),
                    max_sparse_jobs: Some(8),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ),
            ),
            (
                "par-16x16-jobs16",
                AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(16),
                    max_sparse_jobs: Some(16),
                    fallback_policy: ParallelFallbackPolicy::Never,
                }),
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                ),
            ),
        ];

        let mut rows = Vec::new();

        for toolchain in RuntimeTuningToolchain::variants() {
            let toolchain_label = toolchain.label();
            let mut whole_solver = make_combustion_solver(
                n_steps,
                sparse_aot_callback_config(
                    toolchain,
                    AotExecutionPolicy::SequentialOnly,
                    AotChunkingPolicy::default(),
                ),
            );
            let whole_bootstrap_begin = Instant::now();
            let _whole_guard = bootstrap_callable_aot_backend(
                &mut whole_solver,
                &format!("combustion-sparse-aot-callback-{toolchain_label}-whole-{n_steps}"),
            )
            .unwrap_or_else(|err| {
                panic!("{toolchain_label}/whole: sparse AOT callback diagnostic bootstrap failed: {err:?}")
            });
            let whole_bootstrap_ms = whole_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
            let mut whole_bundle =
                sparse_bundle_from_solver_request(&mut whole_solver).unwrap_or_else(|err| {
                    panic!("{toolchain_label}/whole: sparse AOT callback diagnostic bundle failed: {err:?}")
                });
            assert_eq!(
                whole_bundle.effective_backend(),
                SelectedBackendKind::AotCompiled,
                "{toolchain_label}/whole: sparse AOT callback diagnostic should use compiled backend"
            );

            let args = DVector::from_element(whole_bundle.jacobian_shape().1, 0.99);
            let typed = Vectors_type_casting(
                &args,
                crate::symbolic::symbolic_functions_BVP::BvpMatrixBackend::FaerSparseCol
                    .legacy_method()
                    .to_string(),
            );

            let mut whole_residual_samples = Vec::with_capacity(measurement_repeats);
            let mut whole_jacobian_samples = Vec::with_capacity(measurement_repeats);
            for _ in 0..measurement_repeats {
                let (residual_ms, jacobian_ms) = measure_sparse_runtime_callback_throughput(
                    &mut whole_bundle,
                    &*typed,
                    callback_iters,
                );
                whole_residual_samples.push(residual_ms);
                whole_jacobian_samples.push(jacobian_ms);
            }
            let whole_total_samples = whole_residual_samples
                .iter()
                .zip(whole_jacobian_samples.iter())
                .map(|(&residual_ms, &jacobian_ms)| residual_ms + jacobian_ms)
                .collect::<Vec<_>>();
            let whole_total_mean =
                runtime_tuning_aggregate(whole_total_samples.iter().copied()).mean;
            let (
                whole_residual_chunks,
                whole_sparse_chunks,
                whole_residual_jobs,
                whole_sparse_jobs,
            ) = linked_sparse_chunk_counts(&whole_bundle, 1, 1);

            rows.push(CallbackChunkRow {
                config: format!("{toolchain_label}/whole-sequential"),
                bootstrap_ms: whole_bootstrap_ms,
                rayon_workers: rayon::current_num_threads(),
                residual_chunks: whole_residual_chunks,
                sparse_chunks: whole_sparse_chunks,
                residual_jobs: whole_residual_jobs,
                sparse_jobs: whole_sparse_jobs,
                residual_ms: runtime_tuning_aggregate(whole_residual_samples.iter().copied()),
                jacobian_ms: runtime_tuning_aggregate(whole_jacobian_samples.iter().copied()),
                total_ms: runtime_tuning_aggregate(whole_total_samples.iter().copied()),
                speedup_vs_whole: runtime_tuning_aggregate([1.0]),
                residual_diff_vs_whole: 0.0,
                jacobian_diff_vs_whole: 0.0,
            });

            for (label, execution_policy, chunking_policy) in variants.iter().skip(1) {
                let (residual_max_jobs, sparse_max_jobs) = policy_max_jobs(execution_policy);
                let full_label = format!("{toolchain_label}/{label}");
                let mut solver = make_combustion_solver(
                    n_steps,
                    sparse_aot_callback_config(
                        toolchain,
                        execution_policy.clone(),
                        chunking_policy.clone(),
                    ),
                );
                let bootstrap_begin = Instant::now();
                let _guard = bootstrap_callable_aot_backend(
                    &mut solver,
                    &format!("combustion-sparse-aot-callback-{toolchain_label}-{label}-{n_steps}"),
                )
                .unwrap_or_else(|err| {
                    panic!("{full_label}: sparse AOT callback diagnostic bootstrap failed: {err:?}")
                });
                let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
                let mut bundle =
                    sparse_bundle_from_solver_request(&mut solver).unwrap_or_else(|err| {
                        panic!(
                            "{full_label}: sparse AOT callback diagnostic bundle failed: {err:?}"
                        )
                    });
                assert_eq!(
                    bundle.effective_backend(),
                    SelectedBackendKind::AotCompiled,
                    "{full_label}: sparse AOT callback diagnostic should use compiled backend"
                );

                let (residual_diff, jacobian_diff) = compare_sparse_bundles_numerically(
                    &mut whole_bundle,
                    &mut bundle,
                    &args,
                    &format!("combustion-sparse-aot-callback-{toolchain_label}-{label}-{n_steps}"),
                );
                assert!(
                    residual_diff < 1e-6 && jacobian_diff < 1e-6,
                    "{full_label}: chunked callback differs from whole callback, residual_diff={residual_diff}, jacobian_diff={jacobian_diff}"
                );

                let mut residual_samples = Vec::with_capacity(measurement_repeats);
                let mut jacobian_samples = Vec::with_capacity(measurement_repeats);
                for _ in 0..measurement_repeats {
                    let (residual_ms, jacobian_ms) = measure_sparse_runtime_callback_throughput(
                        &mut bundle,
                        &*typed,
                        callback_iters,
                    );
                    residual_samples.push(residual_ms);
                    jacobian_samples.push(jacobian_ms);
                }
                let total_samples = residual_samples
                    .iter()
                    .zip(jacobian_samples.iter())
                    .map(|(&residual_ms, &jacobian_ms)| residual_ms + jacobian_ms)
                    .collect::<Vec<_>>();
                let speedup_samples = total_samples
                    .iter()
                    .map(|&total_ms| whole_total_mean / total_ms.max(f64::EPSILON))
                    .collect::<Vec<_>>();
                let (residual_chunks, sparse_chunks, residual_jobs, sparse_jobs) =
                    linked_sparse_chunk_counts(&bundle, residual_max_jobs, sparse_max_jobs);
                assert!(
                    residual_chunks > 1,
                    "{full_label}: requested chunked sparse AOT residual execution, but linked backend registered only {residual_chunks} residual chunks"
                );
                assert!(
                    sparse_chunks > 1,
                    "{full_label}: requested chunked sparse AOT Jacobian execution, but linked backend registered only {sparse_chunks} Jacobian chunks"
                );
                assert!(
                    residual_jobs > 1,
                    "{full_label}: requested parallel residual execution, but runtime job planner produced only {residual_jobs} job"
                );
                assert!(
                    sparse_jobs > 1,
                    "{full_label}: requested parallel Jacobian execution, but runtime job planner produced only {sparse_jobs} job"
                );

                rows.push(CallbackChunkRow {
                    config: full_label,
                    bootstrap_ms,
                    rayon_workers: rayon::current_num_threads(),
                    residual_chunks,
                    sparse_chunks,
                    residual_jobs,
                    sparse_jobs,
                    residual_ms: runtime_tuning_aggregate(residual_samples.iter().copied()),
                    jacobian_ms: runtime_tuning_aggregate(jacobian_samples.iter().copied()),
                    total_ms: runtime_tuning_aggregate(total_samples.iter().copied()),
                    speedup_vs_whole: runtime_tuning_aggregate(speedup_samples.iter().copied()),
                    residual_diff_vs_whole: residual_diff,
                    jacobian_diff_vs_whole: jacobian_diff,
                });
            }
        }

        println!(
            "[BVP callback parallelism diagnostic] sparse AtomView AOT callbacks, n_steps={n_steps}, callback_iters={callback_iters}, measurement_repeats={measurement_repeats}"
        );
        println!(
            "note: this isolates residual/Jacobian callback evaluation; bootstrap_ms is reported only to expose artifact overhead and is not part of callback throughput."
        );
        println!(
            "{:<28} | {:>12} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<18} | {:>15} | {:>15}",
            "config",
            "bootstrap_ms",
            "workers",
            "res_ch",
            "jac_ch",
            "res_jobs",
            "jac_jobs",
            "residual_ms",
            "jacobian_ms",
            "callback_total_ms",
            "speedup_vs_whole",
            "residual_diff",
            "jacobian_diff",
        );
        println!("{}", "-".repeat(214));
        for row in rows {
            println!(
                "{:<28} | {:>12.3} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<18} | {:>15.6e} | {:>15.6e}",
                row.config,
                row.bootstrap_ms,
                row.rayon_workers,
                row.residual_chunks,
                row.sparse_chunks,
                row.residual_jobs,
                row.sparse_jobs,
                fmt_tuning_short(row.residual_ms),
                fmt_tuning_short(row.jacobian_ms),
                fmt_tuning_short(row.total_ms),
                fmt_tuning_short(row.speedup_vs_whole),
                row.residual_diff_vs_whole,
                row.jacobian_diff_vs_whole,
            );
        }
    }

    #[test]
    #[ignore = "diagnostic stage breakdown inside generate_ms for combustion symbolic backends"]
    fn symbolic_assembly_backends_report_combustion_generate_breakdown_table() {
        #[derive(Debug)]
        struct GenerateBreakdownRow {
            backend: BvpSymbolicAssemblyBackend,
            n_steps: usize,
            discretization_ms: f64,
            symbolic_jacobian_ms: f64,
            sparse_aot_prep_ms: f64,
            total_ms: f64,
        }

        let n_steps_list = [200usize, 300usize];
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();
        let mut rows = Vec::new();

        for &n_steps in &n_steps_list {
            let mut legacy_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_request = legacy_solver.build_solver_request(None, None);
            let legacy_snapshot = measure_symbolic_generation_breakdown_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
            )
            .expect("ExprLegacy combustion breakdown should build");

            let mut atom_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_request = atom_solver.build_solver_request(None, None);
            let atom_snapshot = measure_symbolic_generation_breakdown_with_symbolic_backend(
                atom_request,
                BvpSymbolicAssemblyBackend::AtomView,
            )
            .expect("AtomView combustion breakdown should build");

            for (backend, snapshot) in [
                (BvpSymbolicAssemblyBackend::ExprLegacy, legacy_snapshot),
                (BvpSymbolicAssemblyBackend::AtomView, atom_snapshot),
            ] {
                rows.push(GenerateBreakdownRow {
                    backend,
                    n_steps,
                    discretization_ms: snapshot
                        .get("discretization time")
                        .copied()
                        .unwrap_or_default()
                        * 1_000.0,
                    symbolic_jacobian_ms: snapshot
                        .get("symbolic jacobian time")
                        .copied()
                        .unwrap_or_default()
                        * 1_000.0,
                    sparse_aot_prep_ms: snapshot
                        .get("sparse AOT preparation time")
                        .copied()
                        .unwrap_or_default()
                        * 1_000.0,
                    total_ms: snapshot.get("total time, sec").copied().unwrap_or_default()
                        * 1_000.0,
                });
            }
        }

        println!("[BVP symbolic assembly generate breakdown] combustion ExprLegacy vs AtomView");
        println!(
            "{:<12} | {:>7} | {:>17} | {:>20} | {:>19} | {:>10}",
            "backend",
            "n_steps",
            "discretization_ms",
            "symbolic_jacobian_ms",
            "sparse_aot_prep_ms",
            "total_ms"
        );
        println!("{}", "-".repeat(104));
        for row in &rows {
            let backend = match row.backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            };
            println!(
                "{:<12} | {:>7} | {:>17.3} | {:>20.3} | {:>19.3} | {:>10.3}",
                backend,
                row.n_steps,
                row.discretization_ms,
                row.symbolic_jacobian_ms,
                row.sparse_aot_prep_ms,
                row.total_ms
            );
        }
    }

    #[test]
    #[ignore = "heavy oscillator end-to-end compare for banded lambdify baseline vs AtomView AOT with bootstrap/runtime split"]
    fn oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy() {
        #[derive(Debug)]
        struct Row {
            backend: &'static str,
            setup_ms: f64,
            solve_ms: f64,
            total_ms: f64,
            max_abs_solution: f64,
        }

        let n_steps = 1000usize;

        let lambdify_cfg =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                .with_matrix_backend_override(MatrixBackend::Banded);

        let atom_aot_cfg =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_matrix_backend_override(MatrixBackend::Banded);

        let mut lambdify_solver = make_oscillator_solver(n_steps, lambdify_cfg);
        let lambdify_generate_begin = Instant::now();
        lambdify_solver
            .try_eq_generate(None, None)
            .expect("banded lambdify oscillator generate should succeed");
        let lambdify_setup_ms = lambdify_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let lambdify_solve_begin = Instant::now();
        lambdify_solver
            .try_solve()
            .expect("banded lambdify oscillator solve should succeed");
        let lambdify_solve_ms = lambdify_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let lambdify_solution = lambdify_solver
            .get_result()
            .expect("banded lambdify oscillator solve should produce a solution");

        let mut atom_solver = make_oscillator_solver(n_steps, atom_aot_cfg);
        let (atom_setup_ms, atom_solve_ms, atom_solution) = match solve_with_aot_and_measure(
            &mut atom_solver,
            &format!("oscillator-atomview-aot-banded-vs-lambdify-{n_steps}"),
        ) {
            Ok((_guard, setup_ms, solve_ms)) => {
                let solution = atom_solver
                    .get_result()
                    .expect("banded AtomView AOT oscillator solve should produce a solution")
                    .clone();
                (setup_ms, solve_ms, solution)
            }
            Err(err) if is_aot_environment_issue(&err) => {
                eprintln!(
                    "Skipping heavy oscillator AOT compare due to environment/toolchain issue: {err:?}"
                );
                return;
            }
            Err(err) => {
                panic!("banded AtomView AOT oscillator compare failed unexpectedly: {err:?}")
            }
        };

        let solution_diff = lambdify_solution
            .iter()
            .zip(atom_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            solution_diff < 1e-5,
            "banded oscillator Lambdify vs AtomView AOT disagreement {solution_diff} is too large"
        );

        let rows = [
            Row {
                backend: "Lambdify",
                setup_ms: lambdify_setup_ms,
                solve_ms: lambdify_solve_ms,
                total_ms: lambdify_setup_ms + lambdify_solve_ms,
                max_abs_solution: lambdify_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
            },
            Row {
                backend: "AtomView+AOT",
                setup_ms: atom_setup_ms,
                solve_ms: atom_solve_ms,
                total_ms: atom_setup_ms + atom_solve_ms,
                max_abs_solution: atom_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
            },
        ];

        println!(
            "[BVP end-to-end compare] oscillator banded Lambdify vs AtomView AOT, n_steps={n_steps}"
        );
        println!(
            "{:<14} | {:>12} | {:>10} | {:>10} | {:>16}",
            "backend", "setup_ms", "solve_ms", "total_ms", "max_abs_solution"
        );
        println!("{}", "-".repeat(76));
        for row in rows {
            println!(
                "{:<14} | {:>12.3} | {:>10.3} | {:>10.3} | {:>16.6e}",
                row.backend, row.setup_ms, row.solve_ms, row.total_ms, row.max_abs_solution
            );
        }
        println!(
            "[BVP end-to-end compare] max_diff_lambdify_vs_atomview_aot = {:.6e}",
            solution_diff
        );
    }

    #[test]
    #[ignore = "diagnostic AOT crate emission/materialize/build comparison for combustion symbolic backends"]
    fn symbolic_assembly_backends_report_combustion_aot_crate_build_table() {
        let n_steps_list = [200usize, 300usize];
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();
        let mut rows = Vec::new();

        for &n_steps in &n_steps_list {
            let mut legacy_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            );
            let legacy_request = legacy_solver.build_solver_request(None, None);
            rows.push(
                measure_generated_crate_build_with_symbolic_backend(
                    legacy_request,
                    BvpSymbolicAssemblyBackend::ExprLegacy,
                    n_steps,
                    &format!("combustion-aot-expr-{n_steps}"),
                )
                .expect("ExprLegacy combustion AOT crate generation should succeed"),
            );

            let mut atom_solver = make_combustion_solver(
                n_steps,
                base_config
                    .clone()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
            );
            let atom_request = atom_solver.build_solver_request(None, None);
            rows.push(
                measure_generated_crate_build_with_symbolic_backend(
                    atom_request,
                    BvpSymbolicAssemblyBackend::AtomView,
                    n_steps,
                    &format!("combustion-aot-atom-{n_steps}"),
                )
                .expect("AtomView combustion AOT crate generation should succeed"),
            );
        }

        println!("[BVP symbolic assembly AOT crate build] combustion ExprLegacy vs AtomView");
        println!(
            "{:<12} | {:>7} | {:>11} | {:>11} | {:>11} | {:>8} | {:>11} | {:>10} | {:>14} | {:>10} | {:>10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:<18}",
            "backend",
            "n_steps",
            "jac_prep_ms",
            "lookup_ms",
            "jac_ms",
            "nnz",
            "finalize_ms",
            "module_ms",
            "source_ms",
            "materialize_ms",
            "build_ms",
            "source_kb",
            "blocks",
            "instr",
            "temps",
            "max_blk",
            "outputs",
            "status"
        );
        println!("{}", "-".repeat(234));
        for row in &rows {
            let backend = match row.backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            };
            println!(
                "{:<12} | {:>7} | {:>11} | {:>11} | {:>11} | {:>8} | {:>11} | {:>10} | {:>14} | {:>10} | {:>10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:<18}",
                backend,
                row.n_steps,
                row.jacobian_prepare_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_lookup_prepare_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_jacobian_build_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_nnz
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_finalize_codegen_plan_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.module_build_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.source_emit_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.materialize_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.build_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.source_kb
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.module_blocks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                row.total_block_instructions
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                row.total_block_temps
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                row.max_block_instructions
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                row.total_block_outputs
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                &row.status
            );
        }

        println!();
        println!("[BVP symbolic assembly AOT crate build] atom module pass breakdown");
        println!(
            "{:<12} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
            "backend",
            "n_steps",
            "res_view_ms",
            "res_lower_ms",
            "res_ph_ms",
            "res_reuse_ms",
            "sp_view_ms",
            "sp_lower_ms",
            "sp_ph_ms",
            "sp_reuse_ms"
        );
        println!("{}", "-".repeat(143));
        for row in rows
            .iter()
            .filter(|row| matches!(row.backend, BvpSymbolicAssemblyBackend::AtomView))
        {
            println!(
                "{:<12} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
                "AtomView",
                row.n_steps,
                row.atom_residual_view_collect_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_residual_lower_many_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_residual_peephole_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_residual_reuse_temps_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_view_collect_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_lower_many_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_peephole_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.atom_sparse_reuse_temps_ms
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
            );
        }
    }

    #[test]
    #[ignore = "diagnostic chunk-level IR compare for ExprLegacy vs AtomView combustion lowering"]
    fn symbolic_assembly_backends_report_combustion_chunk_ir_table() {
        let n_steps = 300usize;
        let base_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

        let mut legacy_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_request = legacy_solver.build_solver_request(None, None);
        let (legacy_module, legacy_breakdown) = measure_codegen_module_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        )
        .expect("legacy combustion codegen module should build");

        let mut atom_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_request = atom_solver.build_solver_request(None, None);
        let (atom_module, atom_breakdown) = measure_codegen_module_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        )
        .expect("atom combustion codegen module should build");

        assert_eq!(
            legacy_module.blocks().len(),
            atom_module.blocks().len(),
            "legacy and atom module should produce the same number of blocks"
        );

        let rows = legacy_module
            .blocks()
            .iter()
            .zip(atom_module.blocks().iter())
            .map(|(legacy, atom)| ChunkIrCompareRow {
                fn_name: legacy.fn_name.clone(),
                outputs: legacy.output_count(),
                legacy_instr: legacy.instruction_count(),
                atom_instr: atom.instruction_count(),
                legacy_temps: legacy.temp_count(),
                atom_temps: atom.temp_count(),
            })
            .collect::<Vec<_>>();

        println!(
            "[BVP symbolic assembly chunk IR compare] combustion, n_steps={n_steps}, legacy_instr_total={}, atom_instr_total={}, legacy_temps_total={}, atom_temps_total={}",
            legacy_breakdown.total_block_instructions,
            atom_breakdown.total_block_instructions,
            legacy_breakdown.total_block_temps,
            atom_breakdown.total_block_temps
        );
        println!(
            "{:<36} | {:>7} | {:>12} | {:>10} | {:>12} | {:>10}",
            "fn_name", "outputs", "legacy_instr", "atom_instr", "legacy_temps", "atom_temps"
        );
        println!("{}", "-".repeat(103));
        for row in rows.iter().take(12) {
            println!(
                "{:<36} | {:>7} | {:>12} | {:>10} | {:>12} | {:>10}",
                row.fn_name,
                row.outputs,
                row.legacy_instr,
                row.atom_instr,
                row.legacy_temps,
                row.atom_temps
            );
        }
    }

    #[test]
    //  #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps() {
        println!("[AOT acceptance] default Rust AOT smoke: exact examples, sequential execution");
        let configs = [
            (
                "two-point-64",
                NonlinEquation::TwoPointBVP,
                64usize,
                1.5e-2f64,
            ),
            (
                "two-point-240",
                NonlinEquation::TwoPointBVP,
                240usize,
                6.0e-3f64,
            ),
            ("clairaut-72", NonlinEquation::Clairaut, 72usize, 1.5e-2f64),
            (
                "clairaut-220",
                NonlinEquation::Clairaut,
                220usize,
                1.25e-2f64,
            ),
        ];

        for (label, equation, n_steps, max_tol) in configs {
            let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
            let mut solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                generated_backend_config,
            );
            let _guard = solve_with_aot_and_report(&mut solver, label)
                .expect("sequential AOT exact-example acceptance case should solve");

            let error = match equation {
                NonlinEquation::TwoPointBVP => {
                    max_abs_error_against_exact(&solver, |x| (-x * x / 4.0).exp())
                }
                NonlinEquation::Clairaut => l2_error_against_exact(&solver, |x| {
                    1.0 + (x - 1.0).powi(2) - (x - 1.0).powi(3) / 6.0 + (x - 1.0).powi(4) / 12.0
                }),
                _ => unreachable!("only exact sequential examples are expected here"),
            };

            println!("[AOT exact sequential] {label}: n_steps={n_steps}, error={error:.6e}");
            assert!(
                error < max_tol,
                "{label}: exact-solution error {error} exceeded tolerance {max_tol}"
            );
        }
    }

    #[test]
    //   #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_rust_default_parallel_exact_examples_cover_parallel_modes_and_chunking() {
        println!(
            "[AOT acceptance] default Rust AOT smoke: exact examples, parallel/chunked execution"
        );
        let lane_parallel = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: None,
        };
        let parachute_parallel = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: None,
        };

        let cases = [
            (
                "parachute-parallel-48",
                NonlinEquation::ParachuteEquation,
                48usize,
                parachute_parallel.clone(),
                5.0e-3f64,
            ),
            (
                "lane-emden-parallel-180",
                NonlinEquation::LaneEmden5,
                180usize,
                lane_parallel.clone(),
                3.5e-4f64,
            ),
        ];

        for (label, equation, n_steps, strategy_params, l2_tol) in cases {
            let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(sparse_parallel_policy())
                    .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                        Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    ));
            let mut solver = make_example_solver(
                &equation,
                n_steps,
                Some(strategy_params),
                generated_backend_config,
            );
            let _guard = solve_with_aot_and_report(&mut solver, label)
                .expect("parallel AOT exact-example acceptance case should solve");

            let l2_error = match equation {
                NonlinEquation::ParachuteEquation => {
                    l2_error_against_exact(&solver, |x| (((2.0 * x).exp() + 1.0) / 2.0).ln() - x)
                }
                NonlinEquation::LaneEmden5 => {
                    l2_error_against_exact(&solver, |x| (1.0 + x * x / 3.0).powf(-0.5))
                }
                _ => unreachable!("only parallel analytical examples are expected here"),
            };

            println!("[AOT exact parallel] {label}: n_steps={n_steps}, l2_error={l2_error:.6e}");
            assert!(
                l2_error < l2_tol,
                "{label}: L2 exact-solution error {l2_error} exceeded tolerance {l2_tol}"
            );
        }
    }

    #[test]
    // #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids() {
        println!("[AOT acceptance] default Rust AOT smoke: combustion sequential/parallel grids");
        let sequential_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
        let parallel_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(sparse_parallel_policy())
                .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ));

        let mut sequential = make_combustion_solver(36, sequential_config.clone());
        let _sequential_guard =
            solve_with_aot_and_report(&mut sequential, "combustion-sequential-36")
                .expect("sequential AOT combustion case should solve");
        let sequential_solution = sequential
            .get_result()
            .expect("sequential AOT combustion case should produce a solution");
        assert!(
            sequential_solution.iter().all(|value| value.is_finite()),
            "sequential AOT combustion solution should remain finite"
        );

        let mut parallel = make_combustion_solver(36, parallel_config.clone());
        let _parallel_guard = solve_with_aot_and_report(&mut parallel, "combustion-parallel-36")
            .expect("parallel AOT combustion case should solve");
        let parallel_solution = parallel
            .get_result()
            .expect("parallel AOT combustion case should produce a solution");
        assert!(
            parallel_solution.iter().all(|value| value.is_finite()),
            "parallel AOT combustion solution should remain finite"
        );

        let max_difference = sequential_solution
            .iter()
            .zip(parallel_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        println!(
            "[AOT combustion compare] n_steps=36, max_difference_seq_vs_par={max_difference:.6e}"
        );
        assert!(
            max_difference < 1.0e-4,
            "AOT combustion sequential/parallel disagreement {max_difference} is too large"
        );

        let mut large_sequential = make_combustion_solver(256, sequential_config);
        let _large_sequential_guard =
            solve_with_aot_and_report(&mut large_sequential, "combustion-sequential-256")
                .expect("large-grid sequential AOT combustion case should solve");
        let large_sequential_solution = large_sequential
            .get_result()
            .expect("large-grid sequential AOT combustion case should produce a solution");
        assert!(
            large_sequential_solution
                .iter()
                .all(|value| value.is_finite()),
            "large-grid sequential AOT combustion solution should remain finite"
        );

        let mut large_parallel = make_combustion_solver(256, parallel_config);
        let _large_parallel_guard =
            solve_with_aot_and_report(&mut large_parallel, "combustion-parallel-256")
                .expect("large-grid parallel AOT combustion case should solve");
        let large_solution = large_parallel
            .get_result()
            .expect("large-grid AOT combustion case should produce a solution");
        assert!(
            large_solution.iter().all(|value| value.is_finite()),
            "large-grid parallel AOT combustion solution should remain finite"
        );

        let large_max_difference = large_sequential_solution
            .iter()
            .zip(large_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        println!(
            "[AOT combustion compare] n_steps=256, max_difference_seq_vs_par={large_max_difference:.6e}"
        );
        assert!(
            large_max_difference < 1.0e-4,
            "AOT combustion sequential/parallel disagreement {large_max_difference} is too large"
        );
    }

    #[test]
    fn aot_tcc_smoke_exact_two_point_small_grid_solves() {
        println!(
            "[AOT acceptance] compact TCC smoke: two-point exact BVP, sparse AtomView, sequential execution"
        );
        let generated_backend_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
        let mut solver = make_example_solver(
            &NonlinEquation::TwoPointBVP,
            40,
            Some(SolverParams::default()),
            generated_backend_config,
        );

        match solve_with_aot_and_report(&mut solver, "tcc-smoke-two-point-40") {
            Ok(_guard) => {
                let error = max_abs_error_against_exact(&solver, |x| (-x * x / 4.0).exp());
                println!("[AOT TCC smoke] two-point-40: error={error:.6e}");
                assert!(
                    error < 2.0e-2,
                    "TCC smoke exact-solution error {error} exceeded tolerance"
                );
            }
            Err(err) if is_aot_environment_issue(&err) => {
                eprintln!(
                    "[AOT TCC smoke] skipped because TCC toolchain/artifact environment is unavailable: {err:?}"
                );
            }
            Err(err) => panic!("TCC smoke should solve or report an environment issue: {err:?}"),
        }
    }
    // Historical local timing notes kept only as archaeological context. The live
    // source of truth is the multi-run, multi-toolchain table printed by
    // `aot_combustion_parallel_tuning_reports_runtime_table` and summarized in
    // BVP_DAMP_STORY_TESTS.md.
    /*
            слишком мало чанков: не хватает загрузки
        слишком много чанков: overhead на orchestration начинает съедать выигрыш
        средний режим вроде 8x8 + jobs8 попадает в sweet spot
        И очень важно:

        max_diff_vs_seq = 0 у всех конфигураций
        это именно то, что и нужно было доказать: parallel AOT меняет скорость, но не математику.
        CPU 4 Cores
        [AOT combustion tuning map] scenario=medium-grid, n_steps=128
        config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
        ------------------------------------------------------------------------------------------------
        sequential-baseline      |     128 |     7113.999 |      440.910 |          1.000 |     0.000000e0
        par-4x4-jobs4            |     128 |    15332.448 |      335.703 |          1.313 |     0.000000e0
        par-8x8-jobs8            |     128 |    10177.598 |     1116.167 |          0.395 |     0.000000e0
        par-16x16-jobs16         |     128 |     8069.948 |      261.911 |          1.683 |     0.000000e0
        par-res16-row32-jobs8    |     128 |     6763.878 |      280.503 |          1.572 |     0.000000e0
        [AOT tuning winner] scenario=medium-grid, config=par-16x16-jobs16, n_steps=128, solve_ms=261.911, speedup_vs_seq=1.683, bootstrap_ms=8069.948

        [AOT combustion tuning map] scenario=large-grid, n_steps=256
        config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
        ------------------------------------------------------------------------------------------------
        sequential-baseline      |     256 |    15992.826 |      803.811 |          1.000 |     0.000000e0
        par-4x4-jobs4            |     256 |    36969.672 |     1046.569 |          0.768 |     0.000000e0
        par-8x8-jobs8            |     256 |    22350.296 |      510.894 |          1.573 |     0.000000e0
        par-16x16-jobs16         |     256 |    18268.103 |      956.696 |          0.840 |     0.000000e0
        par-res16-row32-jobs8    |     256 |    13534.274 |      593.440 |          1.354 |     0.000000e0
        [AOT tuning winner] scenario=large-grid, config=par-8x8-jobs8, n_steps=256, solve_ms=510.894, speedup_vs_seq=1.573, bootstrap_ms=22350.296
            CPU 8 Core:
            run 1
        [AOT combustion tuning map] scenario=medium-grid, n_steps=128
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     128 |     2130.826 |      137.532 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     128 |     3685.552 |      116.561 |          1.180 |     0.000000e0
    par-8x8-jobs8            |     128 |     2446.450 |      104.325 |          1.318 |     0.000000e0
    par-16x16-jobs16         |     128 |     1870.474 |      116.204 |          1.184 |     0.000000e0
    par-res16-row32-jobs8    |     128 |     1807.076 |      109.260 |          1.259 |     0.000000e0
    [AOT tuning winner] scenario=medium-grid, config=par-8x8-jobs8, n_steps=128, solve_ms=104.325, speedup_vs_seq=1.318, bootstrap_ms=2446.450

    [AOT combustion tuning map] scenario=large-grid, n_steps=256
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     256 |     3382.138 |      203.362 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     256 |    12063.711 |      172.593 |          1.178 |     0.000000e0
    par-8x8-jobs8            |     256 |     7365.829 |      168.625 |          1.206 |     0.000000e0
    par-16x16-jobs16         |     256 |     4900.391 |      170.448 |          1.193 |     0.000000e0
    par-res16-row32-jobs8    |     256 |     3832.109 |      168.844 |          1.204 |     0.000000e0
    [AOT tuning winner] scenario=large-grid, config=par-8x8-jobs8, n_steps=256, solve_ms=168.625, speedup_vs_seq=1.206, bootstrap_ms=7365.829
    run 2
    [AOT combustion tuning map] scenario=medium-grid, n_steps=128
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     128 |     2044.555 |      127.818 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     128 |     3723.176 |      115.799 |          1.104 |     0.000000e0
    par-8x8-jobs8            |     128 |     2492.751 |      108.581 |          1.177 |     0.000000e0
    par-16x16-jobs16         |     128 |     1891.942 |      107.237 |          1.192 |     0.000000e0
    par-res16-row32-jobs8    |     128 |     1794.337 |      107.782 |          1.186 |     0.000000e0
    [AOT tuning winner] scenario=medium-grid, config=par-16x16-jobs16, n_steps=128, solve_ms=107.237, speedup_vs_seq=1.192, bootstrap_ms=1891.942

    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     256 |     3503.787 |      188.826 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     256 |    12043.939 |      167.994 |          1.124 |     0.000000e0
    par-8x8-jobs8            |     256 |     7444.171 |      167.913 |          1.125 |     0.000000e0
    par-16x16-jobs16         |     256 |     5047.130 |      164.290 |          1.149 |     0.000000e0
    par-res16-row32-jobs8    |     256 |     3913.400 |      176.590 |          1.069 |     0.000000e0
    [AOT tuning winner] scenario=large-grid, config=par-16x16-jobs16, n_steps=256, solve_ms=164.290, speedup_vs_seq=1.149, bootstrap_ms=5047.130

            */
    #[test]
    fn isolated_tuning_solution_payload_round_trips() {
        let solution = DMatrix::from_column_slice(2, 3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let encoded = format!(
            "{ISOLATED_TUNING_SOLUTION_MARKER}\t{}",
            encode_isolated_tuning_solution(&solution)
        );
        assert_eq!(
            decode_isolated_tuning_solution(&encoded),
            solution.iter().copied().collect::<Vec<_>>()
        );
        let metrics = IsolatedColdMetrics {
            total_timer_ms: 1.0,
            symbolic_ms: 2.0,
            linear_ms: 3.0,
            jac_ms: 4.0,
            fun_ms: 5.0,
            cb_residual_values_ms: 6.0,
            cb_jacobian_values_ms: 7.0,
            cb_jacobian_assembly_ms: 8.0,
            residual_actual_jobs: 4.0,
            sparse_jacobian_actual_jobs: 4.0,
            initial_symbolic_jacobian_ms: 9.0,
            post_build_rebind_ms: 10.0,
            aot_artifact_ms: 11.0,
            aot_materialize_ms: 12.0,
            aot_compile_link_ms: 13.0,
            aot_register_link_ms: 14.0,
        };
        let encoded = format!(
            "{ISOLATED_TUNING_METRICS_MARKER}\t{}",
            encode_isolated_cold_metrics(&metrics)
        );
        let decoded = decode_isolated_cold_metrics(&encoded);
        assert_eq!(decoded.initial_symbolic_jacobian_ms, 9.0);
        assert_eq!(decoded.aot_compile_link_ms, 13.0);
        assert_eq!(decoded.residual_actual_jobs, 4.0);
    }

    #[test]
    #[ignore = "heavy combustion sparse AOT runtime tuning map across Rust/gcc/tcc/zig and chunking policies; cold wall-clock rows run in isolated child processes"]
    fn aot_combustion_parallel_tuning_reports_runtime_table() {
        if let Ok(index) = std::env::var(ISOLATED_TUNING_CHILD_INDEX_ENV) {
            let index = index
                .parse::<usize>()
                .expect("isolated tuning child index should be an integer");
            let n_steps = std::env::var(ISOLATED_TUNING_CHILD_STEPS_ENV)
                .expect("isolated tuning child should receive n_steps")
                .parse::<usize>()
                .expect("isolated tuning n_steps should be an integer");
            run_isolated_tuning_child(index, n_steps);
            return;
        }
        run_combustion_tuning_scenario(1000, 4, "medium-grid-multi-toolchain")
            .expect("medium-grid multi-run AOT combustion tuning scenario should solve");
    }

    #[test]
    #[ignore = "practical isolated cold combustion tuning table for Lambdify versus tcc chunking strategies"]
    fn combustion_tcc_chunking_honest_wall_clock_table() {
        let n_steps = 1_000usize;
        let repetitions = 3usize;
        let variants = runtime_tuning_cold_variants();
        let selected = variants
            .iter()
            .enumerate()
            .filter(|(_, (label, _))| label == "lambdify-baseline" || label.starts_with("tcc/"))
            .map(|(index, (label, _))| (index, label.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            selected.len(),
            1 + 1 + runtime_tuning_parallel_cases().len(),
            "narrow cold table should contain Lambdify and every tcc chunking policy"
        );
        println!(
            "[AOT tcc practical cold map] n_steps={n_steps}, repetitions={repetitions}, cooldown_ms={}, cleanup_child_artifacts={}",
            cold_cooldown_ms(),
            clean_cold_artifacts_enabled()
        );

        let mut samples: Vec<(String, IsolatedColdObservation, f64)> = Vec::new();
        for repetition in 0..repetitions {
            let (baseline_index, baseline_label) = &selected[0];
            let baseline = solve_isolated_cold_tuning_variant(*baseline_index, n_steps);
            print_isolated_cold_raw_observation(repetition, baseline_label, &baseline);
            let baseline_solution = baseline.solution.clone();
            samples.push((baseline_label.clone(), baseline, 0.0));
            for (index, label) in selected.iter().skip(1) {
                let observation = solve_isolated_cold_tuning_variant(*index, n_steps);
                print_isolated_cold_raw_observation(repetition, label, &observation);
                let max_diff = baseline_solution
                    .iter()
                    .zip(observation.solution.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_diff < 1.0e-4,
                    "{label}: isolated cold result differs from Lambdify by {max_diff}"
                );
                samples.push((label.clone(), observation, max_diff));
            }
        }

        println!();
        println!("[AOT tcc practical cold map] correctness and wall-clock table");
        println!(
            "{:<26} | {:<22} | {:<18} | {:<18} | {:<18}",
            "config", "honest_e2e_ms [min,max]", "max_diff", "symbolic_ms", "initial_sym_jac"
        );
        println!("{}", "-".repeat(114));
        for (_, label) in &selected {
            let rows = samples
                .iter()
                .filter(|(sample_label, _, _)| sample_label == label)
                .collect::<Vec<_>>();
            let total = runtime_tuning_aggregate(rows.iter().map(|(_, row, _)| row.elapsed_ms));
            let diff = runtime_tuning_aggregate(rows.iter().map(|(_, _, diff)| *diff));
            let symbolic =
                runtime_tuning_aggregate(rows.iter().map(|(_, row, _)| row.metrics.symbolic_ms));
            let sym_jac = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.initial_symbolic_jacobian_ms),
            );
            println!(
                "{:<26} | {:<22} | {:<18} | {:<18} | {:<18}",
                label,
                fmt_tuning_agg(total),
                fmt_tuning_exp(diff),
                fmt_tuning_short(symbolic),
                fmt_tuning_short(sym_jac)
            );
        }

        println!();
        println!(
            "[AOT tcc practical cold map] build and callback stages from the same child solves"
        );
        println!(
            "{:<26} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12}",
            "config",
            "materialize_ms",
            "compile_link_ms",
            "residual_values",
            "jacobian_values",
            "res_jobs",
            "jac_jobs"
        );
        println!("{}", "-".repeat(132));
        for (_, label) in &selected {
            let rows = samples
                .iter()
                .filter(|(sample_label, _, _)| sample_label == label)
                .collect::<Vec<_>>();
            let materialize = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.aot_materialize_ms),
            );
            let compile_link = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.aot_compile_link_ms),
            );
            let residual = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.cb_residual_values_ms),
            );
            let jacobian = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.cb_jacobian_values_ms),
            );
            let res_jobs = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.residual_actual_jobs),
            );
            let jac_jobs = runtime_tuning_aggregate(
                rows.iter()
                    .map(|(_, row, _)| row.metrics.sparse_jacobian_actual_jobs),
            );
            println!(
                "{:<26} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12}",
                label,
                fmt_tuning_short(materialize),
                fmt_tuning_short(compile_link),
                fmt_tuning_short(residual),
                fmt_tuning_short(jacobian),
                fmt_tuning_short(res_jobs),
                fmt_tuning_short(jac_jobs)
            );
        }
    }

    /// Measures what fraction of total solve time is spent in residual+Jacobian eval
    /// vs the linear solve and other Newton overhead.
    ///
    /// If eval_fraction is small (< 0.3), parallelising the eval cannot give
    /// more than 1/(1 - eval_fraction) speedup by Amdahl's law regardless of
    /// how many cores are used.  That is the root cause of the 1.2x ceiling.
    ///
    /// Run with: cargo test diagnose_eval_fraction -- --nocapture
    #[test]
    fn diagnose_eval_fraction_of_solve_time() {
        /*
         use std::hint::black_box;
        use crate::numerical::BVP_Damp::BVP_traits::Jac;
         let n_steps_list = [128usize, 256, 512, 1000];
         let iters = 50usize;

         println!(
             "\n=== Eval fraction of solve time (combustion, iters={iters}) ==="
         );
         println!(
             "{:<8} {:<10} {:<10} {:<12} {:<12} {:<14} {:<14}",
             "n_steps", "vars", "nnz",
             "full_ms", "eval_ms", "eval_frac", "amdahl_ceil"
         );

         for n_steps in n_steps_list {
             let sequential_config =
                 crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                     .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
             let mut solver = make_combustion_solver(n_steps, sequential_config);
             let _guard = bootstrap_callable_aot_backend(
                 &mut solver,
                 &format!("eval-fraction-{n_steps}"),
             )
             .expect("bootstrap should succeed");

             // Full solve timing.
             let t_full = Instant::now();
             solver.try_solve().expect("solve should succeed");
             let full_ms = t_full.elapsed().as_secs_f64() * 1_000.0;

             // Isolate eval cost: call fun+jac directly iters times on the
             // converged solution so we measure hot-path eval, not convergence.
             let solution = solver
                 .get_result()
                 .expect("solution should be present after solve");
             let flat: Vec<f64> = solution.as_slice().to_vec();
             let col = faer::col::ColRef::from_slice(&flat).to_owned();

             let fun = &solver.fun;
             let jac: & Box<dyn Jac> = solver.jac.as_ref().expect("jac should be present");

             let t_eval = Instant::now();
             for _ in 0..iters {
                 let r = fun.call(0.0, &col);
                 black_box(r.len());
                 let j = jac.call(0.0, &col);
                 black_box(j.shape());
             }
             let eval_ms = t_eval.elapsed().as_secs_f64() * 1_000.0 / iters as f64;

             // Estimate Newton iterations from solve time and per-eval cost.
             let vars = 6 * n_steps;
             // nnz estimate: banded structure ~12 nonzeros per row for this problem
             let nnz_est = vars * 12;
             let eval_frac = eval_ms / (full_ms / iters as f64).max(eval_ms);
             // Amdahl ceiling: max speedup if eval is perfectly parallelised
             let amdahl_ceil = if eval_frac >= 1.0 {
                 f64::INFINITY
             } else {
                 1.0 / (1.0 - eval_frac)
             };

             println!(
                 "{:<8} {:<10} {:<10} {:<12.3} {:<12.3} {:<14.3} {:<14.2}",
                 n_steps, vars, nnz_est,
                 full_ms, eval_ms, eval_frac, amdahl_ceil
             );

         }
         */
    }
    /// THE MOST IMPOTANT TEST FOR LINEAR SOLVERS COMPARISON: runs a full combustion-1000 eval+linear-solve and compares
    #[test]
    #[ignore = "heavy combustion-1000 linear-system story for sparse baseline vs consistent superblock solver"]
    fn combustion_1000_linear_system_story_sparse_vs_banded_consistent() {
        #[derive(Debug)]
        struct Row {
            source: &'static str,
            matrix_backend: &'static str,
            variant: String,
            linear_solver: String,
            bootstrap_ms: f64,
            residual_diff: f64,
            jacobian_diff: f64,
            sparse_ms: f64,
            banded_ms: f64,
            layout: String,
            refinement: String,
            direct_rr: f64,
            final_rr: f64,
            solve_rr: f64,
            solve_diff: f64,
            relative_x_diff: f64,
            status: String,
        }

        let n_steps = 1000usize;
        let mut rows = Vec::new();
        println!(
            "[BVP Damp story] starting combustion-1000 linear-system comparison across sparse/banded backends"
        );

        let lambdify_sparse_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                .with_matrix_backend_override(MatrixBackend::SparseCol);

        println!("[BVP Damp story] bootstrapping lambdify sparse baseline");
        let mut baseline_solver = make_combustion_solver(n_steps, lambdify_sparse_config);
        let baseline_bootstrap_begin = Instant::now();
        baseline_solver
            .try_eq_generate(None, None)
            .expect("lambdify sparse combustion-1000 bootstrap should succeed");
        let baseline_bootstrap_ms = baseline_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

        let baseline_args = flattened_initial_guess_state(
            &baseline_solver,
            baseline_solver.values.len() * n_steps,
            "combustion-1000-lambdify-sparse",
        );
        let (baseline_residual, baseline_dense_matrix, baseline_sparse_matrix, _) =
            eval_solver_callback_state(
                &mut baseline_solver,
                &baseline_args,
                MatrixBackend::SparseCol,
                "combustion-1000-lambdify-sparse",
            );
        let baseline_sparse_matrix =
            baseline_sparse_matrix.expect("sparse baseline should produce SparseColMat");
        let baseline_rhs: Vec<f64> = baseline_residual.iter().map(|value| -*value).collect();

        let baseline_sparse_begin = Instant::now();
        let baseline_sparse_solution =
            solve_sparse_lu_for_rhs(&baseline_sparse_matrix, baseline_rhs.as_slice());
        let baseline_sparse_ms = baseline_sparse_begin.elapsed().as_secs_f64() * 1_000.0;
        rows.push(Row {
            source: "Lambdify",
            matrix_backend: "Sparse",
            variant: "ExprLegacy".to_string(),
            linear_solver: "faer_sparse_lu".to_string(),
            bootstrap_ms: baseline_bootstrap_ms,
            residual_diff: 0.0,
            jacobian_diff: 0.0,
            sparse_ms: baseline_sparse_ms,
            banded_ms: 0.0,
            layout: "-".to_string(),
            refinement: "-".to_string(),
            direct_rr: 0.0,
            final_rr: 0.0,
            solve_rr: relative_dense_residual(
                &baseline_dense_matrix,
                &baseline_sparse_solution,
                baseline_rhs.as_slice(),
            ),
            solve_diff: 0.0,
            relative_x_diff: 0.0,
            status: "ok".to_string(),
        });

        let baseline_dense_banded_assembly =
            banded_assembly_from_dense_matrix(&baseline_dense_matrix);
        for solver_choice in BandedStorySolver::variants() {
            let banded_begin = Instant::now();
            let metrics = solve_banded_story_for_rhs(
                &baseline_dense_banded_assembly,
                n_steps,
                baseline_rhs.as_slice(),
                solver_choice,
            );
            let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
            let solve_diff = metrics
                .solution
                .as_ref()
                .map(|solution| {
                    solution
                        .iter()
                        .zip(baseline_sparse_solution.iter())
                        .map(|(lhs, rhs)| (lhs - rhs).abs())
                        .fold(0.0_f64, f64::max)
                })
                .unwrap_or(f64::NAN);
            let rel_x = metrics
                .solution
                .as_ref()
                .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
                .unwrap_or(f64::NAN);
            let solve_rr = metrics
                .solution
                .as_ref()
                .map(|solution| {
                    relative_dense_residual(
                        &baseline_dense_matrix,
                        solution,
                        baseline_rhs.as_slice(),
                    )
                })
                .unwrap_or(f64::NAN);
            rows.push(Row {
                source: "Derived",
                matrix_backend: "Sparse->Banded",
                variant: "DenseBaseline".to_string(),
                linear_solver: metrics.linear_solver,
                bootstrap_ms: 0.0,
                residual_diff: 0.0,
                jacobian_diff: 0.0,
                sparse_ms: 0.0,
                banded_ms,
                layout: metrics.layout,
                refinement: metrics
                    .report
                    .as_ref()
                    .map(|report| {
                        format!(
                            "{}/{}{}",
                            report.accepted_steps,
                            report.requested_steps,
                            if report.refinement_attempted {
                                ""
                            } else {
                                " skipped"
                            }
                        )
                    })
                    .unwrap_or_else(|| "-".to_string()),
                direct_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.direct_relative_residual)
                    .unwrap_or(f64::NAN),
                final_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.final_relative_residual)
                    .unwrap_or(f64::NAN),
                solve_rr,
                solve_diff,
                relative_x_diff: rel_x,
                status: if metrics.status == "ok" {
                    "diag".to_string()
                } else {
                    metrics.status
                },
            });
        }

        let lambdify_banded_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                .with_matrix_backend_override(MatrixBackend::Banded);
        println!("[BVP Damp story] bootstrapping lambdify banded path");
        let mut lambdify_banded_solver = make_combustion_solver(n_steps, lambdify_banded_config);
        let lambdify_banded_bootstrap_begin = Instant::now();
        lambdify_banded_solver
            .try_eq_generate(None, None)
            .expect("lambdify banded combustion-1000 bootstrap should succeed");
        let lambdify_banded_bootstrap_ms =
            lambdify_banded_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
        let lambdify_banded_args = flattened_initial_guess_state(
            &lambdify_banded_solver,
            lambdify_banded_solver.values.len() * n_steps,
            "combustion-1000-lambdify-banded",
        );
        let (lambdify_banded_residual, lambdify_banded_dense, _, lambdify_banded_matrix) =
            eval_solver_callback_state(
                &mut lambdify_banded_solver,
                &lambdify_banded_args,
                MatrixBackend::Banded,
                "combustion-1000-lambdify-banded",
            );
        let lambdify_banded_matrix =
            lambdify_banded_matrix.expect("banded lambdify path should produce BandedMatrixType");
        let lambdify_banded_rhs: Vec<f64> = lambdify_banded_residual
            .iter()
            .map(|value| -*value)
            .collect();
        for solver_choice in BandedStorySolver::variants() {
            let banded_begin = Instant::now();
            let metrics = solve_banded_story_for_rhs(
                &lambdify_banded_matrix.assembly,
                n_steps,
                lambdify_banded_rhs.as_slice(),
                solver_choice,
            );
            let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
            let solve_diff = metrics
                .solution
                .as_ref()
                .map(|solution| {
                    solution
                        .iter()
                        .zip(baseline_sparse_solution.iter())
                        .map(|(lhs, rhs)| (lhs - rhs).abs())
                        .fold(0.0_f64, f64::max)
                })
                .unwrap_or(f64::NAN);
            let rel_x = metrics
                .solution
                .as_ref()
                .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
                .unwrap_or(f64::NAN);
            let solve_rr = metrics
                .solution
                .as_ref()
                .map(|solution| {
                    relative_dense_residual(
                        &lambdify_banded_dense,
                        solution,
                        lambdify_banded_rhs.as_slice(),
                    )
                })
                .unwrap_or(f64::NAN);
            rows.push(Row {
                source: "Lambdify",
                matrix_backend: "Banded",
                variant: "ExprLegacy".to_string(),
                linear_solver: metrics.linear_solver,
                bootstrap_ms: lambdify_banded_bootstrap_ms,
                residual_diff: max_abs_vector_diff(&lambdify_banded_residual, &baseline_residual),
                jacobian_diff: max_abs_matrix_diff(&lambdify_banded_dense, &baseline_dense_matrix),
                sparse_ms: 0.0,
                banded_ms,
                layout: metrics.layout,
                refinement: metrics
                    .report
                    .as_ref()
                    .map(|report| {
                        format!(
                            "{}/{}{}",
                            report.accepted_steps,
                            report.requested_steps,
                            if report.refinement_attempted {
                                ""
                            } else {
                                " skipped"
                            }
                        )
                    })
                    .unwrap_or_else(|| "-".to_string()),
                direct_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.direct_relative_residual)
                    .unwrap_or(f64::NAN),
                final_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.final_relative_residual)
                    .unwrap_or(f64::NAN),
                solve_rr,
                solve_diff,
                relative_x_diff: rel_x,
                status: metrics.status,
            });
        }

        let compiled_sparse_variants = [
            (
                "C-gcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("gcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
            (
                "C-tcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("tcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
            (
                "Zig",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::Zig)
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
        ];

        for (variant, config) in compiled_sparse_variants {
            println!("[BVP Damp story] bootstrapping compiled sparse variant `{variant}`");
            let mut solver = make_combustion_solver(n_steps, config);
            let bootstrap_begin = Instant::now();
            solver
                .try_eq_generate(None, None)
                .expect("compiled sparse combustion-1000 bootstrap should succeed");
            let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

            let args = flattened_initial_guess_state(
                &solver,
                solver.values.len() * n_steps,
                &format!("combustion-1000-sparse-{variant}"),
            );
            let (residual, dense_matrix, sparse_matrix, _) = eval_solver_callback_state(
                &mut solver,
                &args,
                MatrixBackend::SparseCol,
                &format!("combustion-1000-sparse-{variant}"),
            );
            let sparse_matrix =
                sparse_matrix.expect("compiled sparse path should produce SparseColMat");
            let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();

            let sparse_begin = Instant::now();
            let sparse_solution = solve_sparse_lu_for_rhs(&sparse_matrix, rhs.as_slice());
            let sparse_ms = sparse_begin.elapsed().as_secs_f64() * 1_000.0;

            rows.push(Row {
                source: "Compiled",
                matrix_backend: "Sparse",
                variant: variant.to_string(),
                linear_solver: "faer_sparse_lu".to_string(),
                bootstrap_ms,
                residual_diff: max_abs_vector_diff(&residual, &baseline_residual),
                jacobian_diff: max_abs_matrix_diff(&dense_matrix, &baseline_dense_matrix),
                sparse_ms,
                banded_ms: 0.0,
                layout: "-".to_string(),
                refinement: "-".to_string(),
                direct_rr: 0.0,
                final_rr: 0.0,
                solve_rr: relative_dense_residual(&dense_matrix, &sparse_solution, rhs.as_slice()),
                solve_diff: sparse_solution
                    .iter()
                    .zip(baseline_sparse_solution.iter())
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0_f64, f64::max),
                relative_x_diff: relative_x_diff(&sparse_solution, &baseline_sparse_solution),
                status: "ok".to_string(),
            });
        }

        let compiled_banded_variants = [
            (
                "C-gcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("gcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::Banded),
            ),
            (
                "C-tcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("tcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::Banded),
            ),
        ];

        for (variant, config) in compiled_banded_variants {
            println!("[BVP Damp story] bootstrapping compiled banded variant `{variant}`");
            let mut solver = make_combustion_solver(n_steps, config);
            let bootstrap_begin = Instant::now();
            solver
                .try_eq_generate(None, None)
                .expect("compiled banded combustion-1000 bootstrap should succeed");
            let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

            let args = flattened_initial_guess_state(
                &solver,
                solver.values.len() * n_steps,
                &format!("combustion-1000-banded-{variant}"),
            );
            let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
                &mut solver,
                &args,
                MatrixBackend::Banded,
                &format!("combustion-1000-banded-{variant}"),
            );
            let banded_matrix =
                banded_matrix.expect("compiled banded path should produce BandedMatrixType");
            let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();

            for solver_choice in BandedStorySolver::variants() {
                let banded_begin = Instant::now();
                let metrics = solve_banded_story_for_rhs(
                    &banded_matrix.assembly,
                    n_steps,
                    rhs.as_slice(),
                    solver_choice,
                );
                let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
                let solve_diff = metrics
                    .solution
                    .as_ref()
                    .map(|solution| {
                        solution
                            .iter()
                            .zip(baseline_sparse_solution.iter())
                            .map(|(lhs, rhs)| (lhs - rhs).abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(f64::NAN);
                let rel_x = metrics
                    .solution
                    .as_ref()
                    .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
                    .unwrap_or(f64::NAN);
                let solve_rr = metrics
                    .solution
                    .as_ref()
                    .map(|solution| {
                        relative_dense_residual(&dense_matrix, solution, rhs.as_slice())
                    })
                    .unwrap_or(f64::NAN);

                rows.push(Row {
                    source: "Compiled",
                    matrix_backend: "Banded",
                    variant: variant.to_string(),
                    linear_solver: metrics.linear_solver,
                    bootstrap_ms,
                    residual_diff: max_abs_vector_diff(&residual, &baseline_residual),
                    jacobian_diff: max_abs_matrix_diff(&dense_matrix, &baseline_dense_matrix),
                    sparse_ms: 0.0,
                    banded_ms,
                    layout: metrics.layout,
                    refinement: metrics
                        .report
                        .as_ref()
                        .map(|report| {
                            format!(
                                "{}/{}{}",
                                report.accepted_steps,
                                report.requested_steps,
                                if report.refinement_attempted {
                                    ""
                                } else {
                                    " skipped"
                                }
                            )
                        })
                        .unwrap_or_else(|| "-".to_string()),
                    direct_rr: metrics
                        .report
                        .as_ref()
                        .map(|report| report.direct_relative_residual)
                        .unwrap_or(f64::NAN),
                    final_rr: metrics
                        .report
                        .as_ref()
                        .map(|report| report.final_relative_residual)
                        .unwrap_or(f64::NAN),
                    solve_rr,
                    solve_diff,
                    relative_x_diff: rel_x,
                    status: metrics.status,
                });
            }
        }

        println!(
            "[BVP Damp linear story] combustion-1000 sparse baseline vs banded solver variants"
        );
        println!(
            "{:<10} | {:<13} | {:<10} | {:<34} | {:>12} | {:>12} | {:>12} | {:>10} | {:>10} | {:<8} | {:<13} | {:>10} | {:>10} | {:>10} | {:>12} | {:>12} | {:<24}",
            "source",
            "matrix",
            "variant",
            "linear_solver",
            "bootstrap_ms",
            "res_diff",
            "jac_diff",
            "sparse_ms",
            "banded_ms",
            "layout",
            "refinement",
            "direct_rr",
            "final_rr",
            "solve_rr",
            "solve_diff",
            "rel_x_diff",
            "status"
        );
        println!("{}", "-".repeat(262));
        for row in &rows {
            println!(
                "{:<10} | {:<13} | {:<10} | {:<34} | {:>12.3} | {:>12.3e} | {:>12.3e} | {:>10.3} | {:>10.3} | {:<8} | {:<13} | {:>10} | {:>10} | {:>10} | {:>12.3e} | {:>12.3e} | {:<24}",
                row.source,
                row.matrix_backend,
                row.variant,
                row.linear_solver,
                row.bootstrap_ms,
                row.residual_diff,
                row.jacobian_diff,
                row.sparse_ms,
                row.banded_ms,
                row.layout,
                row.refinement,
                fmt_metric(row.direct_rr),
                fmt_metric(row.final_rr),
                fmt_metric(row.solve_rr),
                row.solve_diff,
                row.relative_x_diff,
                row.status
            );
        }

        for row in &rows {
            assert!(
                row.residual_diff < 1e-6,
                "{} {} {} residual diff too large: {}",
                row.source,
                row.matrix_backend,
                row.variant,
                row.residual_diff
            );
            assert!(
                row.jacobian_diff < 1e-6,
                "{} {} {} jacobian diff too large: {}",
                row.source,
                row.matrix_backend,
                row.variant,
                row.jacobian_diff
            );
            if row.matrix_backend == "Sparse" {
                assert!(
                    row.solve_diff < 1e-6,
                    "{} {} {} sparse solve drift too large: {}",
                    row.source,
                    row.matrix_backend,
                    row.variant,
                    row.solve_diff
                );
            } else if row.matrix_backend.contains("Banded") && row.status == "ok" {
                assert!(
                    row.solve_diff.is_finite(),
                    "{} {} {} {} should report finite banded solve_diff",
                    row.source,
                    row.matrix_backend,
                    row.variant,
                    row.linear_solver
                );
            }
        }
    }

    #[test]
    #[ignore = "heavy combustion-1000 end-to-end solve through banded lambdify/AOT backends using lapack-style banded LU + refinement"]
    fn combustion_1000_end_to_end_banded_lapack_refine_statistics() {
        #[derive(Debug)]
        struct EndToEndRow {
            source: &'static str,
            variant: &'static str,
            bootstrap_ms: f64,
            solve_ms: f64,
            total_ms: f64,
            max_abs_solution: f64,
            solve_diff: f64,
            rel_x_diff: f64,
            iterations: usize,
            linear_solves: usize,
            jac_rebuilds: usize,
            linear_timer: String,
            jac_timer: String,
            fun_timer: String,
            status: String,
        }

        fn solution_max_abs(solution: &DMatrix<f64>) -> f64 {
            solution
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
        }

        fn solution_linf_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
            lhs.iter()
                .zip(rhs.iter())
                .map(|(l, r)| (l - r).abs())
                .fold(0.0_f64, f64::max)
        }

        fn solution_rel_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
            solution_linf_diff(lhs, rhs) / solution_max_abs(rhs).max(1.0)
        }

        fn run_variant(
            n_steps: usize,
            source: &'static str,
            variant: &'static str,
            config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
        ) -> (EndToEndRow, Option<DMatrix<f64>>) {
            let total_begin = Instant::now();
            let mut solver = make_combustion_solver(n_steps, config);

            let bootstrap_begin = Instant::now();
            let bootstrap_status = solver.try_eq_generate(None, None);
            let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
            if let Err(err) = bootstrap_status {
                return (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms: 0.0,
                        total_ms: total_begin.elapsed().as_secs_f64() * 1_000.0,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: 0,
                        linear_solves: 0,
                        jac_rebuilds: 0,
                        linear_timer: "-".to_string(),
                        jac_timer: "-".to_string(),
                        fun_timer: "-".to_string(),
                        status: format!("bootstrap_failed({err:?})"),
                    },
                    None,
                );
            }

            let solve_begin = Instant::now();
            let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solve()));
            let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
            let statistics = solver.get_statistics();

            match solve_status {
                Ok(Ok(_)) => match solver.get_result() {
                    Some(solution) => {
                        let max_abs_solution = solution_max_abs(&solution);
                        (
                            EndToEndRow {
                                source,
                                variant,
                                bootstrap_ms,
                                solve_ms,
                                total_ms,
                                max_abs_solution,
                                solve_diff: 0.0,
                                rel_x_diff: 0.0,
                                iterations: stats_count(&statistics, "number of iterations"),
                                linear_solves: stats_count(
                                    &statistics,
                                    "number of solving linear systems",
                                ),
                                jac_rebuilds: stats_count(
                                    &statistics,
                                    "number of jacobians recalculations",
                                ),
                                linear_timer: stats_timer(&statistics, "Linear System"),
                                jac_timer: stats_timer(&statistics, "Jacobian"),
                                fun_timer: stats_timer(&statistics, "Function"),
                                status: "ok".to_string(),
                            },
                            Some(solution),
                        )
                    }
                    None => (
                        EndToEndRow {
                            source,
                            variant,
                            bootstrap_ms,
                            solve_ms,
                            total_ms,
                            max_abs_solution: f64::NAN,
                            solve_diff: f64::NAN,
                            rel_x_diff: f64::NAN,
                            iterations: stats_count(&statistics, "number of iterations"),
                            linear_solves: stats_count(
                                &statistics,
                                "number of solving linear systems",
                            ),
                            jac_rebuilds: stats_count(
                                &statistics,
                                "number of jacobians recalculations",
                            ),
                            linear_timer: stats_timer(&statistics, "Linear System"),
                            jac_timer: stats_timer(&statistics, "Jacobian"),
                            fun_timer: stats_timer(&statistics, "Function"),
                            status: "no_result".to_string(),
                        },
                        None,
                    ),
                },
                Ok(Err(err)) => (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms,
                        total_ms,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: stats_count(&statistics, "number of iterations"),
                        linear_solves: stats_count(&statistics, "number of solving linear systems"),
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        linear_timer: stats_timer(&statistics, "Linear System"),
                        jac_timer: stats_timer(&statistics, "Jacobian"),
                        fun_timer: stats_timer(&statistics, "Function"),
                        status: format!("solve_failed({err:?})"),
                    },
                    None,
                ),
                Err(panic_payload) => {
                    let status = if let Some(message) = panic_payload.downcast_ref::<String>() {
                        format!("solve_panicked({message})")
                    } else if let Some(message) = panic_payload.downcast_ref::<&str>() {
                        format!("solve_panicked({message})")
                    } else {
                        "solve_panicked(non-string payload)".to_string()
                    };
                    (
                        EndToEndRow {
                            source,
                            variant,
                            bootstrap_ms,
                            solve_ms,
                            total_ms,
                            max_abs_solution: f64::NAN,
                            solve_diff: f64::NAN,
                            rel_x_diff: f64::NAN,
                            iterations: stats_count(&statistics, "number of iterations"),
                            linear_solves: stats_count(
                                &statistics,
                                "number of solving linear systems",
                            ),
                            jac_rebuilds: stats_count(
                                &statistics,
                                "number of jacobians recalculations",
                            ),
                            linear_timer: stats_timer(&statistics, "Linear System"),
                            jac_timer: stats_timer(&statistics, "Jacobian"),
                            fun_timer: stats_timer(&statistics, "Function"),
                            status,
                        },
                        None,
                    )
                }
            }
        }

        let n_steps = 1000usize;

        let lambdify_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_lambdify_defaults()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

        let gcc_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

        let tcc_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

        let zig_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

        let runs = [
            ("Lambdify", "ExprLegacy", lambdify_config),
            ("Compiled", "C-gcc", gcc_config),
            ("Compiled", "C-tcc", tcc_config),
            ("Compiled", "Zig", zig_config),
        ];

        let mut rows = Vec::new();
        let mut baseline_solution: Option<DMatrix<f64>> = None;

        for (source, variant, config) in runs {
            let (mut row, solution) = run_variant(n_steps, source, variant, config);
            if source == "Lambdify" && variant == "ExprLegacy" {
                baseline_solution = solution.clone();
            }
            if let (Some(solution), Some(baseline)) =
                (solution.as_ref(), baseline_solution.as_ref())
            {
                row.solve_diff = solution_linf_diff(solution, baseline);
                row.rel_x_diff = solution_rel_diff(solution, baseline);
            }
            rows.push(row);
        }

        println!(
            "[BVP Damp end-to-end] combustion-1000 full solve with banded backends using lapack_style_banded_lu"
        );
        if baseline_solution.is_none() {
            println!(
                "[BVP Damp end-to-end] Lambdify baseline did not converge; solve_diff and rel_x_diff are unavailable for this run"
            );
        }
        println!(
            "{:<10} | {:<10} | {:>12} | {:>10} | {:>10} | {:>12} | {:>12} | {:>12} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<8}",
            "source",
            "variant",
            "bootstrap_ms",
            "solve_ms",
            "total_ms",
            "max_abs_sol",
            "solve_diff",
            "rel_x_diff",
            "iters",
            "linsys",
            "jac_re",
            "linear_timer",
            "jac_timer",
            "fun_timer",
            "status"
        );
        println!("{}", "-".repeat(220));
        for row in &rows {
            println!(
                "{:<10} | {:<10} | {:>12.3} | {:>10.3} | {:>10.3} | {:>12.6e} | {:>12} | {:>12} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<8}",
                row.source,
                row.variant,
                row.bootstrap_ms,
                row.solve_ms,
                row.total_ms,
                row.max_abs_solution,
                fmt_metric(row.solve_diff),
                fmt_metric(row.rel_x_diff),
                row.iterations,
                row.linear_solves,
                row.jac_rebuilds,
                row.linear_timer,
                row.jac_timer,
                row.fun_timer,
                row.status
            );
        }

        assert!(
            !rows.is_empty(),
            "combustion-1000 end-to-end banded diagnostic should produce at least one row"
        );
        if let Some(baseline_solution) = baseline_solution {
            assert!(
                baseline_solution.iter().all(|value| value.is_finite()),
                "lambdify banded combustion-1000 baseline solution should remain finite"
            );
            for row in &rows {
                if row.status == "ok" {
                    assert!(
                        row.max_abs_solution.is_finite(),
                        "{} {} end-to-end banded result should stay finite",
                        row.source,
                        row.variant
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "focused Zig banded AOT bootstrap diagnostic for combustion-1000"]
    fn combustion_1000_compiled_banded_zig_bootstrap_smoke() {
        let n_steps = 1000usize;
        println!("[BVP Damp Zig banded] starting focused combustion-1000 Zig AOT bootstrap");

        let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_codegen_backend(AotCodegenBackend::Zig)
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_matrix_backend_override(MatrixBackend::Banded);

        let mut solver = make_combustion_solver(n_steps, config);
        solver
            .try_eq_generate(None, None)
            .expect("compiled banded Zig combustion-1000 bootstrap should succeed");

        let args = flattened_initial_guess_state(
            &solver,
            solver.values.len() * n_steps,
            "combustion-1000-banded-zig-smoke",
        );
        let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
            &mut solver,
            &args,
            MatrixBackend::Banded,
            "combustion-1000-banded-zig-smoke",
        );
        let banded_matrix =
            banded_matrix.expect("compiled banded Zig path should produce BandedMatrixType");
        let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();
        let metrics = solve_banded_story_for_rhs(
            &banded_matrix.assembly,
            n_steps,
            rhs.as_slice(),
            BandedStorySolver::ConsistentSuperblock {
                nodes_per_superblock: 2,
                refinement_steps: 1,
            },
        );

        println!(
            "[BVP Damp Zig banded] residual_len={}, matrix={}x{}, solver={}, layout={}, refinement={}, direct_rr={:.3e}, final_rr={:.3e}, max|x|={:.3e}, status={}",
            residual.len(),
            dense_matrix.nrows(),
            dense_matrix.ncols(),
            metrics.linear_solver,
            metrics.layout,
            metrics
                .report
                .as_ref()
                .map(|report| format!("{}/{}", report.accepted_steps, report.requested_steps))
                .unwrap_or_else(|| "-".to_string()),
            metrics
                .report
                .as_ref()
                .map(|report| report.direct_relative_residual)
                .unwrap_or(f64::NAN),
            metrics
                .report
                .as_ref()
                .map(|report| report.final_relative_residual)
                .unwrap_or(f64::NAN),
            metrics
                .solution
                .as_ref()
                .map(|solution| solution
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())))
                .unwrap_or(f64::NAN),
            metrics.status
        );

        assert!(
            metrics
                .solution
                .as_ref()
                .map(|solution| solution.iter().all(|value| value.is_finite()))
                .unwrap_or(false),
            "compiled banded Zig combustion-1000 solve should remain finite"
        );
    }

    #[test]
    #[ignore = "focused combustion-1000 diagnostic for lapack-style banded storage/factor/solve"]
    fn diagnose_combustion_1000_lapack_style_banded_path() {
        let n_steps = 1000usize;
        let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                .with_matrix_backend_override(MatrixBackend::Banded);

        println!("[Lapack banded diagnostic] bootstrapping combustion-1000 lambdify banded");
        let mut solver = make_combustion_solver(n_steps, config);
        solver
            .try_eq_generate(None, None)
            .expect("combustion-1000 lambdify banded bootstrap should succeed");

        let args = flattened_initial_guess_state(
            &solver,
            solver.values.len() * n_steps,
            "combustion-1000-lapack-style-diagnostic",
        );

        let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
            &mut solver,
            &args,
            MatrixBackend::Banded,
            "combustion-1000-lapack-style-diagnostic",
        );
        let banded_matrix =
            banded_matrix.expect("combustion-1000 banded path should produce BandedMatrixType");
        let assembly = &banded_matrix.assembly;
        let compact = assembly
            .to_banded()
            .expect("banded assembly should convert to compact storage");

        let dense_from_compact = dense_from_compact_banded(&compact);
        let compact_diff = max_abs_matrix_diff(&dense_matrix, &dense_from_compact);

        let mut lapack = LapackStyleBandedLuFaithful::new(compact.n(), compact.kl(), compact.ku())
            .expect("lapack-style workspace should allocate");
        lapack
            .load_from_banded(&compact)
            .expect("lapack-style workspace should load compact banded matrix");
        let loaded_dense = dense_from_vecvec(&lapack.reconstruct_original_band_dense());
        let load_diff = max_abs_matrix_diff(&dense_matrix, &loaded_dense);

        lapack
            .factor_from(&compact)
            .expect("lapack-style factorization should succeed on combustion-1000 banded Jacobian");
        let factor_rel = lapack
            .factor_residual_relative(&compact)
            .expect("factor residual should be available after factorization");

        let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();
        let mut x_lapack = rhs.clone();
        lapack
            .solve_in_place(&mut x_lapack)
            .expect("lapack-style solve should succeed after factorization");
        let solve_rr = relative_dense_residual(&dense_matrix, &x_lapack, &rhs);

        let sparse = dense_to_sparse_col_mat(&dense_matrix);
        let x_sparse = solve_sparse_lu_for_rhs(&sparse, rhs.as_slice());
        let solve_diff = x_lapack
            .iter()
            .zip(x_sparse.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "[Lapack banded diagnostic] combustion-1000: n={}, kl={}, ku={}, compact_diff={:.3e}, load_diff={:.3e}, factor_rel={:.3e}, solve_rr={:.3e}, solve_diff={:.3e}",
            compact.n(),
            compact.kl(),
            compact.ku(),
            compact_diff,
            load_diff,
            factor_rel,
            solve_rr,
            solve_diff
        );
    }
}

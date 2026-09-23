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
    iterations: usize,
    linear_solves: usize,
    jacobian_rebuilds: usize,
    refinements: usize,
    residual_calls: usize,
    jacobian_calls: usize,
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
    typed_aot_snapshot: Option<BvpAotTelemetrySnapshot>,
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
    jacobian.set_aot_telemetry_mode(request.aot_telemetry_mode);
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
    jacobian.set_aot_telemetry_mode(request.aot_telemetry_mode);
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
    let typed_aot_snapshot = prepared.aot_telemetry_snapshot();

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
            typed_aot_snapshot,
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
                typed_aot_snapshot,
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
        (residual, sparse_jacobian) => jacobian.generate_BVP_with_backend_selection_and_chunking(
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

fn report_top_sparse_bundle_differences(
    lhs: &mut BvpSparseSolverBundle,
    rhs: &mut BvpSparseSolverBundle,
    args: &DVector<f64>,
    label: &str,
) -> (f64, f64) {
    // Keep the diagnostic name descriptive while sharing the canonical
    // callback comparison used by all symbolic frontend stories.
    compare_sparse_bundles_numerically(lhs, rhs, args, label)
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

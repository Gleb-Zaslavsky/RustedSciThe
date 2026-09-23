fn prepared_bvp_jacobian(
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    param_name_refs: Option<&[&str]>,
    param_values: Option<Vec<f64>>,
    banded_linear_solver_config: LinearSolverConfig,
    lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    aot_telemetry_mode: BvpAotTelemetryMode,
    lambdify_execution_policy: BvpLambdifyExecutionPolicy,
) -> Jacobian {
    let mut jacobian_instance = Jacobian::new();
    jacobian_instance.set_symbolic_assembly_backend(symbolic_assembly_backend);
    jacobian_instance.set_params(param_name_refs);
    jacobian_instance.set_param_values(param_values);
    jacobian_instance.set_banded_linear_solver_config(banded_linear_solver_config);
    jacobian_instance.set_lambdify_telemetry_mode(lambdify_telemetry_mode);
    jacobian_instance.set_aot_telemetry_mode(aot_telemetry_mode);
    jacobian_instance.set_lambdify_execution_policy(lambdify_execution_policy);
    jacobian_instance
}

#[allow(clippy::too_many_arguments)]
fn try_generate_sparse_bundle(
    jacobian_instance: Jacobian,
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    param_name_refs: Option<&[&str]>,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
    method: String,
    bandwidth: Option<(usize, usize)>,
    backend_policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_chunking_policy: AotChunkingPolicy,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    match (
        aot_chunking_policy.residual,
        aot_chunking_policy.sparse_jacobian,
    ) {
        (None, None) => jacobian_instance.try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system,
            values,
            arg,
            param_name_refs,
            t0,
            None,
            n_steps,
            h,
            mesh,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
            backend_policy,
            resolver,
        ),
        (residual, sparse_jacobian) => jacobian_instance
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eq_system,
                values,
                arg,
                param_name_refs,
                t0,
                None,
                n_steps,
                h,
                mesh,
                border_conditions,
                bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
                backend_policy,
                resolver,
                residual.unwrap_or(ResidualChunkingStrategy::Whole),
                sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
            ),
    }
}

fn build_policy_can_materialize_auto_chunked_artifact(build_policy: AotBuildPolicy) -> bool {
    matches!(
        build_policy,
        AotBuildPolicy::BuildIfMissing { .. } | AotBuildPolicy::RebuildAlways { .. }
    )
}

fn auto_codegen_chunking_policy_for_sparse_bundle(
    bundle: &BvpSparseSolverBundle,
    original_backend_policy: BackendSelectionPolicy,
    execution_policy: &AotExecutionPolicy,
    build_policy: AotBuildPolicy,
    requested_chunking: AotChunkingPolicy,
) -> Option<AotChunkingPolicy> {
    if requested_chunking != AotChunkingPolicy::default()
        || !matches!(execution_policy, AotExecutionPolicy::Auto)
        || !backend_policy_targets_aot(original_backend_policy)
        || !build_policy_can_materialize_auto_chunked_artifact(build_policy)
    {
        return None;
    }

    let auto_plan = bundle
        .execution
        .selected()
        .prepared_problem
        .auto_parallel_plan();
    if !matches!(
        auto_plan.execution_mode,
        crate::symbolic::codegen::codegen_orchestrator::AutoExecutionMode::Parallel
    ) {
        info!(
            "Auto AOT codegen kept whole callbacks for problem_key={} because residual_reason={}, sparse_reason={}, residual_work/job={}/{}, sparse_work/job={}/{}",
            bundle.execution.selected().problem_key(),
            auto_plan.residual_stage.reason.as_str(),
            auto_plan.sparse_stage.reason.as_str(),
            auto_plan.residual_stage.work_per_job,
            auto_plan.residual_stage.min_work_per_job,
            auto_plan.sparse_stage.work_per_job,
            auto_plan.sparse_stage.min_work_per_job
        );
        return None;
    }

    info!(
        "Auto AOT codegen selected chunked callbacks for problem_key={} with residual_chunking={:?}, sparse_chunking={:?}, residual_jobs={}, sparse_jobs={}, residual_work/job={}, sparse_work/job={}, workers={}",
        bundle.execution.selected().problem_key(),
        auto_plan.residual_chunking,
        auto_plan.sparse_chunking,
        auto_plan.residual_stage.jobs,
        auto_plan.sparse_stage.jobs,
        auto_plan.residual_stage.work_per_job,
        auto_plan.sparse_stage.work_per_job,
        auto_plan.workers
    );
    Some(AotChunkingPolicy::with_parts(
        Some(auto_plan.residual_chunking),
        Some(auto_plan.sparse_chunking),
    ))
}

fn activate_compiled_sparse_bundle_after_aot_build(
    mut bundle: BvpSparseSolverBundle,
    updated_resolver: &AotResolver,
    aot_codegen_backend: AotCodegenBackend,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    let resolution = updated_resolver.resolve_by_problem_key(problem_key.as_str());
    if !resolution.is_compiled() {
        return Err(
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
                problem_key,
                effective_backend: bundle.effective_backend(),
            },
        );
    }

    let mut selected = bundle.execution.selected().clone();
    selected.requested_backend = BackendKind::Aot;
    selected.effective_backend = SelectedBackendKind::AotCompiled;
    selected.aot_resolution = Some(resolution);
    bundle.execution = BvpSparseExecutionPlan::AotCompiled(selected);

    if !(try_link_sparse_runtime_from_resolution(
        &bundle,
        Some(updated_resolver),
        aot_codegen_backend,
    )?) || !bundle.rebind_linked_runtime_callbacks(None, None)
    {
        return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key });
    }

    Ok(bundle)
}

fn sanitize_generated_name(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '_',
        })
        .collect::<String>()
}

fn build_output_parent_for_problem(problem_key: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("generated-aot")
        .join(sanitize_generated_name(problem_key))
}

fn unique_build_output_parent_for_problem(problem_key: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    build_output_parent_for_problem(problem_key)
        .join(format!("build-{}-{nonce}", std::process::id()))
}

fn to_lifecycle_build_profile(profile: AotBuildProfile) -> LifecycleBuildProfile {
    match profile {
        AotBuildProfile::Release => LifecycleBuildProfile::Release,
        AotBuildProfile::Debug => LifecycleBuildProfile::Debug,
    }
}

fn to_c_build_profile(profile: AotBuildProfile) -> CAotBuildProfile {
    match profile {
        AotBuildProfile::Release => CAotBuildProfile::Release,
        AotBuildProfile::Debug => CAotBuildProfile::Debug,
    }
}

fn to_zig_build_profile(profile: AotBuildProfile) -> ZigAotBuildProfile {
    match profile {
        AotBuildProfile::Release => ZigAotBuildProfile::ReleaseFast,
        AotBuildProfile::Debug => ZigAotBuildProfile::Debug,
    }
}

fn to_c_compile_config(compile_config: &AotCompileConfig) -> CAotCompileConfig {
    if *compile_config == AotCompileConfig::dev_fastest() {
        CAotCompileConfig::dev_fastest()
    } else if *compile_config == AotCompileConfig::fast_build() {
        CAotCompileConfig::fast_build()
    } else {
        CAotCompileConfig::production()
    }
}

fn register_sparse_runtime_from_registered_artifact(
    artifact: &crate::symbolic::codegen::codegen_aot_registry::RegisteredAotArtifact,
    matrix_backend: MatrixBackend,
    backend: AotCodegenBackend,
) -> Result<(), String> {
    match backend {
        AotCodegenBackend::Rust => match matrix_backend {
            MatrixBackend::Banded => register_generated_banded_cdylib_backend(artifact).map(|_| ()),
            _ => register_generated_sparse_cdylib_backend(artifact).map(|_| ()),
        },
        AotCodegenBackend::C => match matrix_backend {
            MatrixBackend::Banded => register_generated_c_banded_backend(artifact).map(|_| ()),
            _ => register_generated_c_sparse_backend(artifact).map(|_| ()),
        },
        AotCodegenBackend::Zig => match matrix_backend {
            MatrixBackend::Banded => register_generated_zig_banded_backend(artifact).map(|_| ()),
            _ => register_generated_zig_sparse_backend(artifact).map(|_| ()),
        },
    }
}

fn rust_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
    compile_config: AotCompileConfig,
    atom_profile: AtomOptimizationProfile,
) -> (AotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let crate_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = if selected.matrix_backend == MatrixBackend::Banded
        && selected.prepared_problem.aot_preparation_route()
            == BvpAotPreparationRoute::AtomViewNative
        && selected.prepared_problem.jacobian_strategy == SparseChunkingStrategy::Whole
    {
        selected
            .prepared_problem
            .generated_native_banded_aot_artifact_with_breakdown(
                &crate_name,
                &module_name,
                AotCodegenBackend::Rust,
                atom_profile,
            )
    } else {
        selected
            .prepared_problem
            .generated_aot_artifact_with_breakdown_for_matrix_backend_and_atom_profile(
                &crate_name,
                &module_name,
                AotCodegenBackend::Rust,
                selected.matrix_backend,
                atom_profile,
            )
    };
    let request = AotBuildRequest::new(
        artifact
            .into_rust_crate()
            .expect("Rust backend must emit GeneratedAotCrate"),
        output_parent_dir,
        to_lifecycle_build_profile(profile),
    )
    .with_compile_config(compile_config);
    (request, breakdown)
}

fn c_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
    compile_config: CAotCompileConfig,
    atom_profile: AtomOptimizationProfile,
) -> (CAotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let library_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = if selected.matrix_backend == MatrixBackend::Banded
        && selected.prepared_problem.aot_preparation_route()
            == BvpAotPreparationRoute::AtomViewNative
        && selected.prepared_problem.jacobian_strategy == SparseChunkingStrategy::Whole
    {
        selected
            .prepared_problem
            .generated_native_banded_aot_artifact_with_breakdown(
                &library_name,
                &module_name,
                AotCodegenBackend::C,
                atom_profile,
            )
    } else {
        selected
            .prepared_problem
            .generated_aot_artifact_with_breakdown_for_matrix_backend_and_atom_profile(
                &library_name,
                &module_name,
                AotCodegenBackend::C,
                selected.matrix_backend,
                atom_profile,
            )
    };
    let library_spec = match artifact {
        crate::symbolic::codegen::codegen_aot_driver::GeneratedAotArtifact::C(library) => library,
        _ => unreachable!("C backend must emit GeneratedCAotLibrary"),
    };
    let request =
        CAotBuildRequest::new(library_spec, output_parent_dir, to_c_build_profile(profile))
            .with_compile_config(compile_config);
    (request, breakdown)
}

fn zig_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
    atom_profile: AtomOptimizationProfile,
) -> (ZigAotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let library_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = if selected.matrix_backend == MatrixBackend::Banded
        && selected.prepared_problem.aot_preparation_route()
            == BvpAotPreparationRoute::AtomViewNative
        && selected.prepared_problem.jacobian_strategy == SparseChunkingStrategy::Whole
    {
        selected
            .prepared_problem
            .generated_native_banded_aot_artifact_with_breakdown(
                &library_name,
                &module_name,
                AotCodegenBackend::Zig,
                atom_profile,
            )
    } else {
        selected
            .prepared_problem
            .generated_aot_artifact_with_breakdown_for_matrix_backend_and_atom_profile(
                &library_name,
                &module_name,
                AotCodegenBackend::Zig,
                selected.matrix_backend,
                atom_profile,
            )
    };
    let library_spec = match artifact {
        crate::symbolic::codegen::codegen_aot_driver::GeneratedAotArtifact::Zig(library) => library,
        _ => unreachable!("Zig backend must emit GeneratedZigAotLibrary"),
    };
    (
        ZigAotBuildRequest::new(
            library_spec,
            output_parent_dir,
            to_zig_build_profile(profile),
        ),
        breakdown,
    )
}

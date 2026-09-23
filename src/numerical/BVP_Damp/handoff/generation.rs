/// Generates a complete solver state from symbolic equations and runtime policy.
///
/// The sparse mainline goes through the modern bundle path while non-sparse
/// modes still use the legacy Jacobian handoff for compatibility.
#[allow(clippy::too_many_arguments)]
pub fn generate_damped_solver_state(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    param_names: Option<Vec<String>>,
    param_values: Option<Vec<f64>>,
    arg: String,
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
    aot_execution_policy: AotExecutionPolicy,
    aot_build_policy: AotBuildPolicy,
    aot_compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    aot_chunking_policy: AotChunkingPolicy,
    atom_profile: AtomOptimizationProfile,
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    matrix_backend_override: Option<MatrixBackend>,
    banded_linear_solver_config: LinearSolverConfig,
    lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    aot_telemetry_mode: BvpAotTelemetryMode,
    lambdify_execution_policy: BvpLambdifyExecutionPolicy,
) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
    let handoff_begin = Instant::now();
    let mut handoff_diagnostics = HashMap::new();
    let param_name_refs = parameter_name_refs(param_names.as_ref());
    // AtomView is a symbolic assembly choice, not a Sparse-only feature.
    // Dense must use the same modern bundle when AtomView is selected; the
    // legacy bundle does not contain the Atom-native discretized functions.
    if matches!(method.as_str(), "Sparse" | "Banded")
        || matches!(matrix_backend_override, Some(MatrixBackend::Banded))
        || symbolic_assembly_backend == BvpSymbolicAssemblyBackend::AtomView
    {
        let lifecycle_lock_begin = Instant::now();
        let _lifecycle_guard =
            if aot_lifecycle_needs_serialization(backend_policy, aot_build_policy) {
                Some(bvp_aot_lifecycle_lock().lock().map_err(|_| {
                    BvpBackendIntegrationError::PipelinePanicked(
                        "BVP AOT lifecycle lock poisoned".to_string(),
                    )
                })?)
            } else {
                None
            };
        if _lifecycle_guard.is_some() {
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.aot.lifecycle_lock_wait_ms",
                lifecycle_lock_begin,
            );
        }
        let retry_eq_system = eq_system.clone();
        let retry_values = values.clone();
        let retry_arg = arg.clone();
        let retry_param_values = param_values.clone();
        let retry_mesh = mesh.clone();
        let retry_border_conditions = border_conditions.clone();
        let retry_bounds = bounds.clone();
        let retry_rel_tolerance = rel_tolerance.clone();
        let retry_scheme = scheme.clone();
        let retry_method = method.clone();
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values.clone(),
            banded_linear_solver_config,
            lambdify_telemetry_mode,
            aot_telemetry_mode,
            lambdify_execution_policy,
        );
        let original_backend_policy = backend_policy;
        let backend_policy =
            effective_backend_policy_for_build(original_backend_policy, aot_build_policy);
        let initial_generate_begin = Instant::now();
        let mut bundle = try_generate_sparse_bundle(
            jacobian_instance,
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
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
            aot_chunking_policy,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.initial_generate_wall_ms",
            initial_generate_begin,
        );
        copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "initial");
        if let Some(auto_chunking_policy) = auto_codegen_chunking_policy_for_sparse_bundle(
            &bundle,
            original_backend_policy,
            &aot_execution_policy,
            aot_build_policy,
            aot_chunking_policy,
        ) {
            let jacobian_instance = prepared_bvp_jacobian(
                symbolic_assembly_backend,
                param_name_refs.as_deref(),
                retry_param_values.clone(),
                banded_linear_solver_config,
                lambdify_telemetry_mode,
                aot_telemetry_mode,
                lambdify_execution_policy,
            );
            let auto_regenerate_begin = Instant::now();
            bundle = try_generate_sparse_bundle(
                jacobian_instance,
                retry_eq_system.clone(),
                retry_values.clone(),
                retry_arg.clone(),
                param_name_refs.as_deref(),
                t0,
                n_steps,
                h,
                retry_mesh.clone(),
                retry_border_conditions.clone(),
                retry_bounds.clone(),
                retry_rel_tolerance.clone(),
                retry_scheme.clone(),
                retry_method.clone(),
                bandwidth,
                backend_policy,
                resolver,
                auto_chunking_policy,
            )?;
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.handoff.auto_chunk_regenerate_wall_ms",
                auto_regenerate_begin,
            );
            copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "auto_chunk");
        }
        let build_policy_begin = Instant::now();
        let (bundle, updated_resolver) = enforce_build_policy_on_sparse_bundle(
            bundle,
            resolver,
            aot_build_policy,
            aot_compile_config,
            aot_codegen_backend,
            aot_c_compiler,
            atom_profile,
            &mut handoff_diagnostics,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.build_policy_wall_ms",
            build_policy_begin,
        );
        let bundle = match updated_resolver.as_ref() {
            Some(updated_resolver_ref) if backend_policy_targets_aot(original_backend_policy) => {
                let rebind_begin = Instant::now();
                let bound = activate_compiled_sparse_bundle_after_aot_build(
                    bundle,
                    updated_resolver_ref,
                    aot_codegen_backend,
                )?;
                insert_elapsed_ms(
                    &mut handoff_diagnostics,
                    "generated.handoff.post_build_rebind_wall_ms",
                    rebind_begin,
                );
                bound
            }
            _ => bundle,
        };
        let execution_bind_begin = Instant::now();
        let mut bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.execution_bind_wall_ms",
            execution_bind_begin,
        );
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.total_wall_ms",
            handoff_begin,
        );
        bundle.runtime_diagnostics.extend(handoff_diagnostics);
        Ok(damped_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values,
            banded_linear_solver_config,
            lambdify_telemetry_mode,
            aot_telemetry_mode,
            lambdify_execution_policy,
        );
        let legacy_bundle = jacobian_instance.generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
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
        );
        Ok(damped_state_from_legacy_solver_bundle(legacy_bundle))
    }
}

/// Builds a complete callback/metadata handoff for the frozen solver.
///
/// The sparse mainline goes through the modern bundle path while non-sparse
/// modes still use the legacy Jacobian handoff for compatibility.
#[allow(clippy::too_many_arguments)]
pub fn generate_frozen_solver_state(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    param_names: Option<Vec<String>>,
    param_values: Option<Vec<f64>>,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    scheme: String,
    method: String,
    bandwidth: Option<(usize, usize)>,
    backend_policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_execution_policy: AotExecutionPolicy,
    aot_build_policy: AotBuildPolicy,
    aot_compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    aot_chunking_policy: AotChunkingPolicy,
    atom_profile: AtomOptimizationProfile,
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    matrix_backend_override: Option<MatrixBackend>,
    banded_linear_solver_config: LinearSolverConfig,
    lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    aot_telemetry_mode: BvpAotTelemetryMode,
    lambdify_execution_policy: BvpLambdifyExecutionPolicy,
) -> Result<FrozenGeneratedSolverState, BvpBackendIntegrationError> {
    let handoff_begin = Instant::now();
    let mut handoff_diagnostics = HashMap::new();
    let param_name_refs = parameter_name_refs(param_names.as_ref());
    // Keep AtomView available for Dense as well as Sparse/Banded. Only an
    // explicit ExprLegacy selection should use the compatibility bundle.
    if matches!(method.as_str(), "Sparse" | "Banded")
        || matches!(matrix_backend_override, Some(MatrixBackend::Banded))
        || symbolic_assembly_backend == BvpSymbolicAssemblyBackend::AtomView
    {
        let lifecycle_lock_begin = Instant::now();
        let _lifecycle_guard =
            if aot_lifecycle_needs_serialization(backend_policy, aot_build_policy) {
                Some(bvp_aot_lifecycle_lock().lock().map_err(|_| {
                    BvpBackendIntegrationError::PipelinePanicked(
                        "BVP AOT lifecycle lock poisoned".to_string(),
                    )
                })?)
            } else {
                None
            };
        if _lifecycle_guard.is_some() {
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.aot.lifecycle_lock_wait_ms",
                lifecycle_lock_begin,
            );
        }
        let retry_eq_system = eq_system.clone();
        let retry_values = values.clone();
        let retry_arg = arg.clone();
        let retry_param_values = param_values.clone();
        let retry_mesh = mesh.clone();
        let retry_border_conditions = border_conditions.clone();
        let retry_scheme = scheme.clone();
        let retry_method = method.clone();
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values.clone(),
            banded_linear_solver_config,
            lambdify_telemetry_mode,
            aot_telemetry_mode,
            lambdify_execution_policy,
        );
        let original_backend_policy = backend_policy;
        let backend_policy =
            effective_backend_policy_for_build(original_backend_policy, aot_build_policy);
        let initial_generate_begin = Instant::now();
        let mut bundle = try_generate_sparse_bundle(
            jacobian_instance,
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            n_steps,
            h,
            mesh,
            border_conditions,
            None,
            None,
            scheme,
            method,
            bandwidth,
            backend_policy,
            resolver,
            aot_chunking_policy,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.initial_generate_wall_ms",
            initial_generate_begin,
        );
        copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "initial");
        if let Some(auto_chunking_policy) = auto_codegen_chunking_policy_for_sparse_bundle(
            &bundle,
            original_backend_policy,
            &aot_execution_policy,
            aot_build_policy,
            aot_chunking_policy,
        ) {
            let jacobian_instance = prepared_bvp_jacobian(
                symbolic_assembly_backend,
                param_name_refs.as_deref(),
                retry_param_values.clone(),
                banded_linear_solver_config,
                lambdify_telemetry_mode,
                aot_telemetry_mode,
                lambdify_execution_policy,
            );
            let auto_regenerate_begin = Instant::now();
            bundle = try_generate_sparse_bundle(
                jacobian_instance,
                retry_eq_system.clone(),
                retry_values.clone(),
                retry_arg.clone(),
                param_name_refs.as_deref(),
                t0,
                n_steps,
                h,
                retry_mesh.clone(),
                retry_border_conditions.clone(),
                None,
                None,
                retry_scheme.clone(),
                retry_method.clone(),
                bandwidth,
                backend_policy,
                resolver,
                auto_chunking_policy,
            )?;
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.handoff.auto_chunk_regenerate_wall_ms",
                auto_regenerate_begin,
            );
            copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "auto_chunk");
        }
        let build_policy_begin = Instant::now();
        let (bundle, updated_resolver) = enforce_build_policy_on_sparse_bundle(
            bundle,
            resolver,
            aot_build_policy,
            aot_compile_config,
            aot_codegen_backend,
            aot_c_compiler,
            atom_profile,
            &mut handoff_diagnostics,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.build_policy_wall_ms",
            build_policy_begin,
        );
        let bundle = match updated_resolver.as_ref() {
            Some(updated_resolver_ref) if backend_policy_targets_aot(original_backend_policy) => {
                let rebind_begin = Instant::now();
                let bound = activate_compiled_sparse_bundle_after_aot_build(
                    bundle,
                    updated_resolver_ref,
                    aot_codegen_backend,
                )?;
                insert_elapsed_ms(
                    &mut handoff_diagnostics,
                    "generated.handoff.post_build_rebind_wall_ms",
                    rebind_begin,
                );
                bound
            }
            _ => bundle,
        };
        let execution_bind_begin = Instant::now();
        let mut bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.execution_bind_wall_ms",
            execution_bind_begin,
        );
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.total_wall_ms",
            handoff_begin,
        );
        bundle.runtime_diagnostics.extend(handoff_diagnostics);
        Ok(frozen_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values,
            banded_linear_solver_config,
            lambdify_telemetry_mode,
            aot_telemetry_mode,
            lambdify_execution_policy,
        );
        let legacy_bundle = jacobian_instance.generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            None,
            n_steps,
            h,
            mesh,
            border_conditions,
            None,
            None,
            scheme,
            method,
            bandwidth,
        );
        Ok(frozen_state_from_legacy_solver_bundle(legacy_bundle))
    }
}

/// Returns the current default backend policy for solver-side sparse handoff.
///
/// Sparse mainline is now allowed to prefer compiled AOT artifacts when they
/// become available, while still falling back cleanly to lambdified callbacks.
pub fn backend_policy_for_method(method: &str) -> BackendSelectionPolicy {
    if method == "Sparse" {
        BackendSelectionPolicy::PreferAotThenLambdify
    } else {
        BackendSelectionPolicy::LambdifyOnly
    }
}

fn backend_policy_targets_aot(policy: BackendSelectionPolicy) -> bool {
    matches!(
        policy,
        BackendSelectionPolicy::AotOnly
            | BackendSelectionPolicy::PreferAotThenLambdify
            | BackendSelectionPolicy::PreferAotThenNumeric
    )
}

fn aot_lifecycle_needs_serialization(
    _backend_policy: BackendSelectionPolicy,
    build_policy: AotBuildPolicy,
) -> bool {
    matches!(
        build_policy,
        AotBuildPolicy::BuildIfMissing { .. }
            | AotBuildPolicy::RequirePrebuilt
            | AotBuildPolicy::RebuildAlways { .. }
    )
}

fn effective_backend_policy_for_build(
    backend_policy: BackendSelectionPolicy,
    build_policy: AotBuildPolicy,
) -> BackendSelectionPolicy {
    match build_policy {
        AotBuildPolicy::UseIfAvailable => backend_policy,
        AotBuildPolicy::BuildIfMissing { .. } | AotBuildPolicy::RebuildAlways { .. } => {
            if backend_policy_targets_aot(backend_policy) {
                BackendSelectionPolicy::AotOnly
            } else {
                backend_policy
            }
        }
        AotBuildPolicy::RequirePrebuilt => {
            if backend_policy_targets_aot(backend_policy) {
                BackendSelectionPolicy::AotOnly
            } else {
                backend_policy
            }
        }
    }
}

fn parameter_name_refs(param_names: Option<&Vec<String>>) -> Option<Vec<&str>> {
    param_names.map(|names| names.iter().map(|name| name.as_str()).collect())
}

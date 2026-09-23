fn try_materialize_and_build_sparse_aot_bundle(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    profile: AotBuildProfile,
    compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    atom_profile: AtomOptimizationProfile,
    diagnostics: &mut HashMap<String, String>,
) -> Result<AotResolver, BvpBackendIntegrationError> {
    let selected = bundle.execution.selected();
    let problem_key = selected.problem_key();
    diagnostics.insert(
        "generated.aot.preparation_route".to_string(),
        format!("{:?}", selected.preparation_route()),
    );
    let manifest_begin = Instant::now();
    let manifest = selected
        .prepared_problem
        .manifest_for_matrix_backend(selected.matrix_backend);
    insert_elapsed_ms(diagnostics, "generated.aot.manifest_ms", manifest_begin);
    let mut registry = resolver
        .map(|existing| existing.registry().clone())
        .unwrap_or_else(AotRegistry::new);
    match aot_codegen_backend {
        AotCodegenBackend::Rust => {
            let artifact_begin = Instant::now();
            let (request, breakdown) = rust_sparse_aot_build_request(
                bundle,
                &problem_key,
                profile,
                compile_config,
                atom_profile,
            );
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse Rust AOT crate for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
                &selected.prepared_problem,
            );
            info!(
                "executing sparse Rust AOT build for problem_key={} in crate_dir={}",
                problem_key,
                build.written.crate_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} Rust key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.crate_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
                &selected.prepared_problem,
            );
            let registration_begin = Instant::now();
            let registered = registry
                .register_materialized_build(manifest, &build)
                .clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_banded_cdylib_backend(&registered),
                _ => register_generated_sparse_cdylib_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} Rust AOT build succeeded for problem_key={} but runtime cdylib registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
                insert_aot_stage_elapsed_ms(
                    diagnostics,
                    "generated.aot.register_link_ms",
                    registration_begin,
                    &selected.prepared_problem,
                );
                return Err(BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: format!("Rust runtime registration failed: {err}"),
                });
            }
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
                &selected.prepared_problem,
            );
        }
        AotCodegenBackend::C => {
            let mut c_compile = match profile {
                AotBuildProfile::Debug => CAotCompileConfig::dev_fastest(),
                AotBuildProfile::Release => to_c_compile_config(&compile_config),
            };
            if let Some(compiler) = aot_c_compiler {
                c_compile = c_compile.with_compiler(compiler);
            }
            let artifact_begin = Instant::now();
            let (request, breakdown) =
                c_sparse_aot_build_request(bundle, &problem_key, profile, c_compile, atom_profile);
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse C AOT library for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
                &selected.prepared_problem,
            );
            info!(
                "executing sparse C AOT build for problem_key={} in library_dir={}",
                problem_key,
                build.written.library_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} C key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.library_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
                &selected.prepared_problem,
            );
            let registration_begin = Instant::now();
            let registered = register_c_build_in_registry(&mut registry, manifest, &build).clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_c_banded_backend(&registered),
                _ => register_generated_c_sparse_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} C AOT build succeeded for problem_key={} but runtime registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
                insert_aot_stage_elapsed_ms(
                    diagnostics,
                    "generated.aot.register_link_ms",
                    registration_begin,
                    &selected.prepared_problem,
                );
                return Err(BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: format!("C runtime registration failed: {err}"),
                });
            }
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
                &selected.prepared_problem,
            );
        }
        AotCodegenBackend::Zig => {
            let artifact_begin = Instant::now();
            let (request, breakdown) =
                zig_sparse_aot_build_request(bundle, &problem_key, profile, atom_profile);
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse Zig AOT library for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
                &selected.prepared_problem,
            );
            info!(
                "executing sparse Zig AOT build for problem_key={} in library_dir={}",
                problem_key,
                build.written.library_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} Zig key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.library_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
                &selected.prepared_problem,
            );
            let registration_begin = Instant::now();
            let registered =
                register_zig_build_in_registry(&mut registry, manifest, &build).clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_zig_banded_backend(&registered),
                _ => register_generated_zig_sparse_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} Zig AOT build succeeded for problem_key={} but runtime registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
                insert_aot_stage_elapsed_ms(
                    diagnostics,
                    "generated.aot.register_link_ms",
                    registration_begin,
                    &selected.prepared_problem,
                );
                return Err(BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: format!("Zig runtime registration failed: {err}"),
                });
            }
            insert_aot_stage_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
                &selected.prepared_problem,
            );
        }
    }
    append_typed_aot_telemetry(diagnostics, &selected.prepared_problem);
    info!(
        "sparse {:?} AOT build succeeded for problem_key={} with profile={:?}",
        aot_codegen_backend, problem_key, profile
    );
    Ok(AotResolver::new(registry))
}

fn try_link_sparse_runtime_from_resolution(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    aot_codegen_backend: AotCodegenBackend,
) -> Result<bool, BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    if try_resolve_linked_sparse_backend(problem_key.as_str())
        .map_err(|error| BvpBackendIntegrationError::PipelinePanicked(error.to_string()))?
        .is_some()
    {
        return Ok(true);
    }

    let resolved = bundle
        .resolved_aot_artifact()
        .cloned()
        .or_else(|| resolver.map(|value| value.resolve_by_problem_key(problem_key.as_str())));
    let Some(resolved) = resolved else {
        return Ok(false);
    };
    if !resolved.is_compiled() {
        error!(
            "resolved sparse generated {:?} artifact for problem_key={} is not compiled/callable by contract: {}",
            aot_codegen_backend,
            problem_key,
            resolved.registered.lifecycle_contract_summary()
        );
        return Ok(false);
    }

    match register_sparse_runtime_from_registered_artifact(
        &resolved.registered,
        bundle.execution.selected().matrix_backend,
        aot_codegen_backend,
    ) {
        Ok(_) => Ok(true),
        Err(err) => {
            error!(
                "failed to register sparse generated {:?} runtime for problem_key={}: {}; artifact_contract={}",
                aot_codegen_backend,
                problem_key,
                err,
                resolved.registered.lifecycle_contract_summary()
            );
            Ok(false)
        }
    }
}

fn sparse_runtime_available(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    aot_codegen_backend: AotCodegenBackend,
) -> Result<bool, BvpBackendIntegrationError> {
    if bundle.is_runtime_callable() {
        return Ok(true);
    }
    if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled) {
        try_link_sparse_runtime_from_resolution(bundle, resolver, aot_codegen_backend)
    } else {
        Ok(false)
    }
}

fn enforce_build_policy_on_sparse_bundle(
    bundle: BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    build_policy: AotBuildPolicy,
    compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    atom_profile: AtomOptimizationProfile,
    diagnostics: &mut HashMap<String, String>,
) -> Result<(BvpSparseSolverBundle, Option<AotResolver>), BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    let effective_backend = bundle.effective_backend();
    if matches!(effective_backend, SelectedBackendKind::AotCompiled) {
        record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::Planned);
    }
    info!(
        "enforcing sparse AOT build policy {} for problem_key={} with effective_backend={:?}",
        build_policy.as_str(),
        problem_key,
        effective_backend
    );

    match build_policy {
        AotBuildPolicy::UseIfAvailable => {
            let runtime_available =
                sparse_runtime_available(&bundle, resolver, aot_codegen_backend)?;
            if runtime_available {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheHit);
            } else {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheMiss);
            }
            if matches!(effective_backend, SelectedBackendKind::AotCompiled) && !runtime_available {
                error!(
                    "compiled sparse AOT backend resolved for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key })
            } else {
                info!(
                    "reusing available sparse backend for problem_key={} under UseIfAvailable",
                    problem_key
                );
                Ok((bundle, None))
            }
        }
        AotBuildPolicy::RequirePrebuilt => {
            let runtime_available =
                sparse_runtime_available(&bundle, resolver, aot_codegen_backend)?;
            if runtime_available {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheHit);
            } else {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheMiss);
            }
            if !matches!(effective_backend, SelectedBackendKind::AotCompiled) {
                error!(
                    "RequirePrebuilt requested for problem_key={} but compiled backend is not available: {:?}",
                    problem_key, effective_backend
                );
                Err(
                    BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
                        problem_key,
                        effective_backend,
                    },
                )
            } else if !runtime_available {
                error!(
                    "RequirePrebuilt succeeded in resolution for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key })
            } else {
                info!(
                    "using prebuilt sparse AOT backend for problem_key={}",
                    problem_key
                );
                Ok((bundle, None))
            }
        }
        AotBuildPolicy::BuildIfMissing { .. } => {
            let runtime_available =
                sparse_runtime_available(&bundle, resolver, aot_codegen_backend)?;
            if matches!(effective_backend, SelectedBackendKind::AotCompiled) && runtime_available {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheHit);
                info!(
                    "compiled sparse AOT backend already callable for problem_key={}, skipping build",
                    problem_key
                );
                Ok((bundle, None))
            } else {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::CacheMiss);
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::BuildStarted);
                let profile = match build_policy {
                    AotBuildPolicy::BuildIfMissing { profile } => profile,
                    _ => unreachable!(),
                };
                // On Windows, an already loaded generated cdylib keeps the .dll file locked.
                // If the compiled artifact is present but not currently callable for this bundle,
                // drop any stale linked backend before rebuilding so cargo can overwrite it.
                try_unregister_linked_sparse_backend(problem_key.as_str()).map_err(|error| {
                    BvpBackendIntegrationError::PipelinePanicked(error.to_string())
                })?;
                info!(
                    "compiled sparse AOT backend missing or not callable for problem_key={}, building with profile={:?}",
                    problem_key, profile
                );
                let updated_resolver = match try_materialize_and_build_sparse_aot_bundle(
                    &bundle,
                    resolver,
                    profile,
                    compile_config.clone(),
                    aot_codegen_backend,
                    aot_c_compiler.clone(),
                    atom_profile,
                    diagnostics,
                ) {
                    Ok(resolver) => resolver,
                    Err(error) => {
                        record_bundle_aot_failure(&bundle, &error);
                        return Err(error);
                    }
                };
                let runtime_available = if bundle.is_runtime_callable() {
                    true
                } else {
                    try_link_sparse_runtime_from_resolution(
                        &bundle,
                        Some(&updated_resolver),
                        aot_codegen_backend,
                    )?
                };
                if runtime_available {
                    record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::BuildSucceeded);
                    record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::Published);
                    info!(
                        "compiled sparse AOT backend became callable after BuildIfMissing for problem_key={}",
                        problem_key
                    );
                    Ok((bundle, Some(updated_resolver)))
                } else {
                    error!(
                        "BuildIfMissing completed for problem_key={} but runtime callbacks are still unavailable",
                        problem_key
                    );
                    Err(BvpBackendIntegrationError::AutomaticAotBuildRequested { problem_key })
                }
            }
        }
        AotBuildPolicy::RebuildAlways { .. } => {
            record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::BuildStarted);
            let profile = match build_policy {
                AotBuildPolicy::RebuildAlways { profile } => profile,
                _ => unreachable!(),
            };
            // Forced rebuild must also unload any previously linked generated cdylib;
            // otherwise Windows denies replacing the existing .dll on disk.
            try_unregister_linked_sparse_backend(problem_key.as_str())
                .map_err(|error| BvpBackendIntegrationError::PipelinePanicked(error.to_string()))?;
            info!(
                "forcing sparse AOT rebuild for problem_key={} with profile={:?}",
                problem_key, profile
            );
            let updated_resolver = match try_materialize_and_build_sparse_aot_bundle(
                &bundle,
                resolver,
                profile,
                compile_config,
                aot_codegen_backend,
                aot_c_compiler,
                atom_profile,
                diagnostics,
            ) {
                Ok(resolver) => resolver,
                Err(error) => {
                    record_bundle_aot_failure(&bundle, &error);
                    return Err(error);
                }
            };
            let runtime_available = if bundle.is_runtime_callable() {
                true
            } else {
                try_link_sparse_runtime_from_resolution(
                    &bundle,
                    Some(&updated_resolver),
                    aot_codegen_backend,
                )?
            };
            if runtime_available {
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::BuildSucceeded);
                record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::Published);
                info!(
                    "compiled sparse AOT backend remains callable after forced rebuild for problem_key={}",
                    problem_key
                );
                Ok((bundle, Some(updated_resolver)))
            } else {
                error!(
                    "forced sparse AOT rebuild completed for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::AutomaticAotRebuildRequested { problem_key })
            }
        }
    }
}

fn apply_execution_policy_to_sparse_bundle(
    mut bundle: BvpSparseSolverBundle,
    policy: AotExecutionPolicy,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    info!(
        "applying sparse AOT execution policy {} for problem_key={} with effective_backend={:?}",
        policy.as_str(),
        problem_key,
        bundle.effective_backend()
    );
    match policy {
        AotExecutionPolicy::Auto => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled) {
                let auto_plan = bundle
                    .execution
                    .selected()
                    .prepared_problem
                    .auto_parallel_plan();
                if let Some(config) = auto_plan.executor_config {
                    if !bundle.rebind_linked_runtime_callbacks(None, Some(config)) {
                        return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                            problem_key,
                        });
                    }
                    bundle.refresh_linked_runtime_diagnostics(policy.as_str(), Some(config));
                    info!(
                        "auto-selected sparse parallel runtime binding for problem_key={} with residual_jobs={:?}, sparse_jobs={:?}, residual_chunking={:?}, sparse_chunking={:?}, residual_reason={}, sparse_reason={}, residual_work_per_job={}, sparse_work_per_job={}, min_work_per_job={}, workers={}",
                        problem_key,
                        config.max_residual_jobs,
                        config.max_sparse_jobs,
                        auto_plan.residual_chunking,
                        auto_plan.sparse_chunking,
                        auto_plan.residual_stage.reason.as_str(),
                        auto_plan.sparse_stage.reason.as_str(),
                        auto_plan.residual_stage.work_per_job,
                        auto_plan.sparse_stage.work_per_job,
                        auto_plan.min_work_per_job,
                        auto_plan.workers
                    );
                } else if !bundle.rebind_linked_runtime_callbacks(None, None) {
                    return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                        problem_key,
                    });
                } else {
                    bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
                    info!(
                        "auto-selected sequential sparse runtime binding for problem_key={} with residual_reason={}, sparse_reason={}, residual_work_per_job={}, sparse_work_per_job={}, min_work_per_job={} and workers={}",
                        problem_key,
                        auto_plan.residual_stage.reason.as_str(),
                        auto_plan.sparse_stage.reason.as_str(),
                        auto_plan.residual_stage.work_per_job,
                        auto_plan.sparse_stage.work_per_job,
                        auto_plan.min_work_per_job,
                        auto_plan.workers
                    );
                }
            } else {
                bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
                info!(
                    "keeping default sparse runtime callback binding for problem_key={}",
                    problem_key
                );
            }
        }
        AotExecutionPolicy::SequentialOnly => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled)
                && !bundle.rebind_linked_runtime_callbacks(None, None)
            {
                return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                    problem_key,
                });
            }
            bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
            info!(
                "bound sparse runtime callbacks sequentially for problem_key={}",
                problem_key
            );
        }
        AotExecutionPolicy::Parallel(config) => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled)
                && !bundle.rebind_linked_runtime_callbacks(None, Some(config))
            {
                return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                    problem_key,
                });
            }
            bundle.refresh_linked_runtime_diagnostics(policy.as_str(), Some(config));
            info!(
                "bound sparse runtime callbacks in parallel for problem_key={}",
                problem_key
            );
        }
    }
    if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled) {
        record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::Linked);
        record_bundle_aot_lifecycle(&bundle, BvpAotLifecycleEvent::RuntimeReady);
    }
    Ok(bundle)
}

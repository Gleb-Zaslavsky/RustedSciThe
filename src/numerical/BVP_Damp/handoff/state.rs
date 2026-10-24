/// Converts the assembled sparse symbolic/provider bundle into a damped solver runtime state.
pub fn damped_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> DampedGeneratedSolverState {
    try_damped_state_from_sparse_solver_bundle(
        bundle,
        updated_resolver,
        crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::SolverNative,
    )
    .unwrap_or_else(|error| panic!("{error:?}"))
}

/// Fallible conversion that may replace solver-native callbacks with the opt-in
/// reusable Atom runtime while retaining the generated Banded linear solver.
pub fn try_damped_state_from_sparse_solver_bundle(
    mut bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
    callback_runtime: crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime,
) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
    let selected_backend = bundle.effective_backend();
    let bounds_vec = bundle.bounds_vec.clone().unwrap_or_default();
    let rel_tolerance_vec = bundle.rel_tolerance_vec.clone().unwrap_or_default();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let bc_position_and_value = bundle.bc_position_and_value.clone();
    let mut runtime_diagnostics = bundle.runtime_diagnostics().clone();
    let generation_telemetry = bundle.generation_telemetry;
    let atom_discretization_telemetry = bundle.atom_discretization_telemetry;
    let legacy_lambdify_telemetry = bundle.legacy_lambdify_telemetry.clone();
    let atom_lambdify_telemetry = bundle.atom_lambdify_telemetry.clone();
    let direct_banded_jacobian_telemetry = bundle.direct_banded_jacobian_telemetry.clone();
    let aot_telemetry = bundle.aot_telemetry.clone();
    let parameter_binding = bundle.parameter_binding.clone();

    let (fun, jac) = match callback_runtime {
        crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::SolverNative => {
            let (fun, jac) = bundle.into_runtime_callbacks().ok_or_else(|| {
                BvpBackendIntegrationError::PipelinePanicked(
                    "BVP sparse solver bundle did not provide runtime callbacks".to_string(),
                )
            })?;
            runtime_diagnostics.insert(
                "generated.callback_runtime".to_string(),
                "SolverNative".to_string(),
            );
            (fun, jac)
        }
        crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::PreparedAtom => {
            if selected_backend != SelectedBackendKind::Lambdify {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "callback_runtime".to_string(),
                    value: "PreparedAtom".to_string(),
                    message: format!(
                        "PreparedAtom requires a Lambdify callback bundle, selected {selected_backend:?}"
                    ),
                });
            }
            let discretized = bundle
                .execution
                .selected()
                .prepared_problem
                .atom_discretized_system
                .as_ref()
                .ok_or_else(
                    || BvpBackendIntegrationError::PreparedCallbackPreparationFailed {
                        runtime: "PreparedAtom".to_string(),
                        message: "selected AtomView bundle has no discretized Atom system"
                            .to_string(),
                    },
                )?;
            let delegate = bundle.jacobian_function.take().ok_or_else(|| {
                BvpBackendIntegrationError::PreparedCallbackPreparationFailed {
                    runtime: "PreparedAtom".to_string(),
                    message: "selected Banded bundle has no Jacobian solve delegate".to_string(),
                }
            })?;
            let callbacks =
                crate::numerical::BVP_Damp::prepared_atom_callbacks::try_build_callbacks(
                    discretized,
                    delegate,
                    bandwidth,
                )
                .map_err(|message| {
                    BvpBackendIntegrationError::PreparedCallbackPreparationFailed {
                        runtime: "PreparedAtom".to_string(),
                        message,
                    }
                })?;
            bundle.residual_function.take();
            runtime_diagnostics.insert(
                "generated.callback_runtime".to_string(),
                "PreparedAtom".to_string(),
            );
            runtime_diagnostics.insert(
                "generated.prepared_atom.prepare_wall_ms".to_string(),
                callbacks
                    .preparation_time
                    .as_secs_f64()
                    .mul_add(1000.0, 0.0)
                    .to_string(),
            );
            runtime_diagnostics.insert(
                "generated.prepared_atom.jacobian_nnz".to_string(),
                callbacks.jacobian_nnz.to_string(),
            );
            (callbacks.fun, callbacks.jac)
        }
    };

    Ok(DampedGeneratedSolverState {
        fun,
        jac: Some(jac),
        bounds_vec,
        rel_tolerance_vec,
        variable_string,
        bandwidth,
        bc_position_and_value,
        updated_resolver,
        selected_backend,
        runtime_diagnostics,
        generation_telemetry,
        atom_discretization_telemetry,
        legacy_lambdify_telemetry: if callback_runtime == crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::PreparedAtom { None } else { legacy_lambdify_telemetry },
        atom_lambdify_telemetry: if callback_runtime == crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::PreparedAtom { None } else { atom_lambdify_telemetry },
        direct_banded_jacobian_telemetry: if callback_runtime == crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedCallbackRuntime::PreparedAtom { None } else { direct_banded_jacobian_telemetry },
        aot_telemetry,
        parameter_binding,
    })
}

/// Compatibility helper for callers that still hold a legacy [`Jacobian`] instance.
///
/// New code should prefer bundle-based handoff paths instead of converting from
/// raw Jacobian state directly.
pub fn damped_state_from_legacy_jacobian(
    jacobian_instance: Jacobian,
) -> DampedGeneratedSolverState {
    damped_state_from_legacy_solver_bundle(jacobian_instance.into_legacy_solver_bundle())
}

/// Converts a centralized legacy symbolic bundle into the damped solver runtime state.
pub fn damped_state_from_legacy_solver_bundle(
    bundle: BvpLegacySolverBundle,
) -> DampedGeneratedSolverState {
    DampedGeneratedSolverState {
        fun: bundle.residual_function,
        jac: bundle.jacobian_function,
        bounds_vec: bundle.bounds_vec.unwrap_or_default(),
        rel_tolerance_vec: bundle.rel_tolerance_vec.unwrap_or_default(),
        variable_string: bundle.variable_string,
        bandwidth: bundle.bandwidth.unwrap_or((0, 0)),
        bc_position_and_value: bundle.bc_position_and_value,
        updated_resolver: None,
        selected_backend: SelectedBackendKind::Lambdify,
        runtime_diagnostics: HashMap::new(),
        generation_telemetry: None,
        atom_discretization_telemetry: None,
        legacy_lambdify_telemetry: bundle.legacy_lambdify_telemetry,
        atom_lambdify_telemetry: bundle.atom_lambdify_telemetry,
        direct_banded_jacobian_telemetry: bundle.direct_banded_jacobian_telemetry,
        aot_telemetry: None,
        parameter_binding: bundle.parameter_binding,
    }
}

/// Converts the assembled sparse symbolic/provider bundle into a frozen solver runtime state.
pub fn frozen_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> FrozenGeneratedSolverState {
    let selected_backend = bundle.effective_backend();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let runtime_diagnostics = bundle.runtime_diagnostics().clone();
    let generation_telemetry = bundle.generation_telemetry;
    let atom_discretization_telemetry = bundle.atom_discretization_telemetry;
    let legacy_lambdify_telemetry = bundle.legacy_lambdify_telemetry.clone();
    let atom_lambdify_telemetry = bundle.atom_lambdify_telemetry.clone();
    let direct_banded_jacobian_telemetry = bundle.direct_banded_jacobian_telemetry.clone();
    let parameter_binding = bundle.parameter_binding.clone();

    let (fun, jac) = bundle.into_runtime_callbacks().unwrap_or_else(|| {
        panic!("Frozen BVP sparse solver bundle did not provide runtime callbacks")
    });

    FrozenGeneratedSolverState {
        fun,
        jac: Some(jac),
        variable_string,
        bandwidth,
        updated_resolver,
        selected_backend,
        runtime_diagnostics,
        generation_telemetry,
        atom_discretization_telemetry,
        legacy_lambdify_telemetry,
        atom_lambdify_telemetry,
        direct_banded_jacobian_telemetry,
        parameter_binding,
    }
}

/// Compatibility helper for callers that still hold a legacy [`Jacobian`] instance.
///
/// New code should prefer bundle-based handoff paths instead of converting from
/// raw Jacobian state directly.
pub fn frozen_state_from_legacy_jacobian(
    jacobian_instance: Jacobian,
) -> FrozenGeneratedSolverState {
    frozen_state_from_legacy_solver_bundle(jacobian_instance.into_legacy_solver_bundle())
}

/// Converts a centralized legacy symbolic bundle into the frozen solver runtime state.
pub fn frozen_state_from_legacy_solver_bundle(
    bundle: BvpLegacySolverBundle,
) -> FrozenGeneratedSolverState {
    FrozenGeneratedSolverState {
        fun: bundle.residual_function,
        jac: bundle.jacobian_function,
        variable_string: bundle.variable_string,
        bandwidth: bundle.bandwidth.unwrap_or((0, 0)),
        updated_resolver: None,
        selected_backend: SelectedBackendKind::Lambdify,
        runtime_diagnostics: HashMap::new(),
        generation_telemetry: None,
        atom_discretization_telemetry: None,
        legacy_lambdify_telemetry: bundle.legacy_lambdify_telemetry,
        atom_lambdify_telemetry: bundle.atom_lambdify_telemetry,
        direct_banded_jacobian_telemetry: bundle.direct_banded_jacobian_telemetry,
        parameter_binding: bundle.parameter_binding,
    }
}

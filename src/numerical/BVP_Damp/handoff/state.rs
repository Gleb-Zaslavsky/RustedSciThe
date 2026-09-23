/// Converts the assembled sparse symbolic/provider bundle into a damped solver runtime state.
pub fn damped_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> DampedGeneratedSolverState {
    let selected_backend = bundle.effective_backend();
    let bounds_vec = bundle.bounds_vec.clone().unwrap_or_default();
    let rel_tolerance_vec = bundle.rel_tolerance_vec.clone().unwrap_or_default();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let bc_position_and_value = bundle.bc_position_and_value.clone();
    let runtime_diagnostics = bundle.runtime_diagnostics().clone();
    let generation_telemetry = bundle.generation_telemetry;
    let atom_discretization_telemetry = bundle.atom_discretization_telemetry;
    let legacy_lambdify_telemetry = bundle.legacy_lambdify_telemetry.clone();
    let atom_lambdify_telemetry = bundle.atom_lambdify_telemetry.clone();
    let direct_banded_jacobian_telemetry = bundle.direct_banded_jacobian_telemetry.clone();
    let aot_telemetry = bundle.aot_telemetry.clone();
    let parameter_binding = bundle.parameter_binding.clone();

    let (fun, jac) = bundle
        .into_runtime_callbacks()
        .unwrap_or_else(|| panic!("BVP sparse solver bundle did not provide runtime callbacks"));

    DampedGeneratedSolverState {
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
        legacy_lambdify_telemetry,
        atom_lambdify_telemetry,
        direct_banded_jacobian_telemetry,
        aot_telemetry,
        parameter_binding,
    }
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

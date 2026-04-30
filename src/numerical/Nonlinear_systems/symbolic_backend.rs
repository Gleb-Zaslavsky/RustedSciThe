//! Backend-selection layer for symbolic nonlinear-system problems.
//!
//! This module keeps three concerns separate:
//! - [`crate::numerical::Nonlinear_systems::symbolic`] builds symbolic problems
//!   and prepared dense AOT bridge objects,
//! - generic codegen lifecycle layers own build/registry/resolution metadata,
//! - this module decides which backend branch the nonlinear-system caller
//!   should take next.
//!
//! The goal is to give `Nonlinear_systems` the same architectural shape that
//! already proved useful in the BVP stack:
//! - symbolic setup stays local,
//! - backend policy is explicit,
//! - AOT readiness is resolved through registry metadata,
//! - solver code does not have to understand codegen details.
//!
//! Practical dataflow:
//! - [`crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem`]
//!   stays the symbolic source of truth,
//! - this module asks it for one dense prepared AOT bridge when policy needs
//!   the compiled branch,
//! - shared registry/resolution layers answer whether the compiled artifact is
//!   missing, merely registered, or truly available,
//! - the caller then either keeps the lambdify path or upgrades to the linked
//!   compiled dense backend.

use crate::numerical::Nonlinear_systems::symbolic::{
    PreparedSymbolicNonlinearAotProblem, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
};
use crate::symbolic::codegen::codegen_aot_resolution::{
    AotResolutionStatus, AotResolver, ResolvedAotArtifact,
};
use crate::symbolic::codegen::codegen_provider_api::PreparedProblem;
use log::{info, warn};

/// User-facing backend preference for symbolic nonlinear problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicBackendSelectionPolicy {
    /// Force the existing lambdify backend.
    LambdifyOnly,
    /// Require an AOT backend branch.
    AotOnly,
    /// Prefer AOT when available, otherwise fall back to lambdify.
    PreferAotThenLambdify,
}

/// Effective backend branch chosen for a symbolic nonlinear problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedSymbolicNonlinearBackendKind {
    /// Continue through the current lambdify backend.
    Lambdify,
    /// A compiled AOT artifact is available for the prepared dense problem.
    AotCompiled,
    /// A generated crate is known, but the compiled artifact is still missing.
    AotRegisteredButNotBuilt,
    /// No AOT artifact is known for this prepared dense problem.
    AotMissing,
}

/// Result of symbolic nonlinear backend selection.
#[derive(Debug, Clone)]
pub struct SelectedSymbolicNonlinearBackend<'a> {
    /// User-facing selection policy that was applied.
    pub requested_policy: SymbolicBackendSelectionPolicy,
    /// Effective backend branch after resolution/fallback.
    pub effective_backend: SelectedSymbolicNonlinearBackendKind,
    /// Dense prepared AOT bridge when the selection considered the AOT path.
    pub prepared_aot_problem: Option<PreparedSymbolicNonlinearAotProblem<'a>>,
    /// Optional resolved AOT artifact metadata.
    pub aot_resolution: Option<ResolvedAotArtifact>,
}

impl<'a> SelectedSymbolicNonlinearBackend<'a> {
    /// Returns `true` when the selected branch points to a compiled AOT artifact.
    pub fn is_compiled_aot(&self) -> bool {
        self.effective_backend == SelectedSymbolicNonlinearBackendKind::AotCompiled
    }
}

/// Selects the backend branch for one symbolic nonlinear problem.
///
/// The nonlinear problem itself remains solver-facing and backend-agnostic.
/// This function prepares the dense AOT bridge only when policy requires it,
/// then uses the shared registry/resolution infrastructure to determine
/// whether the compiled branch is already available.
pub fn select_symbolic_nonlinear_backend<'a>(
    problem: &'a SymbolicNonlinearProblem,
    policy: SymbolicBackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_options: SymbolicDenseAotOptions,
) -> SelectedSymbolicNonlinearBackend<'a> {
    match policy {
        SymbolicBackendSelectionPolicy::LambdifyOnly => {
            info!("Selected lambdify backend for symbolic nonlinear problem");
            SelectedSymbolicNonlinearBackend {
                requested_policy: policy,
                effective_backend: SelectedSymbolicNonlinearBackendKind::Lambdify,
                prepared_aot_problem: None,
                aot_resolution: None,
            }
        }
        SymbolicBackendSelectionPolicy::AotOnly
        | SymbolicBackendSelectionPolicy::PreferAotThenLambdify => {
            let prepared_aot_problem = problem.prepare_dense_aot_problem(aot_options);
            let generic_prepared =
                PreparedProblem::dense(prepared_aot_problem.as_prepared_problem());
            let resolution =
                resolver.map(|resolver| resolver.resolve_prepared_problem(&generic_prepared));
            let effective_backend = match resolution.as_ref().map(|resolved| resolved.status) {
                Some(AotResolutionStatus::Compiled) => {
                    info!("Selected compiled AOT backend for symbolic nonlinear problem");
                    SelectedSymbolicNonlinearBackendKind::AotCompiled
                }
                Some(AotResolutionStatus::RegisteredButNotBuilt) => {
                    warn!(
                        "Selected nonlinear AOT backend, but artifact is registered and not built"
                    );
                    SelectedSymbolicNonlinearBackendKind::AotRegisteredButNotBuilt
                }
                Some(AotResolutionStatus::Missing) | None => match policy {
                    SymbolicBackendSelectionPolicy::AotOnly => {
                        warn!("AOT-only nonlinear backend requested, but artifact is missing");
                        SelectedSymbolicNonlinearBackendKind::AotMissing
                    }
                    SymbolicBackendSelectionPolicy::PreferAotThenLambdify => {
                        info!(
                            "AOT artifact missing for symbolic nonlinear problem; falling back to lambdify"
                        );
                        SelectedSymbolicNonlinearBackendKind::Lambdify
                    }
                    SymbolicBackendSelectionPolicy::LambdifyOnly => unreachable!(),
                },
            };

            SelectedSymbolicNonlinearBackend {
                requested_policy: policy,
                effective_backend,
                prepared_aot_problem: Some(prepared_aot_problem),
                aot_resolution: resolution,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::{
        aot_solver_test_guard, linked_dense_resolver_for_problem,
        register_elementary_dense_backend, register_parameterized_dense_backend,
    };
    use crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
        AotBuildProfile, AotBuildRequest,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use std::fs;
    use tempfile::tempdir;

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_expressions_with_options(
            vec![
                Expr::parse_expression("x^2+y^2-10"),
                Expr::parse_expression("x-y-4"),
            ],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_lambdify_backend(),
        )
        .expect("problem should build")
    }

    #[test]
    fn nonlinear_backend_selection_can_stay_on_lambdify() {
        let problem = elementary_problem();
        let selected = select_symbolic_nonlinear_backend(
            &problem,
            SymbolicBackendSelectionPolicy::LambdifyOnly,
            None,
            SymbolicDenseAotOptions::default(),
        );

        assert_eq!(
            selected.effective_backend,
            SelectedSymbolicNonlinearBackendKind::Lambdify
        );
        assert!(selected.prepared_aot_problem.is_none());
        assert!(selected.aot_resolution.is_none());
    }

    #[test]
    fn nonlinear_backend_selection_falls_back_when_aot_is_missing() {
        let resolver = AotResolver::new(AotRegistry::new());
        let problem = elementary_problem();
        let selected = select_symbolic_nonlinear_backend(
            &problem,
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        );

        assert_eq!(
            selected.effective_backend,
            SelectedSymbolicNonlinearBackendKind::Lambdify
        );
        assert!(selected.prepared_aot_problem.is_some());
    }

    #[test]
    fn nonlinear_backend_selection_prefers_compiled_aot_when_registered() {
        let problem = elementary_problem();
        let prepared_bridge = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        let prepared = PreparedProblem::dense(prepared_bridge.as_prepared_problem());
        let manifest = prepared_bridge.manifest();

        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_nonlinear_dense_backend",
                "generated_nonlinear_dense_module",
                &prepared,
            ),
            dir.path(),
            AotBuildProfile::Release,
        )
        .materialize()
        .expect("build request should materialize");
        fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
        fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest, &build);
        let resolver = AotResolver::new(registry);

        let selected = select_symbolic_nonlinear_backend(
            &problem,
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        );

        assert_eq!(
            selected.effective_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert!(selected.is_compiled_aot());
        assert!(selected.prepared_aot_problem.is_some());
        assert!(selected.aot_resolution.is_some());
    }

    #[test]
    fn linked_compiled_dense_backend_evaluates_elementary_problem() {
        let _guard = aot_solver_test_guard();
        let baseline = elementary_problem();
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_elementary_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "y".to_string()]),
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled symbolic problem should build");

        assert_eq!(
            compiled.backend_kind(),
            crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind::Aot
        );
        let x0 = DVector::from_vec(vec![3.0, -1.0]);
        let residual = compiled.residual(&x0).expect("residual");
        let jacobian = compiled.jacobian(&x0).expect("jacobian");
        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 0)], 6.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 1)], -2.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 1)], -1.0, epsilon = 1e-12);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn linked_compiled_dense_backend_evaluates_parameterized_problem() {
        let _guard = aot_solver_test_guard();
        let symbolic = Expr::Symbols("x, y, a");
        let x = symbolic[0].clone();
        let y = symbolic[1].clone();
        let a = symbolic[2].clone();
        let options = SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![2.0]));
        let baseline = SymbolicNonlinearProblem::from_expressions_with_options(
            vec![
                a.clone() * x.clone() + y.clone() - Expr::Const(3.0),
                x.clone() - y.clone(),
            ],
            options.clone(),
        )
        .expect("baseline problem should build");
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_parameterized_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_expressions_with_backend_selection(
            vec![
                a.clone() * x.clone() + y.clone() - Expr::Const(3.0),
                x.clone() - y.clone(),
            ],
            options,
            SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled parameterized symbolic problem should build");

        assert_eq!(
            compiled.backend_kind(),
            crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind::Aot
        );
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let residual = compiled.residual(&x0).expect("residual");
        let jacobian = compiled.jacobian(&x0).expect("jacobian");
        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 1)], -1.0, epsilon = 1e-12);

        unregister_linked_dense_backend(&problem_key);
    }
}

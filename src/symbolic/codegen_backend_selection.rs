//! Thin backend-selection layer for solver-facing symbolic orchestration.
//!
//! This module is meant to be the first place where a future `Jacobian`
//! orchestration path can decide between:
//! - numeric closures,
//! - lambdified symbolic functions,
//! - and AOT-generated backends.
//!
//! It intentionally does not build symbolic tasks, does not generate code, and
//! does not load compiled libraries. Instead, it answers a narrower question:
//! given a prepared problem, a backend preference, and optionally an AOT
//! resolver, which backend branch should the higher layer take next?

use crate::symbolic::codegen_aot_resolution::{
    AotResolutionStatus, AotResolver, ResolvedAotArtifact,
};
use crate::symbolic::codegen_provider_api::{BackendKind, MatrixBackend, PreparedProblem};
use log::{info, warn};

/// User-facing backend preference policy for prepared symbolic problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelectionPolicy {
    NumericOnly,
    LambdifyOnly,
    AotOnly,
    PreferAotThenLambdify,
    PreferAotThenNumeric,
    PreferLambdifyThenNumeric,
}

/// Selected backend branch returned by the selection layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedBackendKind {
    Numeric,
    Lambdify,
    AotCompiled,
    AotRegisteredButNotBuilt,
    AotMissing,
}

/// Selection result that a future symbolic `Jacobian` layer can turn into a
/// concrete provider path.
#[derive(Debug, Clone)]
pub struct SelectedBackend<'a> {
    pub problem: PreparedProblem<'a>,
    pub requested_backend: BackendKind,
    pub effective_backend: SelectedBackendKind,
    pub matrix_backend: MatrixBackend,
    pub aot_resolution: Option<ResolvedAotArtifact>,
}

impl<'a> SelectedBackend<'a> {
    /// Returns `true` when the selected path points at a compiled AOT artifact.
    pub fn is_compiled_aot(&self) -> bool {
        self.effective_backend == SelectedBackendKind::AotCompiled
    }
}

/// Selects the backend branch for a prepared problem under one preference
/// policy. When AOT is involved, the optional resolver determines whether a
/// compiled artifact is already available.
pub fn select_backend<'a>(
    problem: &PreparedProblem<'a>,
    policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
) -> SelectedBackend<'a> {
    let selection = match policy {
        BackendSelectionPolicy::NumericOnly => {
            select_non_aot(problem, SelectedBackendKind::Numeric)
        }
        BackendSelectionPolicy::LambdifyOnly => {
            select_non_aot(problem, SelectedBackendKind::Lambdify)
        }
        BackendSelectionPolicy::AotOnly => select_aot(problem, resolver),
        BackendSelectionPolicy::PreferAotThenLambdify => {
            let selection = select_aot(problem, resolver);
            match selection.effective_backend {
                SelectedBackendKind::AotCompiled
                | SelectedBackendKind::AotRegisteredButNotBuilt => selection,
                SelectedBackendKind::AotMissing => {
                    select_non_aot(problem, SelectedBackendKind::Lambdify)
                }
                _ => unreachable!("AOT selection should only produce AOT states"),
            }
        }
        BackendSelectionPolicy::PreferAotThenNumeric => {
            let selection = select_aot(problem, resolver);
            match selection.effective_backend {
                SelectedBackendKind::AotCompiled
                | SelectedBackendKind::AotRegisteredButNotBuilt => selection,
                SelectedBackendKind::AotMissing => {
                    select_non_aot(problem, SelectedBackendKind::Numeric)
                }
                _ => unreachable!("AOT selection should only produce AOT states"),
            }
        }
        BackendSelectionPolicy::PreferLambdifyThenNumeric => {
            select_non_aot(problem, SelectedBackendKind::Lambdify)
        }
    };

    match selection.effective_backend {
        SelectedBackendKind::AotCompiled => info!(
            "Selected compiled AOT backend for {:?} matrix backend",
            selection.matrix_backend
        ),
        SelectedBackendKind::AotRegisteredButNotBuilt => {
            warn!("Selected AOT backend, but artifact is registered and not built yet")
        }
        SelectedBackendKind::AotMissing => warn!(
            "AOT artifact is missing; effective backend is {:?}",
            selection.requested_backend
        ),
        SelectedBackendKind::Numeric | SelectedBackendKind::Lambdify => info!(
            "Selected {:?} backend for {:?} matrix backend",
            selection.effective_backend, selection.matrix_backend
        ),
    }

    selection
}

fn select_non_aot<'a>(
    problem: &PreparedProblem<'a>,
    effective_backend: SelectedBackendKind,
) -> SelectedBackend<'a> {
    let requested_backend = match effective_backend {
        SelectedBackendKind::Numeric => BackendKind::Numeric,
        SelectedBackendKind::Lambdify => BackendKind::Lambdify,
        _ => unreachable!("non-AOT selection must resolve to numeric or lambdify"),
    };

    SelectedBackend {
        problem: problem.clone(),
        requested_backend,
        effective_backend,
        matrix_backend: problem.matrix_backend(),
        aot_resolution: None,
    }
}

fn select_aot<'a>(
    problem: &PreparedProblem<'a>,
    resolver: Option<&AotResolver>,
) -> SelectedBackend<'a> {
    let resolution = resolver.map(|resolver| resolver.resolve_prepared_problem(problem));
    let effective_backend = match resolution.as_ref().map(|resolved| resolved.status) {
        Some(AotResolutionStatus::Compiled) => SelectedBackendKind::AotCompiled,
        Some(AotResolutionStatus::RegisteredButNotBuilt) => {
            SelectedBackendKind::AotRegisteredButNotBuilt
        }
        Some(AotResolutionStatus::Missing) | None => SelectedBackendKind::AotMissing,
    };

    SelectedBackend {
        problem: problem.clone(),
        requested_backend: BackendKind::Aot,
        effective_backend,
        matrix_backend: problem.matrix_backend(),
        aot_resolution: resolution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
    use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::symbolic_engine::Expr;
    use std::fs;
    use tempfile::tempdir;

    fn sample_prepared_problem() -> PreparedProblem<'static> {
        let residuals = Box::leak(Box::new(vec![Expr::parse_expression("x + 1")]));
        let jacobian = Box::leak(Box::new(vec![vec![Expr::parse_expression("1")]]));
        let vars = Box::leak(Box::new(vec!["x"]));

        PreparedProblem::dense(PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals,
                variables: vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian,
                variables: vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        ))
    }

    #[test]
    fn selection_prefers_compiled_aot_when_artifact_exists() {
        let prepared = sample_prepared_problem();
        let manifest = crate::symbolic::codegen_manifest::PreparedProblemManifest::from(&prepared);
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_backend_select_fixture",
                "generated_backend_select_module",
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

        let selected = select_backend(
            &prepared,
            BackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
        );

        assert_eq!(selected.requested_backend, BackendKind::Aot);
        assert_eq!(selected.effective_backend, SelectedBackendKind::AotCompiled);
        assert!(selected.is_compiled_aot());
    }

    #[test]
    fn selection_falls_back_to_lambdify_when_aot_is_missing() {
        let prepared = sample_prepared_problem();
        let resolver = AotResolver::new(AotRegistry::new());

        let selected = select_backend(
            &prepared,
            BackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
        );

        assert_eq!(selected.requested_backend, BackendKind::Lambdify);
        assert_eq!(selected.effective_backend, SelectedBackendKind::Lambdify);
        assert!(selected.aot_resolution.is_none());
    }

    #[test]
    fn selection_keeps_registered_but_not_built_aot_for_aot_only_policy() {
        let prepared = sample_prepared_problem();
        let manifest = crate::symbolic::codegen_manifest::PreparedProblemManifest::from(&prepared);
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_backend_pending_fixture",
                "generated_backend_select_module",
                &prepared,
            ),
            dir.path(),
            AotBuildProfile::Release,
        )
        .materialize()
        .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest, &build);
        let resolver = AotResolver::new(registry);

        let selected = select_backend(&prepared, BackendSelectionPolicy::AotOnly, Some(&resolver));

        assert_eq!(selected.requested_backend, BackendKind::Aot);
        assert_eq!(
            selected.effective_backend,
            SelectedBackendKind::AotRegisteredButNotBuilt
        );
        assert!(selected.aot_resolution.is_some());
    }
}

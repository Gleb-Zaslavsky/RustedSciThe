//! High-level generated-backend orchestration for nonlinear symbolic problems.
//!
//! This module is the user-facing layer above:
//! - symbolic problem setup,
//! - backend selection,
//! - dense AOT lifecycle materialization/build,
//! - and resolver reuse.
//!
//! It gives `Nonlinear_systems` the same kind of ergonomic surface that the
//! newer BVP stack already has: callers can choose a simple mode such as
//! "defaults", "require prebuilt", or "build if missing" without manually
//! stitching together lifecycle layers.

use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::symbolic::{
    SymbolicBackendConfig, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot::materialize_symbolic_nonlinear_aot_build;
use crate::numerical::Nonlinear_systems::symbolic_backend::{
    SelectedSymbolicNonlinearBackendKind, SymbolicBackendSelectionPolicy,
    select_symbolic_nonlinear_backend,
};
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{AotBuildProfile, AotBuildResult};
use crate::symbolic::symbolic_engine::Expr;
use log::{info, warn};
use std::path::{Path, PathBuf};

/// High-level dense nonlinear generated-backend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseGeneratedBackendMode {
    /// Prefer compiled AOT when possible and otherwise keep the lambdify path.
    #[default]
    Defaults,
    /// Require a prebuilt compiled AOT backend.
    RequirePrebuilt,
    /// Build a release AOT artifact when it is missing.
    BuildIfMissingRelease,
}

/// Build policy for dense nonlinear generated backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicAotBuildPolicy {
    /// Use compiled AOT only when an artifact is already available.
    #[default]
    UseIfAvailable,
    /// Require an existing compiled artifact.
    RequirePrebuilt,
    /// Build the generated crate if the compiled artifact is missing.
    BuildIfMissing { profile: AotBuildProfile },
    /// Always rebuild the generated crate.
    RebuildAlways { profile: AotBuildProfile },
}

/// User-facing configuration for dense nonlinear generated backend orchestration.
#[derive(Debug, Clone, Default)]
pub struct SymbolicGeneratedBackendConfig {
    /// Optional explicit backend policy override.
    pub backend_policy_override: Option<SymbolicBackendSelectionPolicy>,
    /// Optional resolver snapshot reused across calls.
    pub resolver: Option<AotResolver>,
    /// Dense AOT runtime-plan chunking options.
    pub aot_options: SymbolicDenseAotOptions,
    /// Lifecycle build policy.
    pub build_policy: SymbolicAotBuildPolicy,
    /// Parent directory where generated crates should be materialized when a build is requested.
    pub output_parent_dir: Option<PathBuf>,
    /// Optional explicit generated crate name.
    pub crate_name_override: Option<String>,
    /// Optional explicit generated module name.
    pub module_name_override: Option<String>,
}

impl SymbolicGeneratedBackendConfig {
    /// Creates an empty generated-backend configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Production-oriented defaults for dense nonlinear generated backends.
    pub fn defaults() -> Self {
        Self::new().with_backend_policy_override(Some(
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
        ))
    }

    /// Configuration that requires a prebuilt compiled backend.
    pub fn require_prebuilt() -> Self {
        Self::defaults()
            .with_backend_policy_override(Some(SymbolicBackendSelectionPolicy::AotOnly))
            .with_build_policy(SymbolicAotBuildPolicy::RequirePrebuilt)
    }

    /// Configuration that builds a release artifact when it is missing.
    pub fn build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::defaults()
            .with_output_parent_dir(Some(output_parent_dir.into()))
            .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
    }

    /// Creates a configuration from one high-level mode.
    pub fn from_mode(mode: DenseGeneratedBackendMode) -> Self {
        match mode {
            DenseGeneratedBackendMode::Defaults => Self::defaults(),
            DenseGeneratedBackendMode::RequirePrebuilt => Self::require_prebuilt(),
            DenseGeneratedBackendMode::BuildIfMissingRelease => {
                Self::defaults().with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
            }
        }
    }

    /// Sets an explicit backend policy override.
    pub fn with_backend_policy_override(
        mut self,
        backend_policy_override: Option<SymbolicBackendSelectionPolicy>,
    ) -> Self {
        self.backend_policy_override = backend_policy_override;
        self
    }

    /// Installs a resolver snapshot that may be reused across calls.
    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Sets dense nonlinear AOT runtime-plan options.
    pub fn with_aot_options(mut self, aot_options: SymbolicDenseAotOptions) -> Self {
        self.aot_options = aot_options;
        self
    }

    /// Sets the lifecycle build policy.
    pub fn with_build_policy(mut self, build_policy: SymbolicAotBuildPolicy) -> Self {
        self.build_policy = build_policy;
        self
    }

    /// Sets the output directory used by automatic builds.
    pub fn with_output_parent_dir(mut self, output_parent_dir: Option<PathBuf>) -> Self {
        self.output_parent_dir = output_parent_dir;
        self
    }

    /// Overrides the generated crate name used by automatic builds.
    pub fn with_crate_name_override(mut self, crate_name_override: Option<String>) -> Self {
        self.crate_name_override = crate_name_override;
        self
    }

    /// Overrides the generated module name used by automatic builds.
    pub fn with_module_name_override(mut self, module_name_override: Option<String>) -> Self {
        self.module_name_override = module_name_override;
        self
    }

    fn effective_backend_policy(&self) -> SymbolicBackendSelectionPolicy {
        self.backend_policy_override
            .unwrap_or(match self.build_policy {
                SymbolicAotBuildPolicy::RequirePrebuilt => SymbolicBackendSelectionPolicy::AotOnly,
                _ => SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            })
    }

    fn output_parent_dir(&self) -> Result<&Path, SolveError> {
        self.output_parent_dir
            .as_deref()
            .ok_or(SolveError::AotBuildOutputDirMissing)
    }
}

/// Result of preparing one nonlinear symbolic problem through the high-level
/// generated-backend orchestration layer.
pub struct PreparedGeneratedSymbolicProblem {
    /// Final solver-facing symbolic problem.
    pub problem: SymbolicNonlinearProblem,
    /// Effective backend branch that ended up being used.
    pub selected_backend: SelectedSymbolicNonlinearBackendKind,
    /// Updated resolver snapshot after any materialized build/reuse.
    pub updated_resolver: Option<AotResolver>,
    /// Materialized build metadata when this call performed a build step.
    pub build_result: Option<AotBuildResult>,
}

impl PreparedGeneratedSymbolicProblem {
    /// Consumes the orchestration result and returns just the symbolic problem.
    pub fn into_problem(self) -> SymbolicNonlinearProblem {
        self.problem
    }
}

fn generated_names(problem_key: &str, config: &SymbolicGeneratedBackendConfig) -> (String, String) {
    let suffix = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_nonlinear_dense_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_nonlinear_dense_module_{suffix}"));
    (crate_name, module_name)
}

fn select_with_config<'a>(
    problem: &'a SymbolicNonlinearProblem,
    config: &SymbolicGeneratedBackendConfig,
    resolver: Option<&AotResolver>,
) -> crate::numerical::Nonlinear_systems::symbolic_backend::SelectedSymbolicNonlinearBackend<'a> {
    select_symbolic_nonlinear_backend(
        problem,
        config.effective_backend_policy(),
        resolver,
        config.aot_options,
    )
}

fn should_build_for_selection(
    config: &SymbolicGeneratedBackendConfig,
    selected_backend: SelectedSymbolicNonlinearBackendKind,
) -> bool {
    match config.build_policy {
        SymbolicAotBuildPolicy::UseIfAvailable | SymbolicAotBuildPolicy::RequirePrebuilt => false,
        SymbolicAotBuildPolicy::BuildIfMissing { .. } => {
            selected_backend != SelectedSymbolicNonlinearBackendKind::AotCompiled
        }
        SymbolicAotBuildPolicy::RebuildAlways { .. } => true,
    }
}

fn build_profile(policy: SymbolicAotBuildPolicy) -> Option<AotBuildProfile> {
    match policy {
        SymbolicAotBuildPolicy::BuildIfMissing { profile }
        | SymbolicAotBuildPolicy::RebuildAlways { profile } => Some(profile),
        SymbolicAotBuildPolicy::UseIfAvailable | SymbolicAotBuildPolicy::RequirePrebuilt => None,
    }
}

fn perform_requested_build(
    baseline_problem: &SymbolicNonlinearProblem,
    config: &SymbolicGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<AotBuildResult>, Option<AotResolver>), SolveError> {
    let profile = match build_profile(config.build_policy) {
        Some(profile) => profile,
        None => return Ok((None, resolver_snapshot)),
    };

    let prepared = baseline_problem.prepare_dense_aot_problem(config.aot_options);
    let (crate_name, module_name) = generated_names(&prepared.problem_key(), config);
    info!(
        "Materializing dense nonlinear AOT build for crate '{}' with profile {:?}",
        crate_name, profile
    );
    let build = materialize_symbolic_nonlinear_aot_build(
        &crate_name,
        &module_name,
        baseline_problem,
        config.aot_options,
        config.output_parent_dir()?,
        profile,
    )
    .map_err(|err| SolveError::AotBuildFailed(err.to_string()))?;

    let executed = build
        .execute()
        .map_err(|err| SolveError::AotBuildFailed(err.to_string()))?;
    if !executed.succeeded() {
        return Err(SolveError::AotBuildFailed(format!(
            "status={:?}\nstdout:\n{}\nstderr:\n{}",
            executed.status_code, executed.stdout, executed.stderr
        )));
    }

    let mut registry = resolver_snapshot
        .as_ref()
        .map(|resolver| resolver.registry().clone())
        .unwrap_or_default();
    registry.register_materialized_build(prepared.manifest(), &build);
    Ok((Some(build), Some(AotResolver::new(registry))))
}

fn fallback_lambdify_problem(
    equations: Vec<Expr>,
    mut options: SymbolicProblemOptions,
) -> Result<SymbolicNonlinearProblem, SolveError> {
    options.backend_config = SymbolicBackendConfig::lambdify();
    SymbolicNonlinearProblem::from_expressions_with_options(equations, options)
}

impl SymbolicNonlinearProblem {
    /// Builds a symbolic nonlinear problem through the high-level generated-backend layer.
    ///
    /// This is the preferred user-facing path when the caller wants mode/build
    /// policy behavior such as:
    /// - use compiled AOT if already available,
    /// - require a prebuilt artifact,
    /// - or build a dense generated crate automatically when needed.
    pub fn from_expressions_with_generated_backend(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
        config: SymbolicGeneratedBackendConfig,
    ) -> Result<PreparedGeneratedSymbolicProblem, SolveError> {
        let baseline_problem = fallback_lambdify_problem(equations.clone(), options.clone())?;
        let initial_selection =
            select_with_config(&baseline_problem, &config, config.resolver.as_ref());

        let (build_result, resolver_snapshot) =
            if should_build_for_selection(&config, initial_selection.effective_backend) {
                perform_requested_build(&baseline_problem, &config, config.resolver.clone())?
            } else {
                (None, config.resolver.clone())
            };

        let final_selection =
            select_with_config(&baseline_problem, &config, resolver_snapshot.as_ref());
        match final_selection.effective_backend {
            SelectedSymbolicNonlinearBackendKind::Lambdify => {
                Ok(PreparedGeneratedSymbolicProblem {
                    problem: baseline_problem,
                    selected_backend: SelectedSymbolicNonlinearBackendKind::Lambdify,
                    updated_resolver: resolver_snapshot,
                    build_result,
                })
            }
            SelectedSymbolicNonlinearBackendKind::AotCompiled => {
                match SymbolicNonlinearProblem::from_expressions_with_backend_selection(
                    equations.clone(),
                    options.clone(),
                    config.effective_backend_policy(),
                    resolver_snapshot.as_ref(),
                    config.aot_options,
                ) {
                    Ok(problem) => Ok(PreparedGeneratedSymbolicProblem {
                        problem,
                        selected_backend: SelectedSymbolicNonlinearBackendKind::AotCompiled,
                        updated_resolver: resolver_snapshot,
                        build_result,
                    }),
                    Err(SolveError::CompiledAotRuntimeUnavailable(message))
                        if config.effective_backend_policy()
                            == SymbolicBackendSelectionPolicy::PreferAotThenLambdify =>
                    {
                        warn!(
                            "Dense nonlinear compiled AOT artifact exists but no linked runtime is registered; falling back to lambdify: {}",
                            message
                        );
                        Ok(PreparedGeneratedSymbolicProblem {
                            problem: baseline_problem,
                            selected_backend: SelectedSymbolicNonlinearBackendKind::Lambdify,
                            updated_resolver: resolver_snapshot,
                            build_result,
                        })
                    }
                    Err(err) => Err(err),
                }
            }
            SelectedSymbolicNonlinearBackendKind::AotRegisteredButNotBuilt => {
                Err(SolveError::CompiledAotArtifactNotBuilt(
                    "dense nonlinear AOT artifact is registered but not built".to_string(),
                ))
            }
            SelectedSymbolicNonlinearBackendKind::AotMissing => {
                Err(SolveError::CompiledAotArtifactMissing(
                    "dense nonlinear AOT artifact is missing".to_string(),
                ))
            }
        }
    }

    /// String-based convenience wrapper around
    /// [`Self::from_expressions_with_generated_backend`].
    pub fn from_strings_with_generated_backend(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
        config: SymbolicGeneratedBackendConfig,
    ) -> Result<PreparedGeneratedSymbolicProblem, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions_with_generated_backend(expressions, options, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::problem::NonlinearProblem;
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::{
        aot_solver_test_guard, register_elementary_dense_backend,
    };
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use tempfile::tempdir;

    fn elementary_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_lambdify_backend()
    }

    fn elementary_equations() -> Vec<String> {
        vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()]
    }

    #[test]
    fn generated_backend_defaults_fall_back_to_lambdify_when_aot_is_missing() {
        let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults(),
        )
        .expect("defaults should fall back to lambdify");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicNonlinearBackendKind::Lambdify
        );
        assert_eq!(prepared.problem.backend_kind().as_str(), "lambdify");
        assert!(prepared.updated_resolver.is_none());
        assert!(prepared.build_result.is_none());
    }

    #[test]
    fn generated_backend_require_prebuilt_surfaces_missing_artifact() {
        let result = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt(),
        );

        match result {
            Err(SolveError::CompiledAotArtifactMissing(_)) => {}
            Err(other) => panic!("expected missing compiled artifact error, got {other}"),
            Ok(_) => panic!("missing prebuilt artifact should be surfaced"),
        }
    }

    #[test]
    fn generated_backend_build_if_missing_materializes_build_and_updates_resolver() {
        let _guard = aot_solver_test_guard();
        let dir = tempdir().expect("tempdir should exist");
        let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("build-if-missing should succeed");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicNonlinearBackendKind::Lambdify
        );
        assert!(prepared.build_result.is_some());
        let resolver = prepared
            .updated_resolver
            .as_ref()
            .expect("build should update resolver");
        assert_eq!(resolver.registry().len(), 1);
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("resolver should contain one problem key");
        let resolved = resolver.resolve_by_problem_key(&problem_key);
        assert!(resolved.is_compiled());
    }

    #[test]
    fn generated_backend_reuses_updated_resolver_with_linked_runtime() {
        let _guard = aot_solver_test_guard();
        let dir = tempdir().expect("tempdir should exist");
        let first = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("first build should succeed");
        let resolver = first
            .updated_resolver
            .clone()
            .expect("build should produce resolver");
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("resolver should contain one problem key");
        register_elementary_dense_backend(&problem_key);

        let second = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        )
        .expect("second call should reuse compiled resolver");

        assert_eq!(
            second.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert_eq!(second.problem.backend_kind().as_str(), "aot");
        let x0 = DVector::from_vec(vec![3.0, -1.0]);
        let residual = second.problem.residual(&x0).expect("residual");
        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);

        unregister_linked_dense_backend(&problem_key);
    }
}

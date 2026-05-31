//! Driver helpers that turn prepared problems into emitted AOT artifacts.
//!
//! This layer sits between:
//! - prepared solver-facing problems,
//! - generic `CodegenModule` emission,
//! - and the separate generated-crate writer.
//!
//! Its job is to remove manual glue:
//! callers should not need to rebuild a `CodegenModule` by hand once they
//! already have a [`PreparedProblem`](crate::symbolic::codegen_provider_api::PreparedProblem).

use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule, GeneratedBlock};
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::{
    CAotBuildProfile, CAotBuildRequest, CAotBuildResult, CAotCompileConfig, ExecutedCAotBuild,
};
use crate::symbolic::codegen::c_backend::codegen_c_aot_library::GeneratedCAotLibrary;
use crate::symbolic::codegen::codegen_provider_api::{
    PreparedBandedProblem, PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildRequest, AotBuildResult, AotCompileConfig, ExecutedAotBuild,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::{
    ExecutedZigAotBuild, ZigAotBuildProfile, ZigAotBuildRequest, ZigAotBuildResult,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_library::GeneratedZigAotLibrary;
use log::info;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

/// Backend selected for one emitted AOT artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotCodegenBackend {
    Rust,
    C,
    Zig,
}

/// Timing split for lowering one prepared residual/Jacobian problem into a
/// [`CodegenModule`].
///
/// This deliberately sits at the driver layer rather than in BVP tests: the
/// module build stage is shared by sparse, banded, dense, Rust, C, and Zig AOT
/// paths, and it is a real cold-bootstrap cost, not a story-test wrapper.
#[derive(Clone, Debug, Default)]
pub struct PreparedModuleBuildBreakdown {
    pub module_init_ms: f64,
    pub residual_blocks_ms: f64,
    pub jacobian_blocks_ms: f64,
    pub total_ms: f64,
    pub residual_chunks: usize,
    pub jacobian_chunks: usize,
}

impl Default for AotCodegenBackend {
    fn default() -> Self {
        Self::Rust
    }
}

impl AotCodegenBackend {
    pub(crate) fn codegen_language(self) -> CodegenLanguage {
        match self {
            Self::Rust => CodegenLanguage::Rust,
            Self::C => CodegenLanguage::C,
            Self::Zig => CodegenLanguage::Zig,
        }
    }

    fn artifact_kind(self) -> &'static str {
        match self {
            Self::Rust => "Rust crate",
            Self::C => "C library",
            Self::Zig => "Zig library",
        }
    }
}

/// One emitted AOT artifact independent of the final backend language.
#[derive(Debug, Clone)]
pub enum GeneratedAotArtifact {
    Rust(GeneratedAotCrate),
    C(GeneratedCAotLibrary),
    Zig(GeneratedZigAotLibrary),
}

impl GeneratedAotArtifact {
    pub fn backend(&self) -> AotCodegenBackend {
        match self {
            Self::Rust(_) => AotCodegenBackend::Rust,
            Self::C(_) => AotCodegenBackend::C,
            Self::Zig(_) => AotCodegenBackend::Zig,
        }
    }

    pub fn into_rust_crate(self) -> Option<GeneratedAotCrate> {
        match self {
            Self::Rust(crate_spec) => Some(crate_spec),
            Self::C(_) | Self::Zig(_) => None,
        }
    }
}

/// Cross-language build preset for generated AOT artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotBuildPreset {
    Production,
    FastBuild,
    DevFastest,
}

/// One materializable build request independent of the selected backend language.
#[derive(Debug, Clone)]
pub enum GeneratedAotBuildRequest {
    Rust(AotBuildRequest),
    C(CAotBuildRequest),
    Zig(ZigAotBuildRequest),
}

/// One materialized build result independent of the selected backend language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratedAotBuildResult {
    Rust(AotBuildResult),
    C(CAotBuildResult),
    Zig(ZigAotBuildResult),
}

/// One executed build result independent of the selected backend language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutedGeneratedAotBuild {
    Rust(ExecutedAotBuild),
    C(ExecutedCAotBuild),
    Zig(ExecutedZigAotBuild),
}

impl GeneratedAotBuildRequest {
    pub fn materialize(&self) -> io::Result<GeneratedAotBuildResult> {
        match self {
            Self::Rust(request) => request.materialize().map(GeneratedAotBuildResult::Rust),
            Self::C(request) => request.materialize().map(GeneratedAotBuildResult::C),
            Self::Zig(request) => request.materialize().map(GeneratedAotBuildResult::Zig),
        }
    }
}

impl GeneratedAotBuildResult {
    pub fn command_line(&self) -> String {
        match self {
            Self::Rust(result) => result.cargo_command_line(),
            Self::C(result) => result.build_command_line(),
            Self::Zig(result) => result.build_command_line(),
        }
    }

    pub fn workdir(&self) -> PathBuf {
        match self {
            Self::Rust(result) => result.cargo_workdir().to_path_buf(),
            Self::C(result) => result.build_workdir().to_path_buf(),
            Self::Zig(result) => result.build_workdir().to_path_buf(),
        }
    }

    pub fn execute(&self) -> io::Result<ExecutedGeneratedAotBuild> {
        match self {
            Self::Rust(result) => result.execute().map(ExecutedGeneratedAotBuild::Rust),
            Self::C(result) => result.execute().map(ExecutedGeneratedAotBuild::C),
            Self::Zig(result) => result.execute().map(ExecutedGeneratedAotBuild::Zig),
        }
    }
}

impl ExecutedGeneratedAotBuild {
    pub fn succeeded(&self) -> bool {
        match self {
            Self::Rust(build) => build.succeeded(),
            Self::C(build) => build.succeeded(),
            Self::Zig(build) => build.succeeded(),
        }
    }
}

/// Creates one backend-selected build request for a previously emitted artifact.
pub fn generated_aot_build_request_from_artifact(
    artifact: GeneratedAotArtifact,
    output_parent_dir: impl Into<PathBuf>,
    preset: AotBuildPreset,
) -> GeneratedAotBuildRequest {
    let output_parent_dir = output_parent_dir.into();
    match artifact {
        GeneratedAotArtifact::Rust(crate_spec) => {
            let compile_config = match preset {
                AotBuildPreset::Production => AotCompileConfig::production(),
                AotBuildPreset::FastBuild => AotCompileConfig::fast_build(),
                AotBuildPreset::DevFastest => AotCompileConfig::dev_fastest(),
            };
            GeneratedAotBuildRequest::Rust(
                AotBuildRequest::new(crate_spec, output_parent_dir, AotBuildProfile::Release)
                    .with_compile_config(compile_config),
            )
        }
        GeneratedAotArtifact::C(library_spec) => {
            let compile_config = match preset {
                AotBuildPreset::Production => CAotCompileConfig::production(),
                AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
                AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
            };
            GeneratedAotBuildRequest::C(
                CAotBuildRequest::new(library_spec, output_parent_dir, CAotBuildProfile::Release)
                    .with_compile_config(compile_config),
            )
        }
        GeneratedAotArtifact::Zig(library_spec) => {
            let profile = match preset {
                AotBuildPreset::Production => ZigAotBuildProfile::ReleaseFast,
                AotBuildPreset::FastBuild => ZigAotBuildProfile::ReleaseFast,
                AotBuildPreset::DevFastest => ZigAotBuildProfile::Debug,
            };
            GeneratedAotBuildRequest::Zig(ZigAotBuildRequest::new(
                library_spec,
                output_parent_dir,
                profile,
            ))
        }
    }
}

/// Builds a `CodegenModule` for a prepared dense problem.
pub fn codegen_module_from_prepared_dense_problem(
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> CodegenModule {
    info!(
        "Building dense AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let mut module = CodegenModule::new(module_name);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_dense_jacobian_plan(&chunk.plan);
    }
    module
}

fn codegen_module_from_prepared_dense_problem_with_breakdown(
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> (CodegenModule, PreparedModuleBuildBreakdown) {
    info!(
        "Building dense AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let total_begin = Instant::now();
    let init_begin = Instant::now();
    let mut module = CodegenModule::new(module_name);
    let module_init_ms = init_begin.elapsed().as_secs_f64() * 1_000.0;

    let ((residual_blocks, residual_blocks_ms), (jacobian_blocks, jacobian_blocks_ms)) =
        rayon::join(
            || {
                let residual_begin = Instant::now();
                let blocks = problem
                    .residual_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_residual_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, residual_begin.elapsed().as_secs_f64() * 1_000.0)
            },
            || {
                let jacobian_begin = Instant::now();
                let blocks = problem
                    .jacobian_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_dense_jacobian_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, jacobian_begin.elapsed().as_secs_f64() * 1_000.0)
            },
        );
    for block in residual_blocks {
        module.push_generated_block(block);
    }
    for block in jacobian_blocks {
        module.push_generated_block(block);
    }

    let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
    (
        module,
        PreparedModuleBuildBreakdown {
            module_init_ms,
            residual_blocks_ms,
            jacobian_blocks_ms,
            total_ms,
            residual_chunks: problem.residual_plan.chunks.len(),
            jacobian_chunks: problem.jacobian_plan.chunks.len(),
        },
    )
}

fn codegen_module_from_prepared_dense_problem_with_language(
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
    language: CodegenLanguage,
) -> CodegenModule {
    let mut module = CodegenModule::new(module_name).with_language(language);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_dense_jacobian_plan(&chunk.plan);
    }
    module
}

/// Builds a `CodegenModule` for a prepared sparse problem.
pub fn codegen_module_from_prepared_sparse_problem(
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
) -> CodegenModule {
    info!(
        "Building sparse AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let mut module = CodegenModule::new(module_name);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

fn codegen_module_from_prepared_sparse_problem_with_breakdown(
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
) -> (CodegenModule, PreparedModuleBuildBreakdown) {
    info!(
        "Building sparse AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let total_begin = Instant::now();
    let init_begin = Instant::now();
    let mut module = CodegenModule::new(module_name);
    let module_init_ms = init_begin.elapsed().as_secs_f64() * 1_000.0;

    let ((residual_blocks, residual_blocks_ms), (jacobian_blocks, jacobian_blocks_ms)) =
        rayon::join(
            || {
                let residual_begin = Instant::now();
                let blocks = problem
                    .residual_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_residual_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, residual_begin.elapsed().as_secs_f64() * 1_000.0)
            },
            || {
                let jacobian_begin = Instant::now();
                let blocks = problem
                    .jacobian_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_sparse_values_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, jacobian_begin.elapsed().as_secs_f64() * 1_000.0)
            },
        );
    for block in residual_blocks {
        module.push_generated_block(block);
    }
    for block in jacobian_blocks {
        module.push_generated_block(block);
    }

    let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
    (
        module,
        PreparedModuleBuildBreakdown {
            module_init_ms,
            residual_blocks_ms,
            jacobian_blocks_ms,
            total_ms,
            residual_chunks: problem.residual_plan.chunks.len(),
            jacobian_chunks: problem.jacobian_plan.chunks.len(),
        },
    )
}

/// Builds a `CodegenModule` for a prepared banded problem.
///
/// The generated Jacobian chunks still lower to flat values-writing code blocks.
/// The semantic difference between sparse and banded is carried by the prepared
/// problem manifest/structure, while the emitted chunk ABI remains
/// `fn(args, out_values)`.
pub fn codegen_module_from_prepared_banded_problem(
    module_name: &str,
    problem: &PreparedBandedProblem<'_>,
) -> CodegenModule {
    info!(
        "Building banded AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let mut module = CodegenModule::new(module_name);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

fn codegen_module_from_prepared_banded_problem_with_breakdown(
    module_name: &str,
    problem: &PreparedBandedProblem<'_>,
) -> (CodegenModule, PreparedModuleBuildBreakdown) {
    info!(
        "Building banded AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let total_begin = Instant::now();
    let init_begin = Instant::now();
    let mut module = CodegenModule::new(module_name);
    let module_init_ms = init_begin.elapsed().as_secs_f64() * 1_000.0;

    let ((residual_blocks, residual_blocks_ms), (jacobian_blocks, jacobian_blocks_ms)) =
        rayon::join(
            || {
                let residual_begin = Instant::now();
                let blocks = problem
                    .residual_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_residual_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, residual_begin.elapsed().as_secs_f64() * 1_000.0)
            },
            || {
                let jacobian_begin = Instant::now();
                let blocks = problem
                    .jacobian_plan
                    .chunks
                    .iter()
                    .map(|chunk| GeneratedBlock::from_sparse_values_plan(&chunk.plan))
                    .collect::<Vec<_>>();
                (blocks, jacobian_begin.elapsed().as_secs_f64() * 1_000.0)
            },
        );
    for block in residual_blocks {
        module.push_generated_block(block);
    }
    for block in jacobian_blocks {
        module.push_generated_block(block);
    }

    let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
    (
        module,
        PreparedModuleBuildBreakdown {
            module_init_ms,
            residual_blocks_ms,
            jacobian_blocks_ms,
            total_ms,
            residual_chunks: problem.residual_plan.chunks.len(),
            jacobian_chunks: problem.jacobian_plan.chunks.len(),
        },
    )
}

fn codegen_module_from_prepared_sparse_problem_with_language(
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
    language: CodegenLanguage,
) -> CodegenModule {
    let mut module = CodegenModule::new(module_name).with_language(language);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

fn codegen_module_from_prepared_banded_problem_with_language(
    module_name: &str,
    problem: &PreparedBandedProblem<'_>,
    language: CodegenLanguage,
) -> CodegenModule {
    let mut module = CodegenModule::new(module_name).with_language(language);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

/// Builds a `CodegenModule` for any prepared problem.
pub fn codegen_module_from_prepared_problem(
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> CodegenModule {
    match problem {
        PreparedProblem::Dense(problem) => {
            codegen_module_from_prepared_dense_problem(module_name, problem)
        }
        PreparedProblem::Banded(problem) => {
            codegen_module_from_prepared_banded_problem(module_name, problem)
        }
        PreparedProblem::Sparse(problem) => {
            codegen_module_from_prepared_sparse_problem(module_name, problem)
        }
    }
}

/// Builds a `CodegenModule` and reports where the prepared-problem lowering
/// time was spent.
pub fn codegen_module_from_prepared_problem_with_breakdown(
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> (CodegenModule, PreparedModuleBuildBreakdown) {
    match problem {
        PreparedProblem::Dense(problem) => {
            codegen_module_from_prepared_dense_problem_with_breakdown(module_name, problem)
        }
        PreparedProblem::Banded(problem) => {
            codegen_module_from_prepared_banded_problem_with_breakdown(module_name, problem)
        }
        PreparedProblem::Sparse(problem) => {
            codegen_module_from_prepared_sparse_problem_with_breakdown(module_name, problem)
        }
    }
}

fn codegen_module_from_prepared_problem_with_backend(
    module_name: &str,
    problem: &PreparedProblem<'_>,
    backend: AotCodegenBackend,
) -> CodegenModule {
    let language = backend.codegen_language();
    match problem {
        PreparedProblem::Dense(problem) => {
            codegen_module_from_prepared_dense_problem_with_language(module_name, problem, language)
        }
        PreparedProblem::Banded(problem) => {
            codegen_module_from_prepared_banded_problem_with_language(
                module_name,
                problem,
                language,
            )
        }
        PreparedProblem::Sparse(problem) => {
            codegen_module_from_prepared_sparse_problem_with_language(
                module_name,
                problem,
                language,
            )
        }
    }
}

/// Builds one backend-agnostic AOT artifact directly from any prepared problem.
pub fn generated_aot_artifact_from_prepared_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedProblem<'_>,
    backend: AotCodegenBackend,
) -> GeneratedAotArtifact {
    info!(
        "Assembling generated {} '{}' from prepared problem with {:?} matrix backend",
        backend.artifact_kind(),
        artifact_name,
        problem.matrix_backend()
    );
    let module = codegen_module_from_prepared_problem_with_backend(module_name, problem, backend);
    match backend {
        AotCodegenBackend::Rust => GeneratedAotArtifact::Rust(
            GeneratedAotCrate::from_prepared_problem(artifact_name, problem, &module),
        ),
        AotCodegenBackend::C => GeneratedAotArtifact::C(
            GeneratedCAotLibrary::from_prepared_problem(artifact_name, problem, &module),
        ),
        AotCodegenBackend::Zig => GeneratedAotArtifact::Zig(
            GeneratedZigAotLibrary::from_prepared_problem(artifact_name, problem, &module),
        ),
    }
}

/// Builds a generated AOT crate directly from a prepared dense problem.
pub fn generated_aot_crate_from_prepared_dense_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> GeneratedAotCrate {
    info!(
        "Assembling generated dense AOT crate '{}' from prepared problem",
        crate_name
    );
    let module = codegen_module_from_prepared_dense_problem(module_name, problem);
    GeneratedAotCrate::from_prepared_dense_problem(crate_name, problem, &module)
}

/// Builds a generated AOT crate directly from a prepared sparse problem.
pub fn generated_aot_crate_from_prepared_sparse_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
) -> GeneratedAotCrate {
    info!(
        "Assembling generated sparse AOT crate '{}' from prepared problem",
        crate_name
    );
    let module = codegen_module_from_prepared_sparse_problem(module_name, problem);
    GeneratedAotCrate::from_prepared_sparse_problem(crate_name, problem, &module)
}

/// Builds a generated AOT crate directly from any prepared problem.
pub fn generated_aot_crate_from_prepared_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> GeneratedAotCrate {
    generated_aot_artifact_from_prepared_problem(
        crate_name,
        module_name,
        problem,
        AotCodegenBackend::Rust,
    )
    .into_rust_crate()
    .expect("Rust backend must return GeneratedAotCrate")
}

/// Builds a generated C AOT library directly from a prepared dense problem.
pub fn generated_c_aot_library_from_prepared_dense_problem(
    library_name: &str,
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> GeneratedCAotLibrary {
    info!(
        "Assembling generated dense C AOT library '{}' from prepared problem",
        library_name
    );
    let module = codegen_module_from_prepared_dense_problem_with_language(
        module_name,
        problem,
        CodegenLanguage::C,
    );
    GeneratedCAotLibrary::from_prepared_dense_problem(library_name, problem, &module)
}

/// Builds a generated C AOT library directly from any prepared problem.
pub fn generated_c_aot_library_from_prepared_problem(
    library_name: &str,
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> GeneratedCAotLibrary {
    info!(
        "Assembling generated C AOT library '{}' from prepared problem with {:?} matrix backend",
        library_name,
        problem.matrix_backend()
    );
    match generated_aot_artifact_from_prepared_problem(
        library_name,
        module_name,
        problem,
        AotCodegenBackend::C,
    ) {
        GeneratedAotArtifact::C(library) => library,
        GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
            unreachable!("C backend wrapper must return GeneratedCAotLibrary")
        }
    }
}

/// Builds a generated Zig AOT library directly from a prepared problem.
pub fn generated_zig_aot_library_from_prepared_problem(
    library_name: &str,
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> GeneratedZigAotLibrary {
    match generated_aot_artifact_from_prepared_problem(
        library_name,
        module_name,
        problem,
        AotCodegenBackend::Zig,
    ) {
        GeneratedAotArtifact::Zig(library) => library,
        GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
            unreachable!("Zig backend wrapper must return GeneratedZigAotLibrary")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedBandedProblem, PreparedDenseProblem,
        PreparedSparseProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{
        BandedChunkingStrategy, BandedExprEntry, BandedJacobianTask, JacobianTask, ResidualTask,
        SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn dense_driver_builds_module_with_residual_and_dense_jacobian_blocks() {
        let residuals = vec![Expr::parse_expression("x + 1")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let prepared = PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian: &jacobian,
                variables: &vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        );

        let module = codegen_module_from_prepared_dense_problem("dense_module_fixture", &prepared);
        let source = module.emit_source();

        assert!(source.contains("pub mod dense_module_fixture"));
        assert!(source.contains("pub fn eval_residual"));
        assert!(source.contains("pub fn eval_jacobian"));
    }

    #[test]
    fn sparse_driver_builds_generated_crate_from_prepared_problem() {
        let residuals = vec![
            Expr::parse_expression("x + p"),
            Expr::parse_expression("y - p"),
        ];
        let vars = vec!["x", "y"];
        let params = vec!["p"];
        let entry0 = Expr::parse_expression("1");
        let entry1 = Expr::parse_expression("2");
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &entry0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &entry1,
            },
        ];

        let prepared = PreparedProblem::sparse(PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            SparseJacobianTask {
                fn_name: "eval_sparse_values",
                shape: (2, 2),
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(SparseChunkingStrategy::Whole),
        ));

        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_sparse_driver_fixture",
            "sparse_module_fixture",
            &prepared,
        );

        assert_eq!(crate_spec.crate_name, "generated_sparse_driver_fixture");
        assert_eq!(crate_spec.manifest.backend_kind.as_str(), "aot");
        assert_eq!(crate_spec.manifest.matrix_backend.as_str(), "sparse_col");
        assert!(
            crate_spec
                .module_source
                .contains("pub mod sparse_module_fixture")
        );
        assert!(crate_spec.module_source.contains("pub fn eval_residual"));
        assert!(
            crate_spec
                .module_source
                .contains("pub fn eval_sparse_values")
        );
    }

    #[test]
    fn prepared_problem_module_breakdown_preserves_sparse_source() {
        let residuals = vec![
            Expr::parse_expression("x + p"),
            Expr::parse_expression("y - p"),
        ];
        let vars = vec!["x", "y"];
        let params = vec!["p"];
        let entry0 = Expr::parse_expression("1");
        let entry1 = Expr::parse_expression("2");
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &entry0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &entry1,
            },
        ];

        let prepared = PreparedProblem::sparse(PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            SparseJacobianTask {
                fn_name: "eval_sparse_values",
                shape: (2, 2),
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(SparseChunkingStrategy::Whole),
        ));

        let plain = codegen_module_from_prepared_problem("sparse_breakdown_fixture", &prepared);
        let (instrumented, breakdown) = codegen_module_from_prepared_problem_with_breakdown(
            "sparse_breakdown_fixture",
            &prepared,
        );

        assert_eq!(plain.emit_source(), instrumented.emit_source());
        assert_eq!(breakdown.residual_chunks, 1);
        assert_eq!(breakdown.jacobian_chunks, 1);
        assert!(
            breakdown.total_ms
                >= breakdown
                    .residual_blocks_ms
                    .max(breakdown.jacobian_blocks_ms),
            "parallel module timing should include the slower independent lowering branch"
        );
    }

    #[test]
    fn generic_driver_can_emit_c_and_zig_artifacts_from_same_prepared_problem() {
        let residuals = vec![Expr::parse_expression("x + 1")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let prepared = PreparedProblem::dense(PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian: &jacobian,
                variables: &vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        ));

        let c_artifact = generated_aot_artifact_from_prepared_problem(
            "generic_c_fixture",
            "generic_c_module",
            &prepared,
            AotCodegenBackend::C,
        );
        let zig_artifact = generated_aot_artifact_from_prepared_problem(
            "generic_zig_fixture",
            "generic_zig_module",
            &prepared,
            AotCodegenBackend::Zig,
        );

        match c_artifact {
            GeneratedAotArtifact::C(library) => {
                assert_eq!(library.library_name, "generic_c_fixture");
                assert!(library.c_source.contains("eval_residual"));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
                panic!("generic C backend should emit GeneratedCAotLibrary")
            }
        }

        match zig_artifact {
            GeneratedAotArtifact::Zig(library) => {
                assert_eq!(library.library_name, "generic_zig_fixture");
                assert!(library.zig_source.contains("eval_residual"));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
                panic!("generic Zig backend should emit GeneratedZigAotLibrary")
            }
        }
    }

    #[test]
    fn banded_driver_builds_generated_crate_from_prepared_problem() {
        let residuals = vec![
            Expr::parse_expression("x + p"),
            Expr::parse_expression("y - p"),
        ];
        let vars = vec!["x", "y"];
        let params = vec!["p"];
        let e0 = Expr::parse_expression("1");
        let e1 = Expr::parse_expression("2");
        let e2 = Expr::parse_expression("3");
        let entries = vec![
            BandedExprEntry {
                row: 1,
                col: 0,
                diag_offset: -1,
                diag_position: 0,
                expr: &e0,
            },
            BandedExprEntry {
                row: 0,
                col: 0,
                diag_offset: 0,
                diag_position: 0,
                expr: &e1,
            },
            BandedExprEntry {
                row: 1,
                col: 1,
                diag_offset: 0,
                diag_position: 1,
                expr: &e2,
            },
        ];

        let prepared = PreparedProblem::banded(PreparedBandedProblem::new(
            BackendKind::Aot,
            MatrixBackend::Banded,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            BandedJacobianTask {
                fn_name: "eval_banded_values",
                shape: (2, 2),
                kl: 1,
                ku: 0,
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(BandedChunkingStrategy::Whole),
        ));

        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_banded_driver_fixture",
            "banded_module_fixture",
            &prepared,
        );

        assert_eq!(crate_spec.crate_name, "generated_banded_driver_fixture");
        assert_eq!(crate_spec.manifest.backend_kind.as_str(), "aot");
        assert_eq!(crate_spec.manifest.matrix_backend.as_str(), "banded");
        assert!(
            crate_spec
                .module_source
                .contains("pub mod banded_module_fixture")
        );
        assert!(crate_spec.module_source.contains("pub fn eval_residual"));
        assert!(
            crate_spec
                .module_source
                .contains("pub fn eval_banded_values")
        );
    }
}

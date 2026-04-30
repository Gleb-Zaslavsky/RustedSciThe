//pub mod symbolic_functions_BVP_;
pub mod CodegenIR;

pub mod rust_backend;

/// prepared-problem -> CodegenModule / generated-crate driver helpers
pub mod codegen_aot_driver;
/// registry of materialized generated AOT crates and artifact metadata
pub mod codegen_aot_registry;
/// resolution of registered generated AOT artifacts back into solver-facing metadata
pub mod codegen_aot_resolution;
/// process-local registry of statically linked generated AOT backends
pub mod codegen_aot_runtime_link;
/// backend preference selection layer above prepared problems and AOT resolution
pub mod codegen_backend_selection;
// Shared test-only layers used by the staged AOT slices.
#[cfg(test)]
pub(crate) mod codegen_adapters;

#[cfg(test)]
pub(crate) mod codegen_generated_fixtures;

/// owned metadata manifests for prepared AOT problems
pub mod codegen_manifest;
/// sequential and future parallel orchestration over generated chunk functions
pub mod codegen_orchestrator;

/// solver-facing provider traits and backend metadata for future AOT lifecycle
pub mod codegen_provider_api;
/// solver-facing runtime plans for AOT-generated residuals and Jacobians
pub mod codegen_runtime_api;
/// scenario and task descriptions for symbolic code generation
pub mod codegen_tasks;
pub mod c_backend;
pub mod zig_backend;

mod tests;
pub(crate) mod testing_fixtures;

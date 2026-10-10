//! AtomView-native dense nonlinear AOT preparation and source emission.
//!
//! This adapter intentionally stops at the shared generated-crate ABI.  It
//! owns packed residual/Jacobian atoms, lowers them directly through the Atom
//! code generator, and then hands the resulting crate to the existing Rust
//! AOT build/registry lifecycle.  No Atom -> Expr projection is performed.

use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::symbolic::{
    PreparedSymbolicNonlinearAotProblem, SymbolicDenseAotOptions, SymbolicLambdifyFrontend,
    SymbolicNonlinearProblem,
};
use crate::symbolic::bvp::atom_aot::{AtomAotMatrixLayout, AtomAotPreparedPlan};
use crate::symbolic::codegen::codegen_manifest::{
    GeneratedChunkManifest, GeneratedFunctionsManifest, PreparedProblemManifest,
};
use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
use crate::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::{CodegenOutputLayout, SparseChunkingStrategy};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate;
use crate::symbolic::codegen::CodegenIR::{
    AtomOptimizationProfile, AtomTempReusePolicy, CodegenLanguage, CodegenModule, GeneratedBlock,
};
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::state::Symbol;
use ahash::HashMap;
use std::sync::Arc;

/// Builds the packed Atom payload used by the dense nonlinear AOT route.
///
/// The public nonlinear API still accepts `Expr`; this is the one cold
/// boundary where those inputs become AtomView data.  Differentiation and
/// lowering after this function remain entirely on the Atom representation.
pub(crate) fn prepare_atom_dense_plan(
    problem: &SymbolicNonlinearProblem,
    options: SymbolicDenseAotOptions,
) -> Result<AtomAotPreparedPlan, SolveError> {
    if !matches!(options.residual_strategy, ResidualChunkingStrategy::Whole)
        || !matches!(
            options.jacobian_strategy,
            DenseJacobianChunkingStrategy::Whole
        )
    {
        return Err(SolveError::InvalidConfig(
            "AtomView dense AOT currently requires Whole residual and Jacobian strategies"
                .to_owned(),
        ));
    }
    let residuals = problem
        .equations()
        .iter()
        .map(crate::symbolic::View::conversions::expr_to_atom)
        .collect::<Vec<_>>();
    let variable_symbols = problem
        .variables()
        .iter()
        .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();
    let parameter_names = problem
        .parameter_schema()
        .map(|schema| schema.names().to_vec())
        .unwrap_or_default();
    let mut input_names = parameter_names.clone();
    input_names.extend(problem.variables().iter().cloned());
    let input_symbols = input_names
        .iter()
        .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();

    let mut jacobian_entries = Vec::new();
    for (row, residual) in residuals.iter().enumerate() {
        for (col, variable) in variable_symbols.iter().copied().enumerate() {
            let derivative = residual.try_derivative(variable).map_err(|error| {
                SolveError::InvalidConfig(format!(
                    "AtomView AOT Jacobian differentiation failed at ({row}, {col}): {error}"
                ))
            })?;
            if !derivative.is_zero() {
                jacobian_entries.push(SparseAtomJacobianEntry {
                    row,
                    col,
                    value: derivative,
                });
            }
        }
    }

    AtomAotPreparedPlan::from_parts(
        residuals,
        jacobian_entries,
        input_names,
        input_symbols,
        parameter_names.len(),
        AtomAotMatrixLayout::Dense {
            rows: problem.equations().len(),
            cols: problem.variables().len(),
        },
        options.residual_strategy,
        SparseChunkingStrategy::Whole,
    )
    .map_err(|error| SolveError::InvalidConfig(format!("AtomView AOT plan rejected: {error}")))
}

/// Emits a dense AtomView module with the same residual/Jacobian function names
/// and row-major ABI as the ExprLegacy generated route.
pub(crate) fn generated_atom_dense_crate(
    crate_name: &str,
    module_name: &str,
    problem: &SymbolicNonlinearProblem,
    options: SymbolicDenseAotOptions,
) -> Result<GeneratedAotCrate, SolveError> {
    debug_assert_eq!(
        problem.lambdify_frontend(),
        SymbolicLambdifyFrontend::AtomViewNative
    );
    let plan = prepare_atom_dense_plan(problem, options)?;
    let module = atom_dense_module(module_name, &plan)?;
    let (rows, cols) = plan.matrix_layout().shape();
    let functions = GeneratedFunctionsManifest {
        residual_fn_name: "eval_nonlinear_residual".to_owned(),
        residual_chunk_names: vec!["eval_nonlinear_residual".to_owned()],
        residual_chunks: vec![GeneratedChunkManifest {
            fn_name: "eval_nonlinear_residual".to_owned(),
            offset: 0,
            len: plan.residuals().len(),
        }],
        jacobian_fn_name: "eval_nonlinear_jacobian".to_owned(),
        jacobian_chunk_names: vec!["eval_nonlinear_jacobian".to_owned()],
        jacobian_chunks: vec![GeneratedChunkManifest {
            fn_name: "eval_nonlinear_jacobian".to_owned(),
            offset: 0,
            len: rows * cols,
        }],
    };
    let manifest = PreparedProblemManifest::from_atom_aot_plan(
        BackendKind::Aot,
        MatrixBackend::Dense,
        &plan,
        functions,
    );
    Ok(GeneratedAotCrate::from_codegen_module(
        crate_name, &module, manifest,
    ))
}

fn atom_dense_module(
    module_name: &str,
    plan: &AtomAotPreparedPlan,
) -> Result<CodegenModule, SolveError> {
    let vars: Arc<[String]> = plan.input_names().to_vec().into();
    let index = Arc::new(
        plan.input_symbols()
            .iter()
            .enumerate()
            .map(|(position, symbol)| (symbol.id, position))
            .collect::<HashMap<u32, usize>>(),
    );
    let mut module = CodegenModule::new(module_name).with_language(CodegenLanguage::Rust);

    let residual_views = plan
        .residuals()
        .iter()
        .map(Atom::as_view)
        .collect::<Vec<_>>();
    let (residual_block, _) = GeneratedBlock::from_atom_views_with_shared_abi_and_profile(
        "eval_nonlinear_residual",
        &residual_views,
        Arc::clone(&vars),
        Arc::clone(&index),
        Some(CodegenOutputLayout::Vector {
            len: residual_views.len(),
        }),
        AtomOptimizationProfile::Full,
        AtomTempReusePolicy::Auto,
    );
    module.push_generated_block(residual_block);

    let (rows, cols) = plan.matrix_layout().shape();
    let jacobian_views = plan
        .jacobian_entries()
        .iter()
        .map(|entry| entry.value.as_view())
        .collect::<Vec<_>>();
    let offsets = plan
        .jacobian_entries()
        .iter()
        .map(|entry| entry.row * cols + entry.col)
        .collect::<Vec<_>>();
    let (jacobian_block, _) =
        GeneratedBlock::from_atom_views_with_shared_abi_and_profile_and_output_offsets(
            "eval_nonlinear_jacobian",
            &jacobian_views,
            vars,
            index,
            Some(CodegenOutputLayout::Matrix { rows, cols }),
            AtomOptimizationProfile::Full,
            AtomTempReusePolicy::Auto,
            Some(offsets),
        );
    module.push_generated_block(jacobian_block);
    Ok(module)
}

/// Exposes the frontend-aware generation hook to the existing build adapter.
pub(crate) fn generated_atom_dense_crate_for_prepared(
    crate_name: &str,
    module_name: &str,
    problem: &SymbolicNonlinearProblem,
    prepared: &PreparedSymbolicNonlinearAotProblem<'_>,
    options: SymbolicDenseAotOptions,
) -> Result<GeneratedAotCrate, SolveError> {
    let mut generated = generated_atom_dense_crate(crate_name, module_name, problem, options)?;
    // The shared prepared descriptor still uses Expr as the public input
    // boundary.  Keep its structural signature for lifecycle compatibility,
    // while the route tag remains AtomViewNative and prevents Expr artifact
    // reuse.  The generated source itself is entirely Atom-native.
    generated.manifest.expression_signature = prepared.manifest().expression_signature;
    if generated.manifest.problem_key() != prepared.problem_key() {
        return Err(SolveError::InvalidConfig(
            "AtomView AOT generated manifest does not match prepared nonlinear problem key"
                .to_owned(),
        ));
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
    use crate::symbolic::symbolic_engine::Expr;

    fn atom_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_expressions_with_options(
            vec![
                Expr::parse_expression("x^2+y-3"),
                Expr::parse_expression("x-y"),
            ],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_owned(), "y".to_owned()])
                .with_atom_native_frontend(),
        )
        .expect("AtomView problem should build")
    }

    #[test]
    fn atom_aot_emits_shared_dense_abi_without_expr_frontend() {
        let problem = atom_problem();
        let generated = generated_atom_dense_crate(
            "nonlinear_atom_fixture",
            "nonlinear_atom_module",
            &problem,
            SymbolicDenseAotOptions::default(),
        )
        .expect("AtomView AOT source should emit");

        assert_eq!(
            generated.manifest.symbolic_route,
            crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
        );
        assert_eq!(generated.manifest.matrix_backend, MatrixBackend::Dense);
        assert_eq!(
            generated.manifest.functions.residual_fn_name,
            "eval_nonlinear_residual"
        );
        assert_eq!(
            generated.manifest.functions.jacobian_fn_name,
            "eval_nonlinear_jacobian"
        );
        assert!(generated
            .module_source
            .contains("pub fn eval_nonlinear_residual"));
        assert!(generated
            .module_source
            .contains("pub fn eval_nonlinear_jacobian"));
        assert!(!generated.module_source.contains("ExprLegacy"));
    }

    #[test]
    fn atom_aot_manifest_matches_prepared_lifecycle_key() {
        let problem = atom_problem();
        let prepared = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        let mut generated = generated_atom_dense_crate(
            "nonlinear_atom_fixture",
            "nonlinear_atom_module",
            &problem,
            SymbolicDenseAotOptions::default(),
        )
        .expect("AtomView AOT source should emit");
        generated.manifest.expression_signature = prepared.manifest().expression_signature;

        assert_eq!(generated.manifest, prepared.manifest());
        assert_eq!(generated.manifest.problem_key(), prepared.problem_key());
        assert_eq!(
            generated.manifest.symbolic_route,
            crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
        );
    }
}

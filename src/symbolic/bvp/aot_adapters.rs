//! Explicit ownership views for the two BVP AOT preparation routes.
//!
//! `BvpPreparedSparseAotProblem` remains a compatibility bridge because it is
//! used by older code-generation APIs. New lifecycle code should first obtain
//! one of these typed adapters instead of inspecting optional fields and
//! guessing which representation is active.

use std::fmt;

use crate::symbolic::View::bvp_codegen::PreparedSparseAtomBvpCodegen;
use crate::symbolic::bvp::atom_aot::{AtomAotPlanError, AtomAotPreparedPlan};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_engine::Expr;

/// Borrowed ExprLegacy AOT payload.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExprLegacyAotAdapter<'a> {
    /// Residual expressions owned by the compatibility bridge.
    pub residuals: &'a [Expr],
    /// Sparse Jacobian expressions owned by the compatibility bridge.
    pub sparse_entries: &'a [(usize, usize, Expr)],
    /// Matrix shape of the prepared problem.
    pub shape: (usize, usize),
}

/// Borrowed AtomView-native AOT payload.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomViewAotAdapter<'a> {
    /// Atom-native codegen data used by the language emitters.
    pub codegen: &'a PreparedSparseAtomBvpCodegen,
    /// One validated plan shared by manifest and telemetry lifecycle code.
    pub plan: &'a AtomAotPreparedPlan,
}

impl<'a> AtomViewAotAdapter<'a> {
    /// Builds a manifest from the plan owned by this route adapter.
    pub(crate) fn prepared_manifest(
        &self,
        matrix_backend: MatrixBackend,
    ) -> Result<PreparedProblemManifest, AtomAotPlanError> {
        self.codegen
            .prepared_aot_manifest_from_plan(matrix_backend, self.plan)
    }
}

/// Typed route selected by a prepared BVP AOT bridge.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BvpAotAdapter<'a> {
    /// Retained Expr-based compatibility path.
    ExprLegacy(ExprLegacyAotAdapter<'a>),
    /// Canonical AtomView-native path.
    AtomView(AtomViewAotAdapter<'a>),
}

/// Failure to expose a complete typed adapter from a compatibility bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BvpAotAdapterError {
    /// AtomView codegen exists, but its validated plan was not retained.
    AtomViewPlanMissing,
    /// A plan exists without the codegen payload required by emitters.
    AtomViewCodegenMissing,
}

impl fmt::Display for BvpAotAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomViewPlanMissing => {
                f.write_str("AtomView AOT adapter has no validated prepared plan")
            }
            Self::AtomViewCodegenMissing => {
                f.write_str("AtomView AOT adapter has no codegen payload")
            }
        }
    }
}

impl std::error::Error for BvpAotAdapterError {}

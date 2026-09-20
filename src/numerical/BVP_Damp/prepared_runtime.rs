//! Revision tracking and resource ownership for prepared BVP callbacks/runtime.
//!
//! A prepared residual/Jacobian callback is valid only for the problem inputs
//! and solver configuration from which it was built.  Damped and Frozen use
//! the same small value type so their invalidation rules cannot drift apart.

use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::numerical::BVP_Damp::factor_runtime::OwnedLinearFactorRuntime;
use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};

/// Resource-owning part of the common prepared runtime.
///
/// This is the first ownership slice of the shared `PreparedPlan`.  The
/// factor and numeric Jacobian are now owned by one common container for both
/// Damped and Frozen solvers, while callbacks and mesh/layout remain on their
/// historical fields until their own migration is completed.  The `Deref`
/// bridge is intentionally temporary: it preserves the existing internal
/// borrow sites without exposing this container as public API.
#[derive(Default)]
pub(crate) struct BvpPreparedRuntime {
    factor_owner: RefCell<Option<OwnedLinearFactorRuntime>>,
    /// Numeric Jacobian paired with the owned factor generation.
    ///
    /// Keeping both resources in the same container makes invalidation a
    /// single lifecycle operation instead of two solver-local mutations.
    pub(crate) old_jac: Option<Box<dyn MatrixType>>,
}

/// Allocation-free resource state used by lifecycle diagnostics and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpPreparedResourceSnapshot {
    pub(crate) has_numeric_jacobian: bool,
    pub(crate) has_factor: bool,
}

impl BvpPreparedRuntime {
    /// Creates an empty resource-owning runtime.
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Drops the numeric Jacobian and its factor from one generation.
    ///
    /// The revision guard is updated by the solver; this method only owns the
    /// resource transition and therefore stays allocation-free.
    #[inline]
    pub(crate) fn invalidate_numeric_jacobian(&mut self) {
        self.factor_owner.get_mut().take();
        self.old_jac = None;
    }

    /// Returns whether a numeric factor is currently owned by the runtime.
    #[inline]
    pub(crate) fn has_factor(&self) -> bool {
        self.factor_owner.borrow().is_some()
    }

    /// Reports whether the runtime still owns a matching numeric resource set.
    #[inline]
    pub(crate) fn resource_snapshot(&self) -> BvpPreparedResourceSnapshot {
        BvpPreparedResourceSnapshot {
            has_numeric_jacobian: self.old_jac.is_some(),
            has_factor: self.has_factor(),
        }
    }

    /// Borrows the factor for an existing compatibility call site.
    #[inline]
    pub(crate) fn factor(&self) -> Ref<'_, Option<OwnedLinearFactorRuntime>> {
        self.factor_owner.borrow()
    }

    /// Mutably borrows the factor for an existing compatibility call site.
    #[inline]
    pub(crate) fn factor_mut(&self) -> RefMut<'_, Option<OwnedLinearFactorRuntime>> {
        self.factor_owner.borrow_mut()
    }
}

impl Deref for BvpPreparedRuntime {
    type Target = RefCell<Option<OwnedLinearFactorRuntime>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.factor_owner
    }
}

impl DerefMut for BvpPreparedRuntime {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.factor_owner
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreparedRuntimeStamp {
    problem: u64,
    mesh: u64,
    parameters: u64,
    callbacks: u64,
    configuration: u64,
}

/// Common identity captured by every prepared BVP runtime.
///
/// This is deliberately metadata-only: callback ownership remains in the
/// solver/backend bundle, while this value answers whether that bundle still
/// belongs to the current public problem and configuration. Keeping the
/// identity typed prevents Damped and Frozen from growing independent ad-hoc
/// invalidation rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BvpPreparedPlan {
    stamp: PreparedRuntimeStamp,
    fingerprint: Option<PreparedPlanFingerprint>,
    stage: BvpPreparedPlanStage,
}

/// Runtime stage captured by the common prepared-plan guard.
///
/// The stage is intentionally separate from the input revision. A numeric
/// Jacobian may remain valid after a factor is dropped, while any structural
/// input change invalidates the complete prepared runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BvpPreparedPlanStage {
    Unprepared,
    Prepared,
    NumericalJacobianCurrent,
    FactorCurrent,
    Invalidated,
}

/// A cheap identity of the public compatibility inputs captured by a prepared
/// solver plan.
///
/// The solver historically exposes mutable fields for source compatibility, so
/// a caller can bypass revision-tracked setters.  This scalar fingerprint is
/// checked only at the `try_solver_prepared` boundary, never in residual or
/// Jacobian hot paths.  It is therefore a safety net, not runtime telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedPlanFingerprint(pub(crate) u64);

/// Mixes a byte sequence into the deterministic FNV-1a plan fingerprint.
#[inline]
pub(crate) fn fingerprint_bytes(state: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *state ^= u64::from(*byte);
        *state = state.wrapping_mul(0x1000_0000_01b3);
    }
}

/// Adds a debug representation to a prepared-plan fingerprint.
///
/// This helper is intentionally used at preparation boundaries only.  It
/// keeps the compatibility audit independent of the many historical field
/// types without adding `Hash` bounds or allocations to the solve loop.
#[inline]
pub(crate) fn fingerprint_debug<T: std::fmt::Debug>(state: &mut u64, value: &T) {
    fingerprint_bytes(state, format!("{value:?}").as_bytes());
    fingerprint_bytes(state, &[0xff]);
}

/// Tracks the input generations captured by the current prepared runtime.
///
/// The counters are deliberately scalar and allocation-free.  Wrapping is
/// acceptable: a prepared stamp can only become equal again after the same
/// complete generation state has been rebuilt, not merely after one mutation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpRuntimeRevision {
    problem: u64,
    mesh: u64,
    parameters: u64,
    callbacks: u64,
    configuration: u64,
    prepared: Option<BvpPreparedPlan>,
}

impl BvpRuntimeRevision {
    #[inline]
    fn bump(value: &mut u64) {
        *value = value.wrapping_add(1);
    }

    /// Marks equations, unknown layout, argument or boundary-condition data.
    ///
    /// This is intentionally separate from `callbacks_changed`: a prepared
    /// runtime may have identical callback closures but still be invalid for a
    /// changed physical problem.
    #[inline]
    pub(crate) fn problem_changed(&mut self) {
        Self::bump(&mut self.problem);
        self.invalidate_prepared_plan();
    }

    /// Marks a mesh, node layout, or boundary-layout change.
    #[inline]
    pub(crate) fn mesh_changed(&mut self) {
        Self::bump(&mut self.mesh);
        self.invalidate_prepared_plan();
    }

    /// Marks a parameter name or numeric parameter binding change.
    #[inline]
    pub(crate) fn parameters_changed(&mut self) {
        Self::bump(&mut self.parameters);
        self.invalidate_prepared_plan();
    }

    /// Marks a residual/Jacobian callback source change.
    #[inline]
    pub(crate) fn callbacks_changed(&mut self) {
        Self::bump(&mut self.callbacks);
        self.invalidate_prepared_plan();
    }

    /// Marks a backend, evaluator, matrix, or solver-policy change.
    #[inline]
    pub(crate) fn configuration_changed(&mut self) {
        Self::bump(&mut self.configuration);
        self.invalidate_prepared_plan();
    }

    /// Records the input generations captured by a newly prepared runtime.
    #[inline]
    pub(crate) fn mark_prepared(&mut self) {
        self.prepared = Some(BvpPreparedPlan {
            stamp: self.current_stamp(),
            fingerprint: None,
            stage: BvpPreparedPlanStage::Prepared,
        });
    }

    /// Records the revision and the public-input identity captured by a plan.
    #[inline]
    pub(crate) fn mark_prepared_with_fingerprint(&mut self, fingerprint: PreparedPlanFingerprint) {
        self.prepared = Some(BvpPreparedPlan {
            stamp: self.current_stamp(),
            fingerprint: Some(fingerprint),
            stage: BvpPreparedPlanStage::Prepared,
        });
    }

    /// Returns whether the prepared runtime still matches every input class.
    #[inline]
    pub(crate) fn is_current(&self) -> bool {
        self.prepared.is_some_and(|prepared| {
            prepared.stamp == self.current_stamp()
                && prepared.stage != BvpPreparedPlanStage::Invalidated
        })
    }

    /// Returns whether revisions and compatibility inputs still match.
    #[inline]
    pub(crate) fn is_current_with_fingerprint(&self, fingerprint: PreparedPlanFingerprint) -> bool {
        self.is_current()
            && self
                .prepared
                .is_some_and(|prepared| prepared.fingerprint == Some(fingerprint))
    }

    /// Refreshes only the compatibility-input identity after an explicit,
    /// typed numeric rebind. Structural preparation and revision ownership are
    /// unchanged; this is never used for direct public-field mutation.
    #[inline]
    pub(crate) fn refresh_prepared_fingerprint(&mut self, fingerprint: PreparedPlanFingerprint) {
        if self.prepared.is_some() {
            if let Some(prepared) = self.prepared.as_mut() {
                prepared.fingerprint = Some(fingerprint);
                if prepared.stage == BvpPreparedPlanStage::Invalidated {
                    prepared.stage = BvpPreparedPlanStage::Prepared;
                }
            }
        }
    }

    /// Records that a current numerical Jacobian has been produced.
    #[inline]
    pub(crate) fn mark_numeric_jacobian_current(&mut self) {
        let current_stamp = self.current_stamp();
        if let Some(prepared) = self.prepared.as_mut() {
            if prepared.stamp == current_stamp
                && prepared.stage != BvpPreparedPlanStage::Invalidated
            {
                prepared.stage = BvpPreparedPlanStage::NumericalJacobianCurrent;
            }
        }
    }

    /// Records that the factorization belongs to the current Jacobian.
    #[inline]
    pub(crate) fn mark_factor_current(&mut self) {
        let current_stamp = self.current_stamp();
        if let Some(prepared) = self.prepared.as_mut() {
            if prepared.stamp == current_stamp
                && prepared.stage != BvpPreparedPlanStage::Invalidated
            {
                prepared.stage = BvpPreparedPlanStage::FactorCurrent;
            }
        }
    }

    /// Drops only factor validity while preserving the prepared callback/Jacobian plan.
    #[inline]
    pub(crate) fn factor_invalidated(&mut self) {
        if let Some(prepared) = self.prepared.as_mut() {
            if prepared.stage == BvpPreparedPlanStage::FactorCurrent {
                prepared.stage = BvpPreparedPlanStage::NumericalJacobianCurrent;
            }
        }
    }

    /// Returns the current prepared stage for diagnostics and parity tests.
    #[inline]
    pub(crate) fn prepared_stage(&self) -> BvpPreparedPlanStage {
        self.prepared
            .map(|prepared| prepared.stage)
            .unwrap_or(BvpPreparedPlanStage::Unprepared)
    }

    #[inline]
    fn invalidate_prepared_plan(&mut self) {
        if let Some(prepared) = self.prepared.as_mut() {
            prepared.stage = BvpPreparedPlanStage::Invalidated;
        }
    }

    /// Returns the common prepared-plan identity for diagnostics/tests.
    #[inline]
    pub(crate) fn prepared_plan(&self) -> Option<BvpPreparedPlan> {
        self.prepared
    }

    #[inline]
    fn current_stamp(&self) -> PreparedRuntimeStamp {
        PreparedRuntimeStamp {
            problem: self.problem,
            mesh: self.mesh,
            parameters: self.parameters,
            callbacks: self.callbacks,
            configuration: self.configuration,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::numerical::BVP_Damp::factor_runtime::prepare_factor_owner_runtime;
    use nalgebra::DMatrix;

    use super::{
        BvpPreparedPlanStage, BvpPreparedRuntime, BvpRuntimeRevision, PreparedPlanFingerprint,
        fingerprint_debug,
    };

    #[test]
    fn resource_owner_stores_and_invalidates_a_numeric_factor() {
        let mut runtime = BvpPreparedRuntime::new();
        assert!(!runtime.has_factor());

        let owner =
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("test matrix should produce an owned factor runtime");
        *runtime.factor_mut() = Some(owner);

        assert!(runtime.has_factor());
        assert!(runtime.factor().is_some());
        assert_eq!(
            runtime.resource_snapshot(),
            super::BvpPreparedResourceSnapshot {
                has_numeric_jacobian: false,
                has_factor: true,
            }
        );
        runtime.old_jac = Some(Box::new(DMatrix::from_row_slice(1, 1, &[2.0])));
        assert!(runtime.resource_snapshot().has_numeric_jacobian);
        runtime.invalidate_numeric_jacobian();
        assert!(!runtime.has_factor());
        assert!(runtime.old_jac.is_none());
        assert_eq!(runtime.resource_snapshot(), Default::default());
    }

    #[test]
    fn prepared_stamp_becomes_stale_for_each_input_class() {
        let mut revision = BvpRuntimeRevision::default();
        assert!(!revision.is_current());

        revision.mark_prepared();
        assert!(revision.is_current());

        revision.problem_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        revision.mesh_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        revision.parameters_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        revision.callbacks_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        revision.configuration_changed();
        assert!(!revision.is_current());
    }

    #[test]
    fn unchanged_inputs_keep_the_prepared_stamp_current() {
        let mut revision = BvpRuntimeRevision::default();
        revision.mark_prepared();
        assert!(revision.is_current());
        assert_eq!(revision, revision);
    }

    #[test]
    fn fingerprinted_plan_rejects_direct_input_identity_changes() {
        let mut revision = BvpRuntimeRevision::default();
        let mut hash = 0xcbf2_9ce4_8422_2325;
        fingerprint_debug(&mut hash, &("mesh", 8usize));
        let fingerprint = PreparedPlanFingerprint(hash);
        revision.mark_prepared_with_fingerprint(fingerprint);
        assert!(revision.prepared_plan().is_some());
        assert!(revision.is_current_with_fingerprint(fingerprint));
        assert!(!revision.is_current_with_fingerprint(PreparedPlanFingerprint(hash + 1)));
    }

    #[test]
    fn lifecycle_distinguishes_prepared_jacobian_and_factor() {
        let mut revision = BvpRuntimeRevision::default();
        assert_eq!(revision.prepared_stage(), BvpPreparedPlanStage::Unprepared);

        revision.mark_prepared();
        assert_eq!(revision.prepared_stage(), BvpPreparedPlanStage::Prepared);
        revision.mark_numeric_jacobian_current();
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::NumericalJacobianCurrent
        );
        revision.mark_factor_current();
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::FactorCurrent
        );
        revision.factor_invalidated();
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::NumericalJacobianCurrent
        );
        assert!(revision.is_current());

        revision.parameters_changed();
        assert_eq!(revision.prepared_stage(), BvpPreparedPlanStage::Invalidated);
        assert!(!revision.is_current());
    }

    #[test]
    fn numeric_rebind_can_refresh_a_stale_metadata_guard_without_restoring_factor() {
        let mut revision = BvpRuntimeRevision::default();
        revision.mark_prepared_with_fingerprint(PreparedPlanFingerprint(7));
        revision.mark_numeric_jacobian_current();
        revision.mark_factor_current();
        revision.factor_invalidated();

        revision.refresh_prepared_fingerprint(PreparedPlanFingerprint(8));
        assert!(revision.is_current_with_fingerprint(PreparedPlanFingerprint(8)));
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::NumericalJacobianCurrent
        );
    }
}

//! Revision tracking and resource ownership for prepared BVP callbacks/runtime.
//!
//! A prepared residual/Jacobian callback is valid only for the problem inputs
//! and solver configuration from which it was built.  Damped and Frozen use
//! the same small value type so their invalidation rules cannot drift apart.

// Snapshot and revision helpers are consumed by debug lifecycle stories and
// will become part of the complete PreparedPlan owner in a later slice.
#![allow(dead_code)]

use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::numerical::BVP_Damp::factor_runtime::OwnedLinearFactorRuntime;
use std::cell::{Ref, RefCell, RefMut};
use std::ops::{Deref, DerefMut};

/// Resource-owning part of the common prepared runtime.
///
/// This is the first ownership slice of the shared `PreparedPlan`.  The
/// factor and numeric Jacobian are now owned by one common container for both
/// Damped and Frozen solvers, while callbacks and mesh/layout remain on their
/// historical fields until their own migration is completed. The `Deref`
/// bridge is intentionally temporary: it preserves the existing internal
/// borrow sites without exposing this container as public API.
#[derive(Default)]
pub(crate) struct BvpPreparedRuntime {
    factor_owner: RefCell<Option<OwnedLinearFactorRuntime>>,
    /// Fingerprint of the complete prepared input bundle that published the
    /// current layout/Jacobian resources.  This is kept next to the owned
    /// resources so a prepared solve cannot accidentally use a factor from a
    /// different callback or mesh generation.
    prepared_binding: Option<PreparedPlanFingerprint>,
    /// Canonical generated Jacobian layout for the prepared callback bundle.
    layout: BvpPreparedLayout,
    /// Numeric Jacobian paired with the owned factor generation.
    ///
    /// Keeping both resources in the same container makes invalidation a
    /// single lifecycle operation instead of two solver-local mutations.
    pub(crate) old_jac: Option<Box<dyn MatrixType>>,
    /// Generation of the numeric Jacobian currently stored in `old_jac`.
    ///
    /// The generation is intentionally local to the resource owner. It is a
    /// second line of defence below `BvpRuntimeRevision`: even if a future
    /// caller forgets to update the outer revision, installing a new numeric
    /// Jacobian cannot leave the previous factor usable by this container.
    numeric_jacobian_generation: u64,
    /// Generation for which the owned factor was built, if any.
    factor_generation: Option<u64>,
}

/// Layout metadata shared by Lambdify and AOT callback routes.
///
/// The solver structs retain compatibility mirrors for now, but new runtime
/// code reads this value from the prepared owner. Keeping it beside the
/// numeric Jacobian/factor makes layout invalidation explicit and prevents a
/// callback bundle from being paired with a different bandwidth accidentally.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpPreparedLayout {
    variable_string: Vec<String>,
    bandwidth: (usize, usize),
}

/// Allocation-free resource state used by lifecycle diagnostics and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpPreparedResourceSnapshot {
    pub(crate) has_numeric_jacobian: bool,
    pub(crate) has_factor: bool,
    pub(crate) has_layout: bool,
    pub(crate) numeric_jacobian_generation: u64,
    pub(crate) factor_generation: Option<u64>,
    pub(crate) factor_matches_jacobian: bool,
}

/// A single diagnostic view of the prepared plan and its owned resources.
///
/// Keeping these values together prevents diagnostics from accidentally
/// pairing a plan snapshot from one lifecycle transition with resources from
/// another. The snapshot is allocation-free and is not used by callbacks or
/// linear solves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpPreparedRuntimeSnapshot {
    pub(crate) plan: Option<BvpPreparedPlan>,
    pub(crate) resources: BvpPreparedResourceSnapshot,
}

impl BvpPreparedRuntimeSnapshot {
    /// Checks the internal stage/resource invariant without judging whether
    /// the plan is current relative to external solver inputs.
    #[inline]
    pub(crate) fn stage_resources_are_consistent(&self) -> bool {
        match self.plan.map(|plan| plan.stage) {
            None | Some(BvpPreparedPlanStage::Unprepared) => {
                !self.resources.has_numeric_jacobian && !self.resources.has_factor
            }
            Some(BvpPreparedPlanStage::Prepared) => {
                !self.resources.has_numeric_jacobian && !self.resources.has_factor
            }
            Some(BvpPreparedPlanStage::NumericalJacobianCurrent) => {
                self.resources.has_numeric_jacobian && !self.resources.has_factor
            }
            Some(BvpPreparedPlanStage::FactorCurrent) => {
                self.resources.has_numeric_jacobian
                    && self.resources.has_factor
                    && self.resources.factor_matches_jacobian
            }
            Some(BvpPreparedPlanStage::Invalidated) => {
                !self.resources.has_numeric_jacobian && !self.resources.has_factor
            }
        }
    }
}

impl BvpPreparedRuntime {
    /// Creates an empty resource-owning runtime.
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Publishes layout metadata together with a newly prepared callback
    /// bundle. This is a cold-path operation and may move the variable names.
    #[inline]
    pub(crate) fn replace_layout(
        &mut self,
        variable_string: Vec<String>,
        bandwidth: (usize, usize),
    ) {
        // Layout belongs to the callback/Jacobian bundle.  Never allow a
        // newly published layout to inherit a numeric matrix or factor from
        // the previous bundle, even if a caller forgot an explicit clear.
        self.invalidate_numeric_jacobian();
        self.prepared_binding = None;
        self.layout = BvpPreparedLayout {
            variable_string,
            bandwidth,
        };
    }

    /// Drops generated layout metadata during mesh/callback invalidation.
    #[inline]
    pub(crate) fn clear_layout(&mut self) {
        self.invalidate_numeric_jacobian();
        self.prepared_binding = None;
        self.layout = BvpPreparedLayout::default();
    }

    /// Publishes the input identity after callbacks and layout have been
    /// prepared.  This is a cold lifecycle operation; callback evaluation
    /// never recomputes or reads this value.
    #[inline]
    pub(crate) fn publish_prepared_binding(&mut self, fingerprint: PreparedPlanFingerprint) {
        self.prepared_binding = Some(fingerprint);
    }

    /// Refreshes the binding after an explicit numeric parameter rebind.
    ///
    /// A parameter rebind keeps symbolic callbacks and layout reusable, but
    /// the public-plan fingerprint changes.  The numeric Jacobian/factor is
    /// still invalidated separately by the solver.
    #[inline]
    pub(crate) fn refresh_prepared_binding(&mut self, fingerprint: PreparedPlanFingerprint) {
        if self.prepared_binding.is_some() {
            self.prepared_binding = Some(fingerprint);
        }
    }

    /// Checks that the owned numeric resources belong to the requested plan.
    #[inline]
    pub(crate) fn prepared_binding_matches(&self, fingerprint: PreparedPlanFingerprint) -> bool {
        self.prepared_binding == Some(fingerprint)
    }

    /// Returns the canonical prepared Jacobian bandwidth.
    #[inline]
    pub(crate) fn bandwidth(&self) -> (usize, usize) {
        self.layout.bandwidth
    }

    /// Returns the canonical prepared variable ordering.
    #[inline]
    pub(crate) fn variable_string(&self) -> &[String] {
        &self.layout.variable_string
    }

    /// Drops the numeric Jacobian and its factor from one generation.
    ///
    /// The revision guard is updated by the solver; this method only owns the
    /// resource transition and therefore stays allocation-free.
    #[inline]
    pub(crate) fn invalidate_numeric_jacobian(&mut self) {
        self.factor_owner.get_mut().take();
        self.old_jac = None;
        self.numeric_jacobian_generation = self.numeric_jacobian_generation.wrapping_add(1);
        self.factor_generation = None;
    }

    /// Replaces the numeric Jacobian and invalidates its previous factor.
    ///
    /// This is the only production path for publishing a newly evaluated
    /// Jacobian. Keeping the replacement and factor invalidation together
    /// prevents a stale Dense/faer/Banded factor from crossing generations.
    #[inline]
    pub(crate) fn replace_numeric_jacobian(&mut self, jacobian: Box<dyn MatrixType>) {
        self.factor_owner.get_mut().take();
        self.old_jac = Some(jacobian);
        self.numeric_jacobian_generation = self.numeric_jacobian_generation.wrapping_add(1);
        self.factor_generation = None;
    }

    /// Clears the current numeric Jacobian and any factor derived from it.
    #[inline]
    pub(crate) fn clear_numeric_jacobian(&mut self) {
        self.invalidate_numeric_jacobian();
    }

    /// Publishes a factor for the currently stored numeric Jacobian.
    ///
    /// A factor without a Jacobian is never considered current. This keeps
    /// test fixtures and compatibility code from accidentally turning an
    /// orphaned factor into a valid prepared resource.
    #[inline]
    pub(crate) fn replace_factor(&mut self, factor: Option<OwnedLinearFactorRuntime>) {
        let has_jacobian = self.old_jac.is_some();
        let has_factor = factor.is_some();
        *self.factor_owner.get_mut() = factor;
        self.factor_generation =
            (has_jacobian && has_factor).then_some(self.numeric_jacobian_generation);
    }

    /// Removes the published factor and marks its generation stale.
    #[inline]
    pub(crate) fn take_factor(&mut self) -> Option<OwnedLinearFactorRuntime> {
        self.factor_generation = None;
        self.factor_owner.get_mut().take()
    }

    /// Returns the currently published numeric Jacobian.
    #[inline]
    pub(crate) fn jacobian(&self) -> Option<&Box<dyn MatrixType>> {
        self.old_jac.as_ref()
    }

    /// Borrows the optional Jacobian slot for compatibility algorithms that
    /// need to decide whether a recalculation is due.
    #[inline]
    pub(crate) fn jacobian_slot(&self) -> &Option<Box<dyn MatrixType>> {
        &self.old_jac
    }

    /// Returns whether a numeric factor is currently owned by the runtime.
    #[inline]
    pub(crate) fn has_factor(&self) -> bool {
        self.factor_owner.borrow().is_some()
    }

    /// Returns whether the owned factor is valid for the current Jacobian.
    #[inline]
    pub(crate) fn has_current_factor(&self) -> bool {
        self.factor_generation == Some(self.numeric_jacobian_generation) && self.has_factor()
    }

    /// Reports whether the runtime still owns a matching numeric resource set.
    #[inline]
    pub(crate) fn resource_snapshot(&self) -> BvpPreparedResourceSnapshot {
        BvpPreparedResourceSnapshot {
            has_numeric_jacobian: self.old_jac.is_some(),
            has_factor: self.has_factor(),
            has_layout: !self.layout.variable_string.is_empty() || self.layout.bandwidth != (0, 0),
            numeric_jacobian_generation: self.numeric_jacobian_generation,
            factor_generation: self.factor_generation,
            factor_matches_jacobian: self.has_current_factor(),
        }
    }

    /// Captures plan metadata and resource ownership as one diagnostic value.
    #[inline]
    pub(crate) fn runtime_snapshot(
        &self,
        revision: &BvpRuntimeRevision,
    ) -> BvpPreparedRuntimeSnapshot {
        BvpPreparedRuntimeSnapshot {
            plan: revision.prepared_plan(),
            resources: self.resource_snapshot(),
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
    /// Generations of resources that are not fully moved into this owner yet.
    ///
    /// The callback boxes remain compatibility fields for now, but their
    /// publication is represented here as one typed generation.  Artifact and
    /// linked-runtime generations make the transition explicit and prevent a
    /// newly selected artifact from being mistaken for the previous warm
    /// runtime.
    ownership: BvpPreparedOwnership,
    stage: BvpPreparedPlanStage,
}

/// Scalar ownership stamp for the prepared runtime resources.
///
/// This is deliberately metadata-only in the compatibility migration: the
/// actual callback boxes still live in the historical solver/backend bundle.
/// Keeping the generations in the common plan lets both Damped and Frozen
/// enforce the same lifecycle now, while the final callback move can replace
/// the compatibility boxes without changing the invalidation contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BvpPreparedOwnership {
    pub(crate) callback_generation: u64,
    pub(crate) artifact_generation: u64,
    pub(crate) linked_runtime_generation: u64,
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

/// Adds the identity of a compatibility callback to a prepared-plan
/// fingerprint.
///
/// The callback traits intentionally remain object-safe and are still exposed
/// through historical public fields.  Calling a callback to fingerprint its
/// behaviour would be expensive and could have side effects, so the prepared
/// boundary records the trait-object data address instead.  This catches the
/// normal direct `Box` replacement path without adding work to residual or
/// Jacobian evaluation; the closed callback owner remains the next migration
/// step for stronger identity guarantees.
#[inline]
pub(crate) fn fingerprint_callback_ptr<T: ?Sized>(state: &mut u64, callback: Option<&T>) {
    let address = callback
        .map(|callback| callback as *const T as *const () as usize)
        .unwrap_or(0);
    fingerprint_debug(state, &address);
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
    artifact: u64,
    linked_runtime: u64,
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

    /// Marks a newly materialized/selected artifact.  The next preparation
    /// must capture this generation before a prepared solve is accepted.
    #[inline]
    pub(crate) fn artifact_changed(&mut self) {
        Self::bump(&mut self.artifact);
        self.invalidate_prepared_plan();
    }

    /// Marks a linked runtime replacement or unload.  This is separate from
    /// artifact identity because the same artifact can be linked more than
    /// once in a process with different callback/runtime state.
    #[inline]
    pub(crate) fn linked_runtime_changed(&mut self) {
        Self::bump(&mut self.linked_runtime);
        self.invalidate_prepared_plan();
    }

    /// Records the input generations captured by a newly prepared runtime.
    #[inline]
    pub(crate) fn mark_prepared(&mut self) {
        self.prepared = Some(BvpPreparedPlan {
            stamp: self.current_stamp(),
            fingerprint: None,
            ownership: self.current_ownership(),
            stage: BvpPreparedPlanStage::Prepared,
        });
    }

    /// Records the revision and the public-input identity captured by a plan.
    #[inline]
    pub(crate) fn mark_prepared_with_fingerprint(&mut self, fingerprint: PreparedPlanFingerprint) {
        self.prepared = Some(BvpPreparedPlan {
            stamp: self.current_stamp(),
            fingerprint: Some(fingerprint),
            ownership: self.current_ownership(),
            stage: BvpPreparedPlanStage::Prepared,
        });
    }

    /// Returns whether the prepared runtime still matches every input class.
    #[inline]
    pub(crate) fn is_current(&self) -> bool {
        self.prepared.is_some_and(|prepared| {
            prepared.stamp == self.current_stamp()
                && prepared.ownership == self.current_ownership()
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

    /// Returns the resource generations captured by the current plan.
    #[inline]
    pub(crate) fn prepared_ownership(&self) -> Option<BvpPreparedOwnership> {
        self.prepared.map(|plan| plan.ownership)
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

    #[inline]
    fn current_ownership(&self) -> BvpPreparedOwnership {
        BvpPreparedOwnership {
            callback_generation: self.callbacks,
            artifact_generation: self.artifact,
            linked_runtime_generation: self.linked_runtime,
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

        assert!(!runtime.prepared_binding_matches(PreparedPlanFingerprint(11)));
        runtime.publish_prepared_binding(PreparedPlanFingerprint(11));
        assert!(runtime.prepared_binding_matches(PreparedPlanFingerprint(11)));
        assert!(!runtime.prepared_binding_matches(PreparedPlanFingerprint(12)));

        runtime.old_jac = Some(Box::new(DMatrix::from_row_slice(1, 1, &[2.0])));
        let owner =
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("test matrix should produce an owned factor runtime");
        runtime.replace_factor(Some(owner));

        assert!(runtime.has_factor());
        assert!(runtime.factor().is_some());
        assert_eq!(
            runtime.resource_snapshot(),
            super::BvpPreparedResourceSnapshot {
                has_numeric_jacobian: true,
                has_factor: true,
                has_layout: false,
                numeric_jacobian_generation: 0,
                factor_generation: Some(0),
                factor_matches_jacobian: true,
            }
        );
        assert!(runtime.resource_snapshot().has_numeric_jacobian);
        runtime.invalidate_numeric_jacobian();
        assert!(!runtime.has_factor());
        assert!(runtime.old_jac.is_none());
        assert_eq!(
            runtime.resource_snapshot(),
            super::BvpPreparedResourceSnapshot {
                has_numeric_jacobian: false,
                has_factor: false,
                has_layout: false,
                numeric_jacobian_generation: 1,
                factor_generation: None,
                factor_matches_jacobian: false,
            }
        );
    }

    #[test]
    fn replacing_numeric_jacobian_cannot_reuse_the_previous_factor() {
        let mut runtime = BvpPreparedRuntime::new();
        runtime.old_jac = Some(Box::new(DMatrix::from_row_slice(1, 1, &[2.0])));
        let owner = prepare_factor_owner_runtime(
            runtime
                .jacobian()
                .expect("test Jacobian should be present")
                .as_ref(),
            (0, 0),
            None,
        )
        .expect("test matrix should produce an owned factor runtime");
        runtime.replace_factor(Some(owner));
        let before = runtime.resource_snapshot();
        assert!(before.has_factor);
        assert!(before.factor_matches_jacobian);

        runtime.replace_numeric_jacobian(Box::new(DMatrix::from_row_slice(1, 1, &[3.0])));
        let after = runtime.resource_snapshot();
        assert!(after.has_numeric_jacobian);
        assert!(!after.has_factor);
        assert_eq!(after.factor_generation, None);
        assert!(after.numeric_jacobian_generation > before.numeric_jacobian_generation);
        assert!(!after.factor_matches_jacobian);
    }

    #[test]
    fn publishing_a_new_layout_invalidates_matrix_and_factor_resources() {
        let mut runtime = BvpPreparedRuntime::new();
        runtime.replace_numeric_jacobian(Box::new(DMatrix::from_row_slice(1, 1, &[2.0])));
        let owner = prepare_factor_owner_runtime(
            runtime
                .jacobian()
                .expect("test Jacobian should be present")
                .as_ref(),
            (0, 0),
            None,
        )
        .expect("test matrix should produce an owned factor runtime");
        runtime.replace_factor(Some(owner));
        assert!(runtime.resource_snapshot().factor_matches_jacobian);

        runtime.replace_layout(vec!["y_0".to_string()], (1, 2));
        let after_replace = runtime.resource_snapshot();
        assert!(after_replace.has_layout);
        assert_eq!(runtime.bandwidth(), (1, 2));
        assert_eq!(runtime.variable_string(), ["y_0"]);
        assert!(!after_replace.has_numeric_jacobian);
        assert!(!after_replace.has_factor);

        runtime.clear_layout();
        let after_clear = runtime.resource_snapshot();
        assert!(!after_clear.has_layout);
        assert!(!after_clear.has_numeric_jacobian);
        assert!(!after_clear.has_factor);
        assert!(!runtime.prepared_binding_matches(PreparedPlanFingerprint(11)));
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
    fn artifact_and_linked_runtime_generations_invalidate_prepared_plan() {
        let mut revision = BvpRuntimeRevision::default();
        revision.mark_prepared();
        let initial = revision
            .prepared_ownership()
            .expect("prepared plan should expose ownership generations");
        assert!(revision.is_current());

        revision.artifact_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        let after_artifact = revision
            .prepared_ownership()
            .expect("artifact generation should be captured after rebuild");
        assert!(after_artifact.artifact_generation > initial.artifact_generation);
        assert_eq!(
            after_artifact.linked_runtime_generation,
            initial.linked_runtime_generation
        );

        revision.linked_runtime_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        let after_link = revision
            .prepared_ownership()
            .expect("linked runtime generation should be captured after relink");
        assert!(after_link.linked_runtime_generation > after_artifact.linked_runtime_generation);
        assert!(revision.is_current());
    }

    #[test]
    fn callback_generation_is_distinct_from_numeric_factor_lifecycle() {
        let mut revision = BvpRuntimeRevision::default();
        revision.mark_prepared();
        let before = revision.prepared_ownership().unwrap();

        revision.mark_numeric_jacobian_current();
        revision.mark_factor_current();
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::FactorCurrent
        );
        assert!(revision.is_current());

        revision.callbacks_changed();
        assert!(!revision.is_current());
        revision.mark_prepared();
        let after = revision.prepared_ownership().unwrap();
        assert!(after.callback_generation > before.callback_generation);
        assert_eq!(
            after.artifact_generation, before.artifact_generation,
            "callback replacement alone must not pretend to materialize a new artifact"
        );
        assert_eq!(
            revision.prepared_stage(),
            BvpPreparedPlanStage::Prepared,
            "a callback replacement cannot retain the old factor stage"
        );
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
    fn combined_snapshot_rejects_plan_resource_stage_mismatch() {
        let mut runtime = BvpPreparedRuntime::new();
        let mut revision = BvpRuntimeRevision::default();

        revision.mark_prepared();
        assert!(
            runtime
                .runtime_snapshot(&revision)
                .stage_resources_are_consistent()
        );

        runtime.replace_numeric_jacobian(Box::new(DMatrix::from_row_slice(1, 1, &[2.0])));
        revision.mark_numeric_jacobian_current();
        assert!(
            runtime
                .runtime_snapshot(&revision)
                .stage_resources_are_consistent()
        );

        let factor = prepare_factor_owner_runtime(
            runtime
                .jacobian()
                .expect("test Jacobian should be present")
                .as_ref(),
            (0, 0),
            None,
        )
        .expect("test matrix should produce an owned factor runtime");
        runtime.replace_factor(Some(factor));
        revision.mark_factor_current();
        let snapshot = runtime.runtime_snapshot(&revision);
        assert!(snapshot.stage_resources_are_consistent());

        runtime.take_factor();
        revision.factor_invalidated();
        assert!(
            runtime
                .runtime_snapshot(&revision)
                .stage_resources_are_consistent()
        );
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

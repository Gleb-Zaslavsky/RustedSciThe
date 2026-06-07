//! Boundary-aware Banded backend planning for BVP_sci.
//!
//! BVP_sci's Newton matrix has a narrow collocation body and a small set of
//! endpoint boundary-condition rows.  A scalar banded factorization of the whole
//! matrix can be very inefficient because the BC rows couple the first and last
//! mesh nodes, widening the lower scalar band.  This module does not solve the
//! system yet; it classifies the structure so the production backend can choose
//! between full scalar banded, bordered/boundary-aware banded, or sparse
//! fallback deliberately.

use crate::numerical::BVP_sci::BVP_sci_banded::{
    infer_banded_profile, infer_banded_profile_for_row_range, BvpSciBandedProfile,
};
use crate::numerical::BVP_sci::BVP_sci_faer::faer_mat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciBandedRoute {
    /// The whole matrix is compact enough for direct scalar banded LU.
    FullScalarBanded,
    /// The collocation body is compact but boundary rows make the full scalar
    /// band too wide; production code should use a bordered/boundary-aware
    /// algorithm.
    BorderedBanded,
    /// The current structure is not a useful banded candidate.  Keep Sparse LU.
    SparseFallback,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvpSciBorderedBandedProfile {
    pub variable_count: usize,
    pub mesh_points: usize,
    pub parameter_count: usize,
    pub total_size: usize,
    pub collocation_rows: usize,
    pub boundary_rows: usize,
    pub full_scalar: BvpSciBandedProfile,
    pub collocation_scalar: BvpSciBandedProfile,
    pub full_amplification: Option<f64>,
    pub collocation_amplification: Option<f64>,
    pub recommended_route: BvpSciBandedRoute,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpSciBandedRoutePolicy {
    /// Scalar banded storage amplification considered acceptable for the whole
    /// matrix.
    pub max_full_scalar_amplification: f64,
    /// Collocation-only amplification considered compact enough to justify a
    /// future bordered/boundary-aware backend.
    pub max_collocation_amplification: f64,
    /// If boundary rows widen scalar storage by at least this ratio, prefer
    /// bordered/boundary-aware design over full scalar banded.
    pub min_boundary_widening_ratio: f64,
}

impl Default for BvpSciBandedRoutePolicy {
    fn default() -> Self {
        Self {
            max_full_scalar_amplification: 8.0,
            max_collocation_amplification: 4.0,
            min_boundary_widening_ratio: 2.0,
        }
    }
}

pub fn profile_bordered_banded_global_jacobian(
    mat: &faer_mat,
    variable_count: usize,
    mesh_points: usize,
    parameter_count: usize,
    policy: BvpSciBandedRoutePolicy,
) -> BvpSciBorderedBandedProfile {
    let total_size = variable_count * mesh_points + parameter_count;
    let collocation_rows = variable_count * mesh_points.saturating_sub(1);
    let boundary_rows = variable_count + parameter_count;
    let full_scalar = infer_banded_profile(mat);
    let collocation_scalar = infer_banded_profile_for_row_range(mat, 0, collocation_rows);
    let full_amplification = full_scalar.storage_amplification();
    let collocation_amplification = collocation_scalar.storage_amplification();

    let (recommended_route, reason) = recommend_route(
        mat,
        total_size,
        full_amplification,
        collocation_amplification,
        &full_scalar,
        &collocation_scalar,
        policy,
    );

    BvpSciBorderedBandedProfile {
        variable_count,
        mesh_points,
        parameter_count,
        total_size,
        collocation_rows,
        boundary_rows,
        full_scalar,
        collocation_scalar,
        full_amplification,
        collocation_amplification,
        recommended_route,
        reason,
    }
}

fn recommend_route(
    mat: &faer_mat,
    total_size: usize,
    full_amplification: Option<f64>,
    collocation_amplification: Option<f64>,
    full_scalar: &BvpSciBandedProfile,
    collocation_scalar: &BvpSciBandedProfile,
    policy: BvpSciBandedRoutePolicy,
) -> (BvpSciBandedRoute, &'static str) {
    let (nrows, ncols) = mat.shape();
    if nrows != ncols || nrows != total_size {
        return (
            BvpSciBandedRoute::SparseFallback,
            "matrix shape does not match BVP_sci global layout",
        );
    }

    if full_scalar.nonfinite_values > 0 || collocation_scalar.nonfinite_values > 0 {
        return (
            BvpSciBandedRoute::SparseFallback,
            "matrix contains non-finite stored values",
        );
    }

    let Some(full_amp) = full_amplification else {
        return (
            BvpSciBandedRoute::SparseFallback,
            "full matrix has no usable stored structure",
        );
    };

    if full_amp <= policy.max_full_scalar_amplification {
        return (
            BvpSciBandedRoute::FullScalarBanded,
            "full scalar banded storage is compact enough",
        );
    }

    let Some(colloc_amp) = collocation_amplification else {
        return (
            BvpSciBandedRoute::SparseFallback,
            "collocation rows have no usable stored structure",
        );
    };

    let boundary_widening = full_amp / colloc_amp.max(f64::EPSILON);
    if colloc_amp <= policy.max_collocation_amplification
        && boundary_widening >= policy.min_boundary_widening_ratio
    {
        return (
            BvpSciBandedRoute::BorderedBanded,
            "collocation body is compact but boundary rows widen scalar band",
        );
    }

    (
        BvpSciBandedRoute::SparseFallback,
        "neither full scalar nor collocation bordered banded route is attractive",
    )
}

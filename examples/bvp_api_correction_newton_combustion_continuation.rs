//! Synthetic six-state combustion-like BVP, NOT a physical kinetics model.
//!
//! Adapted from BVP_CorrectionNewton/tests/large_workloads.rs's six-state fixture.
//! Teta, q, C0, J0, C1, J1 are mathematical labels, not calibrated temperature,
//! heat flux, species fractions, or conserved physical fluxes. `heat_scale`
//! simply multiplies one nonlinear source. No CorrectionNewton chemistry library is used.
//! Run: cargo run --release --example bvp_api_correction_newton_combustion_continuation -- 256

#[path = "support/correction_newton_continuation.rs"]
mod continuation;

use RustedSciThe::numerical::BVP_api::BvpBoundarySide::{Left, Right};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    continuation::run(continuation::Model {
        names: vec!["Teta", "q", "C0", "J0", "C1", "J1"],
        equations: vec![
            "q",
            "-0.5*q + 0.2*heat_scale*exp(Teta)*C0",
            "J0",
            "-0.3*J0 - 0.5*C0 + 0.1*Teta",
            "J1",
            "-0.2*J1 - 0.3*C1 + 0.05*C0^2",
        ],
        right: 1.0,
        boundaries: vec![
            (Left, 0.0),
            (Right, 0.0),
            (Left, 1.0),
            (Right, 0.0),
            (Left, 0.0),
            (Right, 0.0),
        ],
        initial: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        parameter: "heat_scale",
        parameters: (0..10).map(|i| vec![0.8 + 0.05 * i as f64]).collect(),
        rhs: |scale, y| {
            vec![
                y[1],
                -0.5 * y[1] + 0.2 * scale * y[0].exp() * y[2],
                y[3],
                -0.3 * y[3] - 0.5 * y[2] + 0.1 * y[0],
                y[5],
                -0.2 * y[5] - 0.3 * y[4] + 0.05 * y[2] * y[2],
            ]
        },
        check: |_, mesh, values| {
            let end = mesh.len() - 1;
            assert!(values[(end, 0)] < -0.01, "nontrivial Teta profile");
            assert!(values[(end, 2)] > 1.01, "nontrivial C0 profile");
            assert!(values[(end, 4)] < -0.001, "nontrivial C1 profile");
        },
    })
}

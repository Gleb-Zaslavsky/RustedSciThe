//! Lower-branch Bratu continuation: u'' + lambda*exp(u) = 0.
//!
//! Symmetry reduces u(0)=u(1)=0 to u(0)=0, v(0.5)=0 with v=u'.
//! This gives one endpoint condition per state as required by Cantera.
//! Ten lambda values from 0.3 to 3.0 stay away from the turning point;
//! parameter stepping here is NOT pseudo-arclength continuation.
//! Run: cargo run --release --example bvp_api_cantera_bratu_continuation -- 256

#[path = "support/cantera_continuation.rs"]
mod continuation;

use nalgebra::DMatrix;
use RustedSciThe::numerical::BVP_api::BvpBoundarySide::{Left, Right};

fn check_exact(lambda: f64, mesh: &[f64], values: &DMatrix<f64>) {
    // lambda = 8*a^2/cosh(a)^2 is monotone on [0, 1], which brackets
    // the lower-branch solutions for every lambda in this schedule.
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..60 {
        let a = 0.5 * (low + high);
        if 8.0 * a * a / a.cosh().powi(2) < lambda {
            low = a;
        } else {
            high = a;
        }
    }
    let a = 0.5 * (low + high);
    let h = mesh[1] - mesh[0];
    let mut error = 0.0_f64;
    for (node, &x) in mesh.iter().enumerate() {
        let z = a * (2.0 * x - 1.0);
        let u = 2.0 * (a.cosh() / z.cosh()).ln();
        let v = -4.0 * a * z.tanh();
        error = error
            .max((values[(node, 0)] - u).abs())
            .max((values[(node, 1)] - v).abs());
    }
    // Discretization error is distinct from the Newton correction tolerance.
    assert!(
        error < 20.0 * h * h + 1e-6,
        "Bratu analytic-profile error: {error:e}"
    );
    assert!(values[(mesh.len() - 1, 0)] > 0.0);
    println!("  analytic lower-branch max error={error:.3e}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    continuation::run(continuation::Model {
        names: vec!["u", "v"],
        equations: vec!["v", "-lambda*exp(u)"],
        right: 0.5,
        boundaries: vec![(Left, 0.0), (Right, 0.0)],
        initial: vec![0.0, 0.0],
        parameter: "lambda",
        parameters: (1..=10).map(|i| vec![0.3 * i as f64]).collect(),
        rhs: |lambda, y| vec![y[1], -lambda * y[0].exp()],
        check: check_exact,
    })
}

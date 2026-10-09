//! Frontend-neutral BVP problem contracts.

/// Minimal callback contract consumed by the collocation core.
///
/// The final public API will add explicit analytic-Jacobian and symbolic
/// preparation builders without changing the numerical core's shape.
pub trait BvpSciProblem {
    fn dimension(&self) -> usize;
    fn parameter_dimension(&self) -> usize {
        0
    }
    fn rhs(&self, x: f64, y: &[f64], parameters: &[f64], out: &mut [f64]);
    fn boundary_residual(&self, ya: &[f64], yb: &[f64], parameters: &[f64], out: &mut [f64]);
}

use nalgebra::DVector;
use std::f64;
/// Constants for maximum dimensions support

pub fn lagrange_interpolate(x: f64, x_vals: &DVector<f64>, y_vals: &DVector<f64>) -> f64 {
    let n = x_vals.len();
    let mut result = 0.0;

    for i in 0..n {
        let mut term = y_vals[i];
        for j in 0..n {
            if i != j {
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j]);
            }
        }
        result += term;
    }

    result
}

/// Compute Newton divided difference coefficients

pub fn newton_divided_differences(x_vals: &DVector<f64>, y_vals: &DVector<f64>) -> DVector<f64> {
    let n = x_vals.len();
    let mut coef = y_vals.clone();

    for j in 1..n {
        for i in (j..n).rev() {
            coef[i] = (coef[i] - coef[i - 1]) / (x_vals[i] - x_vals[i - j]);
        }
    }

    coef
}

/// Evaluate Newton interpolating polynomial using Horner's method

pub fn newton_interpolate(x: f64, x_vals: &DVector<f64>, coef: &DVector<f64>) -> f64 {
    let n = coef.len();
    let mut result = coef[n - 1];

    for i in (0..n - 1).rev() {
        result = result * (x - x_vals[i]) + coef[i];
    }

    result
}
////////////////////////////////////////////////////////////////////////////////
/// Interpolation method selector
#[derive(Clone, Copy, Debug)]
pub enum InterpolationMethod {
    Pchip {
        /// Linear or logarithmic interpolation space
        space: InterpolationSpace,
        /// Optional output clamping
        clamp: bool,
    },
}

/// Interpolation space
#[derive(Clone, Copy, Debug)]
pub enum InterpolationSpace {
    /// Interpolate y directly
    Linear,
    /// Interpolate ln(y), then exponentiate
    /// Requires y > 0
    Log,
}

use std::cmp::Ordering;

/// Smallest allowed value in log-space to avoid ln(0)
/// This corresponds to ~1e-300
const LOG_FLOOR: f64 = -690.7755;

/// Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
///
/// This implementation supports:
/// - Linear-space interpolation
/// - Log-space interpolation (for positive quantities)
///
/// Log-space mode guarantees:
/// - Strict positivity
/// - Excellent relative accuracy
/// - Stability over many orders of magnitude
pub struct Pchip {
    /// Node locations (strictly increasing)
    x: Vec<f64>,
    /// Stored y-values (linear OR log-space depending on mode)
    y: Vec<f64>,
    /// Node derivatives
    m: Vec<f64>,
    /// Minimum allowed output (only used if clamping enabled)
    y_min: f64,
    /// Maximum allowed output (only used if clamping enabled)
    y_max: f64,
    /// Interpolation space
    space: InterpolationSpace,
}

impl Pchip {
    /// Construct a PCHIP interpolator
    ///
    /// # Panics
    /// - x.len() < 2
    /// - x not strictly increasing
    /// - x.len() != y.len()
    /// - Log-space selected with non-positive y
    pub fn new(x: &[f64], y: &[f64], space: InterpolationSpace) -> Self {
        assert!(x.len() >= 2, "PCHIP requires at least two points");
        assert_eq!(x.len(), y.len(), "x and y must have same length");

        // Ensure x is strictly increasing
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1], "x must be strictly increasing");
        }

        // Transform y into interpolation space
        let y_transformed: Vec<f64> = match space {
            InterpolationSpace::Linear => y.to_vec(),
            InterpolationSpace::Log => y
                .iter()
                .map(|&v| {
                    assert!(v > 0.0, "Log-space PCHIP requires strictly positive data");
                    v.ln().max(LOG_FLOOR)
                })
                .collect(),
        };

        let n = x.len();

        // Interval widths
        let mut h = vec![0.0; n - 1];
        // Secant slopes
        let mut delta = vec![0.0; n - 1];

        for i in 0..n - 1 {
            h[i] = x[i + 1] - x[i];
            delta[i] = (y_transformed[i + 1] - y_transformed[i]) / h[i];
        }

        // Node derivatives (Fritsch–Carlson)
        let mut m = vec![0.0; n];

        for i in 1..n - 1 {
            if delta[i - 1] * delta[i] > 0.0 {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            } else {
                m[i] = 0.0;
            }
        }

        // Endpoints
        m[0] = delta[0];
        m[n - 1] = delta[n - 2];

        // Clamp bounds (defined in *physical* space)
        let (y_min, y_max) = match space {
            InterpolationSpace::Linear => {
                let min = y.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (min, max)
            }
            InterpolationSpace::Log => {
                let min = y.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (min, max)
            }
        };

        Self {
            x: x.to_vec(),
            y: y_transformed,
            m,
            y_min,
            y_max,
            space,
        }
    }

    /// Evaluate interpolated value
    ///
    /// - Uses cubic Hermite inside domain
    /// - Linear extrapolation outside domain
    pub fn eval(&self, xq: f64, clamp: bool) -> f64 {
        let n = self.x.len();

        let i = match self
            .x
            .binary_search_by(|v| v.partial_cmp(&xq).unwrap_or(Ordering::Less))
        {
            Ok(i) => {
                return self.post_process(self.y[i], clamp);
            }
            Err(i) => {
                if i == 0 {
                    return self.post_process(self.y[0] + self.m[0] * (xq - self.x[0]), clamp);
                } else if i >= n {
                    return self
                        .post_process(self.y[n - 1] + self.m[n - 1] * (xq - self.x[n - 1]), clamp);
                }
                i - 1
            }
        };

        let h = self.x[i + 1] - self.x[i];
        let t = (xq - self.x[i]) / h;

        // Hermite basis
        let h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
        let h10 = t * t * t - 2.0 * t * t + t;
        let h01 = -2.0 * t * t * t + 3.0 * t * t;
        let h11 = t * t * t - t * t;

        let yq =
            h00 * self.y[i] + h10 * h * self.m[i] + h01 * self.y[i + 1] + h11 * h * self.m[i + 1];

        self.post_process(yq, clamp)
    }

    /// Convert back to physical space and apply optional clamping
    #[inline]
    fn post_process(&self, y_internal: f64, clamp: bool) -> f64 {
        let y_phys = match self.space {
            InterpolationSpace::Linear => y_internal,
            InterpolationSpace::Log => y_internal.exp(),
        };

        if clamp {
            y_phys.clamp(self.y_min, self.y_max)
        } else {
            y_phys
        }
    }
}

/* usage
let method = InterpolationMethod::Pchip {
    space: InterpolationSpace::Log, // ideal for mole fractions
    clamp: false,                   // optional safety net
};

let interp = match method {
    InterpolationMethod::Pchip { space, clamp } => {
        let p = Pchip::new(&x_vals, &y_vals, space);
        new_x.iter()
            .map(|&x| p.eval(x, clamp))
            .collect::<Vec<f64>>()
    }
};
*/

#[cfg(test)]
mod tests_PCHIP {
    use super::*;
    const EPS: f64 = 1e-12;
    #[test]
    fn pchip_preserves_monotonicity_linear() {
        // Strictly increasing function
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 0.5, 1.0, 1.5, 2.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        // Dense sampling
        let mut prev = pchip.eval(0.0, false);
        for i in 1..500 {
            let xq = 4.0 * i as f64 / 500.0;
            let yq = pchip.eval(xq, false);

            // Must be non-decreasing
            assert!(
                yq + EPS >= prev,
                "Monotonicity violated: {} < {} at x={}",
                yq,
                prev,
                xq
            );

            // Must stay within bounds
            assert!(yq >= 0.0 - EPS);
            assert!(yq <= 2.0 + EPS);

            prev = yq;
        }
    }

    #[test]
    fn pchip_no_overshoot_nonlinear() {
        // Monotone but strongly nonlinear
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 0.01, 0.5, 1.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        let y_min = *y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let y_max = *y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        for i in 0..500 {
            let xq = 3.0 * i as f64 / 500.0;
            let yq = pchip.eval(xq, false);

            assert!(
                yq >= y_min - EPS && yq <= y_max + EPS,
                "Overshoot detected: y={} at x={}",
                yq,
                xq
            );
        }
    }
    #[test]
    fn pchip_log_space_preserves_positivity() {
        let x = vec![300.0, 600.0, 1000.0, 2000.0];
        let y = vec![1e-12, 1e-9, 1e-6, 1e-3]; // mole fractions

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Log);

        for i in 0..1000 {
            let xq = 300.0 + (2000.0 - 300.0) * i as f64 / 1000.0;
            let yq = pchip.eval(xq, false);

            assert!(
                yq > 0.0,
                "Log-space PCHIP produced non-positive value: {} at x={}",
                yq,
                xq
            );
        }
    }
    #[test]
    fn pchip_log_space_respects_ratios() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1e-6, 1e-4, 1e-2];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Log);

        let y_mid = pchip.eval(2.0, false);
        assert!((y_mid - 1e-4).abs() / 1e-4 < 1e-6);

        let y_quarter = pchip.eval(1.5, false);

        // Should be geometric mean, not arithmetic
        let expected = (1e-6_f64 * 1e-4_f64).sqrt();
        let rel_err = (y_quarter - expected).abs() / expected;

        assert!(
            rel_err < 1e-3,
            "Log-space interpolation is not multiplicative enough"
        );
    }

    #[test]
    fn clamping_works_when_enabled() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.2, 0.5, 0.8];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        let yq = pchip.eval(-10.0, true);

        assert!(yq >= 0.2 && yq <= 0.8, "Clamping failed: yq={}", yq);
    }

    #[test]
    fn pchip_exact_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 2.0, 8.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        for i in 0..x.len() {
            let result = pchip.eval(x[i], false);
            assert!(
                (result - y[i]).abs() < EPS,
                "Not exact at node {}: {} vs {}",
                i,
                result,
                y[i]
            );
        }
    }

    #[test]
    fn pchip_extrapolation_linear() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        // Test left extrapolation
        let left_extrap = pchip.eval(-1.0, false);
        assert!(
            left_extrap.is_finite(),
            "Left extrapolation should be finite"
        );

        // Test right extrapolation
        let right_extrap = pchip.eval(3.0, false);
        assert!(
            right_extrap.is_finite(),
            "Right extrapolation should be finite"
        );
    }

    #[test]
    fn pchip_handles_flat_segments() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 1.0, 1.0, 2.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        // Should handle flat segments without oscillation
        let mid_val = pchip.eval(1.5, false);
        assert!(
            (mid_val - 1.0).abs() < 0.1,
            "Flat segment handling failed: {}",
            mid_val
        );
    }

    #[test]
    fn pchip_log_space_positivity() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1e-10, 1e-5, 1.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Log);

        // All interpolated values should be positive
        for i in 0..100 {
            let xq = 2.0 * i as f64 / 100.0;
            let yq = pchip.eval(xq, false);
            assert!(
                yq > 0.0,
                "Log-space result not positive: {} at x={}",
                yq,
                xq
            );
        }
    }

    #[test]
    fn pchip_reasonable_accuracy() {
        // Test with smooth function - PCHIP should provide reasonable accuracy
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // quadratic

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        // Test at intermediate point
        let test_x = 1.5;
        let expected = test_x * test_x;
        let result = pchip.eval(test_x, false);

        // PCHIP should be reasonably accurate for smooth functions
        let rel_error = (result - expected).abs() / expected;
        assert!(
            rel_error < 0.1,
            "Accuracy test failed: relative error {} too large",
            rel_error
        );
    }

    #[test]
    fn pchip_minimum_points() {
        let x = vec![0.0, 1.0];
        let y = vec![1.0, 3.0];

        let pchip = Pchip::new(&x, &y, InterpolationSpace::Linear);

        // Should work with minimum 2 points
        let result = pchip.eval(0.5, false);
        assert!(
            result.is_finite() && result > 0.0,
            "Minimum points test failed: {}",
            result
        );
    }

    #[test]
    #[should_panic(expected = "PCHIP requires at least two points")]
    fn pchip_panics_single_point() {
        let x = vec![1.0];
        let y = vec![2.0];
        Pchip::new(&x, &y, InterpolationSpace::Linear);
    }

    #[test]
    #[should_panic(expected = "x and y must have same length")]
    fn pchip_panics_mismatched_lengths() {
        let x = vec![0.0, 1.0];
        let y = vec![1.0];
        Pchip::new(&x, &y, InterpolationSpace::Linear);
    }

    #[test]
    #[should_panic(expected = "x must be strictly increasing")]
    fn pchip_panics_non_increasing() {
        let x = vec![0.0, 1.0, 0.5];
        let y = vec![1.0, 2.0, 1.5];
        Pchip::new(&x, &y, InterpolationSpace::Linear);
    }

    #[test]
    #[should_panic(expected = "Log-space PCHIP requires strictly positive data")]
    fn pchip_panics_negative_log() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, -1.0, 2.0];
        Pchip::new(&x, &y, InterpolationSpace::Log);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lagrange_linear_interpolation() {
        // Test linear function: f(x) = 2x + 1
        let x_vals = DVector::from_vec(vec![0.0, 1.0]);
        let y_vals = DVector::from_vec(vec![1.0, 3.0]);

        // Test at known points
        assert_relative_eq!(
            lagrange_interpolate(0.0, &x_vals, &y_vals),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(1.0, &x_vals, &y_vals),
            3.0,
            epsilon = 1e-10
        );

        // Test interpolation
        assert_relative_eq!(
            lagrange_interpolate(0.5, &x_vals, &y_vals),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(2.0, &x_vals, &y_vals),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lagrange_quadratic_interpolation() {
        // Test quadratic function: f(x) = x^2 - 2x + 1 = (x-1)^2
        let x_vals = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let y_vals = DVector::from_vec(vec![1.0, 0.0, 1.0]);

        // Test at known points
        assert_relative_eq!(
            lagrange_interpolate(0.0, &x_vals, &y_vals),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(1.0, &x_vals, &y_vals),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(2.0, &x_vals, &y_vals),
            1.0,
            epsilon = 1e-10
        );

        // Test interpolation and extrapolation
        assert_relative_eq!(
            lagrange_interpolate(0.5, &x_vals, &y_vals),
            0.25,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(1.5, &x_vals, &y_vals),
            0.25,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(3.0, &x_vals, &y_vals),
            4.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lagrange_cubic_interpolation() {
        // Test cubic function: f(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let x_vals = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y_vals = DVector::from_vec(vec![-6.0, 0.0, 0.0, 0.0]);

        // Test at known points
        assert_relative_eq!(
            lagrange_interpolate(0.0, &x_vals, &y_vals),
            -6.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(1.0, &x_vals, &y_vals),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(2.0, &x_vals, &y_vals),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(3.0, &x_vals, &y_vals),
            0.0,
            epsilon = 1e-10
        );

        // Test interpolation
        assert_relative_eq!(
            lagrange_interpolate(1.5, &x_vals, &y_vals),
            0.375,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_newton_divided_differences_linear() {
        // Test linear function: f(x) = 2x + 1
        let x_vals = DVector::from_vec(vec![0.0, 1.0]);
        let y_vals = DVector::from_vec(vec![1.0, 3.0]);

        let coef = newton_divided_differences(&x_vals, &y_vals);

        // For linear function: f[x0] = 1, f[x0,x1] = 2
        assert_relative_eq!(coef[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coef[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_newton_divided_differences_quadratic() {
        // Test quadratic function: f(x) = x^2
        let x_vals = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let y_vals = DVector::from_vec(vec![0.0, 1.0, 4.0]);

        let coef = newton_divided_differences(&x_vals, &y_vals);

        // For f(x) = x^2: f[x0] = 0, f[x0,x1] = 1, f[x0,x1,x2] = 1
        assert_relative_eq!(coef[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(coef[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coef[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_newton_interpolate_linear() {
        // Test linear function: f(x) = 2x + 1
        let x_vals = DVector::from_vec(vec![0.0, 1.0]);
        let y_vals = DVector::from_vec(vec![1.0, 3.0]);
        let coef = newton_divided_differences(&x_vals, &y_vals);

        // Test at known points
        assert_relative_eq!(
            newton_interpolate(0.0, &x_vals, &coef),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(1.0, &x_vals, &coef),
            3.0,
            epsilon = 1e-10
        );

        // Test interpolation and extrapolation
        assert_relative_eq!(
            newton_interpolate(0.5, &x_vals, &coef),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(2.0, &x_vals, &coef),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_newton_interpolate_quadratic() {
        // Test quadratic function: f(x) = x^2 - 2x + 1
        let x_vals = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let y_vals = DVector::from_vec(vec![1.0, 0.0, 1.0]);
        let coef = newton_divided_differences(&x_vals, &y_vals);

        // Test at known points
        assert_relative_eq!(
            newton_interpolate(0.0, &x_vals, &coef),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(1.0, &x_vals, &coef),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(2.0, &x_vals, &coef),
            1.0,
            epsilon = 1e-10
        );

        // Test interpolation
        assert_relative_eq!(
            newton_interpolate(0.5, &x_vals, &coef),
            0.25,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(1.5, &x_vals, &coef),
            0.25,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_newton_vs_lagrange_equivalence() {
        // Both methods should give identical results for the same data
        let x_vals = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y_vals = DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0]); // f(x) = (x+1)^2

        let coef = newton_divided_differences(&x_vals, &y_vals);

        let test_points = vec![-1.0, 0.5, 1.5, 2.5, 4.0];

        for &x in &test_points {
            let lagrange_result = lagrange_interpolate(x, &x_vals, &y_vals);
            let newton_result = newton_interpolate(x, &x_vals, &coef);

            assert_relative_eq!(lagrange_result, newton_result, epsilon = 1e-10,)
        }
    }

    #[test]
    fn test_interpolation_with_uneven_spacing() {
        // Test with unevenly spaced points
        let x_vals = DVector::from_vec(vec![-2.0, -0.5, 1.0, 3.5]);
        let y_vals = DVector::from_vec(vec![4.0, 0.25, 1.0, 12.25]); // f(x) = x^2

        let coef = newton_divided_differences(&x_vals, &y_vals);

        // Test at original points
        for i in 0..x_vals.len() {
            assert_relative_eq!(
                lagrange_interpolate(x_vals[i], &x_vals, &y_vals),
                y_vals[i],
                epsilon = 1e-10
            );
            assert_relative_eq!(
                newton_interpolate(x_vals[i], &x_vals, &coef),
                y_vals[i],
                epsilon = 1e-10
            );
        }

        // Test interpolation at intermediate points
        let test_x = 0.0;
        let expected_y = 0.0; // Since f(x) = x^2

        assert_relative_eq!(
            lagrange_interpolate(test_x, &x_vals, &y_vals),
            expected_y,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(test_x, &x_vals, &coef),
            expected_y,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_single_point_interpolation() {
        // Test with single point (constant function)
        let x_vals = DVector::from_vec(vec![1.0]);
        let y_vals = DVector::from_vec(vec![5.0]);

        let coef = newton_divided_differences(&x_vals, &y_vals);

        // Should return constant value for any x
        assert_relative_eq!(
            lagrange_interpolate(0.0, &x_vals, &y_vals),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(1.0, &x_vals, &y_vals),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            lagrange_interpolate(10.0, &x_vals, &y_vals),
            5.0,
            epsilon = 1e-10
        );

        assert_relative_eq!(
            newton_interpolate(0.0, &x_vals, &coef),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(1.0, &x_vals, &coef),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            newton_interpolate(10.0, &x_vals, &coef),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_interpolation_with_sine_function() {
        // Test with sine function values
        use std::f64::consts::PI;

        let x_vals = DVector::from_vec(vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0]);
        let y_vals = DVector::from_vec(vec![
            0.0,                // sin(0)
            0.5,                // sin(π/6)
            2_f64.sqrt() / 2.0, // sin(π/4)
            3_f64.sqrt() / 2.0, // sin(π/3)
            1.0,                // sin(π/2)
        ]);

        let coef = newton_divided_differences(&x_vals, &y_vals);

        // Test at original points
        for i in 0..x_vals.len() {
            assert_relative_eq!(
                lagrange_interpolate(x_vals[i], &x_vals, &y_vals),
                y_vals[i],
                epsilon = 1e-10
            );
            assert_relative_eq!(
                newton_interpolate(x_vals[i], &x_vals, &coef),
                y_vals[i],
                epsilon = 1e-10
            )
        }
    }
}

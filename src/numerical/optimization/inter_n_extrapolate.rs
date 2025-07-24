use nalgebra::DVector;
#[allow(dead_code)]
fn lagrange_interpolate(x: f64, x_vals: &DVector<f64>, y_vals: &DVector<f64>) -> f64 {
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
#[allow(dead_code)]
fn newton_divided_differences(x_vals: &DVector<f64>, y_vals: &DVector<f64>) -> DVector<f64> {
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
#[allow(dead_code)]
fn newton_interpolate(x: f64, x_vals: &DVector<f64>, coef: &DVector<f64>) -> f64 {
    let n = coef.len();
    let mut result = coef[n - 1];

    for i in (0..n - 1).rev() {
        result = result * (x - x_vals[i]) + coef[i];
    }

    result
}

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

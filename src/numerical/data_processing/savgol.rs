//! # Savitzky-Golay Filter Implementation
//!
//! This module implements the Savitzky-Golay filter, a digital filter that can be applied to
//! a set of digital data points for the purpose of smoothing the data and computing derivatives.
//!
//! ## Mathematical Background
//!
//! The Savitzky-Golay filter is based on local polynomial regression. For each point in the data,
//! a polynomial of degree `poly_order` is fitted to a window of `window_length` points centered
//! on that point. The filtered value is then the value of this polynomial at the center point.
//!
//! ### Mathematical Formulation
//!
//! Given a data series y[i], the filter computes:
//! ```text
//! y_filtered[i] = Σ(j=-n to n) c[j] * y[i+j]
//! ```
//! where:
//! - `c[j]` are the Savitzky-Golay coefficients
//! - `n = (window_length - 1) / 2`
//! - The coefficients are computed by solving a least-squares problem
//!
//! ### Coefficient Calculation
//!
//! The coefficients are found by solving the linear system:
//! ```text
//! A * c = y
//! ```
//! where:
//! - `A` is a Vandermonde matrix with A[i,j] = x[j]^i
//! - `x` are the relative positions within the window
//! - `y` is a unit vector with 1 at position `deriv` (for derivative calculation)
//!
//! ## Algorithmic Implementation
//!
//! 1. **Coefficient Computation**: Solve least-squares problem to find filter coefficients
//! 2. **Convolution**: Apply coefficients to data via convolution
//! 3. **Edge Handling**: Use polynomial fitting for boundary points where full window unavailable
//!
//! ## Applications in Chemical Kinetics
//!
//! - Smoothing noisy experimental data (TGA, DSC)
//! - Computing derivatives for kinetic analysis
//! - Preprocessing data for reaction rate calculations

use lstsq::lstsq;
use nalgebra::{DMatrix, DVector, SVD};

/// Computes factorial of a number recursively.
///
/// # Mathematical Background
/// n! = n × (n-1) × ... × 2 × 1, with 0! = 1 by definition
///
/// # Arguments
/// * `num` - Non-negative integer to compute factorial for
///
/// # Returns
/// Factorial of the input number
fn factorial(num: usize) -> usize {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

/// Computes Savitzky-Golay filter coefficients.
///
/// # Mathematical Background
/// Solves the least-squares problem A*c = y where:
/// - A is a Vandermonde matrix with A[i,j] = x[j]^i
/// - x are relative positions: [-n, -n+1, ..., 0, ..., n-1, n]
/// - y is unit vector with y[deriv] = deriv! (factorial of derivative order)
///
/// # Algorithmic Steps
/// 1. Create position vector centered at window middle
/// 2. Build Vandermonde matrix for polynomial basis
/// 3. Set up right-hand side for derivative calculation
/// 4. Solve linear system using least-squares
///
/// # Arguments
/// * `window` - Window length (must be odd and > poly_order)
/// * `poly_order` - Polynomial degree for local fitting
/// * `deriv` - Derivative order (0 for smoothing, 1 for first derivative, etc.)
///
/// # Returns
/// Vector of filter coefficients or error message
fn SG_coeffs(window: usize, poly_order: usize, deriv: usize) -> Result<DVector<f64>, String> {
    if poly_order >= window {
        return Err("poly_order must be less than window.".to_string());
    }

    let (halflen, rem) = (window / 2, window % 2);

    let pos = match rem {
        0 => halflen as f64 - 0.5,
        _ => halflen as f64,
    };

    if deriv > poly_order {
        return Ok(DVector::from_element(window, 0.0));
    }

    let x = DVector::from_fn(window, |i, _| pos - i as f64);
    let order = DVector::from_fn(poly_order + 1, |i, _| i);
    let mat_a = DMatrix::from_fn(poly_order + 1, window, |i, j| x[j].powf(order[i] as f64));

    let mut y = DVector::from_element(poly_order + 1, 0.0);
    y[deriv] = factorial(deriv) as f64;

    let epsilon = 1e-14;
    let results = lstsq(&mat_a, &y, epsilon)?;
    let solution = results.solution;

    return Ok(solution);
}

/// Computes derivative coefficients of a polynomial.
///
/// # Mathematical Background
/// For polynomial P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
/// The derivative P'(x) = a₁ + 2a₂x + 3a₃x² + ... + naₙx^(n-1)
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients [a₀, a₁, a₂, ...]
///
/// # Returns
/// Derivative coefficients [a₁, 2a₂, 3a₃, ...]
fn poly_derivative(coeffs: &[f64]) -> Vec<f64> {
    coeffs[1..]
        .iter()
        .enumerate()
        .map(|(i, c)| c * (i + 1) as f64)
        .collect()
}

/// Evaluates polynomial at given points using Horner's method.
///
/// # Mathematical Background
/// Evaluates P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ efficiently
///
/// # Arguments
/// * `poly` - Polynomial coefficients [a₀, a₁, a₂, ...]
/// * `values` - Points at which to evaluate polynomial
///
/// # Returns
/// Vector of polynomial values at input points
fn polyval(poly: &[f64], values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|v| {
            poly.iter()
                .enumerate()
                .fold(0.0, |y, (i, c)| y + c * v.powf(i as f64))
        })
        .collect()
}

/// Fits polynomial to edge region and interpolates boundary points.
///
/// # Algorithmic Purpose
/// Handles boundary conditions where full Savitzky-Golay window cannot be applied.
/// Uses polynomial fitting to extrapolate smoothed values at data edges.
///
/// # Arguments
/// * `x` - Original data vector
/// * `window_start` - Start index of fitting window
/// * `window_stop` - End index of fitting window
/// * `interp_start` - Start index of interpolation region
/// * `interp_stop` - End index of interpolation region
/// * `poly_order` - Polynomial degree for fitting
/// * `deriv` - Derivative order
/// * `y` - Mutable reference to output vector
fn fit_edge(
    x: &DVector<f64>,
    window_start: usize,
    window_stop: usize,
    interp_start: usize,
    interp_stop: usize,
    poly_order: usize,
    deriv: usize,
    y: &mut Vec<f64>,
) -> Result<(), String> {
    let x_edge: Vec<f64> = x.as_slice()[window_start..window_stop].to_vec();
    let y_edge: Vec<f64> = (0..window_stop - window_start).map(|i| i as f64).collect();
    let mut poly_coeffs = polyfit(&y_edge, &x_edge, poly_order)?;

    let mut deriv = deriv;
    while deriv > 0 {
        poly_coeffs = poly_derivative(&poly_coeffs);
        deriv -= 1;
    }

    let i: Vec<f64> = (0..interp_stop - interp_start)
        .map(|i| (interp_start - window_start + i) as f64)
        .collect();
    let values = polyval(&poly_coeffs, &i);
    y.splice(interp_start..interp_stop, values);
    Ok(())
}

/// Applies polynomial fitting to both edges of the data.
///
/// # Algorithmic Implementation
/// Calls fit_edge for both left and right boundaries to handle edge effects
/// in Savitzky-Golay filtering.
///
/// # Arguments
/// * `x` - Original data vector
/// * `window` - Window length
/// * `poly_order` - Polynomial degree
/// * `deriv` - Derivative order
/// * `y` - Mutable reference to output vector
fn fit_edges_polyfit(
    x: &DVector<f64>,
    window: usize,
    poly_order: usize,
    deriv: usize,
    y: &mut Vec<f64>,
) -> Result<(), String> {
    let halflen = window / 2;
    fit_edge(x, 0, window, 0, halflen, poly_order, deriv, y)?;
    let n = x.len();
    fit_edge(x, n - window, n, n - halflen, n, poly_order, deriv, y)?;

    Ok(())
}

/// Input parameters for Savitzky-Golay filter.
///
/// # Fields
/// * `data` - Input data slice to be filtered
/// * `window` - Window length (must be odd)
/// * `poly_order` - Polynomial order for local fitting
/// * `derivative` - Derivative order (0 for smoothing)
#[derive(Clone, Debug)]
pub struct SGInput<'a> {
    pub data: &'a [f64],
    pub window: usize,
    pub poly_order: usize,
    pub derivative: usize,
}

/// Applies Savitzky-Golay filter to input data.
///
/// # Mathematical Process
/// 1. Computes filter coefficients via least-squares
/// 2. Applies convolution with coefficients
/// 3. Handles boundary effects with polynomial fitting
///
/// # Algorithmic Steps
/// 1. Validate input parameters
/// 2. Compute Savitzky-Golay coefficients
/// 3. Perform convolution operation
/// 4. Trim convolution artifacts
/// 5. Apply edge correction using polynomial fitting
///
/// # Arguments
/// * `input` - SGInput struct containing data and parameters
///
/// # Returns
/// Filtered data vector or error message
///
/// # Example
/// ```rust
/// use savgol::{SGInput, SG_filter};
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let input = SGInput { data: &data, window: 3, poly_order: 1, derivative: 0 };
/// let filtered = SG_filter(&input).unwrap();
/// ```
pub fn SG_filter(input: &SGInput) -> Result<Vec<f64>, String> {
    if input.window > input.data.len() {
        return Err("window must be less than or equal to the size of the input data".to_string());
    }

    if input.window % 2 == 0 {
        // TODO: figure out how scipy implementation handles the convolution
        // in this case
        return Err("window must be odd".to_string());
    }

    let coeffs = SG_coeffs(input.window, input.poly_order, input.derivative)?;

    let x = DVector::from_vec(input.data.to_vec());

    let y = x.convolve_full(coeffs);

    // trim extra length gained during convolution to mimic scipy convolve1d
    // with mode="constant"
    let padding = y.len() - x.len();
    let padding = padding / 2;
    let y = y.as_slice();
    let mut y = y[padding..y.len().saturating_sub(padding)].to_vec();

    fit_edges_polyfit(&x, input.window, input.poly_order, input.derivative, &mut y)?;
    return Ok(y);
}

/// Fits polynomial to data using least-squares method.
///
/// # Mathematical Background
/// Solves the overdetermined system A*c = b where:
/// - A is Vandermonde matrix: A[i,j] = x[i]^j
/// - c are polynomial coefficients to find
/// - b are the y-values
///
/// Uses SVD decomposition for numerical stability.
///
/// # Arguments
/// * `x_values` - Independent variable values
/// * `y_values` - Dependent variable values
/// * `polynomial_degree` - Degree of polynomial to fit
///
/// # Returns
/// Polynomial coefficients [a₀, a₁, a₂, ...] or error
pub fn polyfit(
    x_values: &[f64],
    y_values: &[f64],
    polynomial_degree: usize,
) -> Result<Vec<f64>, &'static str> {
    let number_of_columns = polynomial_degree + 1;
    let number_of_rows = x_values.len();
    let mut a = DMatrix::zeros(number_of_rows, number_of_columns);

    for (row, &x) in x_values.iter().enumerate() {
        // First column is always 1
        a[(row, 0)] = 1.0f64;

        for col in 1..number_of_columns {
            a[(row, col)] = x.powf(col as f64);
        }
    }

    let b = DVector::from_row_slice(y_values);

    let decomp = SVD::new(a, true, true);

    match decomp.solve(&b, 1e-18f64) {
        Ok(mat) => Ok(mat.data.into()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coeffs_basic() {
        // Test basic coefficient calculation for window=5, poly_order=2, deriv=0
        let c = SG_coeffs(5, 2, 0).unwrap();
        let expected = DVector::from_vec(vec![
            -0.08571428571428572,
            0.34285714285714286,
            0.4857142857142857,
            0.34285714285714286,
            -0.08571428571428572,
        ]);
        assert_eq!(c.len(), expected.len());
        for (a, e) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_coeffs_derivative() {
        // Test first derivative coefficients
        let c = SG_coeffs(5, 2, 1).unwrap();
        let expected = DVector::from_vec(vec![0.2, 0.1, 0.0, -0.1, -0.2]);
        assert_eq!(c.len(), expected.len());
        for (a, e) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_coeffs_symmetry() {
        // Coefficients should be symmetric for even derivative orders
        let c = SG_coeffs(7, 3, 0).unwrap();
        let n = c.len();
        for i in 0..n / 2 {
            assert_relative_eq!(c[i], c[n - 1 - i], max_relative = 1e-12);
        }
    }

    #[test]
    fn test_coeffs_error_cases() {
        // Test error conditions
        assert!(SG_coeffs(3, 3, 0).is_err()); // poly_order >= window
        assert!(SG_coeffs(5, 2, 3).is_ok()); // deriv > poly_order should return zeros

        let c = SG_coeffs(5, 2, 3).unwrap();
        for coeff in c.iter() {
            assert_relative_eq!(*coeff, 0.0, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_filter_on_known_series() {
        // Test filtering on known noisy data
        let input = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
        let y = SG_filter(&SGInput {
            data: &input,
            window: 5,
            poly_order: 2,
            derivative: 0,
        })
        .unwrap();
        let expected = vec![
            1.65714285714,
            3.02857143,
            3.54285714,
            2.85714286,
            0.65714286,
            0.17142857,
            1.0,
            4.05,
            7.97142857,
        ];
        assert_eq!(y.len(), expected.len());
        for (a, e) in y.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-1);
        }
    }

    #[test]
    fn test_filter_derivative() {
        // Test first derivative calculation
        let input = [1.0, 4.0, 9.0, 16.0, 25.0]; // x^2 values
        let y = SG_filter(&SGInput {
            data: &input,
            window: 5,
            poly_order: 2,
            derivative: 1,
        })
        .unwrap();
        // First derivative of x^2 is 2x, so expect approximately [2, 4, 6, 8, 10]
        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(y.len(), expected.len());
        for (a, e) in y.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 0.1);
        }
    }

    #[test]
    fn test_filter_constant_data() {
        // Constant data should remain constant after smoothing
        let input = [5.0; 10];
        let y = SG_filter(&SGInput {
            data: &input,
            window: 5,
            poly_order: 2,
            derivative: 0,
        })
        .unwrap();
        for val in y.iter() {
            assert_relative_eq!(*val, 5.0, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_filter_error_cases() {
        let input = [1.0, 2.0, 3.0];

        // Window too large
        assert!(
            SG_filter(&SGInput {
                data: &input,
                window: 5,
                poly_order: 2,
                derivative: 0
            })
            .is_err()
        );

        // Even window
        assert!(
            SG_filter(&SGInput {
                data: &input,
                window: 4,
                poly_order: 2,
                derivative: 0
            })
            .is_err()
        );
    }

    #[test]
    fn test_filter_linear_function() {
        // y = 3x - 2, should largely pass-through within interior; edges are fit by polyfit
        let data: Vec<f64> = (0..100).map(|i| (3 * i - 2) as f64).collect();
        let y = SG_filter(&SGInput {
            data: &data,
            window: 51,
            poly_order: 5,
            derivative: 0,
        })
        .unwrap();
        assert_eq!(y.len(), data.len());
        // Check some interior points equal the true line exactly for poly order >= 1
        for x in 30..70 {
            let expected = 3.0 * x as f64 - 2.0;
            assert_relative_eq!(y[x], expected, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_polyfit_basic() {
        // Test polynomial fitting on known quadratic
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2
        let coeffs = polyfit(&x, &y, 2).unwrap();

        // Should get coefficients [0, 0, 1] for x^2
        assert_relative_eq!(coeffs[0].abs(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[1].abs(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[2].abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_poly_derivative() {
        // Test derivative of x^3 + 2x^2 + 3x + 4
        let coeffs = vec![4.0, 3.0, 2.0, 1.0]; // [constant, x, x^2, x^3]
        let deriv_coeffs = poly_derivative(&coeffs);
        let expected = vec![3.0, 4.0, 3.0]; // [3, 4x, 3x^2]

        assert_eq!(deriv_coeffs.len(), expected.len());
        for (a, e) in deriv_coeffs.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_polyval() {
        // Test polynomial evaluation
        let coeffs = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        let x_vals = vec![0.0, 1.0, 2.0];
        let y_vals = polyval(&coeffs, &x_vals);
        let expected = vec![1.0, 6.0, 17.0]; // [1, 1+2+3, 1+4+12]

        assert_eq!(y_vals.len(), expected.len());
        for (a, e) in y_vals.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-12);
        }
    }
}

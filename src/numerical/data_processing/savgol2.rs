//! # Dynamic Savitzky-Golay Filter Implementation
//!
//! This module provides an alternative implementation of the Savitzky-Golay filter with
//! dynamic memory allocation and iterator-based input processing.
//!
//! ## Mathematical Background
//!
//! The Savitzky-Golay filter performs local polynomial regression on a sliding window.
//! For each data point, it fits a polynomial of specified degree to surrounding points
//! and uses this polynomial to compute the smoothed value or derivative.
//!
//! ### Core Mathematical Principle
//!
//! Given a window of data points y[i-n], ..., y[i], ..., y[i+n], the filter:
//! 1. Fits polynomial P(x) = a₀ + a₁x + a₂x² + ... + aₖxᵏ
//! 2. Evaluates P(0) for smoothing or P'(0), P''(0), etc. for derivatives
//!
//! ### Coefficient Matrix Solution
//!
//! The filter coefficients solve: A·c = e_d where:
//! - A[i,j] = xⱼⁱ (Vandermonde matrix)
//! - x = [-n, -n+1, ..., 0, ..., n-1, n] (relative positions)
//! - e_d = unit vector with 1 at position d (derivative order)
//! - c = filter coefficients
//!
//! ### Edge Handling Strategy
//!
//! This implementation uses "nearest edge value" padding:
//! - Extends data at boundaries by repeating edge values
//! - Maintains original data length in output
//! - Provides consistent filtering across entire signal
//!
//! ## Algorithmic Features
//!
//! - **Iterator Input**: Accepts any iterator over f64-convertible types
//! - **Dynamic Sizing**: Handles variable-length input streams
//! - **Memory Efficient**: Uses convolution with pre-computed coefficients
//! - **Robust Numerics**: Employs least-squares solver for coefficient computation
//!
//! ## Applications in Experimental Kinetics
//!
//! - Real-time data processing from analytical instruments
//! - Streaming analysis of TGA/DSC measurements
//! - Online derivative calculation for kinetic parameter estimation
//! - Noise reduction in continuous monitoring systems

use nalgebra::{DMatrix, DVector};
use std::borrow::Borrow;
/// Applies Savitzky-Golay filter to dynamic iterator input.
///
/// # Mathematical Process
/// 1. Computes filter coefficients via least-squares solution
/// 2. Pads input data with nearest edge values
/// 3. Applies convolution with computed coefficients
///
/// # Algorithmic Implementation
/// - Collects iterator into vector with capacity estimation
/// - Pads boundaries with (window_length/2) repeated edge values
/// - Performs sliding window convolution
/// - Returns filtered data maintaining original length
///
/// # Arguments
/// * `y` - Iterator over data points (any type convertible to f64)
/// * `window_length` - Size of sliding window (must be odd)
/// * `polyorder` - Polynomial degree for local fitting
/// * `deriv` - Derivative order (None defaults to 0 for smoothing)
/// * `delta` - Spacing between data points (None defaults to 1.0)
///
/// # Returns
/// Vector of filtered values
///
/// # Panics
/// - If window_length is even
/// - If window_length < polyorder + 2
///
/// # Example
/// ```rust
/// use SG2::SG_filter_dyn;
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let filtered = SG_filter_dyn(data.iter(), 3, 1, None, None);
/// ```
pub fn SG_filter_dyn<YI>(
    mut y: YI,
    window_length: usize,
    polyorder: usize,
    deriv: Option<usize>,
    delta: Option<f64>,
) -> Vec<f64>
where
    YI: Iterator,
    YI::Item: Borrow<f64>,
{
    if window_length.is_multiple_of(2) {
        panic!("window_length must be odd")
    }

    if window_length < polyorder + 2 {
        panic!("window_length is too small for the polynomials order")
    }

    let mut fir = SG_coeffs_dyn(window_length, polyorder, deriv, delta)
        .into_iter()
        .collect::<Vec<_>>();

    fir.reverse();

    // Pad with nearest edge value
    let size_hint = y.size_hint();
    let mut data: Vec<f64> = Vec::with_capacity(size_hint.1.unwrap_or(size_hint.0));
    let Some(nearest_borrowed) = y.next() else {
        return vec![];
    };
    let nearest = *nearest_borrowed.borrow();
    data.extend((0..(window_length / 2 + 1)).map(|_| nearest));
    data.extend(y.map(|yi| *yi.borrow()));
    let nearest = *data.last().unwrap();
    data.extend((0..window_length / 2).map(|_| nearest));

    // Convolve the data with the FIR coefficients
    let rslt = data
        .windows(window_length)
        .map(|w| w.iter().zip(fir.iter()).map(|(a, b)| *a * *b).sum::<f64>())
        .collect();

    rslt
}

/// Computes Savitzky-Golay filter coefficients for dynamic implementation.
///
/// # Mathematical Formulation
/// Solves the linear system A·c = y where:
/// - A is Vandermonde matrix: A[i,j] = pos[j]^i for i=0..polyorder, j=0..window_length
/// - pos are relative positions within window: [-n, -n+1, ..., 0, ..., n-1, n]
/// - y is unit vector with factorial(deriv)/delta^deriv at position deriv
///
/// # Algorithmic Steps
/// 1. Calculate relative positions within window
/// 2. Handle even/odd window length positioning
/// 3. Construct Vandermonde matrix for polynomial basis
/// 4. Set up right-hand side for derivative calculation
/// 5. Solve least-squares system for coefficients
///
/// # Arguments
/// * `window_length` - Size of filter window
/// * `polyorder` - Polynomial degree for fitting
/// * `deriv` - Derivative order (None defaults to 0)
/// * `delta` - Data point spacing (None defaults to 1.0)
///
/// # Returns
/// Vector of filter coefficients
///
/// # Panics
/// - If polyorder >= window_length
///
/// # Mathematical Note
/// For derivative order d, the coefficient at position deriv in the solution vector
/// is scaled by d!/Δx^d to account for finite difference scaling.
pub fn SG_coeffs_dyn(
    window_length: usize,
    polyorder: usize,
    deriv: Option<usize>,
    delta: Option<f64>,
) -> Vec<f64> {
    if polyorder >= window_length {
        panic!("polyorder must be less than window_length")
    }

    let half_window = (window_length / 2) as f64;
    let rem = window_length % 2;

    let pos: Vec<f64> = if rem == 0 {
        let f = 0.5f64;
        (0..window_length)
            .map(|i| (half_window - i as f64 - f))
            .collect::<Vec<_>>()
    } else {
        (0..window_length)
            .map(|i| half_window - i as f64)
            .collect::<Vec<_>>()
    };

    //handle the case of default args
    let der = deriv.unwrap_or(0);
    let del = delta.unwrap_or(1.0);

    if der > polyorder {
        return vec![0.0; window_length];
    }

    // Columns are 2m+1 integer positions centered on 0
    // Rows are powers of positions from 0 to polyorder
    // Setting up a Vandermonde matrix for solving A * coeffs = y
    #[allow(non_snake_case)]
    let A = DMatrix::<f64>::from_fn(polyorder + 1, window_length, |i, j| pos[j].powi(i as i32));
    let mut y = DVector::<f64>::from_fn(polyorder + 1, |i, _| if i == der { 1.0 } else { 0.0 });

    y[der] = (factorial(der) as f64) / del.powi(der as i32);

    // Solve the system for the Savitsky-Golay FIR coefficients
    let solve = lstsq::lstsq(&A, &y, 1e-9f64).unwrap();
    solve.solution.data.into()
}

/// Computes factorial using iterative product.
///
/// # Mathematical Definition
/// n! = 1 × 2 × 3 × ... × n, with 0! = 1
///
/// # Implementation
/// Uses iterator product for efficiency and clarity
///
/// # Arguments
/// * `n` - Non-negative integer
///
/// # Returns
/// Factorial of n
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    #[test]
    pub fn can_filter() {
        let v = SG_filter_dyn((0..100).map(|i| i as f64), 11, 2, None, None);
        println!("v = {:?}", v);

        let v = SG_filter_dyn((0..0).map(|i| i as f64), 11, 2, None, None);
        println!("v = {:?}", v);

        let actual_coeff = SG_coeffs_dyn(5, 2, Some(0), Some(1.0));
        println!("coeffs = {:?}", actual_coeff);
        let input = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
        let actual = SG_filter_dyn(input.iter(), 5, 2, None, None);
        println!("actual = {:?}", actual);
        let expected = [
            1.74285714, 3.02857143, 3.54285714, 2.85714286, 0.65714286, 0.17142857, 1.0, 4.6,
            7.97142857,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-5);
        }

        let actual_coeff = SG_coeffs_dyn(5, 2, Some(1), None);
        println!("coeffs = {:?}", actual_coeff);
        let input = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
        let actual = SG_filter_dyn(input.iter(), 5, 2, Some(1), None);
        println!("actual = {:?}", actual);
        let expected = [0.6, 0.3, -0.2, -0.8, -1.0, 0.4, 2.0, 2.6, 2.1];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-5);
        }

        let input = (0..100).map(|i| (3 * i - 2) as f64); //y = 3x - 2
        let actual = SG_filter_dyn(input, 51, 5, None, None);
        let expected = [
            2.45650177,
            4.06008889,
            5.86936071,
            7.87987343,
            10.08430349,
            12.47257185,
            15.03201779,
            17.74762253,
            20.6022825,
            23.57713222,
            26.65191695,
            29.805415,
            33.01590968,
            36.26171099,
            39.52172696,
            42.77608466,
            46.00680095,
            49.19850285,
            52.33919758,
            55.4210924,
            58.44146398,
            61.40357756,
            64.31765571,
            67.20189688,
            70.08354354,
            73.,
            76.,
            79.,
            82.,
            85.,
            88.,
            91.,
            94.,
            97.,
            100.,
            103.,
            106.,
            109.,
            112.,
            115.,
            118.,
            121.,
            124.,
            127.,
            130.,
            133.,
            136.,
            139.,
            142.,
            145.,
            148.,
            151.,
            154.,
            157.,
            160.,
            163.,
            166.,
            169.,
            172.,
            175.,
            178.,
            181.,
            184.,
            187.,
            190.,
            193.,
            196.,
            199.,
            202.,
            205.,
            208.,
            211.,
            214.,
            217.,
            220.,
            222.91645647,
            225.79810312,
            228.6823443,
            231.59642245,
            234.55853602,
            237.5789076,
            240.66080242,
            243.80149716,
            246.99319905,
            250.22391534,
            253.47827305,
            256.73828901,
            259.98409032,
            263.194585,
            266.34808305,
            269.42286778,
            272.3977175,
            275.25237747,
            277.96798222,
            280.52742816,
            282.91569651,
            285.12012658,
            287.13063929,
            288.93991111,
            290.54349823,
        ];

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-5);
        }
    }

    #[test]
    pub fn can_coeffs() {
        let actual = SG_coeffs_dyn(5, 2, None, None);
        let expected = [-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(5, 2, Some(0), Some(1.0));
        let expected = [-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(4, 2, None, None);
        let expected = [-0.0625, 0.5625, 0.5625, -0.0625];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(51, 5, None, None);
        let expected = [
            0.02784785,
            0.01160327,
            -0.00086484,
            -0.00994566,
            -0.01601181,
            -0.01941934,
            -0.02050775,
            -0.01959997,
            -0.01700239,
            -0.0130048,
            -0.00788047,
            -0.00188609,
            0.00473822,
            0.01176888,
            0.01899888,
            0.02623777,
            0.03331167,
            0.04006325,
            0.04635174,
            0.05205294,
            0.05705919,
            0.06127943,
            0.06463912,
            0.0670803,
            0.06856157,
            0.06905808,
            0.06856157,
            0.0670803,
            0.06463912,
            0.06127943,
            0.05705919,
            0.05205294,
            0.04635174,
            0.04006325,
            0.03331167,
            0.02623777,
            0.01899888,
            0.01176888,
            0.00473822,
            -0.00188609,
            -0.00788047,
            -0.0130048,
            -0.01700239,
            -0.01959997,
            -0.02050775,
            -0.01941934,
            -0.01601181,
            -0.00994566,
            -0.00086484,
            0.01160327,
            0.02784785,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 5e-6);
        }

        let actual = SG_coeffs_dyn(21, 8, None, None);
        let expected = [
            0.0125937,
            -0.04897551,
            0.03811252,
            0.04592441,
            -0.01514104,
            -0.06782274,
            -0.05517056,
            0.03024958,
            0.15283999,
            0.25791748,
            0.29894434,
            0.25791748,
            0.15283999,
            0.03024958,
            -0.05517056,
            -0.06782274,
            -0.01514104,
            0.04592441,
            0.03811252,
            -0.04897551,
            0.0125937,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-6);
        }

        //deriv tests
        let actual = SG_coeffs_dyn(5, 2, Some(1), Some(1.0));
        let expected = [2.0e-1, 1.0e-1, 2.07548111e-16, -1.0e-1, -2.0e-1];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(6, 3, Some(1), None);
        let expected = [
            -0.09093915,
            0.4130291,
            0.21560847,
            -0.21560847,
            -0.4130291,
            0.09093915,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(6, 3, Some(2), Some(1.0));
        let expected = [
            0.17857143,
            -0.03571429,
            -0.14285714,
            -0.14285714,
            -0.03571429,
            0.17857143,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 5e-6);
        }
    }
    #[test]
    fn test_basic_filtering() {
        // Test basic smoothing functionality
        let v = SG_filter_dyn((0..100).map(|i| i as f64), 11, 2, None, None);
        assert_eq!(v.len(), 100);

        // Test empty input
        let v = SG_filter_dyn((0..0).map(|i| i as f64), 11, 2, None, None);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_coefficient_calculation() {
        // Test basic coefficient calculation
        let actual_coeff = SG_coeffs_dyn(5, 2, Some(0), Some(1.0));
        let expected = [-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429];
        assert_eq!(actual_coeff.len(), expected.len());
        for (a, e) in actual_coeff.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_smoothing_known_data() {
        // Test smoothing on known noisy data
        let input = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
        let actual = SG_filter_dyn(input.iter(), 5, 2, None, None);
        let expected = [
            1.74285714, 3.02857143, 3.54285714, 2.85714286, 0.65714286, 0.17142857, 1.0, 4.6,
            7.97142857,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_first_derivative() {
        // Test first derivative calculation
        let input = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
        let actual = SG_filter_dyn(input.iter(), 5, 2, Some(1), None);
        let expected = [0.6, 0.3, -0.2, -0.8, -1.0, 0.4, 2.0, 2.6, 2.1];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_linear_function_preservation() {
        // Linear function should be preserved exactly with sufficient polynomial order
        let input = (0..100).map(|i| (3 * i - 2) as f64); // y = 3x - 2
        let actual = SG_filter_dyn(input, 51, 5, None, None);

        // Check interior points are preserved exactly
        for i in 25..75 {
            let expected = 3.0 * i as f64 - 2.0;
            assert_relative_eq!(actual[i], expected, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_coefficient_symmetry() {
        // Smoothing coefficients should be symmetric
        let coeffs = SG_coeffs_dyn(7, 3, Some(0), Some(1.0));
        let n = coeffs.len();
        for i in 0..n / 2 {
            assert_relative_eq!(coeffs[i], coeffs[n - 1 - i], max_relative = 1e-12);
        }
    }

    #[test]
    fn test_derivative_coefficients() {
        // Test various derivative coefficient calculations
        let actual = SG_coeffs_dyn(5, 2, Some(1), Some(1.0));
        let expected = [2.0e-1, 1.0e-1, 2.07548111e-16, -1.0e-1, -2.0e-1];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 1e-7);
        }

        let actual = SG_coeffs_dyn(6, 3, Some(2), Some(1.0));
        let expected = [
            0.17857143,
            -0.03571429,
            -0.14285714,
            -0.14285714,
            -0.03571429,
            0.17857143,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(a, e, max_relative = 5e-6);
        }
    }

    #[test]
    fn test_large_window_coefficients() {
        // Test coefficient calculation for larger windows
        let actual = SG_coeffs_dyn(51, 5, None, None);
        assert_eq!(actual.len(), 51);

        // Coefficients should sum to 1 for smoothing (deriv=0)
        let sum: f64 = actual.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-10);

        // Should be symmetric
        for i in 0..25 {
            assert_relative_eq!(actual[i], actual[50 - i], max_relative = 1e-12);
        }
    }

    #[test]
    fn test_constant_data_preservation() {
        // Constant data should remain unchanged after smoothing
        let input = vec![5.0; 20];
        let filtered = SG_filter_dyn(input.iter(), 7, 3, None, None);
        for val in filtered.iter() {
            assert_relative_eq!(*val, 5.0, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_quadratic_function() {
        // Test on quadratic function y = x^2
        let input: Vec<f64> = (0..21).map(|i| (i as f64).powi(2)).collect();
        let filtered = SG_filter_dyn(input.iter(), 5, 3, None, None);

        // Interior points should be preserved exactly
        for i in 2..19 {
            let expected = (i as f64).powi(2);
            assert_relative_eq!(filtered[i], expected, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_factorial_function() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(6), 720);
    }
}

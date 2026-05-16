use bspline::BSpline;

/// A 2D point struct that implements the Interpolate trait
/// This allows us to interpolate 2D points with the B-spline
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl bspline::Interpolate<f64> for Point2D {
    fn interpolate(&self, other: &Self, t: f64) -> Self {
        Point2D {
            x: self.x * (1.0 - t) + other.x * t,
            y: self.y * (1.0 - t) + other.y * t,
        }
    }
}

/// A 3D point struct that implements the Interpolate trait
/// This allows us to interpolate 3D points with the B-spline
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl bspline::Interpolate<f64> for Point3D {
    fn interpolate(&self, other: &Self, t: f64) -> Self {
        Point3D {
            x: self.x * (1.0 - t) + other.x * t,
            y: self.y * (1.0 - t) + other.y * t,
            z: self.z * (1.0 - t) + other.z * t,
        }
    }
}

/// Quality metrics for interpolation accuracy
#[derive(Debug, Clone, Copy)]
pub struct InterpolationQuality {
    /// Root Mean Square Error
    pub rmse: f64,
    /// Maximum absolute error
    pub max_error: f64,
    /// Mean absolute error
    pub mean_error: f64,
    /// Number of sample points
    pub sample_count: usize,
}

impl InterpolationQuality {
    /// Create a new InterpolationQuality instance
    pub fn new(rmse: f64, max_error: f64, mean_error: f64, sample_count: usize) -> Self {
        Self {
            rmse,
            max_error,
            mean_error,
            sample_count,
        }
    }
}

/// A B-spline interpolator that can interpolate between arbitrary data points
pub struct BSplineInterpolator<T>
where
    T: bspline::Interpolate<f64> + Copy,
{
    degree: usize,
    control_points: Vec<T>,
    knots: Vec<f64>,
    spline: BSpline<T, f64>,
}

impl<T> BSplineInterpolator<T>
where
    T: bspline::Interpolate<f64> + Copy,
{
    /// Create a new B-spline interpolator
    ///
    /// # Arguments
    /// * `degree` - The degree of the B-spline (e.g., 3 for cubic)
    /// * `control_points` - The control points to interpolate
    /// * `knots` - The knot vector (must have length = control_points.len() + degree + 1)
    ///
    /// # Returns
    /// A new BSplineInterpolator instance
    ///
    /// # Panics
    /// If the number of knots is incorrect for the given control points and degree
    pub fn new(degree: usize, control_points: Vec<T>, knots: Vec<f64>) -> Self {
        // Validate input
        if control_points.len() <= degree {
            panic!(
                "Not enough control points for a B-spline of degree {}. Need at least {} control points, but got {}",
                degree,
                degree + 1,
                control_points.len()
            );
        }

        if knots.len() != control_points.len() + degree + 1 {
            panic!(
                "Invalid number of knots. Got {}, expected {}",
                knots.len(),
                control_points.len() + degree + 1
            );
        }

        // Create the B-spline
        let spline = BSpline::new(degree, control_points.clone(), knots.clone());

        Self {
            degree,
            control_points,
            knots,
            spline,
        }
    }

    /// Create a new B-spline interpolator with automatically generated uniform knots
    ///
    /// # Arguments
    /// * `degree` - The degree of the B-spline (e.g., 3 for cubic)
    /// * `control_points` - The control points to interpolate
    ///
    /// # Returns
    /// A new BSplineInterpolator instance with uniform knots
    pub fn with_uniform_knots(degree: usize, control_points: Vec<T>) -> Self {
        let knots = Self::generate_uniform_knots(control_points.len(), degree);
        Self::new(degree, control_points, knots)
    }

    /// Generate a uniform knot vector for the B-spline
    ///
    /// # Arguments
    /// * `control_points_count` - Number of control points
    /// * `degree` - Degree of the B-spline
    ///
    /// # Returns
    /// A vector of knots
    fn generate_uniform_knots(control_points_count: usize, degree: usize) -> Vec<f64> {
        let total_knots = control_points_count + degree + 1;
        let mut knots = Vec::with_capacity(total_knots);

        // Create a clamped knot vector (common for B-splines)
        // First degree+1 knots are 0
        for _ in 0..=degree {
            knots.push(0.0);
        }

        // Middle knots are uniformly spaced
        // We need (total_knots - 2*(degree+1)) middle knots
        let middle_knots_count = total_knots - 2 * (degree + 1);
        for i in 1..=middle_knots_count {
            knots.push(i as f64);
        }

        // Last degree+1 knots are the same value
        let end_value = if middle_knots_count > 0 {
            middle_knots_count as f64 + 1.0
        } else {
            1.0
        };
        for _ in 0..=degree {
            knots.push(end_value);
        }

        knots
    }

    /// Interpolate the B-spline at the given parameter value
    ///
    /// # Arguments
    /// * `t` - Parameter value (must be in the valid range returned by knot_domain)
    ///
    /// # Returns
    /// The interpolated value at parameter t
    pub fn interpolate(&self, t: f64) -> T {
        self.spline.point(t)
    }

    /// Get the valid parameter range for the spline
    ///
    /// # Returns
    /// A tuple (min, max) representing the valid parameter range
    pub fn knot_domain(&self) -> (f64, f64) {
        self.spline.knot_domain()
    }

    /// Get the degree of the B-spline
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the control points
    pub fn control_points(&self) -> &[T] {
        &self.control_points
    }

    /// Get the knot vector
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Interpolate the B-spline at multiple points
    ///
    /// # Arguments
    /// * `point_count` - Number of points to interpolate
    ///
    /// # Returns
    /// A vector of interpolated points
    pub fn interpolate_range(&self, point_count: usize) -> Vec<T> {
        if point_count == 0 {
            return Vec::new();
        }

        let (min_t, max_t) = self.knot_domain();
        let mut points = Vec::with_capacity(point_count);

        for i in 0..point_count {
            // Calculate parameter value in the valid range
            // We need to be careful not to go exactly to max_t to avoid edge cases
            let t = if i == point_count - 1 {
                max_t - f64::EPSILON
            } else {
                min_t + (max_t - min_t) * (i as f64) / (point_count - 1) as f64
            };
            points.push(self.interpolate(t));
        }

        points
    }

    /// Interpolate the B-spline at multiple points with custom parameter values
    ///
    /// # Arguments
    /// * `parameters` - A slice of parameter values to interpolate at
    ///
    /// # Returns
    /// A vector of interpolated points
    pub fn interpolate_at(&self, parameters: &[f64]) -> Vec<T> {
        parameters.iter().map(|&t| self.interpolate(t)).collect()
    }

    /// Get the number of control points
    pub fn control_points_count(&self) -> usize {
        self.control_points.len()
    }

    /// Get the number of knots
    pub fn knots_count(&self) -> usize {
        self.knots.len()
    }

    /// Evaluate interpolation quality by comparing with a reference function
    ///
    /// # Arguments
    /// * `reference_function` - A function that provides the true values
    /// * `sample_count` - Number of points to sample for quality evaluation
    /// * `evaluation_domain` - The domain over which to evaluate (min, max), if None uses knot domain
    ///
    /// # Returns
    /// InterpolationQuality metrics
    pub fn evaluate_quality<F>(
        &self,
        reference_function: F,
        sample_count: usize,
        evaluation_domain: Option<(f64, f64)>,
    ) -> InterpolationQuality
    where
        T: Into<f64> + Copy,
        F: Fn(f64) -> f64,
    {
        if sample_count == 0 {
            return InterpolationQuality::new(0.0, 0.0, 0.0, 0);
        }

        // Use provided domain or default to knot domain
        let (min_t, max_t) = evaluation_domain.unwrap_or_else(|| self.knot_domain());
        let mut sum_squared_error = 0.0;
        let mut max_error = 0.0;
        let mut sum_error = 0.0;

        for i in 0..sample_count {
            let t = min_t + (max_t - min_t) * (i as f64) / (sample_count - 1) as f64;
            let interpolated_value: f64 = self.interpolate(t).into();
            let true_value = reference_function(t);
            let error = (interpolated_value - true_value).abs();

            sum_squared_error += error * error;
            sum_error += error;
            if error > max_error {
                max_error = error;
            }
        }

        let rmse = (sum_squared_error / sample_count as f64).sqrt();
        let mean_error = sum_error / sample_count as f64;

        InterpolationQuality::new(rmse, max_error, mean_error, sample_count)
    }
}
/*
/// For f64 values, we can convert directly to f64
impl Into<f64> for f64 {
    fn into(self) -> f64 {
        self
    }
}
 */
/// For Point2D, we'll use the y-coordinate for quality evaluation
impl Into<f64> for Point2D {
    fn into(self) -> f64 {
        self.y
    }
}

/// For Point3D, we'll use the y-coordinate for quality evaluation
impl Into<f64> for Point3D {
    fn into(self) -> f64 {
        self.y
    }
}

/// Generate a set of control points for demonstration
/// Creates a sine wave pattern for interesting visualization
/// This function is kept for testing purposes
#[cfg(test)]
fn generate_control_points(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let x = i as f64 * 0.5;
            x.sin() * 10.0
        })
        .collect()
}

/// Generate a set of 2D control points for demonstration
/// Creates a circular pattern for interesting visualization
#[cfg(test)]
pub fn generate_2d_control_points(count: usize) -> Vec<Point2D> {
    (0..count)
        .map(|i| {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (count as f64);
            Point2D::new(angle.cos() * 5.0, angle.sin() * 5.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn main_test() {
        println!("B-spline Interpolation Tool");
        println!("===========================");

        // Example 1: 1D interpolation with f64 values
        println!("Example 1: 1D Cubic B-spline interpolation");
        let control_points = vec![0.0_f64, 5.0, 10.0, 5.0, 0.0, -5.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        println!("Control points: {:?}", interpolator.control_points());
        println!(
            "Knot domain: [{}, {}]",
            interpolator.knot_domain().0,
            interpolator.knot_domain().1
        );

        // Interpolate 20 points
        let interpolated_points = interpolator.interpolate_range(20);
        println!("Interpolated 20 points:");
        for (i, point) in interpolated_points.iter().enumerate() {
            println!("  Point {}: {:.4}", i, point);
        }

        // Example 2: 2D point interpolation
        println!("\nExample 2: 2D B-spline interpolation");
        let control_points_2d = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(3.0, 1.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(5.0, -1.0),
        ];

        let interpolator_2d = BSplineInterpolator::with_uniform_knots(3, control_points_2d);
        let interpolated_2d_points = interpolator_2d.interpolate_range(10);

        println!("Interpolated 2D points:");
        for (i, point) in interpolated_2d_points.iter().enumerate() {
            println!("  Point {}: ({:.4}, {:.4})", i, point.x, point.y);
        }

        // Example 3: 3D point interpolation
        println!("\nExample 3: 3D B-spline interpolation");
        let control_points_3d = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(2.0, 0.0, 2.0),
            Point3D::new(3.0, -1.0, 1.0),
            Point3D::new(4.0, 0.0, 0.0),
        ];

        let interpolator_3d = BSplineInterpolator::with_uniform_knots(3, control_points_3d);
        let interpolated_3d_points = interpolator_3d.interpolate_range(8);

        println!("Interpolated 3D points:");
        for (i, point) in interpolated_3d_points.iter().enumerate() {
            println!(
                "  Point {}: ({:.4}, {:.4}, {:.4})",
                i, point.x, point.y, point.z
            );
        }

        // Example 4: Custom parameter interpolation
        println!("\nExample 4: Custom parameter interpolation");
        let control_points_custom = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let interpolator_custom = BSplineInterpolator::with_uniform_knots(2, control_points_custom);

        // Interpolate at specific parameter values
        let custom_parameters = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let custom_points = interpolator_custom.interpolate_at(&custom_parameters);

        println!("Interpolated at custom parameters:");
        for (param, point) in custom_parameters.iter().zip(custom_points.iter()) {
            println!("  t = {}: {:.4}", param, point);
        }

        // Example 5: Quality evaluation for sine function
        println!("\nExample 5: Quality evaluation for sine function");
        let sine_control_points: Vec<f64> = (0..6)
            .map(|i| {
                let x = (i as f64) * std::f64::consts::PI / 5.0;
                x.sin()
            })
            .collect();

        let sine_interpolator = BSplineInterpolator::with_uniform_knots(3, sine_control_points);

        // Evaluate quality against the true sine function over the same domain
        let (min_t, max_t) = sine_interpolator.knot_domain();
        let quality = sine_interpolator.evaluate_quality(|x| x.sin(), 1000, Some((min_t, max_t)));
        println!("Sine function interpolation quality:");
        println!("  RMSE: {:.6}", quality.rmse);
        println!("  Max Error: {:.6}", quality.max_error);
        println!("  Mean Error: {:.6}", quality.mean_error);
    }
    #[test]
    fn test_control_points_generation() {
        let count = 5;
        let points = generate_control_points(count);
        assert_eq!(points.len(), count);
        // Check that points are generated (not all zeros)
        assert!(points.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_knots_generation() {
        let control_points_count = 6;
        let degree = 3;
        let expected_knots_count = control_points_count + degree + 1;

        let control_points = generate_control_points(control_points_count);
        let interpolator = BSplineInterpolator::with_uniform_knots(degree, control_points);
        assert_eq!(interpolator.knots().len(), expected_knots_count);

        // Check that knots are non-decreasing
        let knots = interpolator.knots();
        for i in 1..knots.len() {
            assert!(knots[i] >= knots[i - 1]);
        }
    }

    #[test]
    fn test_bspline_interpolation() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        let points = interpolator.interpolate_range(10);
        assert_eq!(points.len(), 10);

        // Check that all points are finite numbers
        for &point in &points {
            assert!(point.is_finite());
        }
    }

    #[test]
    fn test_bspline_properties() {
        // Test with a simple case: 4 control points, degree 3 (cubic)
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0];
        let knots = vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let interpolator = BSplineInterpolator::new(3, control_points, knots);
        let (min_t, max_t) = interpolator.knot_domain();

        assert_eq!(min_t, 0.0);
        assert_eq!(max_t, 1.0);

        // At the start of the curve, we should get close to the first control point
        let start_point = interpolator.interpolate(0.0);
        assert!((start_point - 0.0).abs() < 0.001);

        // At the end of the curve, we should get close to the last control point
        let end_point = interpolator.interpolate(1.0 - f64::EPSILON);
        assert!((end_point - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolator_with_uniform_knots() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points.clone());

        assert_eq!(interpolator.degree(), 3);
        assert_eq!(interpolator.control_points(), &control_points[..]);
        assert_eq!(interpolator.knots().len(), control_points.len() + 3 + 1);
    }

    #[test]
    fn test_interpolate_range() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);

        let points = interpolator.interpolate_range(5);
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn test_interpolate_range_empty() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);

        let points = interpolator.interpolate_range(0);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_2d_bspline_interpolation() {
        let control_points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 0.0),
        ];

        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);
        let points = interpolator.interpolate_range(5);

        assert_eq!(points.len(), 5);
        for point in &points {
            assert!(point.x.is_finite());
            assert!(point.y.is_finite());
        }
    }

    #[test]
    fn test_3d_bspline_interpolation() {
        let control_points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(2.0, 0.0, 2.0),
        ];

        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);
        let points = interpolator.interpolate_range(5);

        assert_eq!(points.len(), 5);
        for point in &points {
            assert!(point.x.is_finite());
            assert!(point.y.is_finite());
            assert!(point.z.is_finite());
        }
    }

    #[test]
    fn test_interpolate_at() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);

        let parameters = vec![0.0, 0.5, 1.0];
        let points = interpolator.interpolate_at(&parameters);

        assert_eq!(points.len(), 3);
        for &point in &points {
            assert!(point.is_finite());
        }
    }

    #[test]
    fn test_control_points_count() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        assert_eq!(interpolator.control_points_count(), 5);
    }

    #[test]
    fn test_knots_count() {
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        assert_eq!(interpolator.knots_count(), 9); // 5 control points + 3 degree + 1
    }

    #[test]
    fn test_large_dataset_performance() {
        // Test with a large dataset to ensure performance is reasonable
        let control_points: Vec<f64> = (0..10000).map(|i| i as f64 * 0.01).collect();
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        // This should not panic and should complete in a reasonable time
        let start = Instant::now();
        let points = interpolator.interpolate_range(1000);
        let duration = start.elapsed();

        assert_eq!(points.len(), 1000);
        println!(
            "Interpolated 1000 points from 10000 control points in {:?}",
            duration
        );
    }

    #[test]
    fn test_very_large_dataset() {
        // Test with a very large dataset (100,000 points)
        let control_points: Vec<f64> = (0..100000).map(|i| (i as f64).sin() * 10.0).collect();
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        // Interpolate a smaller number of points to avoid memory issues
        let start = Instant::now();
        let points = interpolator.interpolate_range(100);
        let duration = start.elapsed();

        assert_eq!(points.len(), 100);
        println!(
            "Interpolated 100 points from 100000 control points in {:?}",
            duration
        );

        // Verify that all points are finite
        for &point in &points {
            assert!(point.is_finite());
        }
    }

    #[test]
    fn test_interpolation_quality() {
        // Test interpolation quality for a simple linear function
        // Create control points that match a linear function y = x over domain [0, 4]
        let control_points = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        // For a linear function, cubic interpolation should be very accurate
        // Evaluate over the same domain as the control points
        let quality = interpolator.evaluate_quality(|x| x, 1000, Some((0.0, 4.0)));
        println!("Linear function interpolation quality: {:?}", quality);

        // For a linear function interpolated with a cubic spline, error should be very small
        assert!(quality.rmse < 0.01, "RMSE too high: {}", quality.rmse);
        assert!(
            quality.max_error < 0.01,
            "Max error too high: {}",
            quality.max_error
        );
        assert!(
            quality.mean_error < 0.01,
            "Mean error too high: {}",
            quality.mean_error
        );
    }

    #[test]
    fn test_sine_interpolation_quality() {
        // Test interpolation quality for sine function
        // Create control points for sine function over domain [0, π]
        let control_points: Vec<f64> = (0..10)
            .map(|i| {
                let x = (i as f64) * std::f64::consts::PI / 9.0;
                x.sin()
            })
            .collect();

        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        // Evaluate quality against the true sine function over the same domain
        let quality =
            interpolator.evaluate_quality(|x| x.sin(), 1000, Some((0.0, std::f64::consts::PI)));
        println!("Sine function interpolation quality: {:?}", quality);

        // For sine function with 10 control points, error should be reasonably small
        assert!(quality.rmse < 0.05, "RMSE too high: {}", quality.rmse);
        assert!(
            quality.max_error < 0.1,
            "Max error too high: {}",
            quality.max_error
        );
        assert!(
            quality.mean_error < 0.05,
            "Mean error too high: {}",
            quality.mean_error
        );
    }

    #[test]
    fn test_polynomial_interpolation_quality() {
        // Test interpolation quality for a quadratic function
        // Create control points for quadratic function y = x^2 over domain [0, 4.5]
        let control_points: Vec<f64> = (0..10)
            .map(|i| {
                let x = (i as f64) * 0.5;
                x * x // Quadratic function
            })
            .collect();

        let interpolator = BSplineInterpolator::with_uniform_knots(3, control_points);

        // Evaluate quality against the true quadratic function over the same domain
        let quality = interpolator.evaluate_quality(|x| x * x, 1000, Some((0.0, 4.5)));
        println!("Quadratic function interpolation quality: {:?}", quality);

        // For quadratic function with cubic spline, error should be reasonably small
        assert!(quality.rmse < 0.1, "RMSE too high: {}", quality.rmse);
        assert!(
            quality.max_error < 0.2,
            "Max error too high: {}",
            quality.max_error
        );
        assert!(
            quality.mean_error < 0.1,
            "Mean error too high: {}",
            quality.mean_error
        );
    }

    #[test]
    fn test_interpolation_quality_empty() {
        let control_points = vec![0.0_f64, 1.0, 2.0];
        let interpolator = BSplineInterpolator::with_uniform_knots(2, control_points);

        // Test with 0 sample points
        let quality = interpolator.evaluate_quality(|x| x, 0, None);

        assert_eq!(quality.rmse, 0.0);
        assert_eq!(quality.max_error, 0.0);
        assert_eq!(quality.mean_error, 0.0);
        assert_eq!(quality.sample_count, 0);
    }

    #[test]
    fn test_interpolation_quality_struct() {
        let quality = InterpolationQuality::new(0.1, 0.5, 0.3, 100);

        assert_eq!(quality.rmse, 0.1);
        assert_eq!(quality.max_error, 0.5);
        assert_eq!(quality.mean_error, 0.3);
        assert_eq!(quality.sample_count, 100);
    }
}

//! # 2D Animation Module for Bevy
//!
//! This module provides interactive 2D trajectory animation using the Bevy game engine.
//! It displays animated paths with controllable camera, axis visualization, and real-time position tracking.
//!
//! ## Features
//! - Interactive 2D trajectory animation
//! - Mouse-controlled camera (drag to pan, scroll to zoom)
//! - Colored coordinate axes with markers
//! - Real-time position display and axis legend
//! - Customizable axis names and animation speed
//!
//! ## Usage
//!
//! ```rust
//! use nalgebra::{DMatrix, DVector};
//! use your_crate::Utils::animation_2d::*;
//!
//! // Generate sample data (2 rows x N columns matrix)
//! let (positions, times) = generate_circle_2d(1000, 2.0);
//!
//! // Create animation with custom axis names and speed
//! create_2d_animation(
//!     positions,
//!     times,
//!     Some(("X-pos".to_string(), "Y-pos".to_string())),
//!     Some(1.5) // 1.5x speed
//! );
//! ```
//!
//! ## Data Format
//! - `points`: DMatrix<f64> with shape (2, N) where each column is a 2D point
//! - `times`: DVector<f64> with length N containing time values
//!
//! ## Controls
//! - **Left mouse drag**: Pan camera
//! - **Mouse scroll**: Zoom in/out
//! - **ESC**: Close animation window

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};

/// Component that holds animated 2D trajectory data and current animation state
#[derive(Component)]
struct AnimatedLine2D {
    /// 2D points matrix (2 rows x N columns, each column is a point)
    points: DMatrix<f32>,
    /// Time values for each point (length N)
    _times: DVector<f32>,
    /// Current point index being animated (0..N-1)
    current_index: usize,
    /// Timer controlling animation speed between points
    timer: Timer,
}

/// Component for 2D camera control with pan and zoom
#[derive(Component)]
pub struct Camera2DController {
    /// Camera position offset
    pub offset: Vec2,
    /// Zoom level (higher = more zoomed in)
    pub zoom: f32,
}

/// Marker component to identify legend UI elements
#[derive(Component)]
struct LegendMarker;

/// Resource containing all 2D trajectory and display configuration data
#[derive(Resource)]
struct LineData2D {
    /// 2D trajectory points
    points: DMatrix<f32>,
    /// Time values for animation timing
    times: DVector<f32>,
    /// Custom names for X, Y axes
    axis_names: (String, String),
    /// Animation speed multiplier (higher = faster)
    speed_multiplier: f32,
}

/// Creates and runs an interactive 2D trajectory animation
///
/// # Arguments
/// * `points` - 2D trajectory data as DMatrix<f64> with shape (2, N)
/// * `times` - Time values as DVector<f64> with length N
/// * `axes_names` - Optional custom axis names (X, Y). Defaults to ("x", "y")
/// * `speed` - Optional animation speed multiplier. Defaults to 1.0
///
/// # Example
/// ```rust
/// let (sine_points, sine_times) = generate_sine_wave_2d(500, 2.0, 3.0, 10.0);
/// create_2d_animation(
///     sine_points,
///     sine_times,
///     Some(("Time".to_string(), "Amplitude".to_string())),
///     Some(2.0)
/// );
/// ```
pub fn create_2d_animation(
    points: DMatrix<f64>,
    times: DVector<f64>,
    axes_names: Option<(String, String)>,
    speed: Option<f32>,
) {
    // Create Bevy app with default plugins (windowing, rendering, input, etc.)
    App::new()
        .add_plugins(DefaultPlugins)
        // Setup scene once at startup
        .add_systems(Startup, setup_2d_scene)
        // Run animation and camera control every frame
        .add_systems(Update, (animate_2d_line, camera_2d_controller))
        // Store trajectory data as a global resource
        .insert_resource(LineData2D {
            points: points.cast::<f32>(), // Convert f64 to f32 for Bevy compatibility
            times: times.cast::<f32>(),
            axis_names: axes_names.unwrap_or_else(|| ("x".to_string(), "y".to_string())),
            speed_multiplier: speed.unwrap_or(1.0),
        })
        .run(); // Blocks until window is closed
}

/// Bevy startup system that initializes the 2D scene
/// Sets up camera, animated trajectory, and UI elements
fn setup_2d_scene(mut commands: Commands, line_data: Res<LineData2D>) {
    // Set light gray background color
    commands.insert_resource(ClearColor(Color::srgb(0.95, 0.95, 0.98)));

    // Calculate bounding box of trajectory data for optimal camera positioning
    let p = &line_data.points;
    let mut min_vals = [f32::INFINITY; 2];
    let mut max_vals = [f32::NEG_INFINITY; 2];

    // Find min/max values in each dimension
    for col in 0..p.ncols() {
        for row in 0..2 {
            let val = p[(row, col)];
            min_vals[row] = min_vals[row].min(val);
            max_vals[row] = max_vals[row].max(val);
        }
    }

    // Calculate center point and size for camera setup
    let _center = Vec2::new(
        (min_vals[0] + max_vals[0]) / 2.0,
        (min_vals[1] + max_vals[1]) / 2.0,
    );
    let size = ((max_vals[0] - min_vals[0]).powi(2) + (max_vals[1] - min_vals[1]).powi(2)).sqrt();
    let _initial_zoom = if size > 0.0 {
        (2.0 / size).clamp(0.01, 100.0)
    } else {
        1.0
    };

    // Spawn 2D camera with pan/zoom controller
    commands.spawn((
        Camera2d,
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        // Attach the controller so the camera_query in the system finds it
        Camera2DController {
            offset: Vec2::ZERO,
            zoom: 1.0,
        },
    ));

    // Validate input data format
    let t = &line_data.times;
    if p.nrows() != 2 {
        error!("Points must have 2 rows (x,y). got nrows = {}", p.nrows());
        return;
    }
    if p.ncols() == 0 || t.len() == 0 {
        warn!("No points/times provided.");
        return;
    }
    if p.ncols() != t.len() {
        warn!(
            "Number of points (columns) != number of times. points.cols={} times.len={}",
            p.ncols(),
            t.len()
        );
        // Continue with available data
    }

    // Calculate initial animation timing
    let _npoints = p.ncols();
    let initial_dt = if t.len() >= 2 {
        // Use time difference between first two points, scaled by speed
        ((t[1] - t[0]).abs().max(1e-6)) / line_data.speed_multiplier
    } else {
        // Fallback to 100ms per frame, scaled by speed
        0.1_f32 / line_data.speed_multiplier
    };

    // Spawn the animated trajectory entity
    commands.spawn((AnimatedLine2D {
        points: p.clone(),
        _times: t.clone(),
        current_index: 0, // Start at first point
        timer: Timer::from_seconds(initial_dt, TimerMode::Once), // One-shot timer
    },));

    // Print axis information to console for reference
    println!(
        "2D Axes: {} (red), {} (green)",
        line_data.axis_names.0, line_data.axis_names.1
    );
    println!("Controls: Left-click drag to pan, scroll to zoom");

    // Create UI text for real-time position display (bottom-right corner)
    commands
        .spawn((
            Text::new("Position: X:0.0, Y:0.0"),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(20.0), // 20px from bottom
                right: Val::Px(20.0),  // 20px from right
                ..default()
            },
            TextColor(Color::srgb(0.0, 0.0, 0.0)), // Black text
            TextFont {
                font_size: 16.0,
                ..default()
            },
        ))
        .insert(Name::new("PositionText")); // Named for easy querying

    // Create axis legend (above position text)
    commands.spawn((
        Text::new(format!(
            "[R] {} (X-axis)\n[G] {} (Y-axis)",
            line_data.axis_names.0, line_data.axis_names.1
        )),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(60.0), // Above position text
            right: Val::Px(20.0),
            ..default()
        },
        TextColor(Color::srgb(0.6, 0.6, 0.6)), // Gray text
        TextFont {
            font_size: 14.0,
            ..default()
        },
        LegendMarker, // Marker component for identification
    ));
}

/// Bevy system that handles 2D trajectory animation and rendering
/// Runs every frame to update animation state and draw 2D elements
fn animate_2d_line(
    time: Res<Time>,                              // Bevy's time resource for delta time
    mut gizmos: Gizmos,                           // Bevy's immediate-mode 2D drawing API
    mut query: Query<&mut AnimatedLine2D>,        // Query for animated trajectory entities
    line_data: Res<LineData2D>,                   // Global trajectory data
    mut text_query: Query<&mut Text, With<Name>>, // Query for UI text elements
) {
    // Process each animated trajectory (usually just one)
    for mut line in query.iter_mut() {
        let npoints = line.points.ncols();
        if npoints == 0 {
            continue; // Skip empty trajectories
        }

        // Update animation timer with frame delta time
        line.timer.tick(time.delta());

        // Check if it's time to advance to next point
        if line.timer.just_finished() {
            // Calculate point skip based on speed (higher speed = skip more points)
            let skip = (line_data.speed_multiplier as usize).max(1);

            // Advance to next point with wraparound
            line.current_index = (line.current_index + skip) % npoints;

            // Reset timer for next animation step (fixed 50ms base duration)
            let base_duration = 0.05;
            line.timer = Timer::from_seconds(base_duration, TimerMode::Once);
            line.timer.reset();
        }

        // Calculate axis length for proper scaling
        let mut min_vals = [f32::INFINITY; 2];
        let mut max_vals = [f32::NEG_INFINITY; 2];

        // Find min/max in each dimension across all points
        for col in 0..npoints {
            for row in 0..2 {
                let val = line.points[(row, col)];
                min_vals[row] = min_vals[row].min(val);
                max_vals[row] = max_vals[row].max(val);
            }
        }

        // Calculate axis extent based on data bounds with padding
        let x_extent = (max_vals[0] - min_vals[0]).max(1.0) * 0.6;
        let y_extent = (max_vals[1] - min_vals[1]).max(1.0) * 0.6;

        // Draw coordinate axes extending in both directions
        gizmos.line_2d(
            Vec2::new(-x_extent, 0.0),
            Vec2::new(x_extent, 0.0),
            Color::srgb(1.0, 0.0, 0.0),
        ); // X-axis
        gizmos.line_2d(
            Vec2::new(0.0, -y_extent),
            Vec2::new(0.0, y_extent),
            Color::srgb(0.0, 1.0, 0.0),
        ); // Y-axis

        // Draw small circles at axis endpoints
        let circle_radius = (x_extent + y_extent) * 0.01;
        gizmos.circle_2d(
            Vec2::new(x_extent, 0.0),
            circle_radius,
            Color::srgb(1.0, 0.0, 0.0),
        );
        gizmos.circle_2d(
            Vec2::new(-x_extent, 0.0),
            circle_radius,
            Color::srgb(1.0, 0.0, 0.0),
        );
        gizmos.circle_2d(
            Vec2::new(0.0, y_extent),
            circle_radius,
            Color::srgb(0.0, 1.0, 0.0),
        );
        gizmos.circle_2d(
            Vec2::new(0.0, -y_extent),
            circle_radius,
            Color::srgb(0.0, 1.0, 0.0),
        );

        // Draw trajectory path up to current animation point
        for i in 0..line.current_index {
            if i + 1 < npoints {
                // Convert matrix columns to 2D points
                let start = Vec2::new(
                    line.points[(0, i)], // X coordinate
                    line.points[(1, i)], // Y coordinate
                );
                let end = Vec2::new(line.points[(0, i + 1)], line.points[(1, i + 1)]);
                // Draw line segment in cyan
                gizmos.line_2d(start, end, Color::srgb(0.0, 1.0, 1.0));
            }
        }

        // Draw current animation position as a blue circle
        let cur_idx = line.current_index.min(npoints - 1); // Clamp to valid range
        let current_point = Vec2::new(line.points[(0, cur_idx)], line.points[(1, cur_idx)]);
        // Small blue circle to mark current position
        gizmos.circle_2d(current_point, 0.05, Color::srgb(0.0, 0.0, 1.0));

        // Update UI text with current position values
        for mut text in text_query.iter_mut() {
            **text = format!(
                "Position: {}:{:.2}, {}:{:.2}",
                line_data.axis_names.0, current_point.x, line_data.axis_names.1, current_point.y
            );
        }
    }
}

/// Bevy system that handles interactive 2D camera control
/// Provides pan and zoom functionality with mouse input
fn camera_2d_controller(
    mut mouse_motion: MessageReader<MouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut camera_query: Query<(&mut Transform, &mut Camera2DController), With<Camera2d>>,
) {
    for (mut transform, mut controller) in camera_query.iter_mut() {
        // Handle panning
        if mouse_input.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                let pan_speed = 2.0;
                controller.offset.x -= motion.delta.x * pan_speed / controller.zoom; // adjust for zoom
                controller.offset.y += motion.delta.y * pan_speed / controller.zoom;
            }
        }

        // Handle zoom with scroll wheel
        for scroll in scroll_events.read() {
            controller.zoom *= 1.0 + scroll.y * 0.1;
            controller.zoom = controller.zoom.clamp(0.1, 10.0);
        }

        // Apply camera translation and zoom (reciprocal for correct zoom direction)
        transform.translation.x = controller.offset.x;
        transform.translation.y = controller.offset.y;
        transform.scale = Vec3::splat(1.0 / controller.zoom);
    }

    if !mouse_input.pressed(MouseButton::Left) {
        mouse_motion.clear();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for generating test 2D trajectory data
//////////////////////////////////////////////////////////////////////////////////////////

/// Generates a straight line trajectory from (0,0) to (5,0)
///
/// # Arguments
/// * `num_points` - Number of points along the line
///
/// # Returns
/// * Tuple of (positions, times) where positions is 2×N matrix
pub fn generate_line_2d(num_points: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(2, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64; // Normalized time 0..1
        positions[(0, i)] = t * 5.0; // X: linear from 0 to 5
        positions[(1, i)] = 0.0; // Y: constant 0
        times[i] = t;
    }

    (positions, times)
}

/// Generates a circular trajectory
///
/// # Arguments
/// * `num_points` - Number of points around the circle
/// * `radius` - Circle radius
pub fn generate_circle_2d(num_points: usize, radius: f64) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(2, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let angle = t * 2.0 * std::f64::consts::PI; // Full circle
        positions[(0, i)] = radius * angle.cos(); // X: cosine
        positions[(1, i)] = radius * angle.sin(); // Y: sine
        times[i] = t;
    }

    (positions, times)
}

/// Generates a sine wave trajectory along the X-axis
///
/// # Arguments
/// * `num_points` - Number of points along the wave
/// * `amplitude` - Wave amplitude in Y direction
/// * `frequency` - Wave frequency (cycles per unit length)
/// * `length` - Total length along X-axis
pub fn generate_sine_wave_2d(
    num_points: usize,
    amplitude: f64,
    frequency: f64,
    length: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(2, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let x = t * length; // X: linear progression
        positions[(0, i)] = x;
        positions[(1, i)] = amplitude * (frequency * x).sin(); // Y: sine wave
        times[i] = t;
    }

    (positions, times)
}

/// Generates a Lissajous curve (parametric curve)
///
/// # Arguments
/// * `num_points` - Number of points along the curve
/// * `a` - Frequency ratio for X component
/// * `b` - Frequency ratio for Y component
/// * `delta` - Phase shift between X and Y
pub fn generate_lissajous_2d(
    num_points: usize,
    a: f64,
    b: f64,
    delta: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(2, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let angle = t * 2.0 * std::f64::consts::PI;
        positions[(0, i)] = (a * angle).sin(); // X: sine with frequency a
        positions[(1, i)] = (b * angle + delta).sin(); // Y: sine with frequency b and phase
        times[i] = t;
    }

    (positions, times)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_data_generation() {
        // Test line
        let (line_pos, line_times) = generate_line_2d(100);
        assert_eq!(line_pos.nrows(), 2);
        assert_eq!(line_pos.ncols(), 100);
        assert_eq!(line_times.len(), 100);

        // Test circle
        let (circle_pos, _circle_times) = generate_circle_2d(200, 2.0);
        assert_eq!(circle_pos.nrows(), 2);
        assert_eq!(circle_pos.ncols(), 200);

        // Test sine wave
        let (sine_pos, _sine_times) = generate_sine_wave_2d(150, 2.0, 3.0, 10.0);
        assert_eq!(sine_pos.nrows(), 2);
        assert_eq!(sine_pos.ncols(), 150);

        // Test Lissajous
        let (liss_pos, _liss_times) =
            generate_lissajous_2d(300, 3.0, 2.0, std::f64::consts::PI / 2.0);
        assert_eq!(liss_pos.nrows(), 2);
        assert_eq!(liss_pos.ncols(), 300);

        println!("All 2D shapes generated successfully");

        // Uncomment to test animations:
        // create_2d_animation(line_pos, line_times, None, None);
        // create_2d_animation(circle_pos, circle_times, None, None);
        // create_2d_animation(sine_pos, sine_times, None, None);
        // create_2d_animation(liss_pos, liss_times, None, None);
    }
}

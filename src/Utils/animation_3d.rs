//! # 3D Animation Module for Bevy
//!
//! This module provides interactive 3D trajectory animation using the Bevy game engine.
//! It displays animated paths with controllable camera, axis visualization, and real-time position tracking.
//!
//! ## Features
//! - Interactive 3D trajectory animation
//! - Mouse-controlled camera (left-click drag to rotate, scroll to zoom)
//! - Colored coordinate axes with spherical markers
//! - Real-time position display and axis legend
//! - Customizable axis names and animation speed
//!
//! ## Usage
//!
//! ```rust, ignore
//! use nalgebra::{DMatrix, DVector};
//! use your_crate::Utils::animation_3d::*;
//!
//! // Generate sample data (3 rows x N columns matrix)
//! let (positions, times) = generate_helix(1000, 2.0, 5.0, 3.0);
//!
//! // Create animation with custom axis names and speed
//! create_3d_animation(
//!     positions,
//!     times,
//!     Some(("X-pos".to_string(), "Y-pos".to_string(), "Z-pos".to_string())),
//!     Some(2.0) // 2x speed
//! );
//! ```
//!
//! ## Data Format
//! - `points`: DMatrix<f64> with shape (3, N) where each column is a 3D point
//! - `times`: DVector<f64> with length N containing time values
//!
//! ## Controls
//! - **Left mouse drag**: Rotate camera around data center
//! - **Mouse scroll**: Zoom in/out
//! - **ESC**: Close animation window

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use nalgebra::{DMatrix, DVector};

use super::animation::normalize_for_display;
use super::animation::{AnimationInputError, advance_animation_clock, validate_trajectory};

/// Component that holds animated trajectory data and current animation state
#[derive(Component)]
struct AnimatedLine {
    /// 3D points matrix (3 rows x N columns, each column is a point)
    points: DMatrix<f32>,
    /// Current point index being animated (0..N-1)
    current_index: usize,
    /// Sub-frame time already spent in the current sample interval.
    elapsed_in_interval: f64,
}

/// Component for interactive camera control with orbital movement
#[derive(Component)]
pub struct CameraController {
    /// Center point the camera orbits around
    pub center: Vec3,
    /// Distance from center to camera
    pub distance: f32,
    /// Horizontal rotation angle (radians)
    pub yaw: f32,
    /// Vertical rotation angle (radians)
    pub pitch: f32,
}

/// Marker component to identify legend UI elements
#[derive(Component)]
struct LegendMarker;

/// Resource containing all trajectory and display configuration data
#[derive(Resource)]
struct LineData {
    /// 3D trajectory points
    points: DMatrix<f32>,
    /// Seconds between successive displayed samples, including the loop edge.
    frame_intervals: Vec<f32>,
    /// Cached trajectory bounds used by the camera and axes.
    min: Vec3,
    max: Vec3,
    /// Custom names for X, Y, Z axes
    axis_names: (String, String, String),
    /// Transform metadata used to display physical coordinates in the HUD.
    coordinate_origin: [f64; 3],
    units_per_display_unit: f64,
}

/// Creates and runs an interactive 3D trajectory animation
///
/// # Arguments
/// * `points` - 3D trajectory data as DMatrix<f64> with shape (3, N)
/// * `times` - Time values as DVector<f64> with length N
/// * `axes_names` - Optional custom axis names (X, Y, Z). Defaults to ("x", "y", "z")
/// * `speed` - Optional animation speed multiplier. Defaults to 1.0
///
/// # Example
/// ```rust, ignore
/// let (helix_points, helix_times) = generate_helix(500, 1.0, 3.0, 2.0);
/// create_3d_animation(
///     helix_points,
///     helix_times,
///     Some(("Position X".to_string(), "Position Y".to_string(), "Height".to_string())),
///     Some(1.5)
/// );
/// ```
pub fn create_3d_animation(
    points: DMatrix<f64>,
    times: DVector<f64>,
    axes_names: Option<(String, String, String)>,
    speed: Option<f32>,
) {
    if let Err(error) = try_create_3d_animation(points, times, axes_names, speed) {
        eprintln!("Cannot start 3D trajectory animation: {error}");
    }
}

/// Validate and run an interactive 3D trajectory animation.
///
/// Timestamps may be strictly increasing or strictly decreasing. Playback
/// visits every sample and follows adjacent time gaps scaled by `speed`.
/// The Bevy event loop blocks until the window closes; renderer initialization
/// failures remain Bevy/runtime errors and are not converted by this function.
pub fn try_create_3d_animation(
    points: DMatrix<f64>,
    times: DVector<f64>,
    axes_names: Option<(String, String, String)>,
    speed: Option<f32>,
) -> Result<(), AnimationInputError> {
    let speed = speed.unwrap_or(1.0);
    let frame_intervals = validate_trajectory::<3>(&points, &times, speed)?;
    let display = normalize_for_display::<3>(&points);
    let points = display.points;
    let min = Vec3::new(
        (0..points.ncols())
            .map(|column| points[(0, column)])
            .fold(f32::INFINITY, f32::min),
        (0..points.ncols())
            .map(|column| points[(1, column)])
            .fold(f32::INFINITY, f32::min),
        (0..points.ncols())
            .map(|column| points[(2, column)])
            .fold(f32::INFINITY, f32::min),
    );
    let max = Vec3::new(
        (0..points.ncols())
            .map(|column| points[(0, column)])
            .fold(f32::NEG_INFINITY, f32::max),
        (0..points.ncols())
            .map(|column| points[(1, column)])
            .fold(f32::NEG_INFINITY, f32::max),
        (0..points.ncols())
            .map(|column| points[(2, column)])
            .fold(f32::NEG_INFINITY, f32::max),
    );

    // Create Bevy app with default plugins (windowing, rendering, input, etc.)
    App::new()
        .add_plugins(DefaultPlugins)
        // Setup scene once at startup
        .add_systems(Startup, setup_scene)
        // Run animation and camera control every frame
        .add_systems(
            Update,
            (animate_line, camera_controller, exit_animation_on_escape),
        )
        // Store trajectory data as a global resource
        .insert_resource(LineData {
            points,
            frame_intervals,
            min,
            max,
            axis_names: axes_names
                .unwrap_or_else(|| ("x".to_string(), "y".to_string(), "z".to_string())),
            coordinate_origin: display.origin,
            units_per_display_unit: display.units_per_display_unit,
        })
        .run(); // Blocks until window is closed
    Ok(())
}

/// Bevy startup system that initializes the 3D scene
/// Sets up camera, lighting, animated trajectory, and UI elements
fn setup_scene(mut commands: Commands, line_data: Res<LineData>) {
    // Set light gray background color
    commands.insert_resource(ClearColor(Color::srgb(0.95, 0.95, 0.98)));

    // The public entry point validates shape and finiteness before Bevy starts.
    // Calculate center point of data for camera focus
    let center = line_data.min * 0.5 + line_data.max * 0.5;

    // Calculate diagonal size of bounding box
    let extent = line_data.max - line_data.min;
    let size = extent.x.max(extent.y).max(extent.z);

    // Perspective projection needs extra room for diagonal views of the full
    // three-dimensional bounds, especially for attractor-shaped trajectories.
    // Keep tiny trajectories close enough to see; use a safe fallback only
    // when every point is identical.
    let camera_distance = if size > 0.0 { size * 4.0 } else { 2.0 };

    // Spawn 3D camera with orbital controller
    commands.spawn((
        Camera3d::default(),
        // Initial camera position: diagonal view of the data
        Transform::from_xyz(
            center.x + camera_distance,
            center.y + camera_distance,
            center.z + camera_distance,
        )
        .looking_at(center, Vec3::Y), // Look at data center, Y-up orientation
        // Attach camera controller for mouse interaction
        CameraController {
            center,
            distance: camera_distance,
            yaw: 45.0_f32.to_radians(),    // Initial horizontal rotation
            pitch: -35.0_f32.to_radians(), // Initial vertical rotation (looking down)
        },
    ));

    // Add directional light for 3D shading
    commands.spawn((
        DirectionalLight {
            illuminance: 20000.0, // Bright light for clear visibility
            ..default()
        },
        // Light direction: angled from above-right
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            1.0,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));

    let p = &line_data.points;

    // Spawn the animated trajectory entity
    commands.spawn((AnimatedLine {
        points: p.clone(),
        current_index: 0, // Start at first point
        elapsed_in_interval: 0.0,
    },));

    // Print axis information to console for reference
    println!(
        "Axes: {} (red), {} (green), {} (blue)",
        line_data.axis_names.0, line_data.axis_names.1, line_data.axis_names.2
    );
    println!("Controls: Left-click drag to rotate, scroll to zoom");

    // Create UI text for real-time position display (bottom-right corner)
    commands
        .spawn((
            Text::new("Position: X:0.0, Y:0.0, Z:0.0"),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(20.0), // 20px from bottom
                right: Val::Px(20.0),  // 20px from right
                ..default()
            },
            TextColor(Color::srgb(0.0, 0.0, 0.0)), // Black text
            TextFont {
                font_size: FontSize::Px(16.0),
                ..default()
            },
        ))
        .insert(Name::new("PositionText")); // Named for easy querying

    // Create axis legend (above position text)
    commands.spawn((
        Text::new(format!(
            "[R] {} (X-axis)\n[G] {} (Y-axis)\n[B] {} (Z-axis)",
            line_data.axis_names.0, line_data.axis_names.1, line_data.axis_names.2
        )),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(60.0), // Above position text
            right: Val::Px(20.0),
            ..default()
        },
        TextColor(Color::srgb(0.6, 0.6, 0.6)), // Gray text
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        LegendMarker, // Marker component for identification
    ));
}

/// Bevy system that handles trajectory animation and 3D rendering
/// Runs every frame to update animation state and draw 3D elements
fn animate_line(
    time: Res<Time>,                              // Bevy's time resource for delta time
    mut gizmos: Gizmos,                           // Bevy's immediate-mode 3D drawing API
    mut query: Query<&mut AnimatedLine>,          // Query for animated trajectory entities
    line_data: Res<LineData>,                     // Global trajectory data
    mut text_query: Query<&mut Text, With<Name>>, // Query for UI text elements
) {
    // Process each animated trajectory (usually just one)
    for mut line in query.iter_mut() {
        let npoints = line.points.ncols();
        if npoints == 0 {
            continue; // Skip empty trajectories
        }

        // Carry elapsed wall time across sample boundaries instead of losing
        // timer overshoot when adaptive steps are shorter than one render frame.
        (line.current_index, line.elapsed_in_interval) = advance_animation_clock(
            line.current_index,
            line.elapsed_in_interval,
            time.delta().as_secs_f64(),
            &line_data.frame_intervals,
        );

        // Calculate axis length as 120% of maximum data range
        let ranges = [
            line_data.max.x - line_data.min.x,
            line_data.max.y - line_data.min.y,
            line_data.max.z - line_data.min.z,
        ];
        let max_range = ranges.iter().fold(0.0f32, |a, &b| a.max(b));
        let axis_length = if max_range > 0.0 {
            max_range * 1.2
        } else {
            1.2
        };
        let axis_origin = line_data.min * 0.5 + line_data.max * 0.5;

        // Center axes on the data bounds so offset trajectories remain visible.
        let x_end = axis_origin + Vec3::X * axis_length; // Red X-axis
        let y_end = axis_origin + Vec3::Y * axis_length; // Green Y-axis
        let z_end = axis_origin + Vec3::Z * axis_length; // Blue Z-axis

        // Draw axis lines with standard colors
        gizmos.line(axis_origin, x_end, Color::srgb(1.0, 0.0, 0.0)); // Red X
        gizmos.line(axis_origin, y_end, Color::srgb(0.0, 1.0, 0.0)); // Green Y
        gizmos.line(axis_origin, z_end, Color::srgb(0.0, 0.0, 1.0)); // Blue Z

        // Draw small spheres at axis endpoints for clear identification
        let sphere_radius = axis_length * 0.02;
        gizmos.sphere(x_end, sphere_radius, Color::srgb(1.0, 0.0, 0.0));
        gizmos.sphere(y_end, sphere_radius, Color::srgb(0.0, 1.0, 0.0));
        gizmos.sphere(z_end, sphere_radius, Color::srgb(0.0, 0.0, 1.0));

        // Draw trajectory path up to current animation point
        for i in 0..line.current_index {
            if i + 1 < npoints {
                // Convert matrix columns to 3D points
                let start = Vec3::new(
                    line.points[(0, i)], // X coordinate
                    line.points[(1, i)], // Y coordinate
                    line.points[(2, i)], // Z coordinate
                );
                let end = Vec3::new(
                    line.points[(0, i + 1)],
                    line.points[(1, i + 1)],
                    line.points[(2, i + 1)],
                );
                // Draw line segment in cyan
                gizmos.line(start, end, Color::srgb(0.0, 1.0, 1.0));
            }
        }

        // Draw current animation position as a blue sphere
        let cur_idx = line.current_index.min(npoints - 1); // Clamp to valid range
        let current_point = Vec3::new(
            line.points[(0, cur_idx)],
            line.points[(1, cur_idx)],
            line.points[(2, cur_idx)],
        );
        // Small blue sphere to mark current position
        gizmos.sphere(
            current_point,
            (axis_length * 0.01).max(1.0e-6),
            Color::srgb(0.0, 0.0, 1.0),
        );

        // Update UI text with current position values
        for mut text in text_query.iter_mut() {
            let physical_x = f64::from(current_point.x) * line_data.units_per_display_unit
                + line_data.coordinate_origin[0];
            let physical_y = f64::from(current_point.y) * line_data.units_per_display_unit
                + line_data.coordinate_origin[1];
            let physical_z = f64::from(current_point.z) * line_data.units_per_display_unit
                + line_data.coordinate_origin[2];
            **text = format!(
                "Position: {}:{:.2}, {}:{:.2}, {}:{:.2}",
                line_data.axis_names.0,
                physical_x,
                line_data.axis_names.1,
                physical_y,
                line_data.axis_names.2,
                physical_z
            );
        }
    }
}

/// Bevy system that handles interactive camera control
/// Provides orbital camera movement with mouse input
fn camera_controller(
    mut mouse_motion: MessageReader<MouseMotion>, // Mouse movement events
    mut scroll_events: MessageReader<MouseWheel>, // Mouse scroll events
    mouse_input: Res<ButtonInput<MouseButton>>,   // Mouse button states
    mut camera_query: Query<(&mut Transform, &mut CameraController), With<Camera3d>>, // Camera entities
) {
    // Process each camera with controller (usually just one)
    for (mut transform, mut controller) in camera_query.iter_mut() {
        // Handle orbital rotation when left mouse button is held
        if mouse_input.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                // Convert mouse movement to rotation angles
                controller.yaw -= motion.delta.x * 0.01; // Horizontal rotation
                controller.pitch -= motion.delta.y * 0.01; // Vertical rotation
                // Clamp pitch to prevent camera flipping
                controller.pitch = controller.pitch.clamp(-1.5, 1.5);
            }
        }

        // Handle zoom with mouse scroll wheel
        for scroll in scroll_events.read() {
            // Multiply distance by scroll factor (scroll up = zoom in)
            controller.distance *= 1.0 - scroll.y * 0.1;
            // Prevent camera from getting too close or too far
            controller.distance = controller.distance.clamp(1.0e-6, 1.0e9);
        }

        // Calculate new camera position using spherical coordinates
        let x = controller.center.x
            + controller.distance * controller.yaw.cos() * controller.pitch.cos();
        let y = controller.center.y + controller.distance * controller.pitch.sin();
        let z = controller.center.z
            + controller.distance * controller.yaw.sin() * controller.pitch.cos();

        // Apply new position and maintain focus on center
        transform.translation = Vec3::new(x, y, z);
        transform.look_at(controller.center, Vec3::Y); // Always look at data center
    }

    // Clear unused mouse motion events to prevent accumulation
    if !mouse_input.pressed(MouseButton::Left) {
        mouse_motion.clear();
    }
}

/// Close the interactive application with a successful exit when Esc is hit.
fn exit_animation_on_escape(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        app_exit.write(AppExit::Success);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for generating test trajectory data
//////////////////////////////////////////////////////////////////////////////////////////

/// Generates a straight line trajectory from (0,0,0) to (5,0,0)
///
/// # Arguments
/// * `num_points` - Number of points along the line
///
/// # Returns
/// * Tuple of (positions, times) where positions is 3×N matrix
pub fn generate_line(num_points: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(3, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64; // Normalized time 0..1
        positions[(0, i)] = t * 5.0; // X: linear from 0 to 5
        positions[(1, i)] = 0.0; // Y: constant 0
        positions[(2, i)] = 0.0; // Z: constant 0
        times[i] = t;
    }

    (positions, times)
}

/// Generates a circular trajectory in the XY plane
///
/// # Arguments
/// * `num_points` - Number of points around the circle
/// * `radius` - Circle radius
pub fn generate_circle(num_points: usize, radius: f64) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(3, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let angle = t * 2.0 * std::f64::consts::PI; // Full circle
        positions[(0, i)] = radius * angle.cos(); // X: cosine
        positions[(1, i)] = radius * angle.sin(); // Y: sine
        positions[(2, i)] = 0.0; // Z: flat in XY plane
        times[i] = t;
    }

    (positions, times)
}

/// Generates a helical (spiral) trajectory
///
/// # Arguments
/// * `num_points` - Number of points along the helix
/// * `radius` - Helix radius in XY plane
/// * `height` - Total height of helix
/// * `turns` - Number of complete turns
pub fn generate_helix(
    num_points: usize,
    radius: f64,
    height: f64,
    turns: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(3, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let angle = t * turns * 2.0 * std::f64::consts::PI; // Multiple turns
        positions[(0, i)] = radius * angle.cos(); // X: circular motion
        positions[(1, i)] = radius * angle.sin(); // Y: circular motion
        positions[(2, i)] = t * height; // Z: linear rise
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
pub fn generate_sine_wave(
    num_points: usize,
    amplitude: f64,
    frequency: f64,
    length: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut positions = DMatrix::zeros(3, num_points);
    let mut times = DVector::zeros(num_points);

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let x = t * length; // X: linear progression
        positions[(0, i)] = x;
        positions[(1, i)] = amplitude * (frequency * x).sin(); // Y: sine wave
        positions[(2, i)] = 0.0; // Z: flat
        times[i] = t;
    }

    (positions, times)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn generate_lorenz_data(num_points: usize) -> (DMatrix<f32>, DVector<f32>) {
        let mut positions = DMatrix::zeros(3, num_points);
        let mut time = DVector::zeros(num_points);

        let mut x = 1.0;
        let mut y = 1.0;
        let mut z = 1.0;

        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;
        let dt = 0.01;

        for i in 0..num_points {
            let dx = sigma * (y - x);
            let dy = x * (rho - z) - y;
            let dz = x * y - beta * z;

            x += dx * dt;
            y += dy * dt;
            z += dz * dt;

            positions[(0, i)] = x as f32;
            positions[(1, i)] = y as f32;
            positions[(2, i)] = z as f32;
            time[i] = i as f32 * dt as f32;
        }

        (positions, time)
    }

    #[test]
    fn test_lorenz_data_generation() {
        let (positions, times) = generate_lorenz_data(1000);

        // Convert to f64 for the animation function
        let positions_f64 = positions.cast::<f64>();
        let times_f64 = times.cast::<f64>();

        // Verify data structure
        assert_eq!(positions_f64.nrows(), 3);
        assert_eq!(positions_f64.ncols(), 1000);
        assert_eq!(times_f64.len(), 1000);

        // Verify time progression
        assert!(times_f64[1] > times_f64[0]);
        assert!(times_f64[999] > times_f64[0]);

        println!("Lorenz data generated successfully for animation");
        println!(
            "First point: ({}, {}, {})",
            positions_f64[(0, 0)],
            positions_f64[(1, 0)],
            positions_f64[(2, 0)]
        );
        println!(
            "Last point: ({}, {}, {})",
            positions_f64[(0, 999)],
            positions_f64[(1, 999)],
            positions_f64[(2, 999)]
        );
    }

    #[test]
    fn test_geometric_shapes() {
        // Test line
        let (line_pos, line_times) = generate_line(100);
        assert_eq!(line_pos.nrows(), 3);
        assert_eq!(line_pos.ncols(), 100);
        assert_eq!(line_times.len(), 100);

        // Test circle
        let (circle_pos, _circle_times) = generate_circle(200, 2.0);
        assert_eq!(circle_pos.nrows(), 3);
        assert_eq!(circle_pos.ncols(), 200);

        // Test helix
        let (helix_pos, _helix_times) = generate_helix(300, 1.5, 5.0, 3.0);
        assert_eq!(helix_pos.nrows(), 3);
        assert_eq!(helix_pos.ncols(), 300);

        // Test sine wave
        let (sine_pos, _sine_times) = generate_sine_wave(150, 2.0, 3.0, 10.0);
        assert_eq!(sine_pos.nrows(), 3);
        assert_eq!(sine_pos.ncols(), 150);

        println!("All geometric shapes generated successfully");

        // Uncomment to test animations:
        // create_3d_animation(line_pos, line_times, None, None);
        // create_3d_animation(circle_pos, circle_times, None, None);
        // create_3d_animation(helix_pos, helix_times, None, None);
        // create_3d_animation(sine_pos, sine_times, None, None);
    }
}

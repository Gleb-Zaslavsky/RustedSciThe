//! # Bevy 2D Multi-Plot System
//!
//! This module provides a configurable 2D plotting system using the Bevy game engine.
//! Bevy uses an Entity Component System (ECS) architecture which may seem unusual
//! to traditional Rust developers but provides powerful performance and flexibility.
//!
//! ## Key Bevy Concepts Used:
//! - **Systems**: Functions that run every frame or at startup
//! - **Resources**: Global data accessible to all systems
//! - **Components**: Data attached to entities
//! - **Queries**: Way to access entities with specific components
//! - **Commands**: Deferred operations to spawn/despawn entities

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*; // Bevy's prelude imports common types
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Color enumeration for plot lines with predefined color palette
/// Each variant maps to specific RGB values for consistent plotting
#[derive(Debug, Clone, Copy)]
pub enum PlotColor {
    Black,
    White,
    Red,
    Green,
    Blue,
    Yellow,
    Cyan,
    Magenta,
    DarkRed,
    DarkGreen,
    DarkBlue,
    LightRed,
    LightGreen,
    LightBlue,
    Orange,
    Purple,
    Brown,
    Pink,
    Gray,
    DarkGray,
    LightGray,
}

impl PlotColor {
    /// Convert PlotColor enum to Bevy's Color type
    /// Bevy uses sRGB color space with values from 0.0 to 1.0
    pub fn to_bevy_color(self) -> Color {
        match self {
            PlotColor::Black => Color::srgb(0.0, 0.0, 0.0),
            PlotColor::White => Color::srgb(1.0, 1.0, 1.0),
            PlotColor::Red => Color::srgb(1.0, 0.0, 0.0),
            PlotColor::Green => Color::srgb(0.0, 0.8, 0.0),
            PlotColor::Blue => Color::srgb(0.0, 0.0, 1.0),
            PlotColor::Yellow => Color::srgb(1.0, 1.0, 0.0),
            PlotColor::Cyan => Color::srgb(0.0, 1.0, 1.0),
            PlotColor::Magenta => Color::srgb(1.0, 0.0, 1.0),
            PlotColor::DarkRed => Color::srgb(0.5, 0.0, 0.0),
            PlotColor::DarkGreen => Color::srgb(0.0, 0.5, 0.0),
            PlotColor::DarkBlue => Color::srgb(0.0, 0.0, 0.5),
            PlotColor::LightRed => Color::srgb(1.0, 0.5, 0.5),
            PlotColor::LightGreen => Color::srgb(0.5, 1.0, 0.5),
            PlotColor::LightBlue => Color::srgb(0.5, 0.5, 1.0),
            PlotColor::Orange => Color::srgb(1.0, 0.5, 0.0),
            PlotColor::Purple => Color::srgb(0.5, 0.0, 0.5),
            PlotColor::Brown => Color::srgb(0.6, 0.3, 0.1),
            PlotColor::Pink => Color::srgb(1.0, 0.7, 0.8),
            PlotColor::Gray => Color::srgb(0.5, 0.5, 0.5),
            PlotColor::DarkGray => Color::srgb(0.3, 0.3, 0.3),
            PlotColor::LightGray => Color::srgb(0.8, 0.8, 0.8),
        }
    }

    /// Returns a default sequence of colors for automatic plot coloring
    /// Used when no custom color mapping is provided
    pub fn default_sequence() -> Vec<PlotColor> {
        vec![
            PlotColor::Red,
            PlotColor::Green,
            PlotColor::Blue,
            PlotColor::Orange,
            PlotColor::Purple,
            PlotColor::Cyan,
            PlotColor::Magenta,
            PlotColor::Brown,
            PlotColor::Pink,
            PlotColor::DarkRed,
            PlotColor::DarkGreen,
            PlotColor::DarkBlue,
        ]
    }
}

/// Configuration structure for multi-plot display
/// All fields are public for easy customization
pub struct MultiPlotter {
    /// Height of each subplot in relative units
    pub subplot_height: f32,
    /// Vertical spacing between subplots
    pub subplot_spacing: f32,
    /// Number of tick marks on X axis
    pub nticks_x: usize,
    /// Number of tick marks on Y axis  
    pub nticks_y: usize,
    /// Font size for axis labels and numbers
    pub axis_font_size: f32,
    /// Width of plot lines (simulated by drawing multiple offset lines)
    pub line_width: f32,
    /// Optional custom color mapping for specific plot labels
    pub colors: Option<HashMap<String, PlotColor>>,
    /// Optional custom label for X axis (defaults to "x")
    pub x_axis_label: Option<String>,
}

impl Default for MultiPlotter {
    fn default() -> Self {
        Self {
            subplot_height: 2.0,
            subplot_spacing: 0.1,
            nticks_x: 6,
            nticks_y: 5,
            axis_font_size: 12.0,
            line_width: 1.0,
            colors: None,
            x_axis_label: None,
        }
    }
}

/// Bevy Resource containing all plot data
/// Resources are global data accessible to all systems
/// The #[derive(Resource)] macro tells Bevy this can be used as a resource
#[derive(Resource)]
struct MultiPlotData {
    /// X-axis data points (shared across all plots)
    x: DVector<f64>,
    /// Y-axis data matrix (rows = different plots, columns = data points)
    ys: DMatrix<f64>,
    /// Labels for each plot (corresponds to matrix rows)
    labels: Vec<String>,
    /// Configuration settings for the plotter
    config: MultiPlotter,
}

/// Bevy Component for camera control state
/// Components are data attached to entities (in this case, the camera entity)
/// The #[derive(Component)] macro tells Bevy this can be attached to entities
#[derive(Component)]
struct CameraController {
    /// Current zoom level (higher = more zoomed in)
    zoom: f32,
    /// Camera pan offset from origin
    offset: Vec2,
}

impl MultiPlotter {
    /// Create a new MultiPlotter with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Main entry point to create and run the plotting application
    /// This consumes self and starts a Bevy App that runs until window is closed
    pub fn plot_static_multiplot(self, x: DVector<f64>, ys: DMatrix<f64>, labels: Vec<String>) {
        App::new() // Create new Bevy application
            .add_plugins(DefaultPlugins) // Add default Bevy plugins (windowing, rendering, etc.)
            .insert_resource(ClearColor(Color::srgb(1.0, 1.0, 1.0))) // Set white background
            .insert_resource(MultiPlotData {
                x,
                ys,
                labels,
                config: self,
            }) // Add our plot data as resource
            .add_systems(Startup, (setup_camera, setup_axis_labels)) // Systems that run once at startup
            .add_systems(Update, (draw_static_multiplot, camera_controller_system)) // Systems that run every frame
            .run(); // Start the application loop
    }
}

/// Convenience function to plot with default settings
/// Creates a default MultiPlotter and immediately plots the data
pub fn plot_static_multiplot(x: DVector<f64>, ys: DMatrix<f64>, labels: Vec<String>) {
    MultiPlotter::default().plot_static_multiplot(x, ys, labels);
}

/// Bevy startup system to initialize the 2D camera
/// Systems are functions with specific parameter types that Bevy calls automatically
/// - Commands: Used to spawn/despawn entities and components
/// - Res<T>: Immutable access to a resource of type T
fn setup_camera(mut commands: Commands, plot_data: Res<MultiPlotData>) {
    // Extract data from the resource
    let x = &plot_data.x;
    let ys = &plot_data.ys;
    let nplots = ys.nrows();

    // Calculate data bounds for auto-fitting camera
    let x_min = x.min();
    let x_max = x.max();
    let x_range = x_max - x_min;

    let subplot_height = plot_data.config.subplot_height;
    let subplot_spacing = plot_data.config.subplot_spacing;
    let total_height = nplots as f64 * (subplot_height + subplot_spacing) as f64;

    // Calculate initial zoom to fit all content with padding
    let padding = 0.1;
    let width_ratio = x_range;
    let height_ratio = total_height;
    let zoom = 10.0 / (width_ratio.max(height_ratio) * (1.0 + padding)) as f32;

    // Spawn camera entity with required components
    // In Bevy, entities are created by spawning a bundle of components
    commands.spawn((
        Camera2d,                           // Camera component for 2D rendering
        Transform::from_xyz(0.0, 0.0, 1.0), // Position and orientation component
        CameraController {
            // Our custom component for camera control
            zoom,
            offset: Vec2::ZERO,
        },
    ));
}

/// Bevy system for handling camera controls (pan and zoom)
/// This runs every frame and handles user input
/// - MessageReader<T>: Bevy's event system for input events
/// - Res<ButtonInput<T>>: Resource for checking button states
/// - Query<T, F>: Access entities with components T, filtered by F
fn camera_controller_system(
    mut mouse_motion: MessageReader<MouseMotion>, // Mouse movement events
    mut scroll_events: MessageReader<MouseWheel>, // Mouse wheel events
    mouse_input: Res<ButtonInput<MouseButton>>,   // Mouse button states
    mut camera_query: Query<(&mut Transform, &mut CameraController), With<Camera2d>>, // Query for camera entities
) {
    // Iterate through all entities that have both Transform and CameraController components
    for (mut transform, mut controller) in camera_query.iter_mut() {
        // Handle panning with left mouse button
        if mouse_input.pressed(MouseButton::Left) {
            // Process all mouse motion events from this frame
            for motion in mouse_motion.read() {
                let pan_speed = 2.0;
                // Adjust pan speed based on zoom level (more zoomed = slower pan)
                controller.offset.x -= motion.delta.x * pan_speed / controller.zoom;
                controller.offset.y += motion.delta.y * pan_speed / controller.zoom;
            }
        }

        // Handle zooming with mouse wheel
        for scroll in scroll_events.read() {
            controller.zoom *= 1.0 - scroll.y * 0.1; // 10% zoom per scroll step
            controller.zoom = controller.zoom.clamp(0.001, 10000.0); // Prevent extreme zoom levels
        }

        // Apply camera state to Transform component
        // Transform controls the actual camera position and scale in Bevy
        transform.translation.x = controller.offset.x;
        transform.translation.y = controller.offset.y;
        transform.scale = Vec3::splat(1.0 / controller.zoom); // Higher zoom = smaller scale
    }

    // Clear mouse motion events when not panning to prevent accumulation
    if !mouse_input.pressed(MouseButton::Left) {
        mouse_motion.clear();
    }
}

/// Main rendering system that draws plots, axes, and labels every frame
/// This is an Update system, meaning it runs every frame
/// - Gizmos: Bevy's immediate-mode drawing API for lines and shapes
/// - Commands: Used to spawn text entities for labels
fn draw_static_multiplot(mut gizmos: Gizmos, mut commands: Commands, plot: Res<MultiPlotData>) {
    // Extract plot data and configuration
    let x = &plot.x;
    let ys = &plot.ys;

    let nplots = ys.nrows(); // Number of subplots
    let npoints = x.len(); // Number of data points per plot

    // Layout calculations
    let subplot_height = plot.config.subplot_height;
    let subplot_spacing = plot.config.subplot_spacing;
    let total_height = nplots as f32 * (subplot_height + subplot_spacing);
    let top_offset = total_height / 2.0 - subplot_spacing / 2.0; // Center the plot grid

    // Axis tick configuration
    let nticks_x = plot.config.nticks_x;
    let nticks_y = plot.config.nticks_y;

    // Draw each subplot
    for r in 0..nplots {
        let y = ys.row(r); // Get Y data for this subplot

        // Calculate data bounds for normalization
        let min_x = x.min();
        let max_x = x.max();
        let min_y = y.min();
        let max_y = y.max();
        let xrange = (max_x - min_x).max(1e-9); // Prevent division by zero
        let yrange = (max_y - min_y).max(1e-9);

        // Calculate vertical position for this subplot
        let vshift = -(r as f32) * (subplot_height + subplot_spacing) + top_offset;
        let axis_y_world = vshift * 80.0; // Convert to world coordinates

        // Closure to map data coordinates to screen coordinates
        let map = |xi: f64, yi: f64| -> Vec2 {
            // Normalize to [-1, 1] range
            let xn = ((xi - min_x) / xrange * 2.0 - 1.0) as f32;
            let yn = ((yi - min_y) / yrange * 2.0 - 1.0) as f32;
            // Scale to screen coordinates and offset for subplot position
            Vec2::new(xn * 200.0, yn * 100.0 + axis_y_world)
        };

        // Draw coordinate axes using Bevy's Gizmos API
        let axis_color = Color::srgb(0.0, 0.0, 0.0); // Black axes
        // Horizontal axis (X-axis)
        gizmos.line_2d(
            Vec2::new(-220.0, axis_y_world),
            Vec2::new(220.0, axis_y_world),
            axis_color,
        );
        // Vertical axis (Y-axis)
        gizmos.line_2d(
            Vec2::new(0.0, axis_y_world - 120.0),
            Vec2::new(0.0, axis_y_world + 120.0),
            axis_color,
        );

        // Draw X-axis ticks and numeric labels
        for i in 0..=nticks_x {
            let t = i as f64 / nticks_x as f64; // Parameter from 0 to 1
            let xv = min_x + xrange * t; // Actual data value
            let wx = (((xv - min_x) / xrange * 2.0 - 1.0) as f32) * 200.0; // Screen X coordinate
            let wy = axis_y_world; // Screen Y coordinate (on axis)

            // Draw vertical tick mark
            gizmos.line_2d(Vec2::new(wx, wy - 6.0), Vec2::new(wx, wy + 6.0), axis_color);

            // Spawn text entity for numeric label
            // In Bevy, text is rendered by spawning entities with Text components
            commands.spawn((
                Text2d(format!("{:.2}", xv)),            // Text content
                Transform::from_xyz(wx, wy - 18.0, 0.1), // Position (Z=0.1 for layering)
                TextFont {
                    font_size: plot.config.axis_font_size,
                    ..default() // Use default font
                },
                TextColor(Color::srgb(0.1, 0.1, 0.1)), // Dark gray text
            ));
        }

        // Draw Y-axis ticks and numeric labels
        for i in 0..=nticks_y {
            let t = i as f64 / nticks_y as f64; // Parameter from 0 to 1
            let yv = min_y + yrange * t; // Actual data value
            let wy = (((yv - min_y) / yrange * 2.0 - 1.0) as f32) * 100.0 + axis_y_world; // Screen Y coordinate
            let wx_left = -6.0;
            let wx_right = 6.0;

            // Draw horizontal tick mark centered on Y-axis
            gizmos.line_2d(Vec2::new(wx_left, wy), Vec2::new(wx_right, wy), axis_color);

            // Spawn text entity for numeric label (positioned to left of axis)
            commands.spawn((
                Text2d(format!("{:.2}", yv)),
                Transform::from_xyz(-240.0, wy - 6.0, 0.1), // Left of Y-axis
                TextFont {
                    font_size: plot.config.axis_font_size,
                    ..default()
                },
                TextColor(Color::srgb(0.1, 0.1, 0.1)),
            ));
        }

        // Plot lines
        let color = if let Some(ref color_map) = plot.config.colors {
            // Use custom colors if specified
            color_map
                .get(&plot.labels[r])
                .map(|c| c.to_bevy_color())
                .unwrap_or_else(|| {
                    let default_colors = PlotColor::default_sequence();
                    default_colors[r % default_colors.len()].to_bevy_color()
                })
        } else {
            // Use default color sequence
            let default_colors = PlotColor::default_sequence();
            default_colors[r % default_colors.len()].to_bevy_color()
        };
        for i in 0..npoints - 1 {
            let p1 = map(x[i], y[i]);
            let p2 = map(x[i + 1], y[i + 1]);
            // Draw multiple lines for line width effect
            for offset in 0..(plot.config.line_width as i32) {
                let offset_f = offset as f32 - plot.config.line_width * 0.5;
                gizmos.line_2d(
                    Vec2::new(p1.x + offset_f, p1.y + offset_f * 0.5),
                    Vec2::new(p2.x + offset_f, p2.y + offset_f * 0.5),
                    color,
                );
            }
        }
    }
}

/// Bevy startup system to create axis labels
/// This runs once at startup to create persistent text entities
fn setup_axis_labels(mut commands: Commands, plot_data: Res<MultiPlotData>) {
    let nplots = plot_data.ys.nrows();
    let subplot_height = plot_data.config.subplot_height;
    let subplot_spacing = plot_data.config.subplot_spacing;

    // Calculate layout to match draw_static_multiplot positioning
    let total_height = nplots as f32 * (subplot_height + subplot_spacing);
    let top_offset = total_height / 2.0 - subplot_spacing / 2.0;

    let label_color = Color::srgb(0.1, 0.1, 0.1); // Dark gray for labels

    // Create axis labels for each subplot
    for (r, label) in plot_data.labels.iter().enumerate() {
        // Calculate vertical position to match draw_static_multiplot
        let vshift = -(r as f32) * (subplot_height + subplot_spacing) + top_offset;

        // Convert to world coordinates (must match draw_static_multiplot scaling)
        let vshift_world = vshift * 80.0;
        let subplot_half_world = subplot_height * 80.0 * 0.5;

        // Create X-axis label (centered below each subplot)
        let x_label = plot_data.config.x_axis_label.as_deref().unwrap_or("x");
        commands.spawn((
            Text2d(x_label.to_string()),
            Transform::from_xyz(0.0, vshift_world - subplot_half_world - 10.0, 0.1),
            TextFont {
                font_size: 18.0, // Slightly larger for axis labels
                ..default()
            },
            TextColor(label_color),
        ));

        // Create Y-axis label (positioned left of each subplot)
        // Adjust position based on label length to prevent overlap
        let label_length = label.len() as f32;
        let base_offset = -280.0; // Base position left of Y-axis
        let char_width = 8.0; // Approximate character width in pixels
        let label_offset = base_offset - (label_length * char_width);
        commands.spawn((
            Text2d(format!("{}", label)),
            Transform::from_xyz(label_offset, vshift_world, 0.1),
            TextFont {
                font_size: plot_data.config.axis_font_size,
                ..default()
            },
            TextColor(label_color),
        ));
    }
}

use RustedSciThe::Utils::bevy_2d::{MultiPlotter, PlotColor};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

fn main() {  
    println!("Testing static function plotting with Bevy 2D...");

    // Generate sample data
    let n = 200;
    let x = DVector::from_iterator(n, (0..n).map(|i| i as f64 * 0.05));

    // Create multiple functions to plot
    let mut ys = DMatrix::zeros(4, n);
    for i in 0..n {
        let xi = x[i];
        ys[(0, i)] = xi.sin(); // sin(x)
        ys[(1, i)] = xi.cos(); // cos(x)
        ys[(2, i)] = 0.5 * (2.0 * xi).sin(); // 0.5*sin(2x)
        ys[(3, i)] = xi.exp() * 0.1; // 0.1*exp(x)
    }

    let names = vec![
        "sin(x)".to_string(),
        "cos(x)".to_string(),
        "0.5*sin(2x)".to_string(),
        "0.1*exp(x)".to_string(),
    ];

    println!("Plotting {} functions with {} points each", names.len(), n);
    println!("Use left mouse button to pan, scroll wheel to zoom");

    // Create custom color mapping
    let mut colors = HashMap::new();
    colors.insert("sin(x)".to_string(), PlotColor::Red);
    colors.insert("cos(x)".to_string(), PlotColor::DarkGreen);
    colors.insert("0.5*sin(2x)".to_string(), PlotColor::Purple);
    colors.insert("0.1*exp(x)".to_string(), PlotColor::Orange);

    // Launch the interactive plot with custom settings
    let plotter = 
    MultiPlotter {
        subplot_height: 3.0,      // Taller subplots
        subplot_spacing: 4.0,     // More spacing between plots
        nticks_x: 8,             // More X ticks
        nticks_y: 6,             // More Y ticks
        axis_font_size: 14.0,    // Larger font
        line_width: 2.0,         // Thicker lines
        colors: Some(colors),    // Custom colors
        x_axis_label: Some("Y".to_string()),
        ..Default::default()};     // Default settings for other options
    plotter
    .plot_static_multiplot(x, ys, names);
}

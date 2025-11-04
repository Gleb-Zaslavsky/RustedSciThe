use RustedSciThe::Utils::bevy_2d::{MultiPlotter, PlotColor};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
use std::collections::HashMap;

fn main() {  
    println!("Mathematical Functions Visualization");
    println!("===================================");

    // Example 1: Trigonometric functions
    //example_trigonometric();

    // Uncomment to run other examples:
    example_exponential_logarithmic();
    // example_polynomial();
}

#[allow(dead_code)]
fn example_trigonometric() {
    println!("Plotting trigonometric functions...");

    let n = 300;
    let x = DVector::from_iterator(n, (0..n).map(|i| i as f64 * 4.0 * PI / n as f64));

    let mut ys = DMatrix::zeros(3, n);
    for i in 0..n {
        let xi = x[i];
        ys[(0, i)] = xi.sin();
        ys[(1, i)] = xi.cos();
        ys[(2, i)] = (xi / 2.0).tan().clamp(-3.0, 3.0); // Clamp tan to avoid extreme values
    }

    let names = vec![
        "sin(x)".to_string(),
        "cos(x)".to_string(),
        "tan(x/2)".to_string(),
    ];

    // Test with minimal spacing and small fonts
    let mut colors = HashMap::new();
    colors.insert("sin(x)".to_string(), PlotColor::Blue);
    colors.insert("cos(x)".to_string(), PlotColor::LightGreen);
    colors.insert("tan(x/2)".to_string(), PlotColor::Magenta);

    MultiPlotter {
        subplot_height: 1.5,
        subplot_spacing: 2.0,
        nticks_x: 4,
        nticks_y: 3,
        axis_font_size: 10.0,
        line_width: 1.5,
        colors: Some(colors),
        ..Default::default()
    }.plot_static_multiplot(x, ys, names);
}

#[allow(dead_code)]
fn example_exponential_logarithmic() {
    println!("Plotting exponential and logarithmic functions...");

    let n = 200;
    let x = DVector::from_iterator(n, (1..=n).map(|i| i as f64 * 0.1));

    let mut ys = DMatrix::zeros(3, n);
    for i in 0..n {
        let xi = x[i];
        ys[(0, i)] = (-xi).exp();
        ys[(1, i)] = xi.ln();
        ys[(2, i)] = xi.sqrt();
    }

    let names = vec![
        "exp(-x)".to_string(),
        "ln(x)".to_string(),
        "sqrt(x)".to_string(),
    ];

    // Test with large spacing and big fonts
    MultiPlotter {
        subplot_height: 4.0,
        subplot_spacing: 5.0,
        nticks_x: 10,
        nticks_y: 8,
        axis_font_size: 16.0,
        line_width: 3.0,
        colors: None, // Use default colors
        ..Default::default()
    }.plot_static_multiplot(x, ys, names);
}

#[allow(dead_code)]
fn example_polynomial() {
    println!("Plotting polynomial functions...");

    let n = 100;
    let x = DVector::from_iterator(n, (0..n).map(|i| (i as f64 - 50.0) * 0.1));

    let mut ys = DMatrix::zeros(4, n);
    for i in 0..n {
        let xi = x[i];
        ys[(0, i)] = xi; // Linear
        ys[(1, i)] = xi * xi; // Quadratic
        ys[(2, i)] = xi * xi * xi; // Cubic
        ys[(3, i)] = xi * xi * xi * xi; // Quartic
    }

    let names = vec![
        "x".to_string(),
        "x²".to_string(),
        "x³".to_string(),
        "x⁴".to_string(),
    ];

    // Test with custom colors for polynomials
    let mut colors = HashMap::new();
    colors.insert("x".to_string(), PlotColor::Black);
    colors.insert("x²".to_string(), PlotColor::DarkRed);
    colors.insert("x³".to_string(), PlotColor::DarkBlue);
    colors.insert("x⁴".to_string(), PlotColor::Brown);

    MultiPlotter {
        subplot_height: 2.5,
        subplot_spacing: 3.5,
        nticks_x: 7,
        nticks_y: 5,
        axis_font_size: 13.0,
        line_width: 2.5,
        colors: Some(colors),
        ..Default::default()
    }.plot_static_multiplot(x, ys, names);
}

#![allow(warnings)]


use splines::{Spline, Key, Interpolation};
fn interpolate() {
    let keys = vec![
        Key::new(0.0, 0.0, Interpolation::Linear),
        Key::new(1.0, 2.0, Interpolation::Linear),
        Key::new(2.0, 0.0, Interpolation::Linear),
    ];

    let spline = Spline::from_vec(keys);

    let interpolated_value = spline.clamped_sample(1.5);
    match interpolated_value {
        Some(value) => println!("Interpolated value at 1.5: {}", value),
        None => println!("Time is out of bounds"),
    }
    let extrapolated_value = spline.sample(3.0);
    match extrapolated_value {
        Some(value) => println!("Extrapolated value at 3.0: {}", value),
        None => println!("Time is out of bounds"),
    }
}

/* 
impl Interpolate<f64> for Vec<f64> {
   
    
}

#[derive(Debug)]
struct SplinePoint {
    x: f64,
    y: f64,
    dy: f64, // Derivative at the point
}

#[derive(Debug)]
struct CubicSplineSegment {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    x: f64, // Starting x of the segment
}
fn calculate_spline_coefficients(points: &[SplinePoint]) -> Vec<CubicSplineSegment> {
    let n = points.len() - 1;
    let mut segments = Vec::with_capacity(n);

    for i in 0..n {
        let x0 = points[i].x;
        let x1 = points[i + 1].x;
        let y0 = points[i].y;
        let y1 = points[i + 1].y;
        let dy0 = points[i].dy;
        let dy1 = points[i + 1].dy;

        let h = x1 - x0;
        let a = y0;
        let b = dy0;
        let c = (3.0 * (y1 - y0) / h - 2.0 * dy0 - dy1) / h;
        let d = (2.0 * (y0 - y1) / h + dy0 + dy1) / (h * h);

        segments.push(CubicSplineSegment { a, b, c, d, x: x0 });
    }

    segments
}
fn evaluate_spline(segments: &[CubicSplineSegment], x: f64) -> f64 {
    for segment in segments {
        if x >= segment.x && x <= segment.x + (segment.b / segment.c).abs() {
            let dx = x - segment.x;
            return segment.a + segment.b * dx + segment.c * dx * dx + segment.d * dx * dx * dx;
        }
    }
    panic!("x value out of range");
}

use splines::interpolate::{ Interpolation};

#[derive(Debug, Clone, PartialEq)]
pub struct Key<T> {
    pub time: f64,
    pub value: T,
    pub interpolation: Interpolation,
    // Add the following fields for Bézier interpolation
    pub in_tangent: Option<(T, T)>, // (x, y) for 2D, (x, y, z) for 3D, etc.
    pub out_tangent: Option<(T, T)>, // (x, y) for 2D, (x, y, z) for 3D, etc.
}

use splines::{Spline, Key, Interpolation};

impl<T: Clone + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Add<Output = T>> Spline<T> {
    pub fn from_vec(keys: Vec<Key<T>>) -> Self {
        let mut segments = Vec::new();

        for window in keys.windows(3) {
            let (p0, p1, p2) = (window[0].value.clone(), window[1].value.clone(), window[2].value.clone());
            let (t0, t1, t2) = (window[0].time, window[1].time, window[2].time);

            // Calculate Bézier coefficients
            let a = p0;
            let b = -3.0 * p0 + 3.0 * p1;
            let c = 3.0 * p0 - 6.0 * p1 + 3.0 * p2;
            let d = -p0 + 3.0 * p1 - 3.0 * p2 + p2;

            segments.push(Segment {
                a, b, c, d,
                t0, t1, t2,
            });
        }

        Spline { segments }
    }
}

struct Segment<T> {
    a: T,
    b: T,
    c: T,
    d: T,
    t0: f64,
    t1: f64,
    t2: f64,
}
impl<T: Clone + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Add<Output = T>> Spline<T> {
    // ...

    pub fn sample(&self, t: f64) -> Option<T> {
        for segment in &self.segments {
            if t >= segment.t0 && t <= segment.t1 {
                let u = (t - segment.t0) / (segment.t1 - segment.t0);
                let u2 = u * u;
                let u3 = u2 * u;

                let value = segment.a + segment.b * u + segment.c * u2 + segment.d * u3;
                return Some(value);
            }
        }

        None
    }
}

let keys = vec![
    Key {
        time: 0.0,
        value: 0.0,
        interpolation: Interpolation::Bezier(0.5),
        in_tangent: Some((0.0, 1.0)),
        out_tangent: Some((0.0, 1.0)),
    },
    Key {
        time: 1.0,
        value: 2.0,
        interpolation: Interpolation::Linear,
        in_tangent: None,
        out_tangent: None,
    },
    Key {
        time: 2.0,
        value: 0.0,
        interpolation: Interpolation::Linear,
        in_tangent: None,
        out_tangent: None,
    },
];

let spline = Spline::from_vec(keys);

let interpolated_value = spline.clamped_sample(0.5);
match interpolated_value {
    Some(value) => println!("Interpolated value at 0.5: {}", value),
    None => println!("Time is out of bounds"),
}

/*

    let points = vec![
        SplinePoint { x: 0.0, y: 0.0, dy: 1.0 },
        SplinePoint { x: 1.0, y: 1.0, dy: 0.0 },
        SplinePoint { x: 2.0, y: 0.0, dy: -1.0 },
    ];

    let segments = calculate_spline_coefficients(&points);

    let x = 1.5;
    let y = evaluate_spline(&segments, x);

    println!("Spline value at x = {}: y = {}", x, y);
*/
fn B() {
    let keys = vec![
        Key::new(0.0, 0.0, Interpolation::Bezier(0.5) ),
        Key::new(1.0, 2.0, Interpolation::Linear),
        Key::new(2.0, 0.0, Interpolation::Linear),
    ];

    let spline = Spline::from_vec(keys);

    let interpolated_value = spline.clamped_sample(0.5);
    match interpolated_value {
        Some(value) => println!("Interpolated value at 0.5: {}", value),
        None => println!("Time is out of bounds"),
    }
}


*/
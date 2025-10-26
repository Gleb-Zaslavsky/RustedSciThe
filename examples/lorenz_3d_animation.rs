use RustedSciThe::Utils::animation_3d::{create_3d_animation, generate_line};
use RustedSciThe::numerical::BDF::BDF_api::ODEsolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
#[allow(dead_code)]
fn generate_lorenz_data(num_points: usize) -> (DMatrix<f64>, DVector<f64>) {
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

        positions[(0, i)] = x;
        positions[(1, i)] = y;
        positions[(2, i)] = z;
        time[i] = i as f64 * dt;
    }

    (positions, time)
}

fn bdf_lorentz(t_bound: f64) -> (DMatrix<f64>, DVector<f64>) {
    // Lorenz system: x' = σ(y-x), y' = x(ρ-z)-y, z' = xy-βz
    let eq1 = Expr::parse_expression("10*(y-x)");
    let eq2 = Expr::parse_expression("x*(28-z)-y");
    let eq3 = Expr::parse_expression("x*y-8*z/3");
    let eq_system = vec![eq1, eq2, eq3];
    let values = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    // let t_bound = 5.0;
    let max_step = 0.001;
    let rtol = 1e-8;
    let atol = 1e-10;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    assert_eq!(solver.get_status(), "finished");

    let (t, y_result) = solver.get_result();
    (y_result.transpose(), t)
}

fn main() {
    println!("Generating  3D...");
    let (positions, times) = bdf_lorentz(50.0); // generate_lorenz_data(2000);
    //  generate_line(100);
    println!("{} points generated", times.len());

    println!("Use mouse to rotate the view, scroll to zoom");

    create_3d_animation(
        positions,
        times,
        Some(("X".to_string(), "Y".to_string(), "Z".to_string())),
        Some(100.0),
    );
}

use RustedSciThe::Utils::animation_2d::create_2d_animation;
use RustedSciThe::numerical::BDF::BDF_api::ODEsolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn bdf_lorentz_2d(t_bound: f64) -> (DMatrix<f64>, DVector<f64>) {
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
    let max_step = 0.001;
    let rtol = 1e-8;
    let atol = 1e-10;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    assert_eq!(solver.get_status(), "finished");

    let (t, y_result) = solver.get_result();
    
    // Extract only X and Y rows from 3D solution (y_result is 3×N)
    let positions_2d = y_result.transpose().rows(0, 2).into_owned(); // Take first 2 rows
    
    (positions_2d, t)
}

fn main() {
    println!("Generating Lorenz 2D projection (X-Y plane)...");
    let (positions, times) = bdf_lorentz_2d(50.0);
    println!("{} points generated", times.len());
    println!("{:?} solution shape ", positions.shape());
    println!("Use mouse to pan the view, scroll to zoom");

    create_2d_animation(
        positions,
        times,
        Some(("X".to_string(), "Y".to_string())),
        Some(100.0),
    );
}
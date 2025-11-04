/*
r_i' = v_i
v_i' = G * Σ (m_j * (r_j - r_i) / |r_j - r_i|^3) for j ≠ i
let's scale
ms= m / M
rs = r / L
ts = t / T
vs = v / (L / T)
where M is a characteristic mass, L is a characteristic length, T is a characteristic time
then we have
r_i_s' = v_i_s
v_i_s'*(L/T^2) = G * Σ (M*m_j_s *L* (r_j_s - r_i_s) / |r_j_s - r_i_s|^3*L^3) for j ≠ i =>
v_i_s' = (G * M * T^2 / L^3) * Σ (m_j_s * (r_j_s - r_i_s) / |r_j_s - r_i_s|^3) for j ≠ i
G = 6.6743 × 10^-11 м3 кг-1 с-2
M = 1.989×10^30
L = 1.495978707×10^11 m
T = 365.25*24*3600 s = 3.15576×10^7 s
k = G * M * T^2 / L^3 = 6.6743e-11 * 1.989e30 * (3.15576e7)^2 / (1.495978707e11)^3 = 39.47841760435743 this is close to 4π^2

*/
use RustedSciThe::Utils::animation_2d::create_2d_animation;

use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

use RustedSciThe::numerical::ODE_api2::SolverType;
use RustedSciThe::numerical::Radau::Radau_main::RadauOrder;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

#[allow(non_snake_case)]
fn main() {
    let k = 39.47841760435743;
    let m0 = 1.0;
    let m1 = 0.5;
    let m2 = 0.75;
    let m_sum = m0 + m1 + m2;

    let map_of_parameters: HashMap<String, f64> = HashMap::from([
        ("k".to_string(), k),
        ("m0".to_string(), m0),
        ("m1".to_string(), m1),
        ("m2".to_string(), m2),
    ]);
    let R01 = Expr::parse_expression(" ((x0 - x1)^2+(y0 - y1)^2)^0.5");
    let R02 = Expr::parse_expression(" ((x0 - x2)^2+(y0 - y2)^2)^0.5");
    let R12 = Expr::parse_expression(" ((x1 - x2)^2+(y1 - y2)^2)^0.5");
    println!("R01: {}", R01);

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // equations for body 0
    let eq_vx0 = Expr::parse_expression("-k * (m1*(x0 - x1)/R01^3 + m2*(x0 - x2)/R02^3     ) ");
    let eq_x0 = Expr::parse_expression("vx0");
    let eq_vy0 = Expr::parse_expression("-k * (m1*(y0 - y1)/R01^3 + m2*(y0 - y2)/R02^3     ) ");
    let eq_y0 = Expr::parse_expression("vy0");
    let eq_vx0 = eq_vx0
        .substitute_variable("R01", &R01)
        .substitute_variable("R02", &R02)
        .set_variable_from_map(&map_of_parameters);
    let eq_vy0 = eq_vy0
        .substitute_variable("R01", &R01)
        .substitute_variable("R02", &R02)
        .set_variable_from_map(&map_of_parameters);
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // equations for body 1
    let eq_vx1 = Expr::parse_expression("-k * (m0*(x1 - x0)/R01^3 + m2*(x1 - x2)/R12^3     ) ");
    let eq_x1 = Expr::parse_expression("vx1");
    let eq_vy1 = Expr::parse_expression("-k * (m0*(y1 - y0)/R01^3 + m2*(y1 - y2)/R12^3     ) ");
    let eq_y1 = Expr::parse_expression("vy1");
    let eq_vx1 = eq_vx1
        .substitute_variable("R01", &R01)
        .substitute_variable("R12", &R12)
        .set_variable_from_map(&map_of_parameters);
    let eq_vy1 = eq_vy1
        .substitute_variable("R01", &R01)
        .substitute_variable("R12", &R12)
        .set_variable_from_map(&map_of_parameters);
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // equations for body 2
    let eq_vx2 = Expr::parse_expression("-k * (m0*(x2 - x0)/R02^3 + m1*(x2 - x1)/R12^3     ) ");
    let eq_x2 = Expr::parse_expression("vx2");
    let eq_vy2 = Expr::parse_expression("-k * (m0*(y2 - y0)/R02^3 + m1*(y2 - y1)/R12^3     ) ");
    let eq_y2 = Expr::parse_expression("vy2");
    let eq_vx2 = eq_vx2
        .substitute_variable("R12", &R12)
        .substitute_variable("R02", &R02)
        .set_variable_from_map(&map_of_parameters);
    let eq_vy2 = eq_vy2
        .substitute_variable("R02", &R02)
        .substitute_variable("R12", &R12)
        .set_variable_from_map(&map_of_parameters);
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    let eq_sys = vec![
        eq_x0,
        eq_vx0.clone(),
        eq_y0,
        eq_vy0.clone(),
        eq_x1,
        eq_vx1.clone(),
        eq_y1,
        eq_vy1,
        eq_x2,
        eq_vx2.clone(),
        eq_y2,
        eq_vy2,
    ];
    let unknowns = vec![
        "x0".to_string(),
        "vx0".to_string(),
        "y0".to_string(),
        "vy0".to_string(),
        "x1".to_string(),
        "vx1".to_string(),
        "y1".to_string(),
        "vy1".to_string(),
        "x2".to_string(),
        "vx2".to_string(),
        "y2".to_string(),
        "vy2".to_string(),
    ];
    for (yi, eq) in unknowns.clone().iter().zip(eq_sys.clone().iter()) {
        println!("d({})/dt = {}", yi, eq);
    }
    let m0e = Expr::Const(m0);
    let m1e = Expr::Const(m1);
    let m2e = Expr::Const(m2);
    let center_mass_eq_v = (m0e * eq_vx0 + m1e * eq_vx1 + m2e * eq_vx2) / Expr::Const(m_sum);
    let center_mass_eq_v = center_mass_eq_v.simplify_();
    println!(
        "\n \n Center of mass velocity equation: {}",
        center_mass_eq_v
    );
    if m0 == m2 && m1 == m2 {
        // if all masses are equal, the center of mass velocity should be zero at any time
        let Y: Vec<&str> = unknowns.iter().map(|x| x.as_str()).collect();
        let center_mass_eq_v_fun =
            center_mass_eq_v.lambdify_borrowed_thread_safe(Y.as_slice());
        let x0 = 1.0;
        let y0 = 0.0;
        let x1 = 0.0;
        let y1 = 0.0;
        let x2 = 20.0;
        let y2 = 0.0;
        let vx0 = 0.0;
        let vy0 = 0.0;
        let vx1 = 0.0;
        let vy1 = 5.0;
        let vx2 = -0.4;
        let vy2 = 0.0;
        let vect = vec![x0, vx0, y0, vy0, x1, vx1, y1, vy1, x2, vx2, y2, vy2];
        let test_value = center_mass_eq_v_fun(vect.as_slice());
        println!(
            "Center of mass velocity when all masses are equal: {}",
            test_value
        );
        assert!(test_value.abs() < 1e-10);
    }

    // solver parameters
    let arg = "t".to_string();

    // initial conditions
    let t0 = 0.0;
    let x0i = 0.0;
    let y0i = 0.0;
    let x1i = 1.0;
    let y1i = 0.0;
    let x2i = 0.0;
    let y2i = 33.0;
    let vx0i = 0.0;
    let vy0i = 0.0;
    let vx1i = 0.0;
    let vy1i = 5.0;
    let vx2i = -0.4;
    let vy2i = 0.0;
    let vect = vec![
        x0i, vx0i, y0i, vy0i, x1i, vx1i, y1i, vy1i, x2i, vx2i, y2i, vy2i,
    ];
    let y0 = DVector::from_vec(vect.clone());

    let t_bound = 5.0;
    let max_step = 0.001;
    let rtol = 1e-7;
    let atol = 1e-8;
    let solver_name = "BDF".to_string();
    let now = Instant::now();
    let mut solver = match solver_name.as_str() {
        "BDF" => {
            UniversalODESolver::bdf(eq_sys, unknowns, arg, t0, y0, t_bound, max_step, rtol, atol)
        }

        "Radau" => {
            let mut solver = UniversalODESolver::new(
                eq_sys,
                unknowns,
                arg,
                SolverType::Radau(RadauOrder::Order5),
                t0,
                y0,
                t_bound,
            );
            solver.set_max_iterations(1000);
            solver.set_parallel(true);
            solver.set_tolerance(1e-4);
            solver.set_step_size(1e-3);
            solver.initialize();
            solver
        }
        "RK45" => UniversalODESolver::rk45(eq_sys, unknowns, arg, t0, y0, t_bound, 1e-6),
        "AB4" => UniversalODESolver::rk45(eq_sys, unknowns, arg, t0, y0, t_bound, 1e-6),
        "BE" => {
            let solver = UniversalODESolver::backward_euler(
                eq_sys,
                unknowns,
                arg,
                t0,
                y0,
                t_bound,
                1e-6,
                50,
                Some(1e-3),
            );
            solver
        }
        _ => panic!("Unsupported solver: {}", solver_name),
    };

    solver.solve();

    assert_eq!(solver.get_status().unwrap(), "finished");
    println!("Solver time: {:.2?}", now.elapsed().as_millis() as f64);
    let (t, y_result) = solver.get_result();
    let y_result = y_result.unwrap();
    let t = t.unwrap();
    let y_result = y_result.transpose();

    let x0 = y_result.row(0);
    let vx0: DVector<f64> = y_result.row(1).transpose();
    let y0 = y_result.row(2);
    let vy0: DVector<f64> = y_result.row(3).transpose();
    let x1 = y_result.row(4);
    let vx1: DVector<f64> = y_result.row(5).transpose();
    let y1 = y_result.row(6);
    let vy1: DVector<f64> = y_result.row(7).transpose();
    let x2 = y_result.row(8);
    let vx2: DVector<f64> = y_result.row(9).transpose();
    let y2 = y_result.row(10);
    let vy2: DVector<f64> = y_result.row(11).transpose();
    assert_eq!(x0.len(), t.len());
    assert_eq!(y0.len(), t.len());
    assert_eq!(x1.len(), t.len());
    assert_eq!(y1.len(), t.len());
    assert_eq!(x2.len(), t.len());
    assert_eq!(y2.len(), t.len());
    // check initial conditions
    /*
    use approx::assert_relative_eq;
    assert_relative_eq!(x0[0], vect[0], epsilon = 1e-4);
    assert_relative_eq!(y0[0], vect[2], epsilon = 1e-4);
    assert_relative_eq!(x1[0], vect[4], epsilon = 1e-4);
    assert_relative_eq!(y1[0], vect[6], epsilon = 1e-4);
    assert_relative_eq!(x2[0], vect[8], epsilon = 1e-4);
    assert_relative_eq!(y2[0], vect[10], epsilon = 1e-4);
    */
    assert_eq!(vx0.len(), t.len());
    assert_eq!(vy0.len(), t.len());
    assert_eq!(vx1.len(), t.len());
    assert_eq!(vy1.len(), t.len());
    assert_eq!(vx2.len(), t.len());
    assert_eq!(vy2.len(), t.len());

    // center of mass
    let x_center = (m0 * &x0 + m1 * &x1 + m2 * &x2) / m_sum;
    let y_center = (m0 * &y0 + m1 * &y1 + m2 * &y2) / m_sum;
    let _positions_of_0 = DMatrix::from_rows(&[x0, y0]);
    let _positions_of_1 = DMatrix::from_rows(&[x1, y1]);
    let _positions_of_2 = DMatrix::from_rows(&[x2, y2]);
    let positions_of_center = DMatrix::from_rows(&[x_center, y_center]);
    assert_eq!(positions_of_center.ncols(), t.len());
    assert_eq!(positions_of_center.nrows(), 2);
    if m0 == m2 && m1 == m2 {
        println!("Checking center of mass stationary...");
        // Center of mass should be stationary if all masses are equal
        // initial center of mass velocity calculation
        let vx_c: f64 = (vx0i + vx1i + vx2i) / 3.0;
        let vy_c: f64 = (vy0i + vy1i + vy2i) / 3.0;
        // initial center of mass position
        let x_c0 = (x0i + x1i + x2i) / 3.0;
        let y_c0 = (y0i + y1i + y2i) / 3.0;

        for (i, positions_of_center_i) in positions_of_center.column_iter().enumerate() {
            let t_i = t[i];
            let x0_expected = x_c0 + vx_c * t_i;
            let y0_expected = y_c0 + vy_c * t_i;
            let x = positions_of_center_i[0];
            let y = positions_of_center_i[1];
            let r = ((x / x0_expected - 1.0).powi(2) + (y / y0_expected - 1.0).powi(2)).sqrt();
            if r > 1e-3 {
                println!("i = {}, center of mass position deviation r = {}", i, r)
            };
            // assert!(r < 1e-2, "Center of mass is not stationary, r = {}, i = {}", r, i);
        }
    }
    // Total energy conservation check (kinetic + potential)
    println!("\n \n Checking total energy conservation...");
    let v0_2 = vx0.component_mul(&vx0) + vy0.component_mul(&vy0);
    let v1_2 = vx1.component_mul(&vx1) + vy1.component_mul(&vy1);
    let v2_2 = vx2.component_mul(&vx2) + vy2.component_mul(&vy2);

    // Calculate distances between bodies at each time step
    let dx01 = x0.transpose() - x1.transpose();
    let dy01 = y0.transpose() - y1.transpose();
    let r01 = (dx01.component_mul(&dx01) + dy01.component_mul(&dy01)).map(|x| x.sqrt());

    let dx02 = x0.transpose() - x2.transpose();
    let dy02 = y0.transpose() - y2.transpose();
    let r02 = (dx02.component_mul(&dx02) + dy02.component_mul(&dy02)).map(|x| x.sqrt());

    let dx12 = x1.transpose() - x2.transpose();
    let dy12 = y1.transpose() - y2.transpose();
    let r12 = (dx12.component_mul(&dx12) + dy12.component_mul(&dy12)).map(|x| x.sqrt());

    // Kinetic energy
    let kinetic_energy = 0.5 * (m0 * v0_2 + m1 * v1_2 + m2 * v2_2);

    // Potential energy: U = -k*(m0*m1/r01 + m0*m2/r02 + m1*m2/r12)
    // Since k = G*M in scaled units
    let potential_energy = -k
        * (m0 * m1 * r01.map(|r| 1.0 / r)
            + m0 * m2 * r02.map(|r| 1.0 / r)
            + m1 * m2 * r12.map(|r| 1.0 / r));

    // Total energy
    let total_energy = kinetic_energy + potential_energy;
    let total_energy0 = total_energy[0];

    for (i, te) in total_energy.iter().enumerate() {
        let rel_diff = ((*te - total_energy0) / total_energy0).abs();
        if rel_diff > 1e-3 {
            println!("i = {}, Total E = {:.6},", i, te,);
        }
    }

    println!("Creating animation for body 0...");
    let times = t;
    create_2d_animation(
        _positions_of_2,
        times,
        Some(("X".to_string(), "Y".to_string())),
        Some(100.0),
    );
}

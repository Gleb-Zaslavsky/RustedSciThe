#[cfg(test)]
mod tests {
    use super::super::Shooting_simple::BoundaryConditionType;
    use super::super::Shooting_sym_wrap::BVPShooting;
    use crate::symbolic::symbolic_engine::Expr;

    use crate::numerical::Radau::Radau_main::RadauOrder;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;
    use simplelog::*;
    use std::collections::HashMap;
    fn init_logger() {
        let _ = SimpleLogger::init(LevelFilter::Debug, Config::default());
    }
    use nalgebra::DVector;
    #[test]
    fn test_bc_create_dirichlet_dirichlet() {
        init_logger();

        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);

        let bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        let (left_bc, right_bc) = bvp.BC_create();
        println!("left_bc: {:?}, right_bc {:?}", left_bc, right_bc);
        assert_eq!(left_bc.value, 0.0);
        assert_eq!(right_bc.value, 1.0);
        assert!(matches!(left_bc.bc_type, BoundaryConditionType::Dirichlet));
        assert!(matches!(right_bc.bc_type, BoundaryConditionType::Dirichlet));
    }

    #[test]
    fn test_bc_create_dirichlet_neumann() {
        init_logger();

        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0)]);
        boundary_conditions.insert("y1".to_string(), vec![(1, 2.0)]);

        let bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        let (left_bc, right_bc) = bvp.BC_create();

        assert_eq!(left_bc.value, 0.0);
        assert_eq!(right_bc.value, 2.0);
        assert!(matches!(left_bc.bc_type, BoundaryConditionType::Dirichlet));
        assert!(matches!(right_bc.bc_type, BoundaryConditionType::Neumann));
    }

    #[test]
    fn test_simple_linear_bvp() {
        init_logger();

        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        bvp.simple_solve(1.0, 1e-6, 100, 0.01);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;
        assert_abs_diff_eq!(boundpoint[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(boundpoint[1], 1.0, epsilon = 1e-5);
        let y = sol.y;
        let x = sol.x_mesh;
        let y0: DVector<f64> = y.row(0).transpose().into_owned();
        let y1: DVector<f64> = y.row(1).transpose().into_owned();
        for i in 0..x.len() {
            assert_abs_diff_eq!(y0[i], x[i], epsilon = 1e-5);
            assert_abs_diff_eq!(y1[i], 1.0, epsilon = 1e-5);
        }
    }
    #[test]
    fn test_simple_linear_bvp2() {
        init_logger();

        // y'' = 0, y(0) = 0, y'(1) = 1 (solution: y = x)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0)]);
        boundary_conditions.insert("y1".to_string(), vec![(1, 1.0)]);
        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        bvp.simple_solve(1.0, 1e-6, 100, 0.01);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;
        assert_abs_diff_eq!(boundpoint[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(boundpoint[1], 1.0, epsilon = 1e-5);
        let y = sol.y;
        let x = sol.x_mesh;
        let y0: DVector<f64> = y.row(0).transpose().into_owned();
        let y1: DVector<f64> = y.row(1).transpose().into_owned();
        for i in 0..x.len() {
            assert_abs_diff_eq!(y0[i], x[i], epsilon = 1e-5);
            assert_abs_diff_eq!(y1[i], 1.0, epsilon = 1e-5);
        }
    }
    #[test]
    fn test_exponential_bvp() {
        init_logger();

        // y'' - y = 0, y(0) = 0, y(1) = sinh(1)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0_f64.sinh())]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        bvp.simple_solve(1.0, 1e-8, 100, 0.001);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;

        assert_abs_diff_eq!(boundpoint[0], 1.0_f64.sinh(), epsilon = 1e-6);
        assert_abs_diff_eq!(boundpoint[1], 1.0_f64.cosh(), epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_bc_bvp() {
        init_logger();

        // y'' = 0, y(0) = 1, y'(1) = 2 (solution: y = 2x + 1)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 1.0)]);
        boundary_conditions.insert("y1".to_string(), vec![(1, 2.0)]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));
        bvp.simple_solve(2.0, 1e-6, 100, 0.01);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;

        assert_abs_diff_eq!(boundpoint[0], 3.0, epsilon = 1e-5);
        assert_abs_diff_eq!(boundpoint[1], 2.0, epsilon = 1e-5);
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        // println!("y = {:?}", y);
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y[i], 2.0 * x_mesh[i] + 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_harmonic_oscillator() {
        init_logger();

        // y'' + y = 0, y(0) = 1, y(π/2) = 0 (solution: y = cos(x))
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 1.0), (1, 0.0)]);

        let mut bvp = BVPShooting::new(
            eq_vec,
            values,
            arg,
            boundary_conditions,
            (0.0, std::f64::consts::PI / 2.0),
        );
        bvp.simple_solve(0.0, 1e-6, 100, 0.01);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;

        assert_abs_diff_eq!(boundpoint[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(boundpoint[1], -1.0, epsilon = 1e-5); // y'(π/2) = -sin(π/2) = -1
    }
    #[test]
    fn test_linear_with_bdf() {
        use crate::numerical::ODE_api2::{SolverParam, SolverType};
        init_logger();

        // y'' = 0, y(0) = 0, y'(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        //  boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0)]);

        boundary_conditions.insert("y1".to_string(), vec![(1, 1.0)]);
        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));

        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-6)),
            ("atol".to_string(), SolverParam::Float(1e-8)),
            ("max_step".to_string(), SolverParam::Float(1e-3)),
        ]);

        bvp.solve_with_certain_ivp(0.5, 1e-8, 100, 0.001, SolverType::BDF, params);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y0: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let y1: DVector<f64> = y_mesh.row(1).transpose().into_owned();
        println!("sol shape = {:}", y0.len());
        // println!("y0 = {:}", y0.transpose());
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y0[i], x_mesh[i], epsilon = 1e-2);
            assert_abs_diff_eq!(y1[i], 1.0, epsilon = 1e-2);
        }
        assert_abs_diff_eq!(boundpoint[0], 1.0, epsilon = 1e-2);
    }
    #[test]
    fn test_linear_with_bdf2() {
        use crate::numerical::ODE_api2::{SolverParam, SolverType};
        init_logger();

        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));

        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-6)),
            ("atol".to_string(), SolverParam::Float(1e-8)),
            ("max_step".to_string(), SolverParam::Float(1e-3)),
        ]);

        bvp.solve_with_certain_ivp(0.5, 1e-8, 100, 0.001, SolverType::BDF, params);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y0: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let y1: DVector<f64> = y_mesh.row(1).transpose().into_owned();

        println!("sol shape = {:}", y0.len());
        // println!("y0 = {:}", y0.transpose());
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y0[i], x_mesh[i], epsilon = 1e-2);
            assert_abs_diff_eq!(y1[i], 1.0, epsilon = 1e-2);
        }
        assert_abs_diff_eq!(boundpoint[0], 1.0, epsilon = 1e-2);
    }
    #[test]
    fn test_solve_exp_with_bdf() {
        use crate::numerical::ODE_api2::{SolverParam, SolverType};
        init_logger();

        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0, y(1) = sinh(1)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0_f64.sinh())]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));

        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-3)),
            ("atol".to_string(), SolverParam::Float(1e-3)),
            ("max_step".to_string(), SolverParam::Float(1e-4)),
        ]);

        bvp.solve_with_certain_ivp(0.5, 1e-8, 100, 1e-5, SolverType::BDF, params);
        let sol = bvp.get_solution();
        let boundpoint = sol.bound_values;

        //println!("y = {:?}", y);
        // solution y(x) = 1/2 (e^x - e^(-x))
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y0: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let y1: DVector<f64> = y_mesh.row(1).transpose().into_owned();
        // println!("y = {:?}", y0);
        // println!("z = {:?}", z);
        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y0[i], exect, epsilon = 1e-3);
            let exect_z = (x_mesh[i].exp() + (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y1[i], exect_z, epsilon = 1e-3,);
        }
        assert_abs_diff_eq!(boundpoint[0], 1.0_f64.sinh(), epsilon = 1e-3);
        assert_abs_diff_eq!(boundpoint[1], 1.0_f64.cosh(), epsilon = 1e-3);
    }

    #[test]
    fn test_solve_linear_with_radau() {
        use crate::numerical::ODE_api2::{SolverParam, SolverType};
        init_logger();

        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));

        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-6)),
            ("atol".to_string(), SolverParam::Float(1e-8)),
        ]);

        bvp.solve_with_certain_ivp(
            1.0,
            1e-6,
            100,
            0.01,
            SolverType::Radau(RadauOrder::Order5),
            params,
        );
        let sol = bvp.get_solution();

        let boundpoint = sol.bound_values;
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y0: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let y1: DVector<f64> = y_mesh.row(1).transpose().into_owned();

        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y0[i], x_mesh[i], epsilon = 1e-5);
            assert_abs_diff_eq!(y1[i], 1.0, epsilon = 1e-5);
        }
        assert_abs_diff_eq!(boundpoint[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(boundpoint[1], 1.0, epsilon = 1e-5);
    }
    #[test]
    fn test_solve_exp_with_radau() {
        use crate::numerical::ODE_api2::{SolverParam, SolverType};
        init_logger();
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0, y(1) = sinh(1)
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0_f64.sinh())]);

        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, 1.0));

        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-6)),
            ("atol".to_string(), SolverParam::Float(1e-8)),
        ]);

        bvp.solve_with_certain_ivp(
            1.0,
            1e-6,
            100,
            0.01,
            SolverType::Radau(RadauOrder::Order5),
            params,
        );
        let sol = bvp.get_solution();

        let boundpoint = sol.bound_values;
        let x_mesh = bvp.get_x();
        let y_mesh = bvp.get_y();
        let y0: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let y1: DVector<f64> = y_mesh.row(1).transpose().into_owned();

        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y0[i], exect, epsilon = 1e-4);
            let exect_z = (x_mesh[i].exp() + (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y1[i], exect_z, epsilon = 1e-4,);
        }
        assert_abs_diff_eq!(boundpoint[0], 1.0_f64.sinh(), epsilon = 1e-6);
        assert_abs_diff_eq!(boundpoint[1], 1.0_f64.cosh(), epsilon = 1e-6);
    }

    #[test]
    fn test_solve_with_rk45() {
        use crate::numerical::ODE_api2::SolverType;
        init_logger();

        // y'' + y = 0, y(0) = 1, y(π/2) = 0 (solution: y = cos(x))
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 1.0), (1, 0.0)]);

        let mut bvp = BVPShooting::new(
            eq_vec,
            values,
            arg,
            boundary_conditions,
            (0.0, std::f64::consts::PI / 2.0),
        );

        let params = HashMap::new();

        bvp.solve_with_certain_ivp(
            1.0,
            1e-6,
            100,
            1e-4,
            SolverType::NonStiff("RK45".to_string()),
            params,
        );
        let sol = bvp.get_solution();
        let y = sol.y;
        let y0 = y.row(0).into_owned();
        let y1 = y.row(1).transpose().into_owned();
        let x_mesh = bvp.get_x();
        //println!("y0 = {:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-2);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-2);
        }
        // let boundpoint = sol.bound_values;

        //  assert_abs_diff_eq!(boundpoint[0], 0.0, epsilon = 1e-5);
        // assert_abs_diff_eq!(boundpoint[1], -1.0, epsilon = 1e-5);
    }
}

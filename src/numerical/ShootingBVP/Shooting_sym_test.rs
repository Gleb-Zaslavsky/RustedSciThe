#[cfg(test)]
mod tests {
    use super::super::Shooting_sym_wrap::BVPShooting;
    use super::super::Shooting_simple::{BoundaryConditionType, BoundaryCondition};
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;
    use approx::assert_abs_diff_eq;
    use simplelog::*;

    fn init_logger() {
        let _ = SimpleLogger::init(LevelFilter::Debug, Config::default());
    }

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
        
        assert_abs_diff_eq!(bvp.solution[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(bvp.solution[1], 1.0, epsilon = 1e-5);
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
        
        assert_abs_diff_eq!(bvp.solution[0], 1.0_f64.sinh(), epsilon = 1e-6);
        assert_abs_diff_eq!(bvp.solution[1], 1.0_f64.cosh(), epsilon = 1e-6);
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
        
        assert_abs_diff_eq!(bvp.solution[0], 3.0, epsilon = 1e-5); // y(1) = 2*1 + 1 = 1
        assert_abs_diff_eq!(bvp.solution[1], 2.0, epsilon = 1e-5); // y'(1) = 2
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
        
        let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, (0.0, std::f64::consts::PI / 2.0));
        bvp.simple_solve(0.0, 1e-6, 100, 0.01);
        
        assert_abs_diff_eq!(bvp.solution[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(bvp.solution[1], -1.0, epsilon = 1e-5); // y'(π/2) = -sin(π/2) = -1
    }
}
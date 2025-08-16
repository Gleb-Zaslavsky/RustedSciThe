#[cfg(test)]
mod tests {

    use crate::numerical::BVP_sci::BVP_sci_faer::solve_bvp;
    use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat};
    use crate::numerical::BVP_sci::BVP_sci_symb::BVPwrap;
    use crate::numerical::Examples_and_utils::NonlinEquation;

    use crate::symbolic::symbolic_engine::Expr;
    use faer::Row;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    Two-point boundary value problem:
    y'' = -(2/a)*(1 + 2*ln(y))*y
    y(-1) = exp(-1/a), y(1) = exp(-1/a)
    Exact solution: y(x) = exp(-x^2/a)
    */
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    #[test]
    fn exponential_bvp_bc_condition() {
        let a: f64 = 4.0; // Parameter for the equation
        let mut boundary_conditions = HashMap::new();
        let bc_val = ((-1.0 / a) as f64).exp();
        boundary_conditions.insert("y".to_string(), vec![(0usize, bc_val), (1usize, bc_val)]);
        let values = vec!["y".to_string(), "z".to_string()];
        let bc_from_sym = BVPwrap::BC_closure_creater(boundary_conditions, values);
        let bc_val = (-1.0 / a).exp();
        let bc = move |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0] - bc_val, // y(-1) = exp(-1/a)
                1 => yb[0] - bc_val, // y(1) = exp(-1/a)
                _ => 0.0,
            })
        };
        let x = faer_col::from_fn(5, |i| -1.0 + i as f64 * 0.5); // [-1, -0.5, 0, 0.5, 1]
        let bc_val = bc(&x, &x, &x);
        let bc_vsl_sym = bc_from_sym.unwrap()(&x, &x, &x);
        println!("bc_val: {:?}", bc_val);
        println!("bc_vsl_sym: {:?}", bc_vsl_sym);
        assert!(bc_vsl_sym == bc_val);
        assert!(bc_vsl_sym[1] == bc_val[1]);
    }

    #[test]
    fn test_exponential_bvp3() {
        let eqs = vec!["z", "-(2.0/4.0)*(1+2.0*ln( (y) ))*y"];
        let a: f64 = 4.0; // Parameter for the equation
        let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
        let two_point = NonlinEquation::TwoPointBVP;
        let eq_system = vec_eqs;
        let values = two_point.values();
        let mut boundary_conditions = HashMap::new();
        let bc_val = ((-1.0 / a) as f64).exp();
        boundary_conditions.insert("y".to_string(), vec![(0usize, bc_val), (1usize, bc_val)]);

        let start_and_end = two_point.span(None, None);
        let n = 5 as usize;
        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n);
        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-2,
            1000,
            initial_guess,
        );
        let mut Bounds = HashMap::new();
        Bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        bvp_solver.set_additional_parameters(Some(true), Some(Bounds), None);
        let (jacobian, residual_function, bc_func) = bvp_solver.eq_generate();
        ////////////////////////////////////////////////////////////////////////////////////////////////
        let a = 4.0;

        let _fun = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = (*y.get(0, j) as f64).max(1e-10); // Avoid log(0)
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = -(2.0 / a) * (1.0 + 2.0 * y_val.ln()) * y_val; // z' = -(2/a)*(1+2*ln(y))*y
            }
            f
        };

        let bc_val = (-1.0 / a).exp();
        let _bc = move |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0] - bc_val, // y(-1) = exp(-1/a)
                1 => yb[0] - bc_val, // y(1) = exp(-1/a)
                _ => 0.0,
            })
        };

        let x = faer_col::from_fn(5, |i| -1.0 + i as f64 * 0.5); // [-1, -0.5, 0, 0.5, 1]
        let mut y = faer_dense_mat::zeros(2, 5);
        for j in 0..5 {
            let exact_y = 0.0; // (-x_val * x_val / a).exp();
            let exact_z = 0.0; //-2.0 * x_val / a * exact_y;
            *y.get_mut(0, j) = exact_y;
            *y.get_mut(1, j) = exact_z;
        }

        let result = solve_bvp(
            &residual_function,
            &bc_func.unwrap(),
            x.clone(),
            y,
            None,
            None,
            Some(&*jacobian.unwrap()),
            None,
            1e-7,
            2000,
            2,
            None,
            None,
        );

        //   let result1 = result.clone().unwrap().y;
        //  println!("Result of direct solution: {:?}", result1);
        println!("is success {}", result.clone().unwrap().success);
        match result {
            Ok(res) => {
                if res.success {
                    for i in 0..res.x.nrows() {
                        let x_val = res.x[i];
                        let exact = (-x_val * x_val / a).exp();
                        let error = (*res.y.get(0, i) - exact).abs();
                        assert!(
                            error < 1e-7,
                            "Exponential BVP error at x={}: {} vs {}",
                            x_val,
                            res.y.get(0, i),
                            exact
                        );
                    }
                } else {
                    println!("Exponential BVP did not converge: {}", res.status);
                }
            }
            Err(_) => {}
        }
    }
    #[test]
    fn test_exponential_bvp_compare_residuals() {
        let eqs = vec!["z", "-(2.0/4.0)*(1+2.0*ln( (y) ))*y"];
        let a: f64 = 4.0; // Parameter for the equation
        let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
        let two_point = NonlinEquation::TwoPointBVP;
        let eq_system = vec_eqs;
        let values = two_point.values();
        let mut boundary_conditions = HashMap::new();
        let bc_val = ((-1.0 / a) as f64).exp();
        boundary_conditions.insert("y".to_string(), vec![(0usize, bc_val), (1usize, bc_val)]);

        let start_and_end = two_point.span(None, None);
        let n = 5 as usize;
        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n);
        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );
        let mut Bounds = HashMap::new();
        Bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        bvp_solver.set_additional_parameters(Some(true), Some(Bounds), None);
        bvp_solver.solve();
        //////////////////////////compare residual function derived from symbolic and directly defined/////////////////////
        let func_from_sym = &bvp_solver.residual_function;

        // Define the direct function for the BVP
        let fun_direct = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = (*y.get(0, j) as f64).max(1e-10); // Avoid log(0)
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = -(2.0 / a) * (1.0 + 2.0 * y_val.ln()) * y_val; // z' = -(2/a)*(1+2*ln(y))*y
            }
            f
        };
        let x_test = faer_col::from_fn(n, |i| {
            start_and_end.0 + i as f64 * (start_and_end.1 - start_and_end.0) / (n - 1) as f64
        });
        let mut initial_guess_test = faer_dense_mat::zeros(2, x_test.shape().0);
        for j in 0..x_test.shape().0 {
            let x_val = x_test[j];
            let exact_y = (-x_val * x_val / a).exp();
            let exact_z = -2.0 * x_val / a * exact_y;
            *initial_guess_test.get_mut(0, j) = exact_y;
            *initial_guess_test.get_mut(1, j) = exact_z;
        }

        let y_from_sym = func_from_sym(&x_test, &initial_guess_test, &faer_col::zeros(0));
        let y_direct = fun_direct(&x_test, &initial_guess_test, &faer_col::zeros(0));
        let y_from_sym0: Row<f64> = y_from_sym.row(0).to_owned();
        let y_direct0: Row<f64> = y_direct.row(0).to_owned();
        // Compare the two residuals
        for i in 0..y_from_sym0.shape().1 {
            let y_from_sym_val = y_from_sym0[i];
            let y_direct_val = y_direct0[i];
            println!("sim val: {}, direct val: {}", y_from_sym_val, y_direct_val);
            assert!(
                (y_from_sym_val - y_direct_val).abs() < 1e-5,
                "Residuals differ at index {}: {} vs {}",
                i,
                y_from_sym_val,
                y_direct_val
            );
        }
        ////////////////////////end of comparison of residuals////////////////////////////////////////////////////
    }

    #[test]
    fn test_two_point_bvp() {
        /*
        Two-point boundary value problem:
        y'' = -(2/a)*(1 + 2*ln(y))*y
        y(-1) = exp(-1/a), y(1) = exp(-1/a)
        Exact solution: y(x) = exp(-x^2/a)
        */
        let eqs = vec!["z", "-(2.0/4.0)*(1+2.0*ln( (y) ))*y"];
        let a: f64 = 4.0; // Parameter for the equation
        let vec_eqs: Vec<Expr> = Expr::parse_vector_expression(eqs);
        let two_point = NonlinEquation::TwoPointBVP;
        let eq_system = vec_eqs;
        let values = two_point.values();
        let mut boundary_conditions = HashMap::new();
        let bc_val = ((-1.0 / a) as f64).exp();
        boundary_conditions.insert("y".to_string(), vec![(0usize, bc_val), (1usize, bc_val)]);

        let start_and_end = two_point.span(None, None);
        let n = 5 as usize;
        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n);
        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-2,
            1000,
            initial_guess,
        );
        let mut Bounds = HashMap::new();
        Bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        bvp_solver.set_additional_parameters(Some(true), Some(Bounds), None);
        bvp_solver.solve();

        let x_mesh_final = bvp_solver.x_mesh.clone();
        let ressult = " exp(-x^2/4.0) ".to_string();
        let expr = Expr::parse_expression(&ressult);
        let exact_sol_func = expr.lambdify1D();
        let exact_solution: Vec<f64> = x_mesh_final.iter().map(|&x| exact_sol_func(x)).collect();
        println!(
            "\n Exact solution {:?}  of length {}: \n",
            exact_solution,
            exact_solution.len(),
        );
        let sol: DMatrix<f64> = bvp_solver.get_result().unwrap();
        let numer: DVector<f64> = sol.column(0).to_owned().into();
        println!(
            "\n Numerical solution: {:?} of length {} \n",
            numer,
            numer.len()
        );
        for i in 0..x_mesh_final.len() {
            let x_val = x_mesh_final[i];
            let y_numerical = numer[i];
            let y_exact = exact_solution[i];

            assert!(
                (y_numerical - y_exact).abs() < 1e-2,
                "Two-point BVP error at x = {}: {} vs {}",
                x_val,
                y_numerical,
                y_exact
            );
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    /*
    Clairaut's equation: y''' = (x-1)^2 + y^2 + y' - 2
    y(1) = 1, y'(1) = 0, y''(1) = 2
    Exact solution: y = 1 + (x-1)^2 - (1/6)(x-1)^3 + (1/12)(x-1)^4
    */
    //////////////////////////////////////////////////////////////////////////////////////////////
    #[test]
    fn test_clairaut_equation_bvp() {
        /*
        Clairaut's equation: y''' = (x-1)^2 + y^2 + y' - 2
        y(1) = 1, y'(1) = 0, y''(1) = 2
        Exact solution: y = 1 + (x-1)^2 - (1/6)(x-1)^3 + (1/12)(x-1)^4
        */
        let clairaut = NonlinEquation::Clairaut;
        let eq_system = clairaut.setup();
        let values = clairaut.values();
        let boundary_conditions = clairaut.boundary_conditions2();
        let start_and_end = clairaut.span(None, None);
        let n = 100 as usize;

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(3, n);

        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        bvp_solver.solve();
        let x_mesh_final = bvp_solver.x_mesh.clone();
        let ressult = " 1+ (x- 1)^2 - (1/6)*(x-1)^3 + (1/12)*(x-1)^4 ".to_string();
        let expr = Expr::parse_expression(&ressult);
        let exact_sol_func = expr.lambdify1D();
        let exact_solution: Vec<f64> = x_mesh_final.iter().map(|&x| exact_sol_func(x)).collect();

        let sol: DMatrix<f64> = bvp_solver.get_result().unwrap();
        let numer: DVector<f64> = sol.column(0).to_owned().into();
        let mut error = 0.0;
        for i in 0..x_mesh_final.len() {
            let y_numerical = numer[i];
            let y_exact = exact_solution[i];
            error += (y_numerical - y_exact).powi(2);
        }
        let error = error.sqrt() / x_mesh_final.len() as f64;
        println!(" weighted error: {}", error);
        assert!(error < 1e-2, "Clairaut BVP error: {}", error);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    Parachute Equation: y'' + ky'^2 - g = 0
    With initial conditions y(0) = 0 and y'(0) = 0
    y = (1/k)*(ln( (exp(2*(g*k).sqrt*t)+1 )/2) -(g*k).sqrt*t)

    Simplified version: y'' = -y'^2 + 1
     where k = -1.0 and g = -1.0
     y =  -(ln( (exp(t)+1 )/2) - t) = t - ln( (exp(t)+1 )/2)
    */
    #[test]
    fn parachute_bc_condition() {
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 0.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(0usize, 0.0f64)]);
        let values = vec!["z".to_string(), "y".to_string()];
        let bc_from_sym = BVPwrap::BC_closure_creater(BorderConditions, values);
        let bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0], // y(0) = 0
                1 => ya[1], // y'(0) = 0
                _ => 0.0,
            })
        };
        let x = faer_col::from_fn(5, |i| -1.0 + i as f64 * 0.5); // [-1, -0.5, 0, 0.5, 1]
        let bc_val = bc(&x, &x, &x);
        let bc_vsl_sym = bc_from_sym.unwrap()(&x, &x, &x);
        println!("bc_val: {:?}", bc_val);
        println!("bc_vsl_sym: {:?}", bc_vsl_sym);
        assert!(bc_vsl_sym == bc_val);
        assert!(bc_vsl_sym[1] == bc_val[1]);
    }
    #[test]
    fn test_parachute_equation_bvp_compare_residual() {
        // y' = z
        // y''= z' = -y^2 + 1
        let eqs = vec!["z", "-z^2 +1"];
        let parachute = NonlinEquation::ParachuteEquation;
        let eq_system = Expr::parse_vector_expression(eqs);
        let values = parachute.values();
        let boundary_conditions = parachute.boundary_conditions2();
        let start_and_end = parachute.span(None, None);
        let n = 5 as usize;

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, 5);

        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );
        let (_, residual_function_from_sym, _) = bvp_solver.eq_generate();
        ////////////////////////////////////// compare residual function derived from symbolic and directly defined ////////////////////
        let k = 1.0;
        let g = 1.0;

        let fun = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = *y.get(0, j);
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = g - k * y_val * y_val; // z' = g - k*z^2
            }
            f
        };
        let mut y = faer_dense_mat::zeros(2, 4);
        for j in 0..4 {
            *y.get_mut(0, j) = 0.0; // ;
            *y.get_mut(1, j) = 0.0;
        }
        let x_test = faer_col::from_fn(n, |i| {
            start_and_end.0 + i as f64 * (start_and_end.1 - start_and_end.0) / (n - 1) as f64
        });
        let residual_from_sym =
            residual_function_from_sym(&x_test, &y, &faer_col::from_fn(n, |_| 0.0));
        let residual_from_fun = fun(&x_test, &y, &faer_col::from_fn(n, |_| 0.0));
        println!("residual_from_sym: {:?}", residual_from_sym);
        println!("residual_from_fun: {:?}", residual_from_fun);
        assert!(residual_from_sym == residual_from_fun);
    }
    #[test]
    fn test_parachute_equation_bvp_1() {
        // using symbolic residual

        // y' = z
        // y''= z' = -y^2 + 1
        let eqs = vec!["z", "1-z^2"];
        let parachute = NonlinEquation::ParachuteEquation;
        let eq_system = Expr::parse_vector_expression(eqs);
        let values = parachute.values();
        let boundary_conditions = parachute.boundary_conditions2();
        let start_and_end = parachute.span(None, None);
        let n = 5 as usize;

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n);

        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );
        let (jacobian, residual_function_from_sym, bc_func) = bvp_solver.eq_generate();
        ////////////////////////////////////// compare residual function derived from symbolic and directly defined ////////////////////
        let k = 1.0;
        let g = 1.0;

        let _fun = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = g - k * z_val * z_val; // z' = g - k*z^2
            }
            f
        };

        let _bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0], // y(0) = 0
                1 => ya[1], // y'(0) = 0
                _ => 0.0,
            })
        };

        let x = faer_col::from_fn(4, |i| i as f64 * 0.2); // [0, 0.2, 0.4, 0.6]
        let mut y = faer_dense_mat::zeros(2, 4);
        for j in 0..4 {
            *y.get_mut(0, j) = 0.0; // ;
            *y.get_mut(1, j) = 0.0;
        }

        let result = solve_bvp(
            &residual_function_from_sym,
            &bc_func.unwrap(),
            x.clone(),
            y,
            None,
            None,
            Some(&*jacobian.unwrap()),
            None,
            1e-10,
            2000,
            2,
            None,
            None,
        );
        println!(" success: {}", result.clone().unwrap().success);
        match result {
            Ok(res) => {
                if res.success {
                    for i in 0..res.x.nrows() {
                        let t = res.x[i];
                        let sqrt_gk = (g * k).sqrt();
                        let exp_term = (2.0 * sqrt_gk * t).exp();
                        let exact = (1.0 / k) * (((exp_term + 1.0) / 2.0).ln() - sqrt_gk * t);
                        let error = (*res.y.get(0, i) - exact).abs();
                        assert!(
                            error < 1e-7,
                            "Parachute error at t={}: {} vs {}",
                            t,
                            res.y.get(0, i),
                            exact
                        );
                    }
                } else {
                    panic!("solution not found");
                }
            }
            Err(_) => {}
        }
    }

    #[test]
    fn test_parachute_equation_bvp_2() {
        // using symbolic residual

        // y' = z
        // y''= z' = -y^2 + 1
        let eqs = vec!["z", "1-z^2"];
        let parachute = NonlinEquation::ParachuteEquation;
        let eq_system = Expr::parse_vector_expression(eqs);
        let values = parachute.values();
        let boundary_conditions = parachute.boundary_conditions2();
        let start_and_end = parachute.span(None, None);
        let n = 5 as usize;

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n);

        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );
        bvp_solver.solve();
        //////////////////////////////////////////////////////////////////////////
        let g = 1.0;
        let k = 1.0;
        let x_mesh_final = bvp_solver.x_mesh.clone();
        let ressult = "x - ln( (exp(2.0*x) +1 )/2  )  ".to_string();
        let expr = Expr::parse_expression(&ressult);
        let exact_sol_func = expr.lambdify1D();
        let _exact_solution_: Vec<f64> = x_mesh_final.iter().map(|&x| exact_sol_func(x)).collect();

        let sol: DMatrix<f64> = bvp_solver.get_result().unwrap();
        let numer: DVector<f64> = sol.column(0).to_owned().into();

        for i in 0..x_mesh_final.len() {
            let x_val = x_mesh_final[i];
            let y_numerical = numer[i];
            let t = x_val;
            let sqrt_gk = (g * k as f64).sqrt();
            let exp_term = (2.0 * sqrt_gk * t).exp();
            let y_exact = (1.0 / k) * (((exp_term + 1.0) / 2.0).ln() - sqrt_gk * t);
            //let y_exact = -exact_solution_[i];

            assert!(
                (y_numerical - y_exact).abs() < 1e-5,
                "Parachute equation error at x = {}: {} vs {}",
                x_val,
                y_numerical,
                y_exact
            );
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     the Lane-Emden equation of index 5:
    y′′+2xy′+y**5=0,y(0)=1,y′(0)=0
    y'=z
    z'=- 2*x*z - y**5
    With initial conditions y(0)=1,y′(0)=0
    exact solution:
    y = (1+(x^2)/3)^(-0.5)
    */
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Lane Emden equation
    #[test]
    fn test_lane_emden_bvp_compare_residuals() {
        /*
         the Lane-Emden equation of index 5:
        y′′+2xy′+y**5=0,y(0)=1,y′(0)=0
        y'=z
        z'=- 2*x*z - y**5
        With initial conditions y(0)=1,y′(0)=0
        exact solution:
        y = (1+(x^2)/3)^(-0.5)
        */
        let lanemden = NonlinEquation::LaneEmden5;
        let eq_system = lanemden.setup();
        let values = lanemden.values();
        let boundary_conditions = lanemden.boundary_conditions2();
        let start_and_end = lanemden.span(None, None);
        let n = 100 as usize; // number of mesh points

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n); // initial guess for y and z
        // Use new method to create solver instance
        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            2000,
            initial_guess,
        );
        bvp_solver.set_additional_parameters(Some(true), None, None);

        let _ = bvp_solver.eq_generate();

        ///////////////////////////////////////////////////////////////////////////////////
        // direct solution /////////////////////////////

        let fun_direct = |x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = *y.get(0, j);
                let z_val = *y.get(1, j);
                let x_val = x[j];

                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = -2.0 * x_val * z_val - y_val.powi(5); // z' = -2*x*z - y^5
            }
            f
        };
        let fun_from_sym = bvp_solver.residual_function;
        let n = 100;
        let x_test = faer_col::from_fn(n, |i| {
            start_and_end.0 + i as f64 * (start_and_end.1 - start_and_end.0) / (n - 1) as f64
        });
        let mut y = faer_dense_mat::zeros(2, n);
        for j in 0..n {
            let x_val = x_test[j];
            let exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
            *y.get_mut(0, j) = exact;
            *y.get_mut(1, j) = -x_val / 3.0 * (1.0 + x_val * x_val / 3.0).powf(-1.5);
        }

        let res_direct = fun_direct(&x_test, &y, &faer_col::from_fn(1, |_| 0.0));
        let res_from_sym = fun_from_sym(&x_test, &y, &faer_col::from_fn(1, |_| 0.0));
        //  println!("res_direct: {:?}", res_direct);
        //  println!("res_from_sym: {:?}", res_from_sym);
        assert!(res_direct == res_from_sym);
    }
    #[test]
    #[should_panic]
    fn test_lane_emden_bvp() {
        /*
        (1/x^2)*d/dx(x^2*dy/dx) + y^5 = 0
        y''/x^2 + (2/x)y'+ y^5 = 0
        y'' + (2/x)y'+ y^5 = 0
        y' = z
        y'' = z' = -2/x*z - y^5
         the Lane-Emden equation of index 5:
        y′′+2y′/x + y**5=0,y(0)=1,y′(0)=0
        y'=z
        z'=- 2*z/x - y**5
        With initial conditions y(0)=1,y′(0)=0
        exact solution:
        y = (1+(x^2)/3)^(-0.5)

        */
        let lanemden = NonlinEquation::LaneEmden5;
        let eqs = vec!["z", "-2*z/x - y^5"];

        let eq_system = Expr::parse_vector_expression(eqs);
        let values = lanemden.values();
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        boundary_conditions.insert("y".to_string(), vec![(0usize, 0.0f64)]);

        let start_and_end = (0.0, 1.0);
        let n = 100 as usize; // number of mesh points

        let arg = "x".to_owned();
        let initial_guess = DMatrix::zeros(2, n); // initial guess for y and z
        // Use new method to create solver instance
        let mut bvp_solver = BVPwrap::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            arg,
            1e-8,
            2000,
            initial_guess,
        );

        bvp_solver.set_additional_parameters(Some(true), None, None);

        bvp_solver.solve();
        //  bvp_solver.plot_result();
        //  bvp_solver.gnuplot_result();
        //
        let x_mesh_final = bvp_solver.x_mesh.clone();

        ////////////////////////////////////////////////////////////////////////////////
        //
        let ressult = "(1+(x^2)/3)^(-0.5)".to_string();

        let expr = Expr::parse_expression(&ressult);
        let exact_sol_func = expr.lambdify1D();
        let _exact_solution: Vec<f64> = x_mesh_final.iter().map(|&x| exact_sol_func(x)).collect();
        //let exact_impl = |x_val:f64| (1.0 + x_val * x_val / 3.0).powf(-0.5);
        //let exact_solution: Vec<f64> = x_mesh_final.iter().map(|&x| exact_impl(x)).collect();
        let sol: DMatrix<f64> = bvp_solver.get_result().unwrap();
        let numer: DVector<f64> = sol.column(0).to_owned().into();

        for i in 0..x_mesh_final.len() {
            let x_val = x_mesh_final[i];
            let y_numerical = numer[i];
            let y_exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
            let error = (y_numerical - y_exact).abs();
            // exact_solution[i];

            println!(
                "x = {:.3}, y_num = {:.6}, y_exact = {:.6}, error = {:.2e}",
                x_val, y_numerical, y_exact, error
            );
        }
        let res = &bvp_solver.result.clone();

        if res.success {
            //let y = res.y.row(0).clone();
            for i in 0..res.x.nrows() {
                let x_val = res.x[i];
                let exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
                let error = (*res.y.get(0, i) - exact).abs();
                assert!(
                    error < 1e-4,
                    "Lane-Emden error at x[{}]={}: {} vs {}",
                    i,
                    x_val,
                    res.y.get(0, i),
                    exact
                );
            }
        } else {
            panic!("solution not found")
        }
    }
}

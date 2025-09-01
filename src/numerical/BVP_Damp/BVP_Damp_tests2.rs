#[cfg(test)]
mod tets {  
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::{DMatrix, DVector};
    use prettytable::{Table, row};
    use std::collections::HashMap;
    use std::time::Instant;
      #[test] 
    fn test_direct_eq2() {
        // variables
        let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unhnowns_Str:Vec<String> = unknowns_str.iter().map(|s| s.to_string()).collect();
        let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str);
        let Teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let C0 = unknowns[2].clone();
        let J0 = unknowns[3].clone();

        let J1 = unknowns[5].clone();
        // Parameters
        let Q = 3000.0 * 1e3 * 0.034;
        let dT = 600.0;
        let T_scale = 600.0;
        let L: f64 = 3e-4;
        let M0 = 34.2 / 1000.0;
        let Lambda = 0.07;

        let P = 2e6;
   
        let Tm = 1500.0;
        let C1_0 = 1.0;
        let T_initial = 1000.0;
        // problem settings

        // coefficients

 
 
        let Pe_q =  0.0090168  ;

        let D_ro = 2.88e-4  ;
        let Pe_D = 1.50e-3;
        let ro_m_ = M0 * P / (8.314 * Tm);
        // conversion to sym
        let dT_sym = Expr::Const(dT);
        let T_scale_sym = Expr::Const(T_scale);

        let Lambda_sym = Expr::Const(Lambda);
        let Q = Expr::Const(Q);
        let A = Expr::Const(1.3e5);
        let E = Expr::Const(5000.0 * 4.184);
        let M = Expr::Const(M0);
        let R_g = Expr::Const(8.314);
        let ro_m = Expr::Const(ro_m_);
        let qm = Expr::Const(L.powf(2.0) / T_scale);
        let qs = Expr::Const(L.powf(2.0));
        let Pe_q_sym = Expr::Const(Pe_q);
        let ro_D = Expr::Const(D_ro);
        let ro_D = vec![ro_D.clone(), ro_D.clone()];
        let Pe_D = vec![Expr::Const(Pe_D), Expr::Const(Pe_D)];
        let minus = Expr::Const(-1.0);
               let M_reag = Expr::Const(0.342);
        // EQ SYSTEM


        let Rate = A
            * Expr::exp(-E / (R_g * (Teta * T_scale_sym + dT_sym)))
            * C0
            * (ro_m.clone() / M_reag.clone());
        let eq_T = q.clone() / Lambda_sym;
        let eq_q = q * Pe_q_sym - Q * Rate.clone() * qm;
        let eq_C0 = J0.clone() / ro_D[0].clone();
        let eq_J0 = J0 * Pe_D[0].clone()
            - (M.clone() * minus * Rate.clone() * ro_m.clone() / M.clone()) * qs.clone();
        let eq_C1 = J1.clone() / ro_D[1].clone();
        let eq_J1 = J1 * Pe_D[1].clone() - (M.clone() * Rate * ro_m / M) * qs;
                
        let eqs = vec![eq_T, eq_q, eq_C0, eq_J0, eq_C1, eq_J1];
        let eq_and_unknowns = unknowns.clone().into_iter().zip(eqs.clone());
 
        // Pretty print coefficients table
        let mut coeff_table = Table::new();
        coeff_table.add_row(row!["Parameter", "Value"]);
        coeff_table.add_row(row!["Q", format!("{:.2e}", 102000.0)]);
        coeff_table.add_row(row!["dT", format!("{:.1}", 600.0)]);
        coeff_table.add_row(row!["L", format!("{:.2e}", L)]);
        coeff_table.add_row(row!["M0", format!("{:.4}", M0)]);
        coeff_table.add_row(row!["Lambda", format!("{:.3}", Lambda)]);
        coeff_table.add_row(row!["Pe_q", format!("{:.4e}", Pe_q)]);
        coeff_table.add_row(row!["Pe_D", format!("{:?}", Pe_D)]);
        coeff_table.add_row(row!["D_ro", format!("{:.2e}", D_ro)]);
            coeff_table.add_row(row!["ro", format!("{:.2e}", ro_m_)]);

        println!("\n=== COEFFICIENTS ===");
        coeff_table.printstd();

        // Pretty print equations table
        let mut eq_table = Table::new();
        eq_table.add_row(row!["Unknown", "Equation"]);

        for (unknown, equation) in eq_and_unknowns {
            eq_table.add_row(row![format!("{}", unknown), format!("{}", equation)]);
        }
        eq_table.printstd();

    

        ////////////////////////////////////////////
        //solver

            let Teta_initial = (T_initial - dT) / T_scale;
        let BoundaryConditions = HashMap::from([
            ("Teta".to_string(), vec![(0, Teta_initial)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(),  vec![(0, C1_0)]),
            ("J0".to_string(),  vec![(1, 1e-7)]),
            ("C1".to_string(),  vec![(0, 1e-3)]),
            ("J1".to_string(),  vec![(1, 1e-7)]),
        ]);

        let Bounds = HashMap::from([
            ("Teta".to_string(), (0.0, 10.0)),
            ("q".to_string(), (-1e20, 1e20)),
            ("C0".to_string(), (0.0, 1.5)),
            ("J0".to_string(), (-1e2, 1e2)),
            ("C1".to_string(), (0.0, 1.5)),
            ("J1".to_string(), (-1e2, 1e2)),
        ]);
        let n_steps = 20;
        let grid_method = GridRefinementMethod::GrcarSmooke(0.01, 0.01, 1.5);
        // or GridRefinementMethod::Pearson(0.05, 2.5);
        let adaptive =AdaptiveGridConfig{
            version: 1,
            max_refinements: 3,
            grid_method

        };
        let strategy_params = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: Some(0.5),
            adaptive: Some(adaptive),
        };
        
        let rel_tolerance =HashMap::from([
            ("Teta".to_string(), 1e-5),
            ("q".to_string(), 1e-5),
            ("C0".to_string(), 1e-5),
            ("J0".to_string(), 1e-5),
            ("C1".to_string(), 1e-5),
            ("J1".to_string(), 1e-5),
        ]);
                let ig = vec![0.99; n_steps * unknowns.len()];
        let initial_guess = DMatrix::from_vec(unknowns.len(), n_steps, ig);
        let max_iterations = 100;
        let abs_tolerance = 1e-6;
        let loglevel = Some("info".to_string());
        let scheme = "forward".to_string();
        let method = "Sparse".to_string();
        let strategy = "Damped".to_string();
        let linear_sys_method = None;

        // Using the new tolerance helper - much simpler!
        let mut bvp = NRBVP::new(
            eqs.clone(),
            initial_guess,
             unhnowns_Str,
            "x".to_string(),
            BoundaryConditions,
            0.0,
            1.0,
            n_steps,
            scheme,
            strategy,
            Some(strategy_params),
            linear_sys_method,
            method,
            abs_tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            loglevel,
        );
        bvp.dont_save_log(true);
        bvp.solve();
        bvp.gnuplot_result();
        let eq_and_unknowns = unknowns.clone().into_iter().zip(eqs.clone());
        for (unknown, equation) in  eq_and_unknowns{
            println!("unknown: {} | equation: {}", unknown, equation);
        }
        println!("\n=== EQUATIONS SYSTEM ===");
        coeff_table.printstd();
  
    }
    fn test_problem(grid_method: GridRefinementMethod) {
        // variables
        let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unhnowns_Str: Vec<String> = unknowns_str.iter().map(|s| s.to_string()).collect();
        let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str.clone());
        let Teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let C0 = unknowns[2].clone();
        let J0 = unknowns[3].clone();

        let J1 = unknowns[5].clone();
        // Parameters
        let Q = 3000.0 * 1e3 * 0.034;
        let dT = 600.0;
        let T_scale = 600.0;
        let L = 0.3e-4;
        let M0 = 34.2 / 1000.0;
        let Lambda = 0.07;
        let Cp = 0.35 * 4.184 * 1000.0;
        let P = 2e6;
        let T0 = 800.0;
        let Tm = 1500.0;
        let C1_0 = 1.0;
        let T_initial = 1000.0;
        // problem settings

        // coefficients
        let R_G = 8.314;
        let m = 0.077 * (P / 1e5 as f64).powf(0.748) / 100.0;
        let ro0 = M0 * P / (R_G * T0);
        let D = Lambda / (Cp * ro0);
        let Pe_q = (L * Cp * m) / Lambda;

        let D_ro = D * ro0 * (Tm / T0).powf(0.5);
        let Pe_D = m * L / D_ro;
        // conversion to sym
        let dT_sym = Expr::Const(dT);
        let T_scale_sym = Expr::Const(T_scale);

        let Lambda_sym = Expr::Const(Lambda);
        let Q = Expr::Const(Q);
        let A = Expr::Const(1.3e5);
        let E = Expr::Const(5000.0 * 4.184);
        let M = Expr::Const(M0);
        let R_g = Expr::Const(8.314);
        let ro_m = Expr::Const(M0 * P / (8.314 * Tm));
        let qm = Expr::Const(L.powf(2.0) / T_scale);
        let qs = Expr::Const(L.powf(2.0));
        let Pe_q_sym = Expr::Const(Pe_q);
        let ro_D = Expr::Const(D_ro);
        let ro_D = vec![ro_D.clone(), ro_D.clone()];
        let Pe_D = vec![Expr::Const(Pe_D), Expr::Const(Pe_D)];
        let minus = Expr::Const(-1.0);
        // EQ SYSTEM

        let Rate = A
            * Expr::exp(-E / (R_g * (Teta * T_scale_sym + dT_sym)))
            * C0
            * (ro_m.clone() / M.clone());
        let eq_T = q.clone() / Lambda_sym;
        let eq_q = q * Pe_q_sym - Q * Rate.clone() * qm;
        let eq_C0 = J0.clone() / ro_D[0].clone();
        let eq_J0 = J0 * Pe_D[0].clone()
            - (M.clone() * minus * Rate.clone() * ro_m.clone() / M.clone()) * qs.clone();
        let eq_C1 = J1.clone() / ro_D[1].clone();
        let eq_J1 = J1 * Pe_D[1].clone() - (M.clone() * Rate * ro_m / M) * qs;

        let eqs = vec![eq_T, eq_q, eq_C0, eq_J0, eq_C1, eq_J1];
        let eq_and_unknowns = unknowns.clone().into_iter().zip(eqs.clone());

        // Pretty print coefficients table
        let mut coeff_table = Table::new();
        coeff_table.add_row(row!["Parameter", "Value"]);
        coeff_table.add_row(row!["Q", format!("{:.2e}", 102000.0)]);
        coeff_table.add_row(row!["dT", format!("{:.1}", 600.0)]);
        coeff_table.add_row(row!["L", format!("{:.2e}", L)]);
        coeff_table.add_row(row!["M0", format!("{:.4}", M0)]);
        coeff_table.add_row(row!["Lambda", format!("{:.3}", Lambda)]);
        coeff_table.add_row(row!["Pe_q", format!("{:.4e}", Pe_q)]);
        coeff_table.add_row(row!["Pe_D", format!("{:?}", Pe_D)]);
        coeff_table.add_row(row!["D_ro", format!("{:.2e}", D_ro)]);

        println!("\n=== COEFFICIENTS ===");
        coeff_table.printstd();

        // Pretty print equations table
        let mut eq_table = Table::new();
        eq_table.add_row(row!["Unknown", "Equation"]);

        for (unknown, equation) in eq_and_unknowns {
            eq_table.add_row(row![format!("{}", unknown), format!("{}", equation)]);
        }

        println!("\n=== EQUATIONS SYSTEM ===");
        eq_table.printstd();

        ////////////////////////////////////////////
        //solver

        let Teta_initial = (T_initial - dT) / T_scale;
        let BoundaryConditions = HashMap::from([
            ("Teta".to_string(), vec![(0, Teta_initial)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, C1_0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);

        let Bounds = HashMap::from([
            ("Teta".to_string(), (-100.0, 100.0)),
            ("q".to_string(), (-1e20, 1e20)),
            ("C0".to_string(), (-1.0, 1.5)),
            ("J0".to_string(), (-1e20, 1e20)),
            ("C1".to_string(), (-1.0, 1.5)),
            ("J1".to_string(), (-1e20, 1e20)),
        ]);
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        let n_steps = 20;

        let config: AdaptiveGridConfig = AdaptiveGridConfig {
            version: 1,
            max_refinements: 5,
            grid_method: grid_method,
        };
        let strategy_params = Some(SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: Some(config),
        });
        let rel_tolerance = HashMap::from([
            ("Teta".to_string(), 1e-5),
            ("q".to_string(), 1e-5),
            ("C0".to_string(), 1e-5),
            ("J0".to_string(), 1e-5),
            ("C1".to_string(), 1e-5),
            ("J1".to_string(), 1e-5),
        ]);
        let ig = vec![0.99; n_steps * unknowns.len()];
        let initial_guess = DMatrix::from_vec(unknowns.len(), n_steps, ig);
        let max_iterations = 100;
        let abs_tolerance = 1e-6;
        let loglevel = Some("info".to_string());
        let scheme = "forward".to_string();
        let method = "Sparse".to_string();
        let strategy = "Damped".to_string();
        let linear_sys_method = None;

        // Using the new tolerance helper - much simpler!
        let mut bvp = NRBVP::new(
            eqs.clone(),
            initial_guess,
            unhnowns_Str,
            "x".to_string(),
            BoundaryConditions,
            0.0,
            1.0,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            abs_tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            loglevel,
        );
        bvp.dont_save_log(true);
        bvp.solve();
        bvp.gnuplot_result();
        let solution = bvp.get_result().unwrap();
        let eq_and_unknowns = unknowns.clone().into_iter().zip(eqs.clone());
        for (unknown, equation) in eq_and_unknowns {
            println!("unknown: {} | equation: {}", unknown, equation);
        }
        println!("\n=== SOLUTION DEBUG ===");
        println!(
            "Solution matrix shape: {} x {}",
            solution.nrows(),
            solution.ncols()
        );

        // Print first and lust few values for each variable
        for (i, var_name) in unknowns_str.iter().enumerate() {
            if i < solution.ncols() {
                let col = solution.column(i);
                println!(
                    "{}: [first...{:.6}, {:.6}, {:.6}, {:.6} ... last: {:.6}, {:.6}, {:.6}, {:?}]",
                    var_name,
                    col[0],
                    col[1.min(col.len() - 1)],
                    col[2.min(col.len() - 1)],
                    col[3.min(col.len() - 1)],
                    col[col.len() - 4],
                    col[col.len() - 3],
                    col[col.len() - 2],
                    col[col.len() - 1]
                );
            }
        }
        println!("=== END DEBUG ===\n");
        //  let sol_vars =  DMatrix::from_vec(320, 6,  bvp.variable_string);
        //   let v1: DVector<String> = sol_vars.column(0).into();
        //  let v2: DVector<String> = sol_vars.column(1).into();
        //   println!("vec of unknowns {:?}\n {:?}", v1, v2);
      //  let binding = bvp.full_result.clone().unwrap();
    //    let v1: DVector<f64> = binding.row(0).transpose().into();
        // println!("{:?}", v1)
    }

    #[test]
    fn test_BVP_Damp_Grcar() {
        let begin = Instant::now();
        let grid_method = GridRefinementMethod::GrcarSmooke(0.01, 0.01, 2.5);
        test_problem(grid_method);
        println!("duration {}", begin.elapsed().as_secs() as f64)
    }
        #[test]
    fn test_BVP_Damp_Twopnt() {
        let begin = Instant::now();
        let grid_method = GridRefinementMethod::TwoPoint(0.02, 0.02, 2.5);
        test_problem(grid_method);
        println!("duration {}", begin.elapsed().as_secs() as f64)
    }
    #[test]
    fn test_BVP_Damp_Pearson() {
        let grid_method = GridRefinementMethod::Pearson(0.05, 2.5);
        test_problem(grid_method);
    }

    #[test]
    fn test_BVP_Damp_DoublePoints() {
        let begin = Instant::now();
        let grid_method = GridRefinementMethod::DoublePoints;
        test_problem(grid_method);
        println!("duration {}", begin.elapsed().as_secs() as f64)
    }
    #[test]
    fn test_BVP_Damp_Easy() {
        let grid_method = GridRefinementMethod::Easy(0.01);
        test_problem(grid_method);
    }
    #[test]
    fn test_BVP_Damp_Sci() {
        let grid_method = GridRefinementMethod::Sci();
        test_problem(grid_method);
    }
    #[test]
    fn test_BVP_Damp1() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-5;
        let max_iterations = 20;

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 5; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string();

        let grid_method = GridRefinementMethod::GrcarSmooke(0.6, 0.6, 1.5);
        //:Pearson(0.5);
        // GridRefinementMethod::TwoPoint(0.3, 0.5, 1.2);
        //GridRefinementMethod::Easiest(0.2, 1.5);
        let config: AdaptiveGridConfig = AdaptiveGridConfig {
            version: 1,
            max_refinements: 10,
            grid_method: grid_method,
        };
        let strategy_params = Some(SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: Some(config),
        });
        let scheme = "forward".to_string();
        let method = "Sparse".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        let Bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-7.0, 7.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            None,
        );

        println!("solving system");
        nr.dont_save_log(true);
        nr.solve();
        let solution = nr.get_result().unwrap();
    
        // assert_eq!(n, n_steps + 1);
        println!("result = {:?}", solution);
        nr.gnuplot_result();
    }
    #[test]
    fn test_two_point_bvp() {
        let ne = NonlinEquation::TwoPointBVP;

        let eq_system = ne.setup();
        let values = ne.values();
        let arg = "x".to_string();
        let tolerance = 1e-7;
        let max_iterations = 20;
        let t0 = ne.span(None, None).0;
        let t_end = ne.span(None, None).1;
        let n_steps = 1300; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string(); //
        let strategy_params = Some(SolverParams::default());
        let scheme = "forward".to_string();
        let method = "Sparse".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.7; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let BorderConditions = ne.boundary_conditions();
        let Bounds = ne.Bounds();
        println!("eq_syste, {:?}", eq_system.clone());
        println!("values, {:?}", values.clone());
        println!("BorderConditions, {:?}", BorderConditions.clone());
        println!("Bounds, {:?}", Bounds.clone());
        println!("span, {:?}", (t0, t_end));

        let rel_tolerance = ne.rel_tolerance();

        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values.clone(),
            arg,
            BorderConditions.clone(),
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            None,
        );

        println!("solving system");
        nr.dont_save_log(true);
        nr.solve();
        let solution = nr.get_result().unwrap().clone();

        let y_numer: DVector<f64> = solution.column(0).into();

        let x_mesh = nr.x_mesh.clone();
        let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();
        println!("solution: {:?}", y_numer.len());
        println!("x len {}", x_mesh.len());
        nr.gnuplot_result();
        //compare with exact solution
        let y_exact = ne.exact_solution(None, None, Some(n_steps + 2));
        for i in 0..y_exact.len() {
           // let y_exact_ = f64::exp(-(x_mesh[i] as f64).powf(2.0));
            assert!(
                (y_exact[i] - y_numer[i]).abs() < 1e-2,
                "i {}, y_exact: {} y_numer: {}",
                i,
                y_exact[i],
                y_numer[i]
            );
        }
    }

    #[test]
    fn test_Clairaut_bvp() {
        let ne = NonlinEquation::Clairaut;

        let eq_system = ne.setup();
        let values = ne.values();
        let arg = "x".to_string();
        let tolerance = 1e-7;
        let max_iterations = 20;
        let t0 = ne.span(None, None).0;
        let t_end = ne.span(None, None).1;
        let n_steps = 300; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string(); //
        let strategy_params = Some(SolverParams::default());
        let scheme = "forward".to_string();
        let method = "Sparse".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.7; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let BorderConditions = ne.boundary_conditions();
        let Bounds = ne.Bounds();
        println!("eq_syste, {:?}", eq_system.clone());
        println!("values, {:?}", values.clone());
        println!("BorderConditions, {:?}", BorderConditions.clone());
        println!("Bounds, {:?}", Bounds.clone());
        println!("span, {:?}", (t0, t_end));

        let rel_tolerance = ne.rel_tolerance();

        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values.clone(),
            arg,
            BorderConditions.clone(),
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            None,
        );

        println!("solving system");
        nr.dont_save_log(true);
        nr.solve();
        let solution = nr.get_result().unwrap().clone();

        let y_numer: DVector<f64> = solution.column(0).into();

        let x_mesh = nr.x_mesh.clone();
        let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();
        println!("solution: {:?}", y_numer);
        println!("x len {}", x_mesh.len());
        nr.gnuplot_result();

        let ressult = " 1+ (x- 1)^2 - (1/6)*(x-1)^3 + (1/12)*(x-1)^4 ".to_string();
        let expr = Expr::parse_expression(&ressult);
        let exact_sol_func = expr.lambdify1D();
        let exact_solution: Vec<f64> = x_mesh.iter().map(|&x| exact_sol_func(x)).collect();

        let sol: DMatrix<f64> = nr.get_result().unwrap();
        let numer: DVector<f64> = sol.column(0).to_owned().into();
        let mut error = 0.0;
        for i in 0..x_mesh.len() {
            let y_numerical = numer[i];
            let y_exact = exact_solution[i];
            error += (y_numerical - y_exact).powi(2);
        }
        let error = error.sqrt() / x_mesh.len() as f64;
        println!(" weighted error: {}", error);
        assert!(error < 1e-2, "Clairaut BVP error: {}", error);
    }

    fn Parachute_bvp_problem(gridmethod: GridRefinementMethod) {
        let ne = NonlinEquation::ParachuteEquation;

        let eq_system = ne.setup();
        let values = ne.values();
        let arg = "x".to_string();
        let tolerance = 1e-9;
        let max_iterations = 20;
        let t0 = ne.span(None, None).0;
        let t_end = ne.span(None, None).1;
        let n_steps = 40; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string(); //

        let refmethod = AdaptiveGridConfig {
            version: 1,
            max_refinements: 10,
            grid_method: gridmethod,
        };
        let strategy_params = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: Some(refmethod),
        };
        let strategy_params = Some(strategy_params);
        let scheme = "forward".to_string();
        let method = "Sparse".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.7; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let BorderConditions = ne.boundary_conditions();
        let Bounds = ne.Bounds();
        println!("eq_syste, {:?}", eq_system.clone());
        println!("values, {:?}", values.clone());
        println!("BorderConditions, {:?}", BorderConditions.clone());
        println!("Bounds, {:?}", Bounds.clone());
        println!("span, {:?}", (t0, t_end));

        let rel_tolerance = ne.rel_tolerance();

        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values.clone(),
            arg,
            BorderConditions.clone(),
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            None,
        );

        println!("solving system");
        nr.dont_save_log(true);
        nr.solve();
        let solution = nr.get_result().unwrap().clone();
        let y_numer: DVector<f64> = solution.column(0).into();
        let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();
        let x_mesh = nr.x_mesh.clone();
        println!("solution: {:?}", y_numer.len());
        println!("x len {}", x_mesh.len());
       // let y_exect = ne.exact_solution(None, None, Some(n_steps));
        //nr.gnuplot_result();
        //compare with exact solution
        let g = 1.0;
        let k = 1.0;
        for i in 0..x_mesh.len() {
            let x_val = x_mesh[i];
            let y_numerical = y_numer[i];
            let t = x_val;
            let sqrt_gk = (g * k as f64).sqrt();
            let exp_term = (2.0 * sqrt_gk * t).exp();
            let y_exact = (1.0 / k) * (((exp_term + 1.0) / 2.0).ln() - sqrt_gk * t);
            //let y_exact = -exact_solution_[i];

            assert!(
                (y_numerical - y_exact).abs() < 0.5 * 1e-3,
                "Parachute equation error at x = {}: {} vs {}",
                x_val,
                y_numerical,
                y_exact
            );
        }
    }
    #[test]
    fn test_parachute_equation_bvp_pearson() {
        let gridmethod = GridRefinementMethod::Pearson(1e-3, 1.5);
        Parachute_bvp_problem(gridmethod);
    }
    #[test]
    fn test_parachute_equation_bvp_grcar() {
        let gridmethod = GridRefinementMethod::GrcarSmooke(0.002, 0.002, 1.5);
        Parachute_bvp_problem(gridmethod);
    }

    fn lane_emden_bvp_problem(gridmethod: GridRefinementMethod) {
        let ne = NonlinEquation::LaneEmden5;

        let eq_system = ne.setup();
        let values = ne.values();
        let arg = "x".to_string();
        let tolerance = 1e-9;
        let max_iterations = 20;
        let t0 = ne.span(None, None).0;
        let t_end = ne.span(None, None).1;
        let n_steps = 50; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string(); //

        let refmethod = AdaptiveGridConfig {
            version: 1,
            max_refinements: 10,
            grid_method: gridmethod,
        };
        let strategy_params = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: Some(refmethod),
        };
        let strategy_params = Some(strategy_params);
        let scheme = "forward".to_string();
        let method = "Sparse".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.7; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let BorderConditions = ne.boundary_conditions();
        let Bounds = ne.Bounds();
        println!("eq_syste, {:?}", eq_system.clone());
        println!("values, {:?}", values.clone());
        println!("BorderConditions, {:?}", BorderConditions.clone());
        println!("Bounds, {:?}", Bounds.clone());
        println!("span, {:?}", (t0, t_end));

        let rel_tolerance = ne.rel_tolerance();

        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values.clone(),
            arg,
            BorderConditions.clone(),
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            Some(rel_tolerance),
            max_iterations,
            Some(Bounds),
            None,
        );

        println!("solving system");
        nr.dont_save_log(true);
        nr.solve();
        let solution = nr.get_result().unwrap().clone();
        let y_numer: DVector<f64> = solution.column(0).into();
        let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();
        let x_mesh = nr.x_mesh.clone();
        println!("solution: {:?}", y_numer.len());
        println!("x len {}", x_mesh.len());
        // let y_exect = ne.exact_solution(None, None, Some(n_steps));
        //nr.gnuplot_result();
        //compare with exact solution
        for i in 0..x_mesh.len() {
            let x_val = x_mesh[i];
            let y_numerical = y_numer[i];
            let y_exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
            let error = (y_numerical - y_exact).abs();
            // exact_solution[i];
            //    println!(" {}, {}", y_numerical, y_exact);
            assert!(
                error < 1e-4,
                "Lane-Emden equation error at x = {}: {} vs {}",
                x_val,
                y_numerical,
                y_exact
            );
        }
    }
    #[test]
    fn test_lane_emden_bvp_pearson() {
        let gridmethod = GridRefinementMethod::Pearson(0.002, 5.0);
        lane_emden_bvp_problem(gridmethod);
    }
    #[test]
    fn test_lane_emden_bvp_grcar() {
        let begin = Instant::now();
        let gridmethod = GridRefinementMethod::GrcarSmooke(0.002, 0.2, 5.0);
        lane_emden_bvp_problem(gridmethod);
        let duration = begin.elapsed();
        println!(
            "Time elapsed in lane_emden_bvp_problem() is: {:?}",
            duration.as_secs()
        );
    }
}

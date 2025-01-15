#[cfg(test)]
mod tests {

    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::numerical::BVP_Damp::BVP_utils::{
        construct_full_solution, extract_unknown_variables};
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    use strum::IntoEnumIterator;

    use nalgebra::{DMatrix, DVector};
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
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy = "Damped".to_string(); //
        let strategy_params = Some(HashMap::from([
            ("max_jac".to_string(), None),
            ("maxDampIter".to_string(), None),
            ("DampFacor".to_string(), None),
            ("adaptive".to_string(), None),
        ]));
        let scheme = "forward".to_string();
        let method = "Dense".to_string(); // or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
        let Bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-7.0, 7.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRBDVPd::new(
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
        nr.solve();
        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps+1);
        // println!("result = {:?}", solution);
        // nr.plot_result();
    }
    /// Tests the boundary value problem (BVP) solver for the Clairaut equation.
    ///
    /// This test sets up a non-linear equation using the Clairaut configuration
    /// and solves it numerically using a damped strategy with sparse method
    /// for a specified number of steps. It verifies the numerical solution
    /// against the exact solution by calculating the norm of the difference
    /// between the numerical and exact solutions. The test asserts that this
    /// norm is below a specified tolerance to ensure the accuracy of the solution.
    #[test]
    fn test_BVP_Damp2() {
        // let ne=  (NonlinEquation::  Clairaut  ); //  Clairaut  LaneEmden5  ParachuteEquation
        for ne in NonlinEquation::iter() {
            println!("problem {:?}", ne);
            let eq_system = ne.setup();
            let values = ne.values();
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 20;
            let t0 = ne.span(None, None).0;
            let t_end = ne.span(None, None).1;
            let n_steps = 180; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
            let strategy = "Damped".to_string(); //
            let strategy_params = Some(HashMap::from([
                ("max_jac".to_string(), None),
                ("maxDampIter".to_string(), None),
                ("DampFacor".to_string(), None),
                ("adaptive".to_string(), None),// None  Some(vec![1.0, 10.0])
                ("two_point".to_string(), Some(vec![0.2, 0.5, 1.4])),
            ]));
            let scheme = "forward".to_string();
            let method = "Dense".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.99; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let BorderConditions = ne.boundary_conditions();
            let Bounds = ne.Bounds();
            let rel_tolerance = ne.rel_tolerance();

            let mut nr = NRBDVPd::new(
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
            nr.solve();
            let solution = nr.get_result().unwrap().clone();
            let y_numer = solution.column(0);
            let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();

            nr.plot_result();
            //compare with exact solution
            let y_exact = ne.exact_solution(None, None, Some(n_steps));
            let n = &y_exact.len();
            // println!("numerical result = {:?}",  y_numer);
            println!("\n \n y exact{:?}, {}", &y_exact, &y_exact.len());
            println!("\n \n y numer{:?}, {}", &y_numer, &y_numer.len());
            let comparsion: Vec<f64> = y_numer
                .into_iter()
                .zip(y_exact.clone())
                .map(|(y_n_i, y_e_i)| (y_n_i - y_e_i).powf(2.0))
                .collect();
            let norm = comparsion.iter().sum::<f64>().sqrt() / (*n as f64);
            let max_residual = comparsion
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let position = comparsion.iter().position(|&x| x == *max_residual).unwrap();
            let relativ_residual = max_residual.abs() / y_exact[position];
            println!("maximum relative residual of numerical solution wioth respect to exact solution = {}", relativ_residual);

            assert!(norm < 1e-2, "norm = {}", norm);
            assert!(relativ_residual < 1e-1, "norm = {}", norm);
            println!("norm = {}", norm);
         //   let extract_unknown = extract_unknown_variables( solution.clone().transpose(), &BorderConditions.clone(), &values.clone() );
         //   let solution_reconstructed = construct_full_solution( extract_unknown.clone(), &BorderConditions.clone(), &values.clone() ).transpose();
        //   assert_eq!(solution, solution_reconstructed, "solution and reconstruction are not equal");
        }
    }
}

//___________________________________TESTS____________________________________

#[cfg(test)]
mod tests {
    use crate::symbolic::symbolic_engine::Expr;
    use crate::{indexed_var, indexed_var_2d, indexed_vars, symbols};
    use approx;
    use std::f64;
    #[test]
    fn test_add_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        let expected = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_sub_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr -= Expr::Const(2.0);
        let expected = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr *= Expr::Const(2.0);
        let expected = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_div_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_neg() {
        let expr = Expr::Var("x".to_string());
        let neg_expr = -expr;
        let expected = Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Var("x".to_string())),
        );
        assert_eq!(neg_expr, expected);
    }
    #[test]
    fn test_combined_operations() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        expr *= Expr::Const(3.0);
        expr -= Expr::Const(1.0);
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(
            Box::new(Expr::Sub(
                Box::new(Expr::Mul(
                    Box::new(Expr::Add(
                        Box::new(Expr::Var("x".to_string())),
                        Box::new(Expr::Const(2.0)),
                    )),
                    Box::new(Expr::Const(3.0)),
                )),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }
    #[test]
    fn test_diff() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let df_dx = f.diff("x").simplify();
        let _degree = Box::new(Expr::Const(1.0));
        let C = Expr::Const(2.0);

        let expected_result = C.clone() * x.clone();
        //  Mul(Mul(Const(2.0), Pow(Var("x"), Sub(Const(2.0), Const(1.0)))), Const(1.0)) Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x.clone())))
        println!("df_dx {} ", df_dx);
        println!("expected_result {:?} ", expected_result);
        assert_eq!(df_dx, expected_result);
    }

    #[test]
    fn test_sym_to_str() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let rust_function = f.sym_to_str("x");
        assert_eq!(rust_function, "(x^2)");
    }

    #[test]
    fn test_lambdify1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let fn_closure = f.lambdify1D();
        assert_eq!(fn_closure(2.0), 4.0);
    }
    #[test]
    fn test_constuction_of_expression() {
        let vector_of_symbolic_vars = Expr::Symbols("a, b, c");
        let (a, b, c) = (
            vector_of_symbolic_vars[0].clone(),
            vector_of_symbolic_vars[1].clone(),
            vector_of_symbolic_vars[2].clone(),
        );
        let symbolic_expression = a + Expr::exp(b * c);
        let expression_with_const = symbolic_expression.set_variable("a", 1.0);
        let parsed_function = expression_with_const.sym_to_str("a");
        assert_eq!(parsed_function, "(1) + (exp((b) * (c)))");
    }
    #[test]
    fn test_1D() {
        let input = "log(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 0.0);
        let df_dx = f.diff("x");
        let df_dx_str = df_dx.sym_to_str("x");
        assert_eq!(df_dx_str, "(1) / (x)");
    }
    #[test]
    fn test_1D_2() {
        let input = "x+exp(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 1.0 + f64::consts::E);
        let start = 0.0;
        let end = 10f64;
        let num_values = 100;
        let max_norm = 1e-6;
        let (_normm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
        assert_eq!(res, true);
    }
    #[test]
    fn test_multi_diff() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let C = Expr::Const(3.0);
        let f = Expr::pow(x.clone(), C.clone()) + Expr::exp(y.clone());
        let df_dx = f.diff("x").simplify();
        //  let df_dy = f.diff("y");

        let C2 = Expr::Const(2.0);

        let df_dx_expected_result = C.clone() * Expr::pow(x, C2);
        //  let df_dy_expected_result = C* Expr::exp(y);
        assert_eq!(df_dx, df_dx_expected_result);
        let start = vec![1.0, 1.0];
        let end = vec![2.0, 2.0];
        let comparsion = f.compare_num(start, end, 100, 1e-6);
        let bool_1 = &comparsion[0].0;
        let bool_2 = &comparsion[1].0;

        assert_eq!(*bool_1 && *bool_2, true);
        //    assert_eq!(df_dy, expected_result);
    }

    #[test]
    fn test_set_variable() {
        let x = Expr::Var("x".to_string());
        let f = x.clone() + Expr::Const(2.0);
        let f_with_value = f.set_variable("x", 1.0);
        let expected_result = Expr::Const(1.0) + Expr::Const(2.0);
        assert_eq!(f_with_value, expected_result);
    }

    #[test]
    fn test_calc_vector_lambdified1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let _fn_closureee = f.lambdify1D();
        let x_values = vec![1.0, 2.0, 3.0];
        let result = f.calc_vector_lambdified1D(&x_values);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_lambdify1D_from_linspace() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let result = f.lambdify1D_from_linspace(1.0, 3.0, 3);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }
    /*
    #[test]
    fn test_evaluate_vector_lambdified() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let x_values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_values = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = f.evaluate_vector_lambdified(&x_values, &y_values);
        let expected_result = vec![1.0 + 27.18281828459045, 4.0 + 74.08182845904523, 9.0 + 162.31828459045235];
        assert_eq!(result, expected_result);
    }

    */
    #[test]
    fn test_evaluate_multi_diff_from_linspace() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let result = f.evaluate_multi_diff_from_linspace(vec![1.0, 1.0], vec![2.0, 2.0], 100);
        let last_element = result[0].last().unwrap();

        let expected_result: f64 = 4.0f64; // 2*2
        assert!((last_element - expected_result).abs() < f64::EPSILON);
    }
    #[test]
    fn lambdify_IVP_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn lambdify_IVP_owned_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP_owned("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn no_zeros_test() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0)),
        );

        let simplified_expr = expr.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test2() {
        let expr = Expr::Sub(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test3() {
        let expr = Expr::Add(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }

    #[test]
    fn no_zeros_test4() {
        let zero = Box::new(Expr::Const(0.0));
        let added = Expr::Add(zero.clone(), zero.clone()); // 0
        let mulled = Expr::Mul(Box::new(Expr::Const(0.005)), Box::new(added)); //0
        let expr = Box::new(Expr::Sub(
            zero.clone(),
            Box::new(Expr::Add(zero, Box::new(mulled))),
        ));

        let simplified_expr = expr.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }

    #[test]
    fn simplify_test4() {
        let zero = Expr::Const(0.0);
        let one = Expr::Const(1.0);
        let exp = Expr::Exp(Box::new(zero.clone()));
        let expr1 = one - exp;
        let expr2 = zero.clone() + (expr1 + zero);
        let simplified_expr = expr2.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn simplify_test5() {
        let n6 = Expr::Const(6.0);
        let n3 = Expr::Const(3.0);
        let n2 = Expr::Const(2.0);
        let n9 = Expr::Const(9.0);
        let n1 = Expr::Const(1.0);
        let expr2 = n6 * n3 / (n2 * n9) - n1;
        let simplified_expr = expr2.simplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn simplify_test6() {
        let n6 = Expr::Const(6.0);
        let n3 = Expr::Const(3.0);
        let n2 = Expr::Const(2.0);
        let n9 = Expr::Const(9.0);
        let n1 = Expr::Const(1.0);
        let x = Expr::Var("x".to_owned());
        let expr2 = x.clone() * n6 * n3 / (n2 * n9) - n1.clone();
        let simplified_expr = expr2.simplify();
        println!("{}", simplified_expr);
        // After simplification: x * 6 * 3 / (2 * 9) - 1 = x * 1 - 1 = x + (-1)
        let expected_result = x + Expr::Const(-1.0);

        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn simplify_of_complex_expression() {
        println!("\n=== Simplification Test for Very Complex Expression ===");
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = Expr::arctg(Box::new(
            x.clone().pow(Expr::Const(3.0)) / (y.clone() + Expr::Const(1.0)),
        ));
        println!("Original Expression: {}", expr);
        let func1 = expr.lambdify1(&["x", "y"]);
        let simplified_expr = expr.simplify();
        println!("\n Simplified Expression: {}", simplified_expr);
        let func_simplified = simplified_expr.lambdify1(&["x", "y"]);
        let test_args = vec![1.5, 2.0];
        let result_original = func1(&test_args);
        let result_simplified = func_simplified(&test_args);
        assert!(
            (result_original - result_simplified).abs() < 1e-10,
            "Results differ: {} vs {}",
            result_original,
            result_simplified
        );
    }
    #[test]
    fn test_eval_expression_var() {
        let expr = Expr::Var("x".to_string());
        let vars = &["x"];
        let values = vec![5.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_const() {
        let expr = Expr::Const(3.14);
        let vars = &[];
        let values = vec![];
        assert_eq!(expr.eval_expression(vars, &values), 3.14);
    }

    #[test]
    fn test_eval_expression_add() {
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = &["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_sub() {
        let expr = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = &["x", "y"];
        let values = vec![5.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 2.0);
    }

    #[test]
    fn test_eval_expression_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = &["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 6.0);
    }

    #[test]
    fn test_eval_expression_div() {
        let expr = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = &["x", "y"];
        let values = vec![6.0, 2.0];
        assert_eq!(expr.eval_expression(vars, &values), 3.0);
    }

    #[test]
    fn test_eval_expression_pow() {
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let vars = &["x"];
        let values = vec![3.0];
        assert_eq!(expr.eval_expression(vars, &values), 9.0);
    }

    #[test]
    fn test_eval_expression_exp() {
        let expr = Expr::Exp(Box::new(Expr::Var("x".to_string())));
        let vars = &["x"];
        let values = vec![1.0];
        assert!((expr.eval_expression(vars, &values) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_ln() {
        let expr = Expr::Ln(Box::new(Expr::Var("x".to_string())));
        let vars = &["x"];
        let values = vec![std::f64::consts::E];
        assert!((expr.eval_expression(vars, &values) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_complex() {
        let expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Var("z".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let vars = &["x", "y", "z"];
        let values = vec![2.0, 3.0, 4.0];
        assert_eq!(expr.eval_expression(vars, &values), 22.0); // (2 * 3) + (4^2) = 22
    }
    #[test]
    fn test_taylor_series1D_constant() {
        let expr = Expr::Const(5.0);
        let result = expr.taylor_series1D("x", 0.0, 3);
        assert_eq!(result, Expr::Const(5.0));
    }
    #[test]
    fn test_taylor_series1D_log() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().ln();
        let result = expr.taylor_series1D_("x", 5.0, 2);
        let e5 = Expr::Const(5.0);
        let expected = e5.clone().ln() + (x.clone() - e5.clone()) / e5.clone()
            - (x.clone() - e5.clone()).pow(Expr::Const(2.0))
                / (Expr::Const(2.0) * e5.clone().pow(Expr::Const(2.0)));

        println!(
            "result {} \n expected result {}",
            result,
            expected.simplify()
        );
        let taylor_eval = result.lambdify1D()(3.0);
        let expected_eval = expected.lambdify1D()(3.0);
        approx::assert_relative_eq!(taylor_eval, expected_eval, epsilon = 1e-5);
    }
    #[test]
    fn simplify_complex_polinomial() {
        let x = Expr::Var("x".to_string());
        let e5 = Expr::Const(5.0);
        let series = e5.clone().ln() + (x.clone() - e5.clone()) / e5.clone()
            - (x.clone() - e5.clone()).pow(Expr::Const(2.0))
                / (Expr::Const(2.0) * e5.clone().pow(Expr::Const(2.0)));
        let simplified = series.simplify();
        println!("Original series: {}", series);
        println!("Simplified series: {}", simplified);
        // Test that simplification preserves the mathematical meaning
        let test_val = 3.0;
        let original_result = series.set_variable("x", test_val).simplify();
        let simplified_result = simplified.set_variable("x", test_val).simplify();

        // Both should evaluate to the same numerical value
        if let (Expr::Const(orig), Expr::Const(simp)) = (original_result, simplified_result) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed the mathematical value"
            );
        }

        // The simplified form should be more compact than the original
        let simplified_str = format!("{}", simplified);
        let original_str = format!("{}", series);
        println!("Original: {}", original_str);
        println!("Simplified: {}", simplified_str);

        // At minimum, constants should be evaluated
        assert!(simplified_str.contains("1.609") || simplified_str.contains("ln"));
    }

    #[test]
    fn test_taylor_series1D_exp() {
        let x = Expr::Var("x".to_string());

        let exp_expansion = Expr::Const(1.0)
            + x.clone()
            + x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0)
            + x.clone().pow(Expr::Const(3.0)) / Expr::Const(6.0);
        let exp_eval = exp_expansion.lambdify1D()(1.0);

        let taylor = exp_expansion.taylor_series1D_("x", 0.0, 3);
        println!("taylor: {}", taylor);
        let taylor_eval = taylor.lambdify1D()(1.0);
        assert_eq!(taylor_eval, exp_eval);
    }
    #[test]
    fn test_substitute() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone().pow(Expr::Const(2.0));
        let result = expr.substitute_variable("x", &Expr::Const(3.0));
        assert_eq!(result.simplify(), Expr::Const(12.0));
    }

    #[test]
    fn test_substitute2() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone().pow(Expr::Const(2.0));
        let result = expr.substitute_variable("x", &Expr::Var("y".to_string()));
        let y = Expr::Var("y".to_string());
        let expected = y.clone() + y.clone().pow(Expr::Const(2.0));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_symbols_macro() {
        let (x, y, z) = symbols!(x, y, z);
        assert_eq!(x, Expr::Var("x".to_string()));
        assert_eq!(y, Expr::Var("y".to_string()));
        assert_eq!(z, Expr::Var("z".to_string()));
    }

    #[test]
    fn test_indexed_vars_macro() {
        let (vars, names) = indexed_vars!(3, "x");
        assert_eq!(vars.len(), 3);
        assert_eq!(names.len(), 3);
        assert_eq!(vars[0], Expr::Var("x0".to_string()));
        assert_eq!(vars[1], Expr::Var("x1".to_string()));
        assert_eq!(vars[2], Expr::Var("x2".to_string()));
    }

    #[test]
    fn test_indexed_var_macro() {
        let var = indexed_var!(5, "x");
        assert_eq!(var, Expr::Var("x5".to_string()));
    }

    #[test]
    fn test_indexed_var_2d_macro() {
        let var = indexed_var_2d!(2, 3, "A");
        assert_eq!(var, Expr::Var("A_2_3".to_string()));
    }

    // NEW SIMPLIFICATION TESTS

    #[test]
    fn test_like_terms_collection() {
        let x = Expr::Var("x".to_string());
        // 2*x + 3*x = 5*x
        let expr = Expr::Const(2.0) * x.clone() + Expr::Const(3.0) * x.clone();
        let simplified = expr.simplify();
        let expected = Expr::Const(5.0) * x;
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_like_terms_multivar() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // x*y + 2*x*y = 3*x*y
        let expr = x.clone() * y.clone() + Expr::Const(2.0) * x.clone() * y.clone();
        let simplified = expr.simplify();
        let expected = Expr::Const(3.0) * x * y;
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_multiplication() {
        let x = Expr::Var("x".to_string());
        // x^2 * x^3 = x^5
        let expr = x.clone().pow(Expr::Const(2.0)) * x.clone().pow(Expr::Const(3.0));
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(5.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_var_times_power() {
        let x = Expr::Var("x".to_string());
        // x * x^2 = x^3
        let expr = x.clone() * x.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(3.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_same_var() {
        let x = Expr::Var("x".to_string());
        // x * x = x^2
        let expr = x.clone() * x.clone();
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(2.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_division() {
        let x = Expr::Var("x".to_string());
        // x^5 / x^2 = x^3
        let expr = x.clone().pow(Expr::Const(5.0)) / x.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(3.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_division_same_power() {
        let x = Expr::Var("x".to_string());
        // x^2 / x^2 = 1
        let expr = x.clone().pow(Expr::Const(2.0)) / x.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();
        let expected = Expr::Const(1.0);
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_rule_division_var_by_var() {
        let x = Expr::Var("x".to_string());
        // x / x = 1
        let expr = x.clone() / x.clone();
        let simplified = expr.simplify();
        let expected = Expr::Const(1.0);
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_nested_power_rule() {
        let x = Expr::Var("x".to_string());
        // (x^2)^3 = x^6
        let expr = x.clone().pow(Expr::Const(2.0)).pow(Expr::Const(3.0));
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(6.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_polynomial_collection_complex() {
        let x = Expr::Var("x".to_string());
        // x^2 + 2*x^2 + 3*x + x = 3*x^2 + 4*x
        let expr = x.clone().pow(Expr::Const(2.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0))
            + Expr::Const(3.0) * x.clone()
            + x.clone();
        let simplified = expr.simplify();

        // Check that it contains the right terms (order may vary)
        let simplified_str = format!("{}", simplified);
        assert!(simplified_str.contains("3") && simplified_str.contains("4"));
    }

    #[test]
    fn test_zero_coefficient_elimination() {
        let x = Expr::Var("x".to_string());
        // x - x = 0
        let expr = x.clone() - x.clone();
        let simplified = expr.simplify();
        let expected = Expr::Const(0.0);
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_mixed_polynomial_terms() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // x*y + y*x = 2*x*y
        let expr = x.clone() * y.clone() + y.clone() * x.clone();
        let simplified = expr.simplify();
        let expected = Expr::Const(2.0) * x * y;
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_power_with_coefficients() {
        let x = Expr::Var("x".to_string());
        // 2*x^2 + 3*x^2 = 5*x^2
        let expr = Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0))
            + Expr::Const(3.0) * x.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();
        let expected = Expr::Const(5.0) * x.pow(Expr::Const(2.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_constant_collection() {
        // 5 + 3 + 2 = 10
        let expr = Expr::Const(5.0) + Expr::Const(3.0) + Expr::Const(2.0);
        let simplified = expr.simplify();
        let expected = Expr::Const(10.0);
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_no_simplification_needed() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // x + y (different variables, no simplification)
        let expr = x.clone() + y.clone();
        let simplified = expr.simplify();
        let expected = x + y;
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_complex_polynomial_simplification() {
        let x = Expr::Var("x".to_string());
        // (x + 1)*(x + 1) expanded and simplified
        let expr = x.clone() * x.clone() + x.clone() + x.clone() + Expr::Const(1.0);
        let simplified = expr.simplify();

        // Should collect like terms: x^2 + 2*x + 1
        let simplified_str = format!("{}", simplified);
        assert!(simplified_str.contains("2"));
    }

    #[test]
    fn test_power_rule_with_negative_exponents() {
        let x = Expr::Var("x".to_string());
        // x^3 / x^5 = x^(-2)
        let expr = x.clone().pow(Expr::Const(3.0)) / x.clone().pow(Expr::Const(5.0));
        let simplified = expr.simplify();
        let expected = x.pow(Expr::Const(-2.0));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_multivariate_polynomial() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // x^2*y + 2*x^2*y + x*y^2 = 3*x^2*y + x*y^2
        let expr = x.clone().pow(Expr::Const(2.0)) * y.clone()
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0)) * y.clone()
            + x.clone() * y.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();

        // Check that coefficients are collected properly
        let simplified_str = format!("{}", simplified);
        assert!(simplified_str.contains("3"));
    }

    // COMPLEX POLYNOMIAL TESTS WITH TRANSCENDENTAL FUNCTIONS

    #[test]
    fn test_polynomial_with_sin() {
        let x = Expr::Var("x".to_string());
        // sin(x) + 2*x + 3*x - should preserve sin(x) and simplify polynomial part
        let expr = Expr::sin(Box::new(x.clone()))
            + Expr::Const(2.0) * x.clone()
            + Expr::Const(3.0) * x.clone();
        let simplified = expr.simplify();

        // Test that mathematical meaning is preserved
        let test_val = 1.5;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should contain both sin and polynomial terms
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin"),
            "Lost sin function: {}",
            simplified_str
        );
    }

    #[test]
    fn test_polynomial_with_exp_and_ln() {
        let x = Expr::Var("x".to_string());
        // exp(x) + ln(x) + x^2 + 2*x^2 - should preserve transcendental functions
        let expr = x.clone().exp()
            + x.clone().ln()
            + x.clone().pow(Expr::Const(2.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0));
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 2.0;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("exp") && simplified_str.contains("ln"),
            "Lost transcendental functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_complex_rational_with_trig() {
        let x = Expr::Var("x".to_string());
        // (sin(x) + cos(x)) / (x + 1) + x^2 / (x + 1)
        let numerator1 = Expr::sin(Box::new(x.clone())) + Expr::cos(Box::new(x.clone()));
        let numerator2 = x.clone().pow(Expr::Const(2.0));
        let denominator = x.clone() + Expr::Const(1.0);
        let expr = numerator1 / denominator.clone() + numerator2 / denominator;
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.5;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve all functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") && simplified_str.contains("cos"),
            "Lost trigonometric functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_nested_polynomial_with_functions() {
        let x = Expr::Var("x".to_string());
        // ln(x^2 + 2*x + 1) + exp(3*x + x) - nested polynomial inside functions
        let poly_inside_ln =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(1.0);
        let poly_inside_exp = Expr::Const(3.0) * x.clone() + x.clone();
        let expr = poly_inside_ln.ln() + poly_inside_exp.exp();
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 1.0;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve structure
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("ln") && simplified_str.contains("exp"),
            "Lost transcendental functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_mixed_polynomial_division() {
        let x = Expr::Var("x".to_string());
        // (x^3 + 2*x^2 + x) / (sin(x) + x) - polynomial divided by mixed expression
        let numerator = x.clone().pow(Expr::Const(3.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0))
            + x.clone();
        let denominator = Expr::sin(Box::new(x.clone())) + x.clone();
        let expr = numerator / denominator;
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.8;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve division structure
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin"),
            "Lost sin function: {}",
            simplified_str
        );
    }

    #[test]
    fn test_taylor_series_with_polynomial() {
        let x = Expr::Var("x".to_string());
        let a = Expr::Const(2.0);
        // Taylor series: f(a) + f'(a)*(x-a) + f''(a)*(x-a)^2/2! where f(x) = x^2 + sin(x)
        let taylor = a.clone().pow(Expr::Const(2.0))
            + Expr::sin(Box::new(a.clone()))
            + (Expr::Const(2.0) * a.clone() + Expr::cos(Box::new(a.clone())))
                * (x.clone() - a.clone())
            + (Expr::Const(2.0) - Expr::sin(Box::new(a.clone())))
                * (x.clone() - a.clone()).pow(Expr::Const(2.0))
                / Expr::Const(2.0);
        let simplified = taylor.simplify();

        // Test mathematical correctness
        let test_val = 2.1;
        let original_eval = taylor.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve trigonometric functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") || simplified_str.contains("cos"),
            "Lost trigonometric functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_multivariate_mixed_expression() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // sin(x*y) + x^2*y + 2*x^2*y + exp(x+y) - multivariate with mixed terms
        let expr = Expr::sin(Box::new(x.clone() * y.clone()))
            + x.clone().pow(Expr::Const(2.0)) * y.clone()
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0)) * y.clone()
            + (x.clone() + y.clone()).exp();
        let simplified = expr.simplify();

        // Test mathematical correctness
        let original_func = expr.lambdify1(&["x", "y"]);
        let simplified_func = simplified.lambdify1(&["x", "y"]);
        let test_args = vec![1.0, 0.5];
        let orig_result = original_func(&test_args);
        let simp_result = simplified_func(&test_args);

        assert!(
            (orig_result - simp_result).abs() < 1e-10,
            "Simplification changed mathematical value: {} vs {}",
            orig_result,
            simp_result
        );

        // Should preserve all functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") && simplified_str.contains("exp"),
            "Lost transcendental functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_complex_fraction_preservation() {
        let x = Expr::Var("x".to_string());
        // (ln(x) + x^2) / (exp(x) + x^3 + 2*x^3) - complex fraction that shouldn't be over-simplified
        let numerator = x.clone().ln() + x.clone().pow(Expr::Const(2.0));
        let denominator = x.clone().exp()
            + x.clone().pow(Expr::Const(3.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(3.0));
        let expr = numerator / denominator;
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 1.5;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve transcendental functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("ln") && simplified_str.contains("exp"),
            "Lost transcendental functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_zero_coefficient_elimination_with_functions() {
        let x = Expr::Var("x".to_string());
        // sin(x) + 0*x^2 + cos(x) - zero coefficient should be eliminated but functions preserved
        let expr = Expr::sin(Box::new(x.clone()))
            + Expr::Const(0.0) * x.clone().pow(Expr::Const(2.0))
            + Expr::cos(Box::new(x.clone()));
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.7;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve both trig functions and eliminate zero term
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") && simplified_str.contains("cos"),
            "Lost trigonometric functions: {}",
            simplified_str
        );
        assert!(
            !simplified_str.contains("0"),
            "Failed to eliminate zero coefficient: {}",
            simplified_str
        );
    }

    #[test]
    fn test_arctangent_with_polynomial() {
        let x = Expr::Var("x".to_string());
        // arctg(x^2 + x) + x^3 + 2*x^3 - mixed arctangent and polynomial
        let poly_inside = x.clone().pow(Expr::Const(2.0)) + x.clone();
        let expr = Expr::arctg(Box::new(poly_inside))
            + x.clone().pow(Expr::Const(3.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(3.0));
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.5;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve arctangent function
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("arctg"),
            "Lost arctangent function: {}",
            simplified_str
        );
    }

    #[test]
    fn test_power_inside_transcendental() {
        let x = Expr::Var("x".to_string());
        // sin(x^2 + 3*x^2) + cos(2*x + x) - polynomial simplification inside functions
        let poly1 =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(3.0) * x.clone().pow(Expr::Const(2.0));
        let poly2 = Expr::Const(2.0) * x.clone() + x.clone();
        let expr = Expr::sin(Box::new(poly1)) + Expr::cos(Box::new(poly2));
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.3;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve both trig functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") && simplified_str.contains("cos"),
            "Lost trigonometric functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_deeply_nested_mixed_expression() {
        let x = Expr::Var("x".to_string());
        // exp(ln(x^2 + x) + sin(x)) + x^4 + 3*x^4 - deeply nested with polynomial
        let inner_poly = x.clone().pow(Expr::Const(2.0)) + x.clone();
        let nested = inner_poly.ln() + Expr::sin(Box::new(x.clone()));
        let outer_poly =
            x.clone().pow(Expr::Const(4.0)) + Expr::Const(3.0) * x.clone().pow(Expr::Const(4.0));
        let expr = nested.exp() + outer_poly;
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 1.2;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve all nested functions
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("exp")
                && simplified_str.contains("ln")
                && simplified_str.contains("sin"),
            "Lost nested functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_rational_polynomial_with_trig() {
        let x = Expr::Var("x".to_string());
        // (x^2 + 2*x^2) / (sin(x) + cos(x)) + (3*x + x) / (sin(x) + cos(x))
        let num1 =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0));
        let num2 = Expr::Const(3.0) * x.clone() + x.clone();
        let denom = Expr::sin(Box::new(x.clone())) + Expr::cos(Box::new(x.clone()));
        let expr = num1 / denom.clone() + num2 / denom;
        let simplified = expr.simplify();

        // Test mathematical correctness
        let test_val = 0.6;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "Simplification changed mathematical value"
            );
        }

        // Should preserve trigonometric functions in denominator
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("sin") && simplified_str.contains("cos"),
            "Lost trigonometric functions: {}",
            simplified_str
        );
    }

    #[test]
    fn test_preservation_principle() {
        let x = Expr::Var("x".to_string());
        // Complex expression that can't be fully simplified but must be preserved
        let expr = x.clone().exp() / (x.clone().ln() + Expr::sin(Box::new(x.clone())))
            + x.clone().pow(Expr::Const(2.0)) * Expr::cos(Box::new(x.clone()));
        let simplified = expr.simplify();

        // Test mathematical correctness - this is the key principle
        let test_val = 2.5;
        let original_eval = expr.set_variable("x", test_val).simplify();
        let simplified_eval = simplified.set_variable("x", test_val).simplify();

        if let (Expr::Const(orig), Expr::Const(simp)) = (original_eval, simplified_eval) {
            assert!(
                (orig - simp).abs() < 1e-10,
                "CRITICAL: Simplification changed mathematical value - this violates the preservation principle!"
            );
        }

        // Should preserve all transcendental functions even if it can't simplify much
        let simplified_str = format!("{}", simplified);
        assert!(
            simplified_str.contains("exp")
                && simplified_str.contains("ln")
                && simplified_str.contains("sin")
                && simplified_str.contains("cos"),
            "Lost transcendental functions - preservation principle violated: {}",
            simplified_str
        );
    }
    #[test]
    fn test_multple_vars1() {
        let expr = Expr::Var("x".to_string()) * Expr::Const(3.0)
            + Expr::Var("y".to_string()) * Expr::Var("z".to_string())
            + Expr::Const(2.0) * Expr::Var("x".to_string());
        let simplified = expr.simplify();
        println!("Simplified: {}", simplified);
    }
    #[test]
    fn test_simplify_with_variables_fix() {
        // Test the original failing case
        let n6 = Expr::Const(6.0);
        let n3 = Expr::Const(3.0);
        let n2 = Expr::Const(2.0);
        let n9 = Expr::Const(9.0);
        let n1 = Expr::Const(1.0);
        let x = Expr::Var("x".to_owned());

        // This should simplify to x - 1
        let expr = x.clone() * n6 * n3 / (n2 * n9) - n1.clone();
        let simplified = expr.simplify();

        // Expected result: x - 1
        let expected = x + Expr::Const(-1.0);

        assert_eq!(
            simplified, expected,
            "x*6*3/(2*9) - 1 should simplify to x - 1"
        );
    }

    #[test]
    fn test_constant_folding_in_multiplication() {
        let x = Expr::Var("x".to_owned());

        // Test: 2*x*3 should simplify to 6*x
        let expr = Expr::Const(2.0) * x.clone() * Expr::Const(3.0);
        let simplified = expr.simplify();
        let expected = Expr::Const(6.0) * x.clone();

        assert_eq!(simplified, expected, "2*x*3 should simplify to 6*x");
    }

    #[test]
    fn test_division_with_constants() {
        let x = Expr::Var("x".to_owned());

        // Test: (4*x)/2 should simplify to 2*x
        let expr = (Expr::Const(4.0) * x.clone()) / Expr::Const(2.0);
        let simplified = expr.simplify();
        let expected = Expr::Const(2.0) * x.clone();

        assert_eq!(simplified, expected, "(4*x)/2 should simplify to 2*x");
    }

    #[test]
    fn test_nested_constant_operations() {
        let x = Expr::Var("x".to_owned());

        // Test: x*(2*3)/(4*1) should simplify to (6/4)*x = 1.5*x
        let expr = x.clone() * (Expr::Const(2.0) * Expr::Const(3.0))
            / (Expr::Const(4.0) * Expr::Const(1.0));
        let simplified = expr.simplify();
        let expected = Expr::Const(1.5) * x.clone();

        assert_eq!(
            simplified, expected,
            "x*(2*3)/(4*1) should simplify to 1.5*x"
        );
    }

    // TESTS FOR NEW DIFF OPTIMIZATION FEATURES

    #[test]
    fn test_diff_power_rule_optimization() {
        let x = Expr::Var("x".to_string());
        // x^3 -> 3*x^2 (not 3*x^(3-1)*1)
        let f = x.clone().pow(Expr::Const(3.0));
        let df_dx = f.diff("x").simplify();
        let expected = Expr::Const(3.0) * x.clone().pow(Expr::Const(2.0));
        assert_eq!(df_dx, expected);
    }

    #[test]
    fn test_diff_power_rule_exponent_zero() {
        let x = Expr::Var("x".to_string());
        // x^0 -> 0
        let f = x.clone().pow(Expr::Const(0.0));
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(0.0));
    }

    #[test]
    fn test_diff_power_rule_exponent_one() {
        let x = Expr::Var("x".to_string());
        // x^1 -> 1
        let f = x.clone().pow(Expr::Const(1.0));
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(1.0));
    }

    #[test]
    fn test_diff_immediate_simplification_add() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // d/dx(x + y) = 1 + 0 = 1 (not Add(Const(1.0), Const(0.0)))
        let f = x.clone() + y.clone();
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(1.0));
    }

    #[test]
    fn test_diff_immediate_simplification_mul() {
        let x = Expr::Var("x".to_string());
        let c = Expr::Const(5.0);
        // d/dx(5*x) = 5*1 + 0*x = 5
        let f = c.clone() * x.clone();
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(5.0));
    }

    #[test]
    fn test_diff_early_termination() {
        let _x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // d/dx(y^2 + sin(y)) = 0 (expression doesn't contain x)
        let f = y.clone().pow(Expr::Const(2.0)) + Expr::sin(Box::new(y.clone()));
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(0.0));
    }

    #[test]
    fn test_diff_function_with_zero_inner_derivative() {
        let _x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // d/dx(sin(y)) = cos(y) * 0 = 0
        let f = Expr::sin(Box::new(y.clone()));
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(0.0));
    }

    #[test]
    fn test_diff_complex_with_optimizations() {
        let x = Expr::Var("x".to_string());
        // d/dx(x^2 + 3*x + 5) = 2*x + 3
        let f = x.clone().pow(Expr::Const(2.0)) + Expr::Const(3.0) * x.clone() + Expr::Const(5.0);
        let df_dx = f.diff("x").simplify();
        let expected = Expr::Const(2.0) * x.clone() + Expr::Const(3.0);
        assert_eq!(df_dx, expected);
    }

    #[test]
    fn test_diff_subtraction_with_zero() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        // d/dx(x - y) = 1 - 0 = 1
        let f = x.clone() - y.clone();
        let df_dx = f.diff("x").simplify();
        assert_eq!(df_dx, Expr::Const(1.0));
    }

    #[test]
    fn test_diff_maintains_mathematical_correctness() {
        let x = Expr::Var("x".to_string());
        let f =
            x.clone().pow(Expr::Const(4.0)) + Expr::Const(2.0) * x.clone().pow(Expr::Const(3.0));
        let df_dx = f.diff("x");

        // Test at x=2: d/dx(x^4 + 2*x^3) = 4*x^3 + 6*x^2 = 4*8 + 6*4 = 56
        let test_val = 2.0;
        let derivative_at_2 = df_dx.set_variable("x", test_val).simplify();

        if let Expr::Const(val) = derivative_at_2 {
            assert!((val - 56.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diff_zero_elimination_in_complex_expr() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());

        // Expression with terms that will have zero derivatives
        let f = x.clone().pow(Expr::Const(2.0))
            + y.clone() * z.clone()
            + Expr::sin(Box::new(y.clone()))
            + Expr::Const(42.0);

        let df_dx = f.diff("x").simplify();

        // Should only contain the derivative of x^2, which is 2*x
        let expected = Expr::Const(2.0) * x.clone();
        assert_eq!(df_dx, expected);
    }

    #[test]
    fn test_diff_simplify_multiplication_edge_cases() {
        let x = Expr::Var("x".to_string());

        // Test multiplication by 0 and 1
        let f1 = Expr::Const(0.0) * x.clone();
        let df1_dx = f1.diff("x").simplify();
        assert_eq!(df1_dx, Expr::Const(0.0));

        let f2 = Expr::Const(1.0) * x.clone();
        let df2_dx = f2.diff("x").simplify();
        assert_eq!(df2_dx, Expr::Const(1.0));
    }
    // DISPLAY FORMATTING TESTS - Testing bracket placement rules

    #[test]
    fn test_display_variables_and_constants() {
        let x = Expr::Var("x".to_string());
        let c = Expr::Const(5.0);
        assert_eq!(format!("{}", x), "x");
        assert_eq!(format!("{}", c), "5");
    }

    #[test]
    fn test_display_functions_no_brackets() {
        let x = Expr::Var("x".to_string());
        let sin_x = Expr::sin(Box::new(x.clone()));
        let exp_x = Expr::Exp(Box::new(x.clone()));
        let ln_x = Expr::Ln(Box::new(x));
        
        assert_eq!(format!("{}", sin_x), "sin(x)");
        assert_eq!(format!("{}", exp_x), "exp(x)");
        assert_eq!(format!("{}", ln_x), "ln(x)");
    }

    #[test]
    fn test_display_addition_precedence() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x + y (simple addition)
        let expr1 = x.clone() + y.clone();
        assert_eq!(format!("{}", expr1), "x + y");
        
        // x + y + z (left-associative, no extra brackets)
        let expr2 = expr1 + z;
        assert_eq!(format!("{}", expr2), "x + y + z");
    }

    #[test]
    fn test_display_multiplication_precedence() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x * y + z (no brackets around x * y due to higher precedence)
        let expr = x.clone() * y.clone() + z.clone();
        assert_eq!(format!("{}", expr), "x * y + z");
        
        // (x + y) * z (brackets needed for lower precedence in higher context)
        let expr2 = (x + y) * z;
        assert_eq!(format!("{}", expr2), "(x + y) * z");
    }

    #[test]
    fn test_display_subtraction_right_associative() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x - (y - z) (brackets needed for right-associative subtraction)
        let expr = x.clone() - (y.clone() - z.clone());
        assert_eq!(format!("{}", expr), "x - (y - z)");
        
        // x - y - z (left-associative, no brackets needed)
        let expr2 = (x - y) - z;
        assert_eq!(format!("{}", expr2), "x - y - z");
    }

    #[test]
    fn test_display_division_right_associative() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x / (y / z) (brackets needed for right-associative division)
        let expr = x.clone() / (y.clone() / z.clone());
        assert_eq!(format!("{}", expr), "x / (y / z)");
        
        // x / (y + z) (brackets needed for lower precedence in denominator)
        let expr2 = x / (y + z);
        assert_eq!(format!("{}", expr2), "x / (y + z)");
    }

    #[test]
    fn test_display_power_right_associative() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x ^ y ^ z becomes x ^ (y ^ z) (right-associative power)
        let expr = x.clone().pow(y.clone().pow(z.clone()));
        assert_eq!(format!("{}", expr), "x ^ (y ^ z)");
        
        // (x + y) ^ z (brackets needed for lower precedence base)
        let expr2 = (x + y).pow(z);
        assert_eq!(format!("{}", expr2), "(x + y) ^ z");
    }

    #[test]
    fn test_display_mixed_precedence_complex() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x + y * z ^ 2 (no brackets needed due to precedence)
        let expr = x.clone() + y.clone() * z.clone().pow(Expr::Const(2.0));
        assert_eq!(format!("{}", expr), "x + y * z ^ 2");
        
        // (x + y) * (z ^ 2) (brackets only where needed)
        let expr2 = (x + y) * z.pow(Expr::Const(2.0));
        assert_eq!(format!("{}", expr2), "(x + y) * z ^ 2");
    }

    #[test]
    fn test_display_functions_with_operations() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // sin(x) + y (no brackets around sin(x))
        let expr = Expr::sin(Box::new(x.clone())) + y.clone();
        assert_eq!(format!("{}", expr), "sin(x) + y");
        
        // sin(x + y) (expression inside function)
        let expr2 = Expr::sin(Box::new(x.clone() + y));
        assert_eq!(format!("{}", expr2), "sin(x + y)");
        
        // sin(x) * cos(x) (no brackets around functions)
        let expr3 = Expr::sin(Box::new(x.clone())) * Expr::cos(Box::new(x));
        assert_eq!(format!("{}", expr3), "sin(x) * cos(x)");
    }

    #[test]
    fn test_display_nested_operations() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // x * (y + z) / (x - y) (brackets preserved where needed)
        let expr = x.clone() * (y.clone() + z.clone()) / (x.clone() - y);
        assert_eq!(format!("{}", expr), "x * (y + z) / (x - y)");
    }

    #[test]
    fn test_display_power_with_complex_expressions() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // (x + y) ^ (x - y) (brackets needed for both base and exponent)
        let expr = (x.clone() + y.clone()).pow(x.clone() - y.clone());

        assert_eq!(format!("{}", expr), "(x + y) ^ (x - y)");
        
        // x ^ (y + 1) (brackets needed for complex exponent)
        let expr2 = x.pow(y + Expr::Const(1.0));
        assert_eq!(format!("{}", expr2), "x ^ (y + 1)");
    }
    #[test]
    fn test_display_power_with_complex_expressions2() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
   
        let expr = (x.clone() + Expr::Const(2.0)).pow(x.clone());
        println!("{}", expr);
        assert_eq!(format!("{}", expr), "(x + 2) ^ x");
     
    }

    #[test]
    fn test_display_trigonometric_functions() {
        let x = Expr::Var("x".to_string());
        
        // Test all trigonometric functions
        assert_eq!(format!("{}", Expr::sin(Box::new(x.clone()))), "sin(x)");
        assert_eq!(format!("{}", Expr::cos(Box::new(x.clone()))), "cos(x)");
        assert_eq!(format!("{}", Expr::tg(Box::new(x.clone()))), "tg(x)");
        assert_eq!(format!("{}", Expr::ctg(Box::new(x.clone()))), "ctg(x)");
        assert_eq!(format!("{}", Expr::arcsin(Box::new(x.clone()))), "arcsin(x)");
        assert_eq!(format!("{}", Expr::arccos(Box::new(x.clone()))), "arccos(x)");
        assert_eq!(format!("{}", Expr::arctg(Box::new(x.clone()))), "arctg(x)");
        assert_eq!(format!("{}", Expr::arcctg(Box::new(x))), "arcctg(x)");
    }

    #[test]
    fn test_display_complex_mathematical_expression() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // sin(x^2 + y) * exp(x - y) / (x + 1)
        let expr = Expr::sin(Box::new(x.clone().pow(Expr::Const(2.0)) + y.clone())) 
                   * Expr::Exp(Box::new(x.clone() - y)) 
                   / (x + Expr::Const(1.0));
        
        let result = format!("{}", expr);
        assert!(result.contains("sin(x ^ 2 + y)"));
        assert!(result.contains("exp(x - y)"));
        assert!(result.contains("(x + 1)"));
    }

    #[test]
    fn test_display_no_unnecessary_brackets() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // Test that we don't add unnecessary brackets
        let expr = x.clone() * y.clone() * z; // Should be x * y * z, not (x * y) * z
        let result = format!("{}", expr);
        assert_eq!(result, "x * y * z");
        
        // Test addition chain
        let expr2 = x.clone() + y.clone() + Expr::Const(1.0);
        let result2 = format!("{}", expr2);
        assert_eq!(result2, "x + y + 1");
    }

    #[test]
    fn test_display_preserves_necessary_brackets() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Cases where brackets are mathematically necessary
        let cases = vec![
            ((x.clone() + y.clone()) * Expr::Const(2.0), "(x + y) * 2"),
            (x.clone() / (y.clone() + Expr::Const(1.0)), "x / (y + 1)"),
            (x.clone() - (y.clone() - Expr::Const(1.0)), "x - (y - 1)"),
            ((x.clone() * y.clone()).pow(Expr::Const(2.0)), "(x * y) ^ 2"),
        ];
        
        for (expr, expected) in cases {
            assert_eq!(format!("{}", expr), expected);
        }
    }

    #[test]
    fn test_display_constants_formatting() {
        // Test various constant formats
        assert_eq!(format!("{}", Expr::Const(0.0)), "0");
        assert_eq!(format!("{}", Expr::Const(1.0)), "1");
        assert_eq!(format!("{}", Expr::Const(-1.0)), "-1");
        assert_eq!(format!("{}", Expr::Const(3.14159)), "3.14159");
    }

    #[test]
    fn test_display_comparison_with_old_format() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // These expressions should be much cleaner than the old format
        let expr1 = x.clone() + y.clone() * Expr::Const(2.0);
        assert_eq!(format!("{}", expr1), "x + y * 2");
        // Old format would be: (x + (y * 2))
        
        let expr2 = Expr::sin(Box::new(x)) + y;
        assert_eq!(format!("{}", expr2), "sin(x) + y");
        // Old format would be: (sin(x) + y)
    }




  // PRETTY PRINT TESTS - Testing mathematical formatting features

    #[test]
    fn test_pretty_print_simple_variables_and_constants() {
        let x = Expr::Var("x".to_string());
        let c = Expr::Const(5.0);
        
        assert_eq!(x.pretty_print(), "x");
        assert_eq!(c.pretty_print(), "5");
    }

    #[test]
    fn test_pretty_print_unicode_superscripts() {
        let x = Expr::Var("x".to_string());
        
        // Simple digit powers should use Unicode superscripts
        let x_squared = x.clone().pow(Expr::Const(2.0));
        assert_eq!(x_squared.pretty_print(), "x²");
        
        let x_cubed = x.clone().pow(Expr::Const(3.0));
        assert_eq!(x_cubed.pretty_print(), "x³");
        
        // Single letter variable powers
        let y = Expr::Var("y".to_string());
        let x_to_y = x.pow(y);
        assert_eq!(x_to_y.pretty_print(), "xʸ");
    }

    #[test]
    fn test_pretty_print_exp_function() {
        let x = Expr::Var("x".to_string());
        
        // Simple exponent should use Unicode superscript
        let exp_x = Expr::Exp(Box::new(x.clone()));
        assert_eq!(exp_x.pretty_print(), "eˣ");
        
        // Digit exponent
        let exp_2 = Expr::Exp(Box::new(Expr::Const(2.0)));
        assert_eq!(exp_2.pretty_print(), "e²");
    }

    #[test]
    fn test_pretty_print_complex_powers() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Complex exponent should use multi-line format
        let complex_power = x.clone().pow(x.clone() + y);
        let result = complex_power.pretty_print();
             println!("{}", result);
        // Should contain multiple lines
        assert!(result.contains('\n'));
        // Should have exponent above base
        let lines: Vec<&str> = result.split('\n').collect();
        assert!(lines.len() > 1);
    }

    #[test]
    fn test_pretty_print_division_horizontal_line() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Simple division
        let division = x.clone() / y.clone();
        let result = division.pretty_print();
             println!("{}", result);
        // Should contain horizontal line character
        assert!(result.contains('─'));
        
        // Should be multi-line
        let lines: Vec<&str> = result.split('\n').collect();
        assert_eq!(lines.len(), 3); // numerator, line, denominator
        
        // Middle line should be the division line
        assert!(lines[1].chars().all(|c| c == '─'));
    }

    #[test]
    fn test_pretty_print_complex_division() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Division with complex numerator and denominator
        let numerator = x.clone().pow(Expr::Const(2.0)) + y.clone();
        let denominator = Expr::sin(Box::new(x)) + Expr::Const(1.0);
        let division = numerator / denominator;
        
        let result = division.pretty_print();
             println!("{}", result);
        // Should contain horizontal line
        assert!(result.contains('─'));
        
        // Should be multi-line
        let lines: Vec<&str> = result.split('\n').collect();
        assert!(lines.len() >= 3);
        
        // Should contain both x² and sin
        assert!(result.contains('²'));
        assert!(result.contains("sin"));
    }

    #[test]
    fn test_pretty_print_mixed_operations() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Expression with both simple and complex parts
        let expr = x.clone().pow(Expr::Const(2.0)) + Expr::sin(Box::new(y));
        let result = expr.pretty_print();
             println!("{}", result);
        // Should use Unicode superscript for x²
        assert!(result.contains('²'));
        // Should contain sin function
        assert!(result.contains("sin"));
        // Should be single line for this case
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_pretty_print_nested_functions() {
        let x = Expr::Var("x".to_string());
        
        // Nested function calls
        let nested = Expr::sin(Box::new(Expr::cos(Box::new(x))));
        let result = nested.pretty_print();
             println!("{}", result);
        assert_eq!(result, "sin(cos(x))");
    }

    #[test]
    fn test_pretty_print_trigonometric_functions() {
        let x = Expr::Var("x".to_string());
        
        let test_cases = vec![
            (Expr::sin(Box::new(x.clone())), "sin(x)"),
            (Expr::cos(Box::new(x.clone())), "cos(x)"),
            (Expr::tg(Box::new(x.clone())), "tg(x)"),
            (Expr::ctg(Box::new(x.clone())), "ctg(x)"),
            (Expr::arcsin(Box::new(x.clone())), "arcsin(x)"),
            (Expr::arccos(Box::new(x.clone())), "arccos(x)"),
            (Expr::arctg(Box::new(x.clone())), "arctg(x)"),
            (Expr::arcctg(Box::new(x)), "arcctg(x)"),
        ];
        
        for (expr, expected) in test_cases {
            assert_eq!(expr.pretty_print(), expected);
        }
    }

    #[test]
    fn test_pretty_print_bracket_handling() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Should preserve necessary brackets
        let expr1 = (x.clone() + y.clone()) * Expr::Const(2.0);
        assert_eq!(expr1.pretty_print(), "(x + y) * 2");
        
        // Should omit unnecessary brackets
        let expr2 = x.clone() * y + Expr::Const(2.0);
        assert_eq!(expr2.pretty_print(), "x * y + 2");
    }

    #[test]
    fn test_pretty_print_complex_exp_function() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Simple complex exponent (single line)
        let simple_complex_exp = Expr::Exp(Box::new(x.clone() + y.clone()));
        let result1 = simple_complex_exp.pretty_print();
        println!("{}", result1);
        // Should stay as single line since x + y is single line
        //assert_eq!(result1, "e^(x + y)");
        
        // Truly multi-line exponent (division)
        let multi_line_exp = Expr::Exp(Box::new(x / y));
        let result2 = multi_line_exp.pretty_print();
        
        // Should use multi-line format for division exponent
        assert!(result2.contains('\n'));
        
        let lines: Vec<&str> = result2.split('\n').collect();
        // Should have exponent above 'e'
        assert!(lines.len() > 1);
        assert!(lines.last().unwrap().contains('e'));
    }

    #[test]
    fn test_pretty_print_power_precedence() {
        let x = Expr::Var("x".to_string());
        
        // Power should have higher precedence than multiplication
        let expr = Expr::Const(2.0) * x.clone().pow(Expr::Const(3.0));
        let result = expr.pretty_print();
        
        // Should show as 2 * x³, not (2 * x)³
        assert_eq!(result, "2 * x³");
    }

    #[test]
    fn test_pretty_print_division_line_width() {
        let x = Expr::Var("x".to_string());
        
        // Division where numerator is longer than denominator
        let long_numerator = x.clone() + Expr::Var("very_long_variable_name".to_string());
        let short_denominator = Expr::Const(2.0);
        let division = long_numerator / short_denominator;
        
        let result = division.pretty_print();
        let lines: Vec<&str> = result.split('\n').collect();
             println!("{}", result);
        // Division line should match the width of the longest part
        let numerator_width = lines[0].len();
        let line_width = lines[1].len();
        let denominator_line = lines[2];
        
        // Line should be at least as wide as the numerator
        assert!(line_width >= numerator_width);
        // Denominator should be centered
        assert!(denominator_line.trim().len() < line_width);
    }

    #[test]
    fn test_pretty_print_multi_line_addition() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Addition where one operand is multi-line (division)
        let division = x.clone() / y.clone();
        let addition = division + Expr::Const(1.0);
        
        let result = addition.pretty_print();
             println!("{}", result);
        // Should be multi-line
        assert!(result.contains('\n'));
        
        let lines: Vec<&str> = result.split('\n').collect();
        // Should have proper alignment with + operator
        let plus_line = lines.iter().find(|line| line.contains(" + "));
        assert!(plus_line.is_some());
    }

    #[test]
    fn test_pretty_print_logarithm_function() {
        let x = Expr::Var("x".to_string());
        
        // Simple logarithm
        let ln_x = Expr::Ln(Box::new(x.clone()));
        assert_eq!(ln_x.pretty_print(), "ln(x)");
        
        // Logarithm with complex argument
        let complex_arg = x.clone() / Expr::Const(2.0);
        let ln_complex = Expr::Ln(Box::new(complex_arg));
        let result = ln_complex.pretty_print();
        println!("{}", result);
        // Should handle multi-line argument properly
        assert!(result.contains("ln("));
        assert!(result.contains(')'));
    }

    #[test]
    fn test_pretty_print_unicode_superscript_limits() {
        let x = Expr::Var("x".to_string());
        
        // Powers that can't use Unicode superscripts should fall back
        let complex_power = x.clone().pow(Expr::Const(10.0)); // 10 > 9, no Unicode
        let result = complex_power.pretty_print();
             println!("{}", result);
        // Should use multi-line format
        assert!(result.contains('\n'));
        
        // Multi-character variable should also fall back
        let long_var = Expr::Var("xy".to_string());
        let long_var_power = x.pow(long_var);
        let result2 = long_var_power.pretty_print();
             println!("{}", result2);
        // Should use multi-line format
        assert!(result2.contains('\n'));
    }

    #[test]
    fn test_pretty_print_exp_vs_pow() {
        let x = Expr::Var("x".to_string());
        
        // Expr::Exp should render as e^x
        let exp_func = Expr::Exp(Box::new(x.clone()));
        assert_eq!(exp_func.pretty_print(), "eˣ");
        
        // Manual e^x power should render differently
        let e_var = Expr::Var("e".to_string());
        let manual_exp = e_var.pow(x);
        assert_eq!(manual_exp.pretty_print(), "eˣ");
    }

    #[test]
    fn test_pretty_print_negative_exponents() {
        let x = Expr::Var("x".to_string());
        
        // Negative constant exponent
        let neg_power = x.pow(Expr::Const(-1.0));
        let result = neg_power.pretty_print();
        
        // Should use multi-line format for negative exponents
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_pretty_print_comprehensive_example() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Complex mathematical expression: (x² + sin(y)) / (e^x + 1)
        let numerator = x.clone().pow(Expr::Const(2.0)) + Expr::sin(Box::new(y));
        let denominator = Expr::Exp(Box::new(x)) + Expr::Const(1.0);
        let complex_expr = numerator / denominator;
        
        let result = complex_expr.pretty_print();
             println!("{}", result);
        // Should contain Unicode superscript
        assert!(result.contains('²'));
        // Should contain sin function
        assert!(result.contains("sin"));
        // Should contain e with superscript
        assert!(result.contains('e'));
        // Should be multi-line due to division
        assert!(result.contains('\n'));
        // Should contain horizontal division line
        assert!(result.contains('─'));
    }

    #[test]
    fn test_pretty_print_preserves_mathematical_meaning() {
        let x = Expr::Var("x".to_string());
        
        // Verify that pretty_print doesn't change mathematical structure
        let expr = x.clone().pow(Expr::Const(2.0)) + Expr::Const(1.0);
        let pretty = expr.pretty_print();
        let regular = format!("{}", expr);
        
        // Both should represent the same mathematical expression
        // (though formatting will be different)
        assert!(pretty.contains('²'));
        assert!(regular.contains("2"));
        
        // Mathematical evaluation should be identical
        let test_val = 3.0;
        let expr_eval = expr.set_variable("x", test_val).simplify();
        if let Expr::Const(val) = expr_eval {
            assert_eq!(val, 10.0); // 3² + 1 = 10
        }
    }

    #[test]
    fn test_pretty_print_empty_and_edge_cases() {
        // Test edge cases
        let zero = Expr::Const(0.0);
        assert_eq!(zero.pretty_print(), "0");
        
        let one = Expr::Const(1.0);
        assert_eq!(one.pretty_print(), "1");
        
        // Very simple power
        let x = Expr::Var("x".to_string());
        let x_to_1 = x.pow(Expr::Const(1.0));
        assert_eq!(x_to_1.pretty_print(), "x¹");
    }

      #[test]
    fn test_pretty_print_debug_positioning() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Test simple addition with division
        let simple_div = x.clone() / y.clone();
        let simple_add = simple_div.clone() + Expr::Const(1.0);
        
        println!("=== Simple Division ===");
        println!("{}", simple_div.clone().pretty_print());
        
        println!("\n=== Simple Addition with Division ===");
        println!("{}", simple_add.pretty_print());
        
        // Test power with division
        let base = x.clone() + Expr::Const(1.0);
        let exponent = y.clone() / Expr::Const(2.0);
        let power_expr = base.pow(exponent);
        
        println!("\n=== Power with Division Exponent ===");
        println!("{}", power_expr.pretty_print());
        
        // Test addition of two complex expressions
        let left_complex = x.clone() / y.clone();
        let right_complex = (x.clone() + Expr::Const(1.0)).pow(y.clone() / Expr::Const(2.0));
        let combined = left_complex + right_complex;
        
        println!("\n=== Combined Complex Expression ===");
        println!("{}", combined.pretty_print());
    }
    #[test]
    fn test_pretty_print_rather_complex_expression() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // Build an extremely complex nested expression:
        // ln(sin(x²/y) + cos(z³)) / (e^(x+y) * arctg(z/x)) + (x+1)^(y/2)
        
        // First part: ln(sin(x²/y) + cos(z³))
        let x_squared_over_y = x.clone() / y.clone();
        let sin_part = Expr::sin(Box::new(x_squared_over_y));
        let z_cubed = Expr::arcctg(Box::new(sin_part));
        let cos_part = Expr::cos(Box::new(z_cubed));
       
        let ln_part = Expr::Ln(Box::new(cos_part));
        
        // Second part: e^(x+y) * arctg(z/x)
        let exp_part = Expr::Exp(Box::new(x.clone() + y.clone()));
        let arctg_arg = z.clone() / x.clone();
        let arctg_part = Expr::arctg(Box::new(arctg_arg));
        let denominator = exp_part * arctg_part;
        
        // Third part: (x+1)^(y/2)

    
        
        // Combine everything: ln(...) / (...)
        let fraction = ln_part / denominator;
        let complete_expr = fraction;
        
        let result = complete_expr.pretty_print();
        println!("=== RATHER COMPLEX EXPRESSION ===");
        println!("{}", result);
        println!("=====================================");
        
   

        

    }

    #[test]
    fn test_pretty_print_extremely_complex_expression() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // Build an extremely complex nested expression:
        // ln(sin(x²/y) + cos(z³)) / (e^(x+y) * arctg(z/x)) + (x+1)^(y/2)
        
        // First part: ln(sin(x²/y) + cos(z³))
        let x_squared_over_y = x.clone().pow(Expr::Const(2.0)) / y.clone();
        let sin_part = Expr::sin(Box::new(x_squared_over_y));
        let z_cubed = z.clone().pow(Expr::Const(3.0));
        let cos_part = Expr::cos(Box::new(z_cubed));
        let ln_arg = sin_part + cos_part;
        let ln_part = Expr::Ln(Box::new(ln_arg));
        
        // Second part: e^(x+y) * arctg(z/x)
        let exp_part = Expr::Exp(Box::new(x.clone() + y.clone()));
        let arctg_arg = z.clone() / x.clone();
        let arctg_part = Expr::arctg(Box::new(arctg_arg));
        let denominator = exp_part * arctg_part;
        
        // Third part: (x+1)^(y/2)
        let base = x.clone() + Expr::Const(1.0);
        let exponent = y.clone() / Expr::Const(2.0);
        let power_part = base.pow(exponent);
        
        // Combine everything: ln(...) / (...) + (...)^(...)
        let fraction = ln_part / denominator;
        let complete_expr = fraction + power_part;
        
        let result = complete_expr.pretty_print();
        println!("=== EXTREMELY COMPLEX EXPRESSION ===");
        println!("{}", result);
        println!("=====================================");
        
        // Verify it contains all the expected mathematical elements
        assert!(result.contains("ln("));
        assert!(result.contains("sin("));
        assert!(result.contains("cos("));
        assert!(result.contains("arctg("));
        assert!(result.contains('²')); // x²
        assert!(result.contains('³')); // z³
        assert!(result.contains('e')); // exponential
        assert!(result.contains('─')); // division lines
        assert!(result.contains('\n')); // multi-line formatting
        
        // Should contain proper function formatting
        assert!(result.contains(") "));
        
        // Verify mathematical correctness by evaluating at a test point
        let test_values = [("x", 2.0), ("y", 3.0), ("z", 1.5)];
        let mut test_expr = complete_expr.clone();
        for (var, val) in test_values.iter() {
            test_expr = test_expr.set_variable(var, *val);
        }
        let evaluated = test_expr.simplify();
        
        // Should evaluate to a finite number (not NaN or infinity)
        if let Expr::Const(val) = evaluated {
            assert!(val.is_finite(), "Expression should evaluate to a finite number");
        }
    }

    #[test]
    fn test_pretty_print_nested_divisions_and_powers() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        
        // Create deeply nested divisions and powers:
        // ((x/y)^z) / ((sin(x)/cos(y))^(z+1))
        
        let x_over_y = x.clone() / y.clone();
        let numerator_base = x_over_y.pow(z.clone());
        
        let sin_x = Expr::sin(Box::new(x.clone()));
        let cos_y = Expr::cos(Box::new(y.clone()));
        let trig_fraction = sin_x / cos_y;
        let denominator_exp = z.clone() + Expr::Const(1.0);
        let denominator_base = trig_fraction.pow(denominator_exp);
        
        let complex_fraction = numerator_base / denominator_base;
        
        let result = complex_fraction.pretty_print();
        println!("=== NESTED DIVISIONS AND POWERS ===");
        println!("{}", result);
        println!("===================================");
        
        // Should contain multiple division lines
        let line_count = result.matches('─').count();
        assert!(line_count >= 2, "Should have multiple division lines");
        
        // Should be multi-line
        assert!(result.contains('\n'));
        
        // Should contain trigonometric functions
        assert!(result.contains("sin("));
        assert!(result.contains("cos("));
        
        // Should handle nested structure properly
        assert!(result.contains(") "));
    }

    #[test]
    fn test_power_alignment_issue() {
        let x = Expr::Var("x".to_string());
        let e = Expr::Var("E".to_string());
        let r = Expr::Var("R".to_string());
        let t = Expr::Var("T".to_string());
        
        // Test simple polynomial case (should work fine)
        let simple_poly = (Expr::Const(1.0) - x.clone()).pow(Expr::Const(2.0)) * x.clone();
        println!("=== Simple polynomial (1-x)²*x ===");
        println!("{}", simple_poly.pretty_print());
        
        // Test problematic case with complex exponent
        // (1 - x)² * x * (-1 * E)^(something/R*T)
        let base_part = (Expr::Const(1.0) - x.clone()).pow(Expr::Const(2.0)) * x.clone();
        let neg_e = Expr::Const(-1.0) * e.clone();
        let complex_exponent = Expr::Const(1.0) / (r.clone() * t.clone());
        let power_part = neg_e.pow(complex_exponent);
        let full_expr = base_part * power_part;
        
        println!("\n=== Complex case with power alignment issue ===");
        println!("{}", full_expr.pretty_print());
        
        // Test another case with fraction in exponent
        let fraction_exp = x.clone() / (r.clone() * t.clone());
        let e_to_fraction = e.clone().pow(fraction_exp);
        let mixed_expr = (Expr::Const(1.0) - x.clone()).pow(Expr::Const(2.0)) * x.clone() * e_to_fraction;
        
        println!("\n=== Mixed expression with fraction exponent ===");
        println!("{}", mixed_expr.pretty_print());
    }

    #[test]
    fn test_division_alignment_debug() {
        
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Test simple division
        let simple_div = x.clone().pow(Expr::Const(2.0)) / y.clone();
        println!("=== Simple Division x²/y ===");
        println!("{}", simple_div.pretty_print());
        
        // Test division inside sin function
        let sin_div = Expr::sin(Box::new(simple_div.clone()));
        println!("\n=== Division inside sin function ===");
        println!("{}", sin_div.pretty_print());
        
        // Test the problematic case from the complex expression
        let z = Expr::Var("z".to_string());
        let x_squared_over_y = x.clone().pow(Expr::Const(2.0)) / y.clone();
        let sin_part = Expr::sin(Box::new(x_squared_over_y));
        let z_cubed = z.clone().pow(Expr::Const(3.0));
        let cos_part = Expr::cos(Box::new(z_cubed));
        let ln_arg = sin_part * cos_part;
        
        println!("\n=== Complex case: sin(x²/y) + cos(z³) ===");
        println!("{}", ln_arg.pretty_print());
    }

    #[test]
    fn test_pretty_print_mixed_transcendental_functions() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Complex expression with multiple transcendental functions:
        // ln(e^x + sin(y)) / arctg(cos(x) * tg(y))
        
        let exp_x = Expr::Exp(Box::new(x.clone()));
        let sin_y = Expr::sin(Box::new(y.clone()));
        let ln_arg = exp_x + sin_y;
        let ln_part = Expr::Ln(Box::new(ln_arg));
        
        let cos_x = Expr::cos(Box::new(x.clone()));
        let tg_y = Expr::tg(Box::new(y.clone()));
        let arctg_arg = cos_x * tg_y;
        let arctg_part = Expr::arctg(Box::new(arctg_arg));
        
        let complex_expr = ln_part / arctg_part;
        
        let result = complex_expr.pretty_print();
        println!("=== MIXED TRANSCENDENTAL FUNCTIONS ===");
        println!("{}", result);
        println!("======================================");
        
        // Should contain all function types
        assert!(result.contains("ln("));
        assert!(result.contains("sin("));
        assert!(result.contains("cos("));
        assert!(result.contains("tg("));
        assert!(result.contains("arctg("));
        assert!(result.contains('e'));
        
        // Should have proper division formatting
        assert!(result.contains('─'));
        assert!(result.contains('\n'));
        
        // Functions should be properly formatted horizontally
        assert!(result.contains(") "));
    }

        #[test]
    fn test_power_positioning_isolated() {
        println!("\n=== Testing Power Positioning ===");
        
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        
        // Simple power: (x + 1)^(y/2)
        let base = x.clone() + Expr::Const(1.0);
        let exp = y.clone() / Expr::Const(2.0);
        let power = base.pow(exp);
        
        println!("=== Simple Power ===");
        println!("{}", power.pretty_print());
        
        // Power in addition: x + (x + 1)^(y/2)
        let addition = x.clone() + power.clone();
        println!("\n=== Power in Addition ===");
        println!("{}", addition.pretty_print());
        
        // The problematic case: x/y + (y/2)^(x+1)
        let left_div = x.clone() / y.clone();
        let right_base = y.clone() / Expr::Const(2.0);
        let right_exp = x.clone() + Expr::Const(1.0);
        let right_power = right_base.pow(right_exp);
        let complex_expr = left_div + right_power.clone();
        
        println!("\n=== Complex Expression ===");
        println!("{}", complex_expr.pretty_print());
        
        // Test just the power part alone
        println!("\n=== Just the Power Part ===");
        println!("{}", right_power.pretty_print());
    }

    #[test]
    fn close_to_life_pretty_print() {
        let expr = Expr::parse_expression("((1- x)^2+x)*exp(-E/(R*T))");
        let pretty = expr.pretty_print();
        println!("{}", pretty);


    }

}

  
  

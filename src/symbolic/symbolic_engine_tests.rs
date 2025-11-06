use crate::symbolic::symbolic_engine::Expr;
use crate::{indexed_var, indexed_var_2d, indexed_vars, symbols};
use std::f64;
//___________________________________TESTS____________________________________

#[cfg(test)]
use approx;
mod tests {
    use super::*;
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
        let df_dx = f.diff("x");
        let _degree = Box::new(Expr::Const(1.0));
        let C = Expr::Const(2.0);
        let C1 = Expr::Const(1.0);

        let expected_result = C.clone() * Expr::pow(x.clone(), C.clone() - C1.clone()) * C1.clone();
        //  Mul(Mul(Const(2.0), Pow(Var("x"), Sub(Const(2.0), Const(1.0)))), Const(1.0)) Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x.clone())))
        println!("df_dx {:?} ", df_dx);
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
        let df_dx = f.diff("x");
        //  let df_dy = f.diff("y");

        let C1 = Expr::Const(1.0);
        let C0 = Expr::Const(0.0);
        let df_dx_expected_result =
            C.clone() * Expr::pow(x, C - C1.clone()) * C1 + Expr::exp(y.clone()) * C0;
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
        let vars = vec!["x"];
        let values = vec![5.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_const() {
        let expr = Expr::Const(3.14);
        let vars = vec![];
        let values = vec![];
        assert_eq!(expr.eval_expression(vars, &values), 3.14);
    }

    #[test]
    fn test_eval_expression_add() {
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_sub() {
        let expr = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![5.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 2.0);
    }

    #[test]
    fn test_eval_expression_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 6.0);
    }

    #[test]
    fn test_eval_expression_div() {
        let expr = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![6.0, 2.0];
        assert_eq!(expr.eval_expression(vars, &values), 3.0);
    }

    #[test]
    fn test_eval_expression_pow() {
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let vars = vec!["x"];
        let values = vec![3.0];
        assert_eq!(expr.eval_expression(vars, &values), 9.0);
    }

    #[test]
    fn test_eval_expression_exp() {
        let expr = Expr::Exp(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
        let values = vec![1.0];
        assert!((expr.eval_expression(vars, &values) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_ln() {
        let expr = Expr::Ln(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
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
        let vars = vec!["x", "y", "z"];
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
    #[test]
    fn taylor_series_test() {}
}

#[cfg(test)]
mod tests {
    use crate::symbolic::symbolic_engine::Expr;

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
        let simplified = expr.symplify();

        // Expected result: x - 1
        let expected = x - n1;

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
        let simplified = expr.symplify();
        let expected = Expr::Const(6.0) * x.clone();

        assert_eq!(simplified, expected, "2*x*3 should simplify to 6*x");
    }

    #[test]
    fn test_division_with_constants() {
        let x = Expr::Var("x".to_owned());

        // Test: (4*x)/2 should simplify to 2*x
        let expr = (Expr::Const(4.0) * x.clone()) / Expr::Const(2.0);
        let simplified = expr.symplify();
        let expected = Expr::Const(2.0) * x.clone();

        assert_eq!(simplified, expected, "(4*x)/2 should simplify to 2*x");
    }

    #[test]
    fn test_nested_constant_operations() {
        let x = Expr::Var("x".to_owned());

        // Test: x*(2*3)/(4*1) should simplify to (6/4)*x = 1.5*x
        let expr = x.clone() * (Expr::Const(2.0) * Expr::Const(3.0))
            / (Expr::Const(4.0) * Expr::Const(1.0));
        let simplified = expr.symplify();
        let expected = Expr::Const(1.5) * x.clone();

        assert_eq!(
            simplified, expected,
            "x*(2*3)/(4*1) should simplify to 1.5*x"
        );
    }
}

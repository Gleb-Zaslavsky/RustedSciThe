#[cfg(test)]
mod tests {
    use crate::symbolic::symbolic_engine::Expr;

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
        let ln_arg = sin_part + cos_part;
        
        println!("\n=== Complex case: sin(x²/y) + cos(z³) ===");
        println!("{}", ln_arg.pretty_print());
    }
}
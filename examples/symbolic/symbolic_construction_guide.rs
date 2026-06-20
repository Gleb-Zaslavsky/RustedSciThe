use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // A tiny symbolic workflow: create variables, build an expression, rename,
    // substitute and pretty-print the result.
    let (x, y, z) = RustedSciThe::symbols!(x, y, z);

    let expr = x.clone() + y.clone() * Expr::exp(z.clone());
    println!("expr (debug)        = {:?}", expr);
    println!("expr (pretty)       = {}", expr.pretty_print());

    let renamed = expr.rename_variable("z", "theta");
    println!("renamed (pretty)    = {}", renamed.pretty_print());

    let substituted = renamed
        .substitute_variable("theta", &Expr::Const(2.0))
        .set_variable("x", 1.5);
    println!("substituted (debug) = {:?}", substituted);
    println!("substituted (pretty)= {}", substituted.pretty_print());

    let mut params = std::collections::HashMap::new();
    params.insert("y".to_string(), 3.0);
    let fully_numeric = substituted.set_variable_from_map(&params);
    println!("numeric (pretty)    = {}", fully_numeric.pretty_print());
}

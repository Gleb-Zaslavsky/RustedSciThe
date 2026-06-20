use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // Короткий разбор производной: дифференцируем, упрощаем, вычисляем.
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(3.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(1.0);
    println!("f(x)        = {}", expr.pretty_print());

    let df = expr.diff("x");
    println!("f'(x) raw   = {}", df.pretty_print());

    let df_simplified = df.simplify();
    println!("f'(x) simp  = {}", df_simplified.pretty_print());

    let probe = 2.0;
    let f_at = expr.lambdify1D()(probe);
    let df_at = df_simplified.lambdify1D()(probe);
    println!("f({probe})   = {f_at}");
    println!("f'({probe})  = {df_at}");

    let check = expr.compare_num1D("x", 0.0, 4.0, 16, 1e-8);
    println!("compare_num1D = {:?}", check);
}

use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // Сначала упрощаем, потом вычисляем через lambdify.
    let x = Expr::Var("x".to_string());
    let expr = (x.clone() + x.clone()) * (x.clone() - Expr::Const(1.0));

    println!("original  = {}", expr.pretty_print());
    let simplified = expr.simplify();
    println!("simplified = {}", simplified.pretty_print());

    let f = simplified.lambdify1D();
    for value in [0.0, 1.0, 2.0, 3.0] {
        println!("f({value}) = {}", f(value));
    }

    let samples = simplified.lambdify1D_from_linspace(0.0, 3.0, 4);
    println!("samples   = {:?}", samples);
}

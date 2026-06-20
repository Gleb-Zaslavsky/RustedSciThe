use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // Indexed variables are convenient when building matrices or vector-valued
    // formulas directly.
    let (matrix, names) = Expr::IndexedVars2D(2, 3, "A");
    println!("matrix = {:?}", matrix);
    println!("names  = {:?}", names);

    let flat = Expr::IndexedVars2Dflat(2, 3, "A");
    println!("flat[0] = {:?}", flat[0]);
    println!("flat[5] = {:?}", flat[5]);

    let (u0, u1, u2) = RustedSciThe::symbols!(u0, u1, u2);
    let vector_expr = u0.clone() + u1.clone() * u2.clone();
    println!("vector_expr (pretty) = {}", vector_expr.pretty_print());

    let matrix_entry = matrix[1][2].clone() + vector_expr;
    println!("matrix_entry (pretty) = {}", matrix_entry.pretty_print());
}

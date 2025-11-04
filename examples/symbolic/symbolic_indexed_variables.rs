use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // INDEXED VARIABLES
    let (matrix_of_indexed, vec_of_names) = Expr::IndexedVars2D(1, 10, "x");
    println!("matrix_of_indexed = {:?} \n", matrix_of_indexed);
    println!("vec_of_names = {:?} \n", vec_of_names);
}
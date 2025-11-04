use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() { 
    // SOME USEFUL FEATURES 
    // a symbolic function can be defined in a more straightforward way without parsing expression
    // first define symbolic variables
    let vector_of_symbolic_vars = Expr::Symbols("a, b, c");
    println!("vector_of_symbolic_vars = {:?}", vector_of_symbolic_vars);
    let (a, b, c) = (
        vector_of_symbolic_vars[0].clone(),
        // consruct symbolic expression
        vector_of_symbolic_vars[1].clone(),
        vector_of_symbolic_vars[2].clone(),
    );
    let symbolic_expression = a + Expr::exp(b * c);
    println!("symbolic_expression = {:?}", symbolic_expression);
    // if you want to change a variable inti constant:
    let expression_with_const = symbolic_expression.set_variable("a", 1.0);
    println!("expression_with_const = {:?}", expression_with_const);
    let parsed_function = expression_with_const.sym_to_str("a");
    println!("{}, sym to string:", parsed_function);
}
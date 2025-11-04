use RustedSciThe::symbolic::symbolic_functions::Jacobian;

fn main() {
    // JACOBIAN
    // instance of Jacobian structure
    let mut Jacobian_instance = Jacobian::new();
    // function of 2 or more arguments
    let vec_of_expressions = vec!["2*x^3+y".to_string(), "1.0".to_string()];
    // set vector of functions
    Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
    // set vector of variables
    //  Jacobian_instance.set_varvecor_from_str("x, y");
    Jacobian_instance.set_variables(vec!["x", "y"]);
    // calculate symbolic jacobian
    Jacobian_instance.calc_jacobian();
    // transform into human...kind of readable form
    Jacobian_instance.readable_jacobian();
    // generate jacobian made of regular rust functions
    Jacobian_instance.jacobian_generate(vec!["x", "y"]);

    println!(
        "Jacobian_instance: functions  {:?}. Variables {:?}",
        Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables
    );
    println!(
        "Jacobian_instance: Jacobian  {:?} readable {:?}.",
        Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian
    );
    for i in 0..Jacobian_instance.symbolic_jacobian.len() {
        for j in 0..Jacobian_instance.symbolic_jacobian[i].len() {
            println!(
                "Jacobian_instance: Jacobian  {} row  {} colomn {:?}",
                i, j, Jacobian_instance.symbolic_jacobian[i][j]
            );
        }
    }
    // calculate element of jacobian (just for control)
    let ij_element =
        Jacobian_instance.calc_ij_element(0, 0, vec!["x", "y"], vec![10.0, 2.0]);
    println!("ij_element = {:?} \n", ij_element);
    // evaluate jacobian to numerical values
    Jacobian_instance.evaluate_func_jacobian(&vec![10.0, 2.0]);
    println!("Jacobian = {:?} \n", Jacobian_instance.evaluated_jacobian);
    // lambdify and evaluate function vector to numerical values
    Jacobian_instance.lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
    println!(
        "function vector = {:?} \n",
        Jacobian_instance.evaluated_functions
    );
    // or first lambdify
    Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
    // then evaluate
    Jacobian_instance.evaluate_funvector_lambdified(vec![10.0, 2.0]);
    println!(
        "function vector after evaluate_funvector_lambdified = {:?} \n",
        Jacobian_instance.evaluated_functions
    );
    // evaluate jacobian to nalgebra matrix format
    Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
    println!(
        "Jacobian_DMatrix = {:?} \n",
        Jacobian_instance.evaluated_jacobian_DMatrix
    );
    // evaluate function vector to nalgebra matrix format
    Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
    println!(
        "function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n",
        Jacobian_instance.evaluated_functions_DVector
    );
}
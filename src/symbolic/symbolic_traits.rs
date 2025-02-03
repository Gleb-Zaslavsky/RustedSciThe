// SYMBOLIC TRAITS //////////////////////////////////////////////////////////////////
// This module contains the traits for the symbolic engine.
// The traits are implemented for the native engine in the symbolic_engine.rs file
// add other engines here as needed


use crate::symbolic::symbolic_engine::Expr;
use std::any::Any;
use std::collections::HashMap;
use std::f64;

pub trait SymbolicType: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_symbolic_type(&self) -> Box<dyn SymbolicType>;
    fn diff(&self, str_variable: &str) -> Box<dyn SymbolicType>;
    fn simplify(&self) -> Box<dyn SymbolicType>;
    fn lambdify_owned(&self, variable_str: Vec<String>) -> Box<dyn Fn(Vec<f64>) -> f64>;
    fn rename_variables(&self, variables: &HashMap<String, String>) -> Box<dyn SymbolicType>;
    fn set_variable(&self, variable: &str, value: f64) -> Box<dyn SymbolicType>;
    fn convert_to_string(&self) -> String;
    fn indexed_matrix(
        self,
        num_vars: usize,
        values: Vec<String>,
    ) -> (Vec<Vec<Box<dyn SymbolicType>>>, Vec<Vec<String>>);
    fn to_native(&self) -> Expr;
    fn clone_box(&self)-> Box<dyn SymbolicType>;
    fn get_type(&self) -> String;
}
///////////////// IMPLEMENTATION OF THE TRAIT FOR THE NATIVE ENGINE /////////////////////////
impl SymbolicType for Expr {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_symbolic_type(&self) -> Box<dyn SymbolicType> {
        Box::new(self.clone())
    }
    fn diff(&self, str_variable: &str) -> Box<dyn SymbolicType> {
        let partial = Expr::diff(&self.clone(), str_variable);
        Box::new(partial)
    }
    fn simplify(&self) -> Box<dyn SymbolicType> {
        let simplified = self.symplify();
        Box::new(simplified)
    }
    fn lambdify_owned(&self, variable_str: Vec<String>) -> Box<dyn Fn(Vec<f64>) -> f64> {
        let lambdified = Expr::lambdify_owned(
            self.to_owned(),
            variable_str
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .clone(),
        );
        Box::new(lambdified)
    }
    fn rename_variables(&self, variables: &HashMap<String, String>) -> Box<dyn SymbolicType> {
        let renamed = self.rename_variables(variables);
        Box::new(renamed)
    }
    fn set_variable(&self, variable: &str, value: f64) -> Box<dyn SymbolicType> {
        let res = self.set_variable(variable, value);
        Box::new(res)
    }
    fn convert_to_string(&self) -> String {
        let to_string = self.to_string();
        to_string
    }
    fn indexed_matrix(
        self,
        num_vars: usize,
        values: Vec<String>,
    ) -> (Vec<Vec<Box<dyn SymbolicType>>>, Vec<Vec<String>>) {
        let (indexed_matrix, matrix_of_names) = Expr::IndexedVarsMatrix(num_vars, values.clone());
        let indexed_matrix: Vec<Vec<Box<dyn SymbolicType>>> = indexed_matrix
            .into_iter()
            .map(|vec_x| {
                vec_x
                    .into_iter()
                    .map(|x| Box::new(x) as Box<dyn SymbolicType>)
                    .collect()
            })
            .collect();
        (indexed_matrix, matrix_of_names)
    }
    fn to_native(&self) -> Expr {
        if let Some(native) = self.as_any().downcast_ref::<Expr>(){
            native.to_owned()
        } else {
            panic!("Type mismatch: expected Expr, got {}", self.get_type());
        }

    }
    fn clone_box(&self) -> Box<dyn SymbolicType> {
        Box::new(self.clone())
    }
    fn get_type(&self) -> String {
        "native".to_string()
    }
}

pub fn SymbolicType_casting<T: SymbolicType>(expr: T, desired_type: String) -> Box<dyn SymbolicType> {
    let res = if desired_type == "native" {
        Box::new(expr) as Box<dyn SymbolicType>
    } else {
        panic!("Unknown constant type")
    };
    res
}

/////////////////////////////////////////////////////////////////////////////////////////
// FACTORY METHODS  ////////////////////////////////////////////////////////////////////
// Add an enum to represent different symbolic engine types
pub enum SymbolicEngineType {
    Native,
    // Add other engines here as needed
    // 
}
// Create a factory trait for symbolic expressions. Added &self to the trait methods to make them object-safe
pub trait SymbolicFactory: Send + Sync { // Send + Sync is needed for the factory method to be thread-safe
    fn create_constant(&self, value: f64) -> Box<dyn SymbolicType>;
    fn create_variable(&self, name: String) -> Box<dyn SymbolicType>;
    fn parse_expression(&self, expr_str: &str) -> Box<dyn SymbolicType>;
}

// Implement factory for Native engine
pub struct NativeSymbolicFactory;

impl SymbolicFactory for NativeSymbolicFactory {
    fn create_constant(&self, value: f64) -> Box<dyn SymbolicType> {
        Box::new(Expr::Const(value))
    }

    fn create_variable(&self, name: String) -> Box<dyn SymbolicType> {
        Box::new(Expr::Var(name))
    }

    fn parse_expression(&self, expr_str: &str) -> Box<dyn SymbolicType> {
        Box::new(Expr::parse_expression(expr_str))
    }
}
// factory method to create the appropriate factory based on engine type
pub fn get_symbolic_factory(engine_type: SymbolicEngineType) -> &'static dyn SymbolicFactory {
    match engine_type {
        SymbolicEngineType::Native => &NativeSymbolicFactory,
        // Add other engines here
    }
}
pub fn symbolic_backend_from_string(engine_type: String) -> &'static dyn SymbolicFactory {
    match engine_type.as_str() {
        "native" => &NativeSymbolicFactory,
        _ => panic!("Unknown symbolic engine type"),
        // Add other engines here
    }
}

// Usage example:
// let factory = get_symbolic_factory(SymbolicEngineType::Native);
// let constant = factory.create_constant(42.0);
// let variable = factory.create_variable("x".to_string());

/*
To add a new symbolic engine, you would:
Add a new variant to SymbolicEngineType
2. Create a new implementation of SymbolicType for your engine's expression type
Create a new factory implementation for your engine
4. Add the new engine to the get_symbolic_factory match stateme
*/



 //___________________________________TESTS____________________________________

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_generic() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let f_cast = Box::new(f) as Box<dyn SymbolicType>;
        let df_dx = &f_cast.diff("x");
        let df_dx_native = df_dx.to_native();

      
        let _degree = Box::new(Expr::Const(1.0));
        let C = Expr::Const(2.0);
        let C1 = Expr::Const(1.0);

        let expected_result = C.clone() * Expr::pow(x.clone(), C.clone() - C1.clone()) * C1.clone();
        //  Mul(Mul(Const(2.0), Pow(Var("x"), Sub(Const(2.0), Const(1.0)))), Const(1.0)) Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x.clone())))
        println!("df_dx {:?} ",  df_dx_native);
        println!("expected_result {:?} ", expected_result);
        assert_eq!( df_dx_native, expected_result);
    }

    #[test] // Basic factory creation and usage
    fn test_get_symbolic_factory() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        let constant = factory.create_constant(42.0);
        let variable = factory.create_variable("x".to_string());
        
        // Test constant
        if let Some(expr) = constant.as_any().downcast_ref::<Expr>() {
            assert_eq!(*expr, Expr::Const(42.0));
        } else {
            panic!("Expected Expr type for constant");
        }

        // Test variable
        if let Some(expr) = variable.as_any().downcast_ref::<Expr>() {
            assert_eq!(*expr, Expr::Var("x".to_string()));
        } else {
            panic!("Expected Expr type for variable");
        }
    }

    // Expression parsing tests
    #[test]
    fn test_parse_expression() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        
        // Test simple expressions
        let expr = factory.parse_expression("x + 1");
        let result = expr.convert_to_string();
        assert_eq!(result, "(x + 1)");

        // Test more complex expressions
        let expr = factory.parse_expression("x * exp(y)");
        let result = expr.convert_to_string();
        assert_eq!(result, "(x * exp(y))");
    }

    // Test symbolic operations using factory-created expressions
    #[test]
    fn test_symbolic_operations() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        
        // Create expression x^2
        let x = factory.create_variable("x".to_string());
      //  let two = factory.create_constant(2.0);
        
        // Test differentiation
        let derivative = x.diff("x");
        let result = derivative.convert_to_string();
        assert_eq!(result, "1");

        // Test evaluation
        let expr = factory.parse_expression("x");
        let evaluated = expr.set_variable("x", 2.0).simplify();
      
        if let Some(native_expr) = evaluated.as_any().downcast_ref::<Expr>() {
            println!("\n \n evaluated {:?} ", native_expr.to_string());
            match native_expr {
                Expr::Const(value) => assert_eq!(*value, 2.0),
                _ => panic!("Expected constant value after evaluation"),
            }
        }
    }

    // Test error cases
    #[test]
    #[should_panic(expected = "Invalid expression")]
    fn test_invalid_expression_parsing() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        factory.parse_expression("@invalid@");
    }

    #[test]
    fn test_variable_substitution() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        
        let expr = factory.parse_expression("x + y");
        let mut substitutions = HashMap::new();
        substitutions.insert("x".to_string(), "a".to_string());
        substitutions.insert("y".to_string(), "b".to_string());
        
        let renamed = expr.rename_variables(&substitutions);
        let result = renamed.convert_to_string();
        assert_eq!(result, "(a + b)");
    }

    // Test numerical evaluation
    #[test]
    fn test_numerical_evaluation() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        
        let expr = factory.parse_expression("x^2");
        let func = expr.lambdify_owned(vec!["x".to_string()]);
        
        assert_eq!(func(vec![2.0]), 4.0);
        assert_eq!(func(vec![3.0]), 9.0);
        assert_eq!(func(vec![-2.0]), 4.0);
    }

    // Test composition of expressions
    #[test]
    fn test_expression_composition() {
        let factory = get_symbolic_factory(SymbolicEngineType::Native);
        
        let x = factory.create_variable("x".to_string());
        let _y = factory.create_variable("y".to_string());
        
        // Create expression: exp(x * y)
        let product = x.clone_box();  // Need to implement proper multiplication
        let expr = factory.parse_expression(&format!("exp({})", product.convert_to_string()));
        
        // Test the resulting expression
        let result = expr.convert_to_string();
        assert_eq!(result, "exp(x)");
    }
}

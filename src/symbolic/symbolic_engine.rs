#![allow(non_camel_case_types)]

use std::collections::HashMap;

use std::f64;
use std::fmt;

// Define an enum to represent different types of symbolic expressions

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var(String),
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
}

// Implement Display for pretty printing

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(val) => write!(f, "{}", val),
            Expr::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            Expr::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            Expr::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
            Expr::Div(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
            Expr::Pow(base, exp) => write!(f, "({} ^ {})", base, exp),
            Expr::Exp(expr) => write!(f, "exp({})", expr),
            Expr::Ln(expr) => write!(f, "ln({})", expr),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(self.boxed(), rhs.boxed())
    }
}
impl std::ops::AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        *self = Expr::Add(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::SubAssign for Expr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Expr::Sub(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Expr::Mul(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::DivAssign for Expr {
    fn div_assign(&mut self, rhs: Self) {
        *self = Expr::Div(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(self))
    }
}

/*


impl  std::f64:: for Expr {
    type Output = Self;
    fn exp(self) -> Self::Output {
        Expr::Exp(self.boxed())
    }
}

    ipml std::f64::::pow for Expr {
    type Output = Self;

    fn pow(self, rhs: Self) -> Self::Output {
        Expr::Pow(self.boxed(), rhs.boxed())
    }
}

impl f64::ops::ln for Expr {
    type Output = Self;
    fn ln(self) -> Self::Output {
        Expr::Ln(self.boxed())
    }
}
    */
// Implement differentiation, based on the recursive definition

impl Expr {
    /// BASIC FEATURES

    /// create new variables from string
    pub fn Symbols(symbols: &str) -> Vec<Expr> {
        let symbols = symbols.to_string();
        let vec_trimmed: Vec<String> = symbols.split(',').map(|s| s.trim().to_string()).collect();
        let vector_of_symbolic_vars: Vec<Expr> = vec_trimmed
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| Expr::Var(s.to_string()))
            .collect();
        vector_of_symbolic_vars
    }
    /// change a variable to a constant  
    pub fn set_variable(&self, var: &str, value: f64) -> Expr {
        match self {
            Expr::Var(name) if name == var => Expr::Const(value),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.set_variable(var, value)),
                Box::new(exp.set_variable(var, value)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.set_variable(var, value))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.set_variable(var, value))),
            _ => self.clone(),
        }
    }

    /// change a variables to a constant from a map
    pub fn set_variable_from_map(&self, var_map: &HashMap<String, f64>) -> Expr {
        match self {
            Expr::Var(name) if var_map.contains_key(name) => Expr::Const(var_map[name]),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.set_variable_from_map(var_map)),
                Box::new(exp.set_variable_from_map(var_map)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.set_variable_from_map(var_map))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.set_variable_from_map(var_map))),
            _ => self.clone(),
        }
    }
    /// rename variable
    pub fn rename_variable(&self, old_var: &str, new_var: &str) -> Expr {
        match self {
            Expr::Var(name) if name == old_var => Expr::Var(new_var.to_string()),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.rename_variable(old_var, new_var)),
                Box::new(exp.rename_variable(old_var, new_var)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.rename_variable(old_var, new_var))),
            _ => self.clone(),
        }
    }
    /// rename variables from a map
    pub fn rename_variables(&self, var_map: &HashMap<String, String>) -> Expr {
        match self {
            Expr::Var(name) if var_map.contains_key(name) => Expr::Var(var_map[name].to_string()),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.rename_variables(var_map)),
                Box::new(exp.rename_variables(var_map)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.rename_variables(var_map))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.rename_variables(var_map))),
            _ => self.clone(),
        }
    }
    // just shortcut for box
    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn var_expr(&mut self, var: &str) -> Expr {
        let expr = Expr::Var(var.to_string());
        *self = expr.clone();
        expr
    }

    pub fn const_expr(&mut self, val: f64) {
        *self = Expr::Const(val);
    }
    // implementing different functions that are not part of std
    pub fn exp(mut self) -> Expr {
        self = Expr::Exp(self.boxed());
        self
    }
    pub fn ln(mut self) -> Expr {
        self = Expr::Ln(self.boxed());
        self
    }
    pub fn log10(mut self) -> Expr {
        self = Expr::Ln(self.boxed())/Expr::Const(2.30258509);
        self
    }
    pub fn pow(mut self, rhs: Expr) -> Expr {
        self = Expr::Pow(self.boxed(), rhs.boxed());
        self
    }
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Const(val) => val == &0.0,
            _ => false,
        }
    }



    ///__________________________________INDEXED VARIABLES____________________________________
    pub fn IndexedVar(index: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}{}", var_name, index);
        Expr::Var(indexed_var_name)
    }

    pub fn IndexedVars(num_vars: usize, var_name: &str) -> (Vec<Expr>, Vec<String>) {
        let vec_of_expr = (0..num_vars)
            .map(|i| Expr::IndexedVar(i, var_name))
            .collect();
        let vec_of_names = (0..num_vars)
            .map(|i| format!("{}_{}", var_name, i))
            .collect();
        (vec_of_expr, vec_of_names)
    }
    pub fn IndexedVarsMatrix(
        num_vars: usize,
        var_names: Vec<String>,
    ) -> (Vec<Vec<Expr>>, Vec<Vec<String>>) {
        let mut matrix = Vec::new();
        let mut matrix_of_expr = Vec::new();
        for i in 0..num_vars {
            let mut matrix_i = Vec::new();
            let mut matrix_of_expr_i = Vec::new();
            for j in 0..var_names.len() {
                let indexed_var_name = format!("{}_{}", var_names[j], i);
                matrix_i.push(indexed_var_name.clone());
                matrix_of_expr_i.push(Expr::Var(indexed_var_name));
            }
            matrix.push(matrix_i);
            matrix_of_expr.push(matrix_of_expr_i);
        }
        (matrix_of_expr, matrix)
    }
    // 2D indexation"x2_315", "Z21_235"
    pub fn IndexedVar2D(index_row: usize, index_col: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}_{}_{}", var_name, index_row, index_col);
        Expr::Var(indexed_var_name)
    }

    pub fn IndexedVars2D(
        num_rows: usize,
        num_cols: usize,
        var_name: &str,
    ) -> (Vec<Vec<Expr>>, Vec<String>) {
        let mut vec_of_names: Vec<String> = Vec::new();
        let matrix = (0..num_rows)
            .map(|i| {
                (0..num_cols)
                    .map(|j| {
                        let indexed_var_name = format!("{}_{}_{}", var_name, i, j);

                        Expr::Var(indexed_var_name)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for i in 0..num_rows {
            for j in 0..num_cols {
                let indexed_var_name = format!("{}_{}_{}", var_name, i, j);
                vec_of_names.push(indexed_var_name);
            }
        }
        (matrix, vec_of_names)
    }
    pub fn IndexedVars2Dflat(num_rows: usize, num_cols: usize, var_name: &str) -> Vec<Expr> {
        (0..num_rows)
            .flat_map(|i| (0..num_cols).map(move |j| Expr::IndexedVar2D(i, j, var_name)))
            .collect()
    }
    //___________________________________SYMPLIFICATE____________________________________
    //  function to symplify the symbolic expression in sych way: if in expression there is a
    // subexpression Mul(Expr::Const(0.0), ...something ) the function turns all subexpression into Expr::Const(0.0)
    #[allow(dead_code)]
    fn nozeros(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) && simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Add(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Sub(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) && simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Sub(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Mul(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) || simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Mul(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Div(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Pow(base, exp) => Expr::Pow(Box::new(base.nozeros()), Box::new(exp.nozeros())),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.nozeros())),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.nozeros())),
        }
    } // nozeros

    pub fn simplify_numbers(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                    (lhs, rhs) => Expr::Add(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                    (lhs, rhs) => Expr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                    (lhs, rhs) => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a / b),
                    (lhs, rhs) => Expr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.simplify_numbers()),
                Box::new(exp.simplify_numbers()),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.simplify_numbers())),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.simplify_numbers())),
        }
    }
    pub fn simplify_(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) =>Expr::Const(a + b),// (a) + (b) = (a + b) 
                    (Expr::Const(0.0), _) => rhs, // x + 0 = x
                    (_, Expr::Const(0.0)) => lhs,//  0 + x = x
                    _ => Expr::Add(Box::new(lhs), Box::new(rhs) ),
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) =>Expr::Const(a - b),// (a) - (b) = (a - b)
                    (_, Expr::Const(0.0)) => lhs, // x - 0 = x
                    _ =>Expr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),// (a) * (b) = (a * b)
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) =>Expr::Const(0.0), // 0 * x = 0 or 0*x = 0
                    (Expr::Const(1.0), _) => rhs, // 1 * x = x
                    (_, Expr::Const(1.0)) => lhs, // x * 1 = x
                    _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Expr::Const(a / b),// (a) / (b) = (a / b)
                    (Expr::Const(0.0), _) => Expr::Const(0.0),// (0.0) / x = 0.0
                    (_, Expr::Const(1.0)) => lhs,// x / 1.0 = x
                    _ => Expr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Pow(base, exp) => {
                let base = base.simplify_();
                let exp = exp.simplify_();
                match (&base, &exp) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a.powf(*b)), // (a) ^ (b) = (a ^ b)
                    (_, Expr::Const(0.0)) => Expr::Const(1.0), // x ^ 0 = 1
                    (_, Expr::Const(1.0)) => base, // x ^ 1 = x
                    (Expr::Const(0.0), _) => Expr::Const(0.0), // 0 ^ x = 0
                    (Expr::Const(1.0), _) => Expr::Const(1.0), // 1 ^ x = 1
                    _ => Expr::Pow(Box::new(base), Box::new(exp)),
                }
            }
            Expr::Exp(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) if a!=&0.0 =>Expr::Const(a.exp()),
                    Expr::Const(0.0) => Expr::Const(1.0),
                    _ => Expr::Exp(Box::new(expr)),
                }
            }
            Expr::Ln(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(1.0) => Expr::Const(0.0),
                    Expr::Const(a) if *a > 0.0 => Expr::Const(a.ln()),
                    _ => Expr::Ln(Box::new(expr)),
                }
            }
        }
    }
    pub fn symplify(&self) -> Expr {
        //let zeros_proceeded = self.nozeros().simplify_numbers();
        let zeros_proceeded = self.simplify_();
        zeros_proceeded
    }


}

//___________________________________TESTS____________________________________

#[cfg(test)]
use approx;
mod tests {
    use super::*;
    #[test]
    fn test_add_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        let expected = Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_sub_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr -= Expr::Const(2.0);
        let expected = Expr::Sub(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr *= Expr::Const(2.0);
        let expected = Expr::Mul(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_div_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_neg() {
        let expr = Expr::Var("x".to_string());
        let neg_expr = -expr;
        let expected = Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(Expr::Var("x".to_string())));
        assert_eq!(neg_expr, expected);
    }
    #[test]
    fn test_combined_operations() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        expr *= Expr::Const(3.0);
        expr -= Expr::Const(1.0);
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(
            Box::new(Expr::Sub(
                Box::new(Expr::Mul(
                    Box::new(Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)))),
                    Box::new(Expr::Const(3.0))
                )),
                Box::new(Expr::Const(1.0))
            )),
            Box::new(Expr::Const(2.0))
        );
        assert_eq!(expr, expected);
    }
    #[test]
    fn test_diff() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let df_dx = f.diff("x");
        let _degree = Box::new(Expr::Const(1.0));
        let C = Expr::Const(2.0);
        let C1 = Expr::Const(1.0);

        let expected_result = C.clone() * Expr::pow(x.clone(), C.clone() - C1.clone()) * C1.clone();
        //  Mul(Mul(Const(2.0), Pow(Var("x"), Sub(Const(2.0), Const(1.0)))), Const(1.0)) Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x.clone())))
        println!("df_dx {:?} ", df_dx);
        println!("expected_result {:?} ", expected_result);
        assert_eq!(df_dx, expected_result);
    }

    #[test]
    fn test_sym_to_str() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let rust_function = f.sym_to_str("x");
        assert_eq!(rust_function, "(x^2)");
    }

    #[test]
    fn test_lambdify1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let fn_closure = f.lambdify1D();
        assert_eq!(fn_closure(2.0), 4.0);
    }
    #[test]
    fn test_constuction_of_expression() {
        let vector_of_symbolic_vars = Expr::Symbols("a, b, c");
        let (a, b, c) = (
            vector_of_symbolic_vars[0].clone(),
            vector_of_symbolic_vars[1].clone(),
            vector_of_symbolic_vars[2].clone(),
        );
        let symbolic_expression = a + Expr::exp(b * c);
        let expression_with_const = symbolic_expression.set_variable("a", 1.0);
        let parsed_function = expression_with_const.sym_to_str("a");
        assert_eq!(parsed_function, "(1) + (exp((b) * (c)))");
    }
    #[test]
    fn test_1D() {
        let input = "log(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 0.0);
        let df_dx = f.diff("x");
        let df_dx_str = df_dx.sym_to_str("x");
        assert_eq!(df_dx_str, "(1) / (x)");
    }
    #[test]
    fn test_1D_2() {
        let input = "x+exp(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 1.0 + f64::consts::E);
        let start = 0.0;
        let end = 10f64;
        let num_values = 100;
        let max_norm = 1e-6;
        let (_normm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
        assert_eq!(res, true);
    }
    #[test]
    fn test_multi_diff() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let C = Expr::Const(3.0);
        let f = Expr::pow(x.clone(), C.clone()) + Expr::exp(y.clone());
        let df_dx = f.diff("x");
        //  let df_dy = f.diff("y");

        let C1 = Expr::Const(1.0);
        let C0 = Expr::Const(0.0);
        let df_dx_expected_result =
            C.clone() * Expr::pow(x, C - C1.clone()) * C1 + Expr::exp(y.clone()) * C0;
        //  let df_dy_expected_result = C* Expr::exp(y);
        assert_eq!(df_dx, df_dx_expected_result);
        let start = vec![1.0, 1.0];
        let end = vec![2.0, 2.0];
        let comparsion = f.compare_num(start, end, 100, 1e-6);
        let bool_1 = &comparsion[0].0;
        let bool_2 = &comparsion[1].0;

        assert_eq!(*bool_1 && *bool_2, true);
        //    assert_eq!(df_dy, expected_result);
    }

    #[test]
    fn test_set_variable() {
        let x = Expr::Var("x".to_string());
        let f = x.clone() + Expr::Const(2.0);
        let f_with_value = f.set_variable("x", 1.0);
        let expected_result = Expr::Const(1.0) + Expr::Const(2.0);
        assert_eq!(f_with_value, expected_result);
    }

    #[test]
    fn test_calc_vector_lambdified1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let _fn_closureee = f.lambdify1D();
        let x_values = vec![1.0, 2.0, 3.0];
        let result = f.calc_vector_lambdified1D(&x_values);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_lambdify1D_from_linspace() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let result = f.lambdify1D_from_linspace(1.0, 3.0, 3);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }
    /*
    #[test]
    fn test_evaluate_vector_lambdified() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let x_values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_values = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = f.evaluate_vector_lambdified(&x_values, &y_values);
        let expected_result = vec![1.0 + 27.18281828459045, 4.0 + 74.08182845904523, 9.0 + 162.31828459045235];
        assert_eq!(result, expected_result);
    }

    */
    #[test]
    fn test_evaluate_multi_diff_from_linspace() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let result = f.evaluate_multi_diff_from_linspace(vec![1.0, 1.0], vec![2.0, 2.0], 100);
        let last_element = result[0].last().unwrap();

        let expected_result: f64 = 4.0f64; // 2*2
        assert!((last_element - expected_result).abs() < f64::EPSILON);
    }
    #[test]
    fn lambdify_IVP_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn lambdify_IVP_owned_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP_owned("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn no_zeros_test() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0)),
        );

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test2() {
        let expr = Expr::Sub(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test3() {
        let expr = Expr::Add(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }

    #[test]
    fn no_zeros_test4() {
        let zero = Box::new(Expr::Const(0.0));
        let added = Expr::Add(zero.clone(), zero.clone()); // 0
        let mulled = Expr::Mul(Box::new(Expr::Const(0.005)), Box::new(added)); //0
        let expr = Box::new(Expr::Sub(
            zero.clone(),
            Box::new(Expr::Add(zero, Box::new(mulled))),
        ));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn test_eval_expression_var() {
        let expr = Expr::Var("x".to_string());
        let vars = vec!["x"];
        let values = vec![5.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_const() {
        let expr = Expr::Const(3.14);
        let vars = vec![];
        let values = vec![];
        assert_eq!(expr.eval_expression(vars, &values), 3.14);
    }

    #[test]
    fn test_eval_expression_add() {
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_sub() {
        let expr = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![5.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 2.0);
    }

    #[test]
    fn test_eval_expression_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 6.0);
    }

    #[test]
    fn test_eval_expression_div() {
        let expr = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![6.0, 2.0];
        assert_eq!(expr.eval_expression(vars, &values), 3.0);
    }

    #[test]
    fn test_eval_expression_pow() {
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let vars = vec!["x"];
        let values = vec![3.0];
        assert_eq!(expr.eval_expression(vars, &values), 9.0);
    }

    #[test]
    fn test_eval_expression_exp() {
        let expr = Expr::Exp(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
        let values = vec![1.0];
        assert!((expr.eval_expression(vars, &values) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_ln() {
        let expr = Expr::Ln(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
        let values = vec![std::f64::consts::E];
        assert!((expr.eval_expression(vars, &values) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_complex() {
        let expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Var("z".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let vars = vec!["x", "y", "z"];
        let values = vec![2.0, 3.0, 4.0];
        assert_eq!(expr.eval_expression(vars, &values), 22.0); // (2 * 3) + (4^2) = 22
    }
    #[test]
    fn test_taylor_series1D_constant() {
        let expr = Expr::Const(5.0);
        let result = expr.taylor_series1D("x", 0.0, 3);
        assert_eq!(result, Expr::Const(5.0));
    }
    #[test]
    fn test_taylor_series1D_log() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().ln();
        let result = expr.taylor_series1D_("x", 5.0, 2);
        let e5 = Expr::Const(5.0);
        let expected = e5.clone().ln() + (x.clone() - e5.clone()) /  e5.clone() - (x.clone() - e5.clone()).pow(Expr::Const(2.0)) / (Expr::Const(2.0)* e5.clone().pow(Expr::Const(2.0)) );  
        println!("{} \n {}", result, expected.symplify());
        let taylor_eval = result.lambdify1D()(3.0);
        let expected_eval = expected.lambdify1D()(3.0);
        approx::assert_relative_eq!(taylor_eval, expected_eval, epsilon=1e-5);
    }
    #[test]
    fn test_taylor_series1D_exp() {
        let x = Expr::Var("x".to_string());

        let exp_expansion  = Expr::Const(1.0) + x.clone() + x.clone().pow(Expr::Const(2.0))/Expr::Const(2.0) +  x.clone().pow(Expr::Const(3.0))/Expr::Const(6.0);
        let exp_eval = exp_expansion.lambdify1D()(1.0);
       
        let taylor = exp_expansion.taylor_series1D_("x", 0.0, 3);
         println!("taylor: {}", taylor);
        let taylor_eval = taylor.lambdify1D()(1.0);
        assert_eq!(taylor_eval, exp_eval);
    }


}

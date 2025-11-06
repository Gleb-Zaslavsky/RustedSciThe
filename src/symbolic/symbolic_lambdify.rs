use crate::symbolic::symbolic_engine::Expr;
use std::f64::consts::PI;
const LAMBDIFY_METHOD: usize = 0;

impl Expr {
    /// LAMBDIFICATION - Converting Symbolic Expressions to Executable Functions

    /// Converts a single-variable symbolic expression into an executable Rust closure.
    ///
    /// This is the core method for numerical computation, transforming symbolic math
    /// into high-performance executable code. The resulting closure can be called
    /// repeatedly with different input values.
    ///
    /// # Returns
    /// Boxed closure that takes f64 input and returns f64 output
    ///
    /// # Performance Notes
    /// - Creates optimized closure with minimal overhead
    /// - Recursive structure mirrors expression tree for efficiency
    /// - No runtime parsing or interpretation
    ///
    /// # Examples
    /// ```rust, ignore
    /// let x = Expr::Var("x".to_string());
    /// let f = x.pow(Expr::Const(2.0)); // x^2
    /// let func = f.lambdify1D();
    /// assert_eq!(func(3.0), 9.0);
    /// ```
    pub fn lambdify1D(&self) -> Box<dyn Fn(f64) -> f64> {
        let vars = self.all_arguments_are_variables();
        if vars.len() == 1 {
            let var_name = &vars[0];
            let compiled_func = self.lambdify_borrowed_thread_safe(&[var_name]);
            Box::new(move |x| compiled_func(&[x]))
        } else if vars.is_empty() {
            // Constant expression
            let compiled_func = self.lambdify_borrowed_thread_safe(&[]);
            Box::new(move |_| compiled_func(&[]))
        } else {
            panic!(
                "lambdify1D can only be used with expressions containing exactly one variable, found: {:?}",
                vars
            );
        }
    } // end of lambdify1D
    pub fn lambdify1Dlegacy(&self) -> Box<dyn Fn(f64) -> f64> {
        match self {
            Expr::Var(_) => Box::new(|x| x),
            Expr::Const(val) => {
                let val = *val;
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) + rhs_fn(x))
            }

            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) - rhs_fn(x))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) * rhs_fn(x))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) / rhs_fn(x))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify1D();
                let exp_fn = exp.lambdify1D();
                Box::new(move |x| base_fn(x).powf(exp_fn(x)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(1.0 / x).tan())
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| PI / 2.0 - expr_fn(x).atan())
            }
        } // end of match
    } // end of lambdify1D
    #[inline(always)]
    pub fn lambdify_borrowed_thread_safe(
        &self,
        vars: &[&str],
    ) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
        match LAMBDIFY_METHOD {
            0 => self.lambdify1(vars),

            _ => self.lambdify2(vars),
        }
    }

    #[inline(always)]
    pub fn lambdify1(&self, vars: &[&str]) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                Box::new(move |args| args[index])
            }
            Expr::Const(val) => {
                let val = *val;
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lf = lhs.lambdify_borrowed_thread_safe(vars);
                let rf = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lf(args) + rf(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lf = lhs.lambdify_borrowed_thread_safe(vars);
                let rf = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lf(args) - rf(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lf = lhs.lambdify_borrowed_thread_safe(vars);
                let rf = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lf(args) * rf(args))
            }
            Expr::Div(lhs, rhs) => {
                let lf = lhs.lambdify_borrowed_thread_safe(vars);
                let rf = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lf(args) / rf(args))
            }
            Expr::Pow(b, e) => {
                let bf = b.lambdify_borrowed_thread_safe(vars);
                let ef = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| bf(args).powf(ef(args)))
            }
            Expr::Exp(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).exp())
            }
            Expr::Ln(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).ln())
            }
            Expr::sin(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).sin())
            }
            Expr::cos(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).cos())
            }
            Expr::tg(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).tan())
            }
            Expr::ctg(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| 1.0 / f(args).tan())
            }
            Expr::arcsin(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).asin())
            }
            Expr::arccos(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).acos())
            }
            Expr::arctg(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| f(args).atan())
            }
            Expr::arcctg(e) => {
                let f = e.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| (PI / 2.0) - f(args).atan())
            }
        }
    } // end of lambdify
    /// Convenience method that automatically detects variables and creates a closure.
    ///
    /// Extracts all variables from the expression and creates a lambdified function
    /// with variables ordered alphabetically. Eliminates need to manually specify
    /// variable names.
    ///
    /// # Returns
    /// Boxed closure where input vector positions correspond to alphabetically
    /// sorted variable names
    ///
    /// # Usage
    /// Ideal for quick evaluation when variable order doesn't matter
    pub fn lambdify_wrapped(&self) -> Box<dyn Fn(Vec<f64>) -> f64 + '_> {
        let vars_ = self.all_arguments_are_variables();
        let vars = vars_.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        let clo = self.lambdify_borrowed_thread_safe(vars.as_slice());
        let y = Box::new(move |x: Vec<f64>| clo(x.as_slice()));
        y
    }
    #[inline(always)]
    pub fn lambdify2(&self, vars: &[&str]) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
        let compiled = self.compile(vars);
        let closure = compiled.as_closure();
        Box::new(closure)
    } // end of lambdify
}

#[derive(Clone, Debug)]
pub enum Lambda {
    Var(usize),
    Const(f64),
    Add(Box<Lambda>, Box<Lambda>),
    Sub(Box<Lambda>, Box<Lambda>),
    Mul(Box<Lambda>, Box<Lambda>),
    Div(Box<Lambda>, Box<Lambda>),
    Pow(Box<Lambda>, Box<Lambda>),
    Exp(Box<Lambda>),
    Ln(Box<Lambda>),
    Sin(Box<Lambda>),
    Cos(Box<Lambda>),
    Tg(Box<Lambda>),
    Ctg(Box<Lambda>),
    ArcSin(Box<Lambda>),
    ArcCos(Box<Lambda>),
    ArcTg(Box<Lambda>),
    ArcCtg(Box<Lambda>),
}

impl Expr {
    pub fn compile(&self, vars: &[&str]) -> Lambda {
        match self {
            Expr::Var(name) => {
                let idx = vars.iter().position(|&v| v == name).unwrap();
                Lambda::Var(idx)
            }
            Expr::Const(v) => Lambda::Const(*v),
            Expr::Add(a, b) => Lambda::Add(Box::new(a.compile(vars)), Box::new(b.compile(vars))),
            Expr::Sub(a, b) => Lambda::Sub(Box::new(a.compile(vars)), Box::new(b.compile(vars))),
            Expr::Mul(a, b) => Lambda::Mul(Box::new(a.compile(vars)), Box::new(b.compile(vars))),
            Expr::Div(a, b) => Lambda::Div(Box::new(a.compile(vars)), Box::new(b.compile(vars))),
            Expr::Pow(a, b) => Lambda::Pow(Box::new(a.compile(vars)), Box::new(b.compile(vars))),
            Expr::Exp(e) => Lambda::Exp(Box::new(e.compile(vars))),
            Expr::Ln(e) => Lambda::Ln(Box::new(e.compile(vars))),
            Expr::sin(e) => Lambda::Sin(Box::new(e.compile(vars))),
            Expr::cos(e) => Lambda::Cos(Box::new(e.compile(vars))),
            Expr::tg(e) => Lambda::Tg(Box::new(e.compile(vars))),
            Expr::ctg(e) => Lambda::Ctg(Box::new(e.compile(vars))),
            Expr::arcsin(e) => Lambda::ArcSin(Box::new(e.compile(vars))),
            Expr::arccos(e) => Lambda::ArcCos(Box::new(e.compile(vars))),
            Expr::arctg(e) => Lambda::ArcTg(Box::new(e.compile(vars))),
            Expr::arcctg(e) => Lambda::ArcCtg(Box::new(e.compile(vars))),
        }
    }
}

impl Lambda {
    #[inline(always)]
    pub fn eval(&self, args: &[f64]) -> f64 {
        match self {
            Lambda::Var(i) => args[*i],
            Lambda::Const(v) => *v,
            Lambda::Add(a, b) => a.eval(args) + b.eval(args),
            Lambda::Sub(a, b) => a.eval(args) - b.eval(args),
            Lambda::Mul(a, b) => a.eval(args) * b.eval(args),
            Lambda::Div(a, b) => a.eval(args) / b.eval(args),
            Lambda::Pow(a, b) => a.eval(args).powf(b.eval(args)),
            Lambda::Exp(e) => e.eval(args).exp(),
            Lambda::Ln(e) => e.eval(args).ln(),
            Lambda::Sin(e) => e.eval(args).sin(),
            Lambda::Cos(e) => e.eval(args).cos(),
            Lambda::Tg(e) => e.eval(args).tan(),
            Lambda::Ctg(e) => 1.0 / e.eval(args).tan(),
            Lambda::ArcSin(e) => e.eval(args).asin(),
            Lambda::ArcCos(e) => e.eval(args).acos(),
            Lambda::ArcTg(e) => e.eval(args).atan(),
            Lambda::ArcCtg(e) => (PI / 2.0) - e.eval(args).atan(),
        }
    }

    /// Optional API for compatibility with previous closure-based code
    pub fn as_closure(self) -> impl Fn(&[f64]) -> f64 + Send + Sync {
        move |args| self.eval(args)
    }
}

impl Expr {
    //____________________________________________________________________________________________________________________________
    //                    INITIAL VALUE PROBLEM (IVP) SPECIALIZATION
    //____________________________________________________________________________________________________________________________

    /// Creates closure specialized for Initial Value Problems with time-dependent functions.
    ///
    /// In IVPs, functions typically have the form f(t, y1, y2, ...) where t is the
    /// independent variable (time) and y1, y2, ... are state variables. This method
    /// creates a closure that separates the time argument from state variables.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically time "t" or "x")
    /// * `vars` - State variable names in order
    ///
    /// # Returns
    /// Closure taking (time_value, state_vector) and returning f64
    ///
    /// # Usage
    /// Essential for ODE solvers where dy/dt = f(t, y)
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = parse_expression("t + y1 + y2"); // f(t, y1, y2)
    /// let func = expr.lambdify_IVP("t", vec!["y1", "y2"]);
    /// let result = func(1.0, vec![2.0, 3.0]); // f(1.0, 2.0, 3.0) = 6.0
    /// ```
    pub fn lambdify_IVP(
        &self,
        arg: &str,
        vars: Vec<&str>,
    ) -> Box<dyn Fn(f64, Vec<f64>) -> f64 + '_> {
        let mut x = vec![arg];
        x.extend(vars); // extend vars.clone();

        let f = self.lambdify_borrowed_thread_safe(&x);

        let f_closure: Box<dyn Fn(f64, Vec<f64>) -> f64> = Box::new(move |x, y_vec| {
            // Assuming y_vec has at least one element
            let mut x_y_vec = vec![x];
            x_y_vec.extend(y_vec); // extend vars.clone();

            // Call the original closure with x and y
            f(x_y_vec.as_slice())
        });
        f_closure
    }

    /// Owned version of lambdify_IVP that consumes the expression.
    ///
    /// Similar to lambdify_IVP but takes ownership of the expression, allowing
    /// the returned closure to have 'static lifetime. Useful for ODE solvers
    /// that need to store the function long-term.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically time)
    /// * `vars` - State variable names in order
    ///
    /// # Returns
    /// Closure with 'static lifetime taking (time, state_vector) -> f64
    ///
    /// # Ownership
    /// Consumes self, preventing further use of the original expression
    pub fn lambdify_IVP_owned(
        self,
        arg: &str,
        vars: Vec<&str>,
    ) -> Box<dyn Fn(f64, Vec<f64>) -> f64> {
        let mut x = vec![arg];
        x.extend(vars); // extend vars.clone();
        //  println!("x {:?}",&x);
        let f = self.lambdify_borrowed_thread_safe(&x);

        let f_closure: Box<dyn Fn(f64, Vec<f64>) -> f64> = Box::new(move |x, y_vec| {
            // Assuming y_vec has at least one element
            let mut x_y_vec = vec![x];
            x_y_vec.extend(y_vec); // extend vars.clone();

            // Call the original closure with x and y
            f(x_y_vec.as_slice())
        });
        f_closure
    }
}
/////////////////////////////TESTS/////////////////////
pub fn parse_very_complex_expression() -> Expr {
    let s = " (0.000002669 * (28.0 * T)^0.5) /
        (13.3225 * ((1.16145 / ((T / 98.1) ^ 0.14874))
                  + (0.52487 / exp(0.7732 * (T / 98.1)))
                  + (2.16178 / exp(2.43787 * (T / 98.1)))
                  + ((0.2 * 0.0 ^ 2) / (T / 98.1))))";
    let s = Expr::parse_expression(s);
    let s1 = s.diff("T");
    let s2 = s1.diff("T");
    s2
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn test_lambdify1d_single_variable() {
        let x = Expr::Var("x".to_string());
        let func = x.lambdify1D();
        assert_eq!(func(5.0), 5.0);
    }

    #[test]
    fn test_lambdify1d_constant() {
        let c = Expr::Const(42.0);
        let func = c.lambdify1D();
        assert_eq!(func(100.0), 42.0);
    }

    #[test]
    fn test_lambdify1d_polynomial() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0); // x^2 + 2x + 1
        let func = expr.lambdify1D();
        assert_eq!(func(3.0), 16.0); // 9 + 6 + 1 = 16
    }

    #[test]
    fn test_lambdify1d_trigonometric() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(x));
        let func = expr.lambdify1D();
        assert!((func(0.0) - 0.0).abs() < 1e-10);
        assert!((func(PI / 2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lambdify1d_exponential() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Exp(Box::new(x));
        let func = expr.lambdify1D();
        assert!((func(0.0) - 1.0).abs() < 1e-10);
        assert!((func(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    #[should_panic(
        expected = "lambdify1D can only be used with expressions containing exactly one variable"
    )]
    fn test_lambdify1d_multiple_variables_panic() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = x + y;
        let _func = expr.lambdify1D();
    }
    #[test]
    fn test_lambdify_wrapped() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0); // x^2 + 2x + 1
        let func = expr.lambdify_wrapped();
        assert_eq!(func(vec![3.0]), 16.0); // 9 + 6 + 1 = 16
    }
    #[test]
    fn lambdify1d_comapare() {
        let expr = parse_very_complex_expression();
        let start = Instant::now();
        let vars = expr.all_arguments_are_variables();
        let vars_extracting_time = start.elapsed();
        //println!("expr {:?}", expr);
        let start = Instant::now();
        let func = expr.lambdify1Dlegacy();
        let x = func(1.0);
        let end = start.elapsed();

        let start = Instant::now();
        let func = expr.lambdify1D();
        let y = func(1.0);
        let duration = start.elapsed();
        assert_eq!(x, y);
        println!("\n lambdify1d_comapare {:?}", duration);
        println!("\n lambdify1d_comapare legacy {:?}", end);
        println!(
            "\n vars_extracting_time {:?}, vars {:?}",
            vars_extracting_time, vars
        );
    }
}

use crate::symbolic::symbolic_engine::Expr;
use std::fmt;

use std::collections::HashMap;
/// Enum to represent the root finding methods
#[derive(Debug, Clone)]
pub enum RootFindingMethod {
    Bisection,
    Secant,
    NewtonRaphson,
}

/// Error types for root finding methods
#[derive(Debug, Clone)]
pub enum RootFindingError {
    MaxIterationsReached,
    InvalidInterval,
    FunctionNotContinuous,
    DerivativeZero,
    ToleranceNotMet,
    InvalidInput(String),
}

impl fmt::Display for RootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RootFindingError::MaxIterationsReached => write!(f, "Maximum iterations reached"),
            RootFindingError::InvalidInterval => write!(f, "Invalid interval for bisection method"),
            RootFindingError::FunctionNotContinuous => {
                write!(f, "Function is not continuous in the given interval")
            }
            RootFindingError::DerivativeZero => write!(f, "Derivative is zero"),
            RootFindingError::ToleranceNotMet => write!(f, "Required tolerance not met"),
            RootFindingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
//////////////////////////////////TRAITS AND IMPLEMENTATIONS/////////////////////////////////
impl std::error::Error for RootFindingError {}

/// Trait for representing a nonlinear equation f(x) = 0
pub trait NonlinearFunction {
    /// Evaluate the function at point x
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate the derivative at point x (optional, for methods that can use it)
    fn derivative(&self, _x: f64) -> Option<f64> {
        None
    }

    /// Get function name for debugging/logging
    fn name(&self) -> &str {
        "unnamed_function"
    }
}

/// Simple function wrapper for closures
pub struct ClosureFunction<F>
where
    F: Fn(f64) -> f64,
{
    func: F,
    name: String,
}

impl<F> ClosureFunction<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(func: F, name: String) -> Self {
        Self { func, name }
    }
}

impl<F> NonlinearFunction for ClosureFunction<F>
where
    F: Fn(f64) -> f64,
{
    fn evaluate(&self, x: f64) -> f64 {
        (self.func)(x)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Function wrapper with analytical derivative
pub struct FunctionWithDerivative<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    func: F,
    derivative_func: D,
    name: String,
}

impl<F, D> FunctionWithDerivative<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub fn new(func: F, derivative_func: D, name: String) -> Self {
        Self {
            func,
            derivative_func,
            name,
        }
    }
}

impl<F, D> NonlinearFunction for FunctionWithDerivative<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    fn evaluate(&self, x: f64) -> f64 {
        (self.func)(x)
    }

    fn derivative(&self, x: f64) -> Option<f64> {
        Some((self.derivative_func)(x))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

////////////////////////SYMBOLIC FUNCTIONS//////////////////////////////////////////
/// Symbolic function wrapper that uses symbolic expressions
pub struct SymbolicFunction {
    original_expr: Expr, // Store original expression without parameter substitution
    expr: Expr,
    derivative_expr: Option<Expr>,
    variable: String,
    name: String,
    func: Box<dyn Fn(f64) -> f64>,
    derivative_func: Option<Box<dyn Fn(f64) -> f64>>,
    parameters: HashMap<String, f64>,
}

impl SymbolicFunction {
    /// Create a new symbolic function from a string expression
    pub fn from_string(
        expr_str: &str,
        variable: &str,
        name: Option<String>,
    ) -> Result<Self, RootFindingError> {
        let expr = Expr::parse_expression(expr_str);
        let func_name = name.unwrap_or_else(|| format!("symbolic_function({})", expr_str));

        Self::from_expr(expr, variable, Some(func_name))
    }

    /// Create a new symbolic function from an Expr
    pub fn from_expr(
        expr: Expr,
        variable: &str,
        name: Option<String>,
    ) -> Result<Self, RootFindingError> {
        let func_name = name.unwrap_or_else(|| "symbolic_function".to_string());

        // Compute derivative symbolically
        let derivative_expr = expr.diff(variable);

        // Convert to lambdified functions
        let func = expr.lambdify1D();
        let derivative_func = derivative_expr.lambdify1D();

        Ok(Self {
            original_expr: expr.clone(), // Store original
            expr: expr.clone(),
            derivative_expr: Some(derivative_expr),
            variable: variable.to_string(),
            name: func_name,
            func,
            derivative_func: Some(derivative_func),
            parameters: HashMap::new(),
        })
    }

    /// Create symbolic function with parameters
    pub fn from_string_with_params(
        expr_str: &str,
        variable: &str,
        parameters: HashMap<String, f64>,
        name: Option<String>,
    ) -> Result<Self, RootFindingError> {
        let expr = Expr::parse_expression(expr_str);
        let func_name = name.unwrap_or_else(|| format!("symbolic_function({})", expr_str));

        Self::from_expr_with_params(expr, variable, parameters, Some(func_name))
    }

    /// Create symbolic function from Expr with parameters
    pub fn from_expr_with_params(
        expr: Expr,
        variable: &str,
        parameters: HashMap<String, f64>,
        name: Option<String>,
    ) -> Result<Self, RootFindingError> {
        let func_name = name.unwrap_or_else(|| "symbolic_function_with_params".to_string());

        // Substitute parameters into the expression
        let expr_with_params = expr.set_variable_from_map(&parameters);

        // Compute derivative symbolically
        let derivative_expr = expr_with_params.diff(variable);

        // Convert to lambdified functions
        let func = expr_with_params.lambdify1D();
        let derivative_func = derivative_expr.lambdify1D();

        Ok(Self {
            original_expr: expr.clone(), // Store original without parameters
            expr: expr_with_params,
            derivative_expr: Some(derivative_expr),
            variable: variable.to_string(),
            name: func_name,
            func,
            derivative_func: Some(derivative_func),
            parameters,
        })
    }

    /// Update parameters and regenerate functions
    pub fn set_parameters(
        &mut self,
        parameters: HashMap<String, f64>,
    ) -> Result<(), RootFindingError> {
        self.parameters = parameters;

        // Use original expression and apply new parameters
        let expr_with_params = self.original_expr.set_variable_from_map(&self.parameters);
        let derivative_expr = expr_with_params.diff(&self.variable);

        // Update the current expression
        self.expr = expr_with_params.clone();
        self.derivative_expr = Some(derivative_expr.clone());

        // Regenerate lambdified functions
        self.func = expr_with_params.lambdify1D();
        self.derivative_func = Some(derivative_expr.lambdify1D());

        Ok(())
    }

    /// Get the symbolic expression as string
    pub fn expression_string(&self) -> String {
        self.expr.sym_to_str(&self.variable)
    }

    /// Get the derivative expression as string
    pub fn derivative_string(&self) -> Option<String> {
        self.derivative_expr
            .as_ref()
            .map(|expr| expr.sym_to_str(&self.variable))
    }
}

impl NonlinearFunction for SymbolicFunction {
    fn evaluate(&self, x: f64) -> f64 {
        (self.func)(x)
    }

    fn derivative(&self, x: f64) -> Option<f64> {
        self.derivative_func.as_ref().map(|f| f(x))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

///////////////////////////////////////////SETTERS AND GETTERS///////////////////////////////////////////
/// Result structure for root finding methods
#[derive(Debug, Clone)]
pub struct RootFindingResult {
    pub root: f64,
    pub function_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub method: String,
}

/// Configuration for root finding methods
#[derive(Debug, Clone)]
pub struct RootFindingConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub verbose: bool,
}

impl Default for RootFindingConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 100,
            verbose: false,
        }
    }
}

/// Main structure for scalar root finding methods
pub struct ScalarRootFinder {
    config: RootFindingConfig,
}

impl ScalarRootFinder {
    /// Create a new ScalarRootFinder with default configuration
    pub fn new() -> Self {
        Self {
            config: RootFindingConfig::default(),
        }
    }

    /// Create a new ScalarRootFinder with custom configuration
    pub fn with_config(config: RootFindingConfig) -> Self {
        Self { config }
    }

    /// Set tolerance for convergence
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.config.tolerance = tolerance;
    }

    /// Set maximum number of iterations
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.config.max_iterations = max_iterations;
    }

    /// Enable or disable verbose output
    pub fn set_verbose(&mut self, verbose: bool) {
        self.config.verbose = verbose;
    }
    /////////////////////////////IMPLEMENTING SYMBOLIC FUNCTIONS///////////////////////////////////////////
    /// Solve using symbolic expression string
    pub fn solve_symbolic_str(
        &self,
        expr_str: &str,
        variable: &str,
        method: RootFindingMethod,
        initial_guess: f64,
        search_range: Option<(f64, f64)>,
        parameters: Option<HashMap<String, f64>>,
    ) -> Result<RootFindingResult, RootFindingError> {
        let symbolic_func = if let Some(params) = parameters {
            SymbolicFunction::from_string_with_params(expr_str, variable, params, None)?
        } else {
            SymbolicFunction::from_string(expr_str, variable, None)?
        };

        self.solve_with_method(&symbolic_func, method, initial_guess, search_range)
    }

    /// Solve using symbolic Expr
    pub fn solve_symbolic_expr(
        &self,
        expr: Expr,
        variable: &str,
        method: RootFindingMethod,
        initial_guess: f64,
        search_range: Option<(f64, f64)>,
        parameters: Option<HashMap<String, f64>>,
    ) -> Result<RootFindingResult, RootFindingError> {
        let symbolic_func = if let Some(params) = parameters {
            SymbolicFunction::from_expr_with_params(expr, variable, params, None)?
        } else {
            SymbolicFunction::from_expr(expr, variable, None)?
        };

        self.solve_with_method(&symbolic_func, method, initial_guess, search_range)
    }

    /// Solve with specific method
    pub fn solve_with_method<F>(
        &self,
        function: &F,
        method: RootFindingMethod,
        initial_guess: f64,
        search_range: Option<(f64, f64)>,
    ) -> Result<RootFindingResult, RootFindingError>
    where
        F: NonlinearFunction,
    {
        match method {
            RootFindingMethod::Bisection => {
                if let Some((a, b)) = search_range {
                    self.bisection(function, a, b)
                } else {
                    Err(RootFindingError::InvalidInput(
                        "Bisection method requires search range".to_string(),
                    ))
                }
            }
            RootFindingMethod::Secant => {
                let x1 = initial_guess + 0.01 * initial_guess.abs().max(1.0);
                self.secant(function, initial_guess, x1)
            }
            RootFindingMethod::NewtonRaphson => self.newton_raphson(function, initial_guess),
        }
    }

    /////////////////////////////////////////METHODS///////////////////////////////////////////

    /// Bisection method for finding roots
    /// Requires that f(a) and f(b) have opposite signs
    pub fn bisection<F>(
        &self,
        function: &F,
        mut a: f64,
        mut b: f64,
    ) -> Result<RootFindingResult, RootFindingError>
    where
        F: NonlinearFunction,
    {
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }

        let fa = function.evaluate(a);
        let fb = function.evaluate(b);

        // Check if the function values have opposite signs
        if fa * fb > 0.0 {
            return Err(RootFindingError::InvalidInterval);
        }

        // Check if we already have a root at the endpoints
        if fa.abs() < self.config.tolerance {
            return Ok(RootFindingResult {
                root: a,
                function_value: fa,
                iterations: 0,
                converged: true,
                method: "bisection".to_string(),
            });
        }

        if fb.abs() < self.config.tolerance {
            return Ok(RootFindingResult {
                root: b,
                function_value: fb,
                iterations: 0,
                converged: true,
                method: "bisection".to_string(),
            });
        }

        let mut iterations = 0;
        let mut c: f64;
        let mut fc: f64;

        if self.config.verbose {
            println!("Bisection method for function: {}", function.name());
            println!("Initial interval: [{}, {}]", a, b);
            println!("Tolerance: {}", self.config.tolerance);
        }

        while iterations < self.config.max_iterations {
            c = (a + b) / 2.0;
            fc = function.evaluate(c);

            if self.config.verbose {
                println!(
                    "Iteration {}: x = {:.10}, f(x) = {:.2e}, interval = [{:.6}, {:.6}]",
                    iterations + 1,
                    c,
                    fc,
                    a,
                    b
                );
            }

            // Check for convergence
            if fc.abs() < self.config.tolerance || (b - a) / 2.0 < self.config.tolerance {
                return Ok(RootFindingResult {
                    root: c,
                    function_value: fc,
                    iterations: iterations + 1,
                    converged: true,
                    method: "bisection".to_string(),
                });
            }

            // Update interval
            if function.evaluate(a) * fc < 0.0 {
                b = c;
            } else {
                a = c;
            }

            iterations += 1;
        }

        Err(RootFindingError::MaxIterationsReached)
    }

    /// Secant method for finding roots
    /// Requires two initial guesses x0 and x1
    pub fn secant<F>(
        &self,
        function: &F,
        mut x0: f64,
        mut x1: f64,
    ) -> Result<RootFindingResult, RootFindingError>
    where
        F: NonlinearFunction,
    {
        let mut f0 = function.evaluate(x0);
        let mut f1 = function.evaluate(x1);

        if self.config.verbose {
            println!("Secant method for function: {}", function.name());
            println!("Initial guesses: x0 = {}, x1 = {}", x0, x1);
            println!("Tolerance: {}", self.config.tolerance);
        }

        // Check if we already have a root
        if f0.abs() < self.config.tolerance {
            return Ok(RootFindingResult {
                root: x0,
                function_value: f0,
                iterations: 0,
                converged: true,
                method: "secant".to_string(),
            });
        }

        if f1.abs() < self.config.tolerance {
            return Ok(RootFindingResult {
                root: x1,
                function_value: f1,
                iterations: 0,
                converged: true,
                method: "secant".to_string(),
            });
        }

        let mut iterations = 0;

        while iterations < self.config.max_iterations {
            // Check if the denominator is too small
            if (f1 - f0).abs() < 1e-15 {
                return Err(RootFindingError::DerivativeZero);
            }

            // Calculate next approximation using secant formula
            let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
            let f2 = function.evaluate(x2);

            if self.config.verbose {
                println!(
                    "Iteration {}: x = {:.10}, f(x) = {:.2e}",
                    iterations + 1,
                    x2,
                    f2
                );
            }

            // Check for convergence
            if f2.abs() < self.config.tolerance || (x2 - x1).abs() < self.config.tolerance {
                return Ok(RootFindingResult {
                    root: x2,
                    function_value: f2,
                    iterations: iterations + 1,
                    converged: true,
                    method: "secant".to_string(),
                });
            }

            // Update for next iteration
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f2;

            iterations += 1;
        }

        Err(RootFindingError::MaxIterationsReached)
    }

    /// Newton-Raphson method (bonus method that uses derivative if available)
    pub fn newton_raphson<F>(
        &self,
        function: &F,
        mut x: f64,
    ) -> Result<RootFindingResult, RootFindingError>
    where
        F: NonlinearFunction,
    {
        if self.config.verbose {
            println!("Newton-Raphson method for function: {}", function.name());
            println!("Initial guess: x0 = {}", x);
            println!("Tolerance: {}", self.config.tolerance);
        }

        let mut iterations = 0;

        while iterations < self.config.max_iterations {
            let fx = function.evaluate(x);

            if self.config.verbose {
                println!(
                    "Iteration {}: x = {:.10}, f(x) = {:.2e}",
                    iterations + 1,
                    x,
                    fx
                );
            }

            // Check for convergence
            if fx.abs() < self.config.tolerance {
                return Ok(RootFindingResult {
                    root: x,
                    function_value: fx,
                    iterations: iterations + 1,
                    converged: true,
                    method: "newton_raphson".to_string(),
                });
            }

            // Get derivative
            let fpx = match function.derivative(x) {
                Some(deriv) => deriv,
                None => {
                    // Use numerical differentiation if analytical derivative is not available
                    let h = 1e-8;
                    (function.evaluate(x + h) - function.evaluate(x - h)) / (2.0 * h)
                }
            };

            // Check if derivative is too small
            if fpx.abs() < 1e-15 {
                return Err(RootFindingError::DerivativeZero);
            }

            // Newton-Raphson update
            let x_new = x - fx / fpx;

            // Check for convergence in x
            if (x_new - x).abs() < self.config.tolerance {
                return Ok(RootFindingResult {
                    root: x_new,
                    function_value: function.evaluate(x_new),
                    iterations: iterations + 1,
                    converged: true,
                    method: "newton_raphson".to_string(),
                });
            }

            x = x_new;
            iterations += 1;
        }

        Err(RootFindingError::MaxIterationsReached)
    }

    /// Hybrid method that tries different approaches
    pub fn solve<F>(
        &self,
        function: &F,
        initial_guess: f64,
        search_range: Option<(f64, f64)>,
    ) -> Result<RootFindingResult, RootFindingError>
    where
        F: NonlinearFunction,
    {
        // Try Newton-Raphson first if derivative is available
        if function.derivative(initial_guess).is_some() {
            if let Ok(result) = self.newton_raphson(function, initial_guess) {
                return Ok(result);
            }
        }

        // Try secant method with perturbed initial guess
        let x1 = initial_guess + 0.01 * initial_guess.abs().max(1.0);
        if let Ok(result) = self.secant(function, initial_guess, x1) {
            return Ok(result);
        }

        // Finally try bisection if search range is provided
        if let Some((a, b)) = search_range {
            return self.bisection(function, a, b);
        }

        Err(RootFindingError::ToleranceNotMet)
    }
}

impl Default for ScalarRootFinder {
    fn default() -> Self {
        Self::new()
    }
}

// ... (previous code remains the same until the convenience functions)

// Convenience functions for quick usage
pub fn bisection<F>(function: F, a: f64, b: f64, tolerance: f64) -> Result<f64, RootFindingError>
where
    F: Fn(f64) -> f64,
{
    let func = ClosureFunction::new(function, "bisection_function".to_string());
    let mut solver = ScalarRootFinder::new();
    solver.set_tolerance(tolerance);
    let result = solver.bisection(&func, a, b)?;
    Ok(result.root)
}

pub fn secant<F>(function: F, x0: f64, x1: f64, tolerance: f64) -> Result<f64, RootFindingError>
where
    F: Fn(f64) -> f64,
{
    let func = ClosureFunction::new(function, "secant_function".to_string());
    let mut solver = ScalarRootFinder::new();
    solver.set_tolerance(tolerance);
    let result = solver.secant(&func, x0, x1)?;
    Ok(result.root)
}
/////////////////////////////////////////TESTS////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Helper function to check if two floats are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_closure_function() {
        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());
        assert_eq!(func.evaluate(2.0), 0.0);
        assert_eq!(func.evaluate(-2.0), 0.0);
        assert_eq!(func.evaluate(0.0), -4.0);
        assert_eq!(func.name(), "x^2 - 4");
    }

    #[test]
    fn test_function_with_derivative() {
        let func = FunctionWithDerivative::new(
            |x| x * x - 4.0,
            |x| 2.0 * x,
            "x^2 - 4 with derivative".to_string(),
        );
        assert_eq!(func.evaluate(2.0), 0.0);
        assert_eq!(func.derivative(2.0), Some(4.0));
        assert_eq!(func.derivative(3.0), Some(6.0));
    }

    #[test]
    fn test_bisection_simple_quadratic() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        // Test positive root
        let result = solver.bisection(&func, 0.0, 3.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "bisection");

        // Test negative root
        let result = solver.bisection(&func, -3.0, 0.0).unwrap();
        assert!(approx_equal(result.root, -2.0, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_bisection_cubic() {
        let solver = ScalarRootFinder::new();
        // f(x) = x^3 - x - 1, root approximately at x = 1.324717957
        let func = ClosureFunction::new(|x| x * x * x - x - 1.0, "x^3 - x - 1".to_string());

        let result = solver.bisection(&func, 1.0, 2.0).unwrap();
        let expected_root = 1.324717957244746;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_bisection_trigonometric() {
        let solver = ScalarRootFinder::new();
        // f(x) = sin(x), root at x = π
        let func = ClosureFunction::new(|x| x.sin(), "sin(x)".to_string());

        let result = solver.bisection(&func, 3.0, 4.0).unwrap();
        assert!(approx_equal(result.root, PI, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_bisection_invalid_interval() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x * x + 1.0, "x^2 + 1".to_string());

        // This function has no real roots, so bisection should fail
        let result = solver.bisection(&func, -1.0, 1.0);
        assert!(matches!(result, Err(RootFindingError::InvalidInterval)));
    }

    #[test]
    fn test_bisection_root_at_endpoint() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x - 2.0, "x - 2".to_string());

        // Root is exactly at the right endpoint
        let result = solver.bisection(&func, 1.0, 2.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_secant_simple_quadratic() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        // Test positive root
        let result = solver.secant(&func, 1.0, 3.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "secant");

        // Test negative root
        let result = solver.secant(&func, -1.0, -3.0).unwrap();
        assert!(approx_equal(result.root, -2.0, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_secant_cubic() {
        let solver = ScalarRootFinder::new();
        // f(x) = x^3 - x - 1
        let func = ClosureFunction::new(|x| x * x * x - x - 1.0, "x^3 - x - 1".to_string());

        let result = solver.secant(&func, 1.0, 2.0).unwrap();
        let expected_root = 1.324717957244746;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);

        let result = solver.bisection(&func, 1.0, 2.0).unwrap();
        let expected_root = 1.324717957244746;
        println!("result.root Bisection: {}", result.root);
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);

        let result = solver.newton_raphson(&func, 1.0).unwrap();
        let expected_root = 1.324717957244746;
        println!("result.root Newton: {}", result.root);
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_secant_exponential() {
        let solver = ScalarRootFinder::new();
        // f(x) = e^x - 2, root at x = ln(2)
        let func = ClosureFunction::new(|x| x.exp() - 2.0, "e^x - 2".to_string());

        let result = solver.secant(&func, 0.0, 1.0).unwrap();
        let expected_root = 2.0_f64.ln();
        assert!(approx_equal(result.root, expected_root, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_secant_root_at_initial_guess() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x - 2.0, "x - 2".to_string());

        // Root is exactly at the first initial guess
        let result = solver.secant(&func, 2.0, 3.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_secant_derivative_zero_error() {
        let solver = ScalarRootFinder::new();
        // Function that will cause f1 - f0 to be very small
        let func = ClosureFunction::new(|_x| 1.0, "constant function".to_string());

        let result = solver.secant(&func, 1.0, 1.0001);
        assert!(matches!(result, Err(RootFindingError::DerivativeZero)));
    }

    #[test]
    fn test_newton_raphson_with_derivative() {
        let solver = ScalarRootFinder::new();
        let func = FunctionWithDerivative::new(
            |x| x * x - 4.0,
            |x| 2.0 * x,
            "x^2 - 4 with derivative".to_string(),
        );

        let result = solver.newton_raphson(&func, 1.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "newton_raphson");
    }

    #[test]
    fn test_newton_raphson_without_derivative() {
        let solver = ScalarRootFinder::new();
        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        // Should use numerical differentiation
        let result = solver.newton_raphson(&func, 1.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_newton_raphson_complex_function() {
        let solver = ScalarRootFinder::new();
        // f(x) = x^3 - 2x - 5, f'(x) = 3x^2 - 2
        let func = FunctionWithDerivative::new(
            |x| x * x * x - 2.0 * x - 5.0,
            |x| 3.0 * x * x - 2.0,
            "x^3 - 2x - 5".to_string(),
        );

        let result = solver.newton_raphson(&func, 2.0).unwrap();
        let expected_root = 2.094551481542327; // Approximate root
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_solver_configuration() {
        let config = RootFindingConfig {
            tolerance: 1e-6,
            max_iterations: 50,
            verbose: false,
        };
        let solver = ScalarRootFinder::with_config(config);
        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        let result = solver.bisection(&func, 0.0, 3.0).unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-6));
        assert!(result.converged);
    }

    #[test]
    fn test_solver_max_iterations() {
        let mut solver = ScalarRootFinder::new();
        solver.set_max_iterations(5); // Very low iteration limit
        solver.set_tolerance(1e-15); // Very high precision requirement

        let func = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        let result = solver.bisection(&func, 0.0, 3.0);
        assert!(matches!(
            result,
            Err(RootFindingError::MaxIterationsReached)
        ));
    }

    #[test]
    fn test_hybrid_solve_method() {
        let solver = ScalarRootFinder::new();

        // Test with function that has derivative
        let func_with_deriv = FunctionWithDerivative::new(
            |x| x * x - 4.0,
            |x| 2.0 * x,
            "x^2 - 4 with derivative".to_string(),
        );

        let result = solver
            .solve(&func_with_deriv, 1.0, Some((0.0, 3.0)))
            .unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert_eq!(result.method, "newton_raphson");

        // Test with function without derivative
        let func_without_deriv = ClosureFunction::new(|x| x * x - 4.0, "x^2 - 4".to_string());

        let result = solver
            .solve(&func_without_deriv, 1.0, Some((0.0, 3.0)))
            .unwrap();
        assert!(approx_equal(result.root, 2.0, 1e-9));
        // Should use secant method since no analytical derivative
        assert_eq!(result.method, "secant");
    }

    #[test]
    fn test_convenience_functions() {
        // Test convenience bisection function
        let root = bisection(|x| x * x - 4.0, 0.0, 3.0, 1e-10).unwrap();
        assert!(approx_equal(root, 2.0, 1e-10));

        // Test convenience secant function
        let root = secant(|x| x * x - 4.0, 1.0, 3.0, 1e-10).unwrap();
        assert!(approx_equal(root, 2.0, 1e-10));
    }

    #[test]
    fn test_real_world_examples() {
        let solver = ScalarRootFinder::new();

        // Example 1: Finding where two curves intersect
        // f(x) = x^2 and g(x) = 2x + 3, solve x^2 - 2x - 3 = 0
        // Roots are x = 3 and x = -1
        let intersection_func =
            ClosureFunction::new(|x| x * x - 2.0 * x - 3.0, "x^2 - 2x - 3".to_string());

        let positive_root = solver.bisection(&intersection_func, 0.0, 5.0).unwrap();
        assert!(approx_equal(positive_root.root, 3.0, 1e-10));

        let negative_root = solver.bisection(&intersection_func, -5.0, 0.0).unwrap();
        assert!(approx_equal(negative_root.root, -1.0, 1e-10));

        // Example 2: Finding break-
        // even point for a cost and revenue function\
    }
}

///////////////////////////////SYMOBOLIC TESTING////////////////////////////////////
#[cfg(test)]
mod symbolic_tests {
    use super::*;
    use std::collections::HashMap;
    use std::f64::consts::E;

    // Helper function to check if two floats are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_symbolic_function_creation_from_string() {
        // Test simple quadratic function
        let func =
            SymbolicFunction::from_string("x^2 - 4", "x", Some("quadratic".to_string())).unwrap();
        assert_eq!(func.name(), "quadratic");
        assert!(approx_equal(func.evaluate(2.0), 0.0, 1e-10));
        assert!(approx_equal(func.evaluate(-2.0), 0.0, 1e-10));
        assert!(approx_equal(func.evaluate(0.0), -4.0, 1e-10));

        // Test derivative
        assert!(func.derivative(2.0).is_some());
        assert!(approx_equal(func.derivative(2.0).unwrap(), 4.0, 1e-10));
        assert!(approx_equal(func.derivative(3.0).unwrap(), 6.0, 1e-10));
    }

    #[test]
    fn test_symbolic_function_with_parameters() {
        let mut params = HashMap::new();
        params.insert("a".to_string(), 2.0);
        params.insert("b".to_string(), -8.0);

        // f(x) = a*x^2 + b = 2*x^2 - 8, roots at x = ±2
        let func = SymbolicFunction::from_string_with_params(
            "a*x^2 + b",
            "x",
            params,
            Some("parametric_quadratic".to_string()),
        )
        .unwrap();

        assert!(approx_equal(func.evaluate(2.0), 0.0, 1e-10));
        assert!(approx_equal(func.evaluate(-2.0), 0.0, 1e-10));
        assert!(approx_equal(func.evaluate(0.0), -8.0, 1e-10));

        // Test derivative: f'(x) = 2*a*x = 4*x
        assert!(approx_equal(func.derivative(2.0).unwrap(), 8.0, 1e-10));
        assert!(approx_equal(func.derivative(1.0).unwrap(), 4.0, 1e-10));
    }

    #[test]
    fn test_symbolic_function_parameter_update() {
        let mut params = HashMap::new();
        params.insert("a".to_string(), 1.0);

        let mut func =
            SymbolicFunction::from_string_with_params("a*x^2 - 4", "x", params, None).unwrap();

        // Initially a=1, so f(x) = x^2 - 4, root at x=2
        assert!(approx_equal(func.evaluate(2.0), 0.0, 1e-10));

        // Update parameter a=4, so f(x) = 4*x^2 - 4, root at x=1
        let mut new_params = HashMap::new();
        new_params.insert("a".to_string(), 4.0);
        func.set_parameters(new_params).unwrap();
        // 4*1^2 - 4,
        println!("func.evaluate(1.0) ,{}", func.evaluate(1.0));
        assert!(approx_equal(func.evaluate(1.0), 0.0, 1e-10));
        assert!(approx_equal(func.evaluate(2.0), 12.0, 1e-10));
    }

    #[test]
    fn test_solve_symbolic_quadratic_newton_raphson() {
        let solver = ScalarRootFinder::new();

        // Solve x^2 - 4 = 0 using Newton-Raphson
        let result = solver
            .solve_symbolic_str(
                "x^2 - 4",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.0, // initial guess
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "newton_raphson");
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_symbolic_quadratic_bisection() {
        let solver = ScalarRootFinder::new();

        // Solve x^2 - 4 = 0 using bisection
        let result = solver
            .solve_symbolic_str(
                "x^2 - 4",
                "x",
                RootFindingMethod::Bisection,
                1.0,              // initial guess (not used for bisection)
                Some((0.0, 3.0)), // search range
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "bisection");
    }

    #[test]
    fn test_solve_symbolic_quadratic_secant() {
        let solver = ScalarRootFinder::new();

        // Solve x^2 - 4 = 0 using secant method
        let result = solver
            .solve_symbolic_str(
                "x^2 - 4",
                "x",
                RootFindingMethod::Secant,
                1.0, // initial guess
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
        assert_eq!(result.method, "secant");
    }

    #[test]
    fn test_solve_symbolic_cubic() {
        // fails
        let solver = ScalarRootFinder::new();

        // Solve x^3 - x - 1 = 0, root approximately at x = 1.324717957
        let result = solver
            .solve_symbolic_str(
                "x^3 - (x + 1)",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.5, // initial guess
                None,
                None,
            )
            .unwrap();
        println!("result.root: {}", result.root);
        let expected_root = 1.324717957244746;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }
    #[test]
    fn test_solve_symbolic_cubic2() {
        // fails
        let solver = ScalarRootFinder::new();

        // Solve x^3 - x - 1 = 0, root approximately at x = 1.324717957
        let result = solver
            .solve_symbolic_str(
                "x^3 - (x + 1)",
                "x",
                RootFindingMethod::Secant,
                1.5, // initial guess
                None,
                None,
            )
            .unwrap();
        println!("result.root: {}", result.root);
        let expected_root = 1.324717957244746;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }
    #[test]
    fn test_solve_symbolic_cubic3() {
        // fails
        let solver = ScalarRootFinder::new();

        // Solve x^3 - x - 1 = 0, root approximately at x = 1.324717957
        let result = solver
            .solve_symbolic_str(
                "x^3 - (x + 1)",
                "x",
                RootFindingMethod::Bisection,
                1.5,                // initial guess
                Some((-10.0, 3.0)), // search range
                None,
            )
            .unwrap();
        println!("result.root: {}", result.root);
        let expected_root = 1.324717957244746;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn SymbolicFunction_from_string_with_params() {
        let func = SymbolicFunction::from_string("x^3-(x+1)", "x", None).unwrap();
        let F = func.func;
        let exp = func.expr;
        let f = F(0.0);
        println!("f: {}", f);
        println!("exp: {}", exp);
    }
    #[test]
    fn test_solve_symbolic_exponential() {
        let solver = ScalarRootFinder::new();

        // Solve exp(x) - 2 = 0, root at x = ln(2)
        let result = solver
            .solve_symbolic_str(
                "exp(x) - 2",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.0, // initial guess
                None,
                None,
            )
            .unwrap();

        let expected_root = 2.0_f64.ln();
        assert!(approx_equal(result.root, expected_root, 1e-10));
        assert!(result.converged);
    }
    /*
       #[test]
       fn test_solve_symbolic_trigonometric() {
           let solver = ScalarRootFinder::new();

           // Solve sin(x) = 0, looking for root near π
           let result = solver.solve_symbolic_str(
               "s",
               "x",
               RootFindingMethod::Bisection,
               3.0, // initial guess
               Some((3.0, 4.0)), // search range around π
               None
           ).unwrap();

           assert!(approx_equal(result.root, PI, 1e-10));
           assert!(result.converged);
       }
    */
    #[test]
    fn test_solve_symbolic_with_parameters() {
        let solver = ScalarRootFinder::new();

        let mut params = HashMap::new();
        params.insert("a".to_string(), 3.0);
        params.insert("b".to_string(), -12.0);

        // Solve a*x^2 + b = 0 => 3*x^2 - 12 = 0 => x^2 = 4 => x = ±2
        let result = solver
            .solve_symbolic_str(
                "a*x^2 + b",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.0, // initial guess
                None,
                Some(params),
            )
            .unwrap();

        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_solve_symbolic_complex_polynomial() {
        let solver = ScalarRootFinder::new();

        // Solve x^4 - 10*x^2 + 9 = 0
        // This factors as (x^2 - 1)(x^2 - 9) = 0
        // Roots are x = ±1, ±3
        let result = solver
            .solve_symbolic_str(
                "x^4 - 10*x^2 + 9",
                "x",
                RootFindingMethod::NewtonRaphson,
                0.5, // initial guess to find root at x=1
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, 1.0, 1e-9));
        assert!(result.converged);

        // Test for another root
        let result2 = solver
            .solve_symbolic_str(
                "x^4 - 10*x^2 + 9",
                "x",
                RootFindingMethod::NewtonRaphson,
                2.5, // initial guess to find root at x=3
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result2.root, 3.0, 1e-9));
        assert!(result2.converged);
    }

    #[test]
    fn test_solve_symbolic_rational_function() {
        let solver = ScalarRootFinder::new();

        // Solve (x^2 - 4)/(x + 1) = 0
        // This has roots where numerator = 0 and denominator ≠ 0
        // So roots are x = ±2 (x = -1 is excluded)
        let result = solver
            .solve_symbolic_str(
                "(x^2 - 4)/(x + 1)",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.0, // initial guess
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, 2.0, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_solve_symbolic_logarithmic() {
        let solver = ScalarRootFinder::new();

        // Solve ln(x) - 1 = 0, root at x = e
        let result = solver
            .solve_symbolic_str(
                "ln(x) - 1",
                "x",
                RootFindingMethod::Secant,
                2.0, // initial guess
                None,
                None,
            )
            .unwrap();

        assert!(approx_equal(result.root, E, 1e-10));
        assert!(result.converged);
    }

    #[test]
    fn test_solve_symbolic_mixed_functions() {
        let solver = ScalarRootFinder::new();

        // Solve x*exp(x) - 1 = 0
        // This is the equation for finding the inverse of x*e^x
        let result = solver
            .solve_symbolic_str(
                "x*exp(x) - 1",
                "x",
                RootFindingMethod::NewtonRaphson,
                0.5, // initial guess
                None,
                None,
            )
            .unwrap();

        // The root is approximately 0.567143290409784
        let expected_root = 0.567143290409784;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_solve_symbolic_expr_object() {
        // fails
        let solver = ScalarRootFinder::new();

        // Create symbolic expression directly
        let expr = Expr::parse_expression("x^3 - (2*x + 5)");

        let result = solver
            .solve_symbolic_expr(
                expr,
                "x",
                RootFindingMethod::NewtonRaphson,
                2.0, // initial guess
                None,
                None,
            )
            .unwrap();
        println!("result.root: {}", result.root);
        // Expected root is approximately 2.094551481542327
        let expected_root = 2.094551481542327;
        assert!(approx_equal(result.root, expected_root, 1e-9));
        assert!(result.converged);
    }

    #[test]
    fn test_symbolic_function_expressions_strings() {
        let func = SymbolicFunction::from_string("x^2 + 3*x - 4", "x", None).unwrap();

        let expr_str = func.expression_string();
        let deriv_str = func.derivative_string().unwrap();

        // Check that we can get string representations
        assert!(expr_str.contains("x"));
        assert!(deriv_str.contains("x"));

        println!("Expression: {}", expr_str);
        println!("Derivative: {}", deriv_str);
    }

    #[test]
    fn test_solve_symbolic_with_different_tolerances() {
        let mut solver = ScalarRootFinder::new();
        solver.set_tolerance(1e-15); // Very high precision

        let result = solver
            .solve_symbolic_str(
                "x^2 - 2",
                "x",
                RootFindingMethod::NewtonRaphson,
                1.0,
                None,
                None,
            )
            .unwrap();

        // Root should be sqrt(2)
        let expected_root = 2.0_f64.sqrt();
        assert!(approx_equal(result.root, expected_root, 1e-14));
        assert!(result.converged);
    }
    /*
    #[test]
    fn test_solve_symbolic_bisection_requires_range() {
        let solver = ScalarRootFinder::new();

        // Bisection should fail without search range
        let result = solver.solve_symbolic_str(
            "x^2 - 4",
            "x",
            RootFindingMethod::Bisection,
            1.0,
            None, // No search range provided
            None
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            RootFindingError::InvalidInput(msg) => {
                assert!(msg.contains("Bisection method requires search range"));
            }
        }
    }
    */
}

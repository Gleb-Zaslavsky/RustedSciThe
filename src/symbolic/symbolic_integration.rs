use crate::symbolic::symbolic_engine::Expr;
use gauss_quad::{GaussHermite, GaussLaguerre, GaussLegendre};
use std::collections::HashMap;
impl Expr {
    /// SYMBOLIC INTEGRATION

    /// Main integration method - integrates with respect to a variable
    /// Returns the indefinite integral (without constant of integration)
    ///  This module deals with simple integrals.
    pub fn integrate(&self, var: &str) -> Result<Expr, String> {
        match self {
            // ∫ c dx = c*x
            Expr::Const(c) => Ok(Expr::Const(*c) * Expr::Var(var.to_string())),

            // ∫ x dx = x²/2, ∫ y dx = y*x (if y ≠ x)
            Expr::Var(name) => {
                if name == var {
                    Ok(Expr::Pow(
                        Box::new(Expr::Var(var.to_string())),
                        Box::new(Expr::Const(2.0)),
                    ) / Expr::Const(2.0))
                } else {
                    Ok(Expr::Var(name.clone()) * Expr::Var(var.to_string()))
                }
            }

            // ∫ (f + g) dx = ∫ f dx + ∫ g dx
            Expr::Add(lhs, rhs) => {
                let lhs_int = lhs.integrate(var)?;
                let rhs_int = rhs.integrate(var)?;
                Ok(lhs_int + rhs_int)
            }

            // ∫ (f - g) dx = ∫ f dx - ∫ g dx
            Expr::Sub(lhs, rhs) => {
                let lhs_int = lhs.integrate(var)?;
                let rhs_int = rhs.integrate(var)?;
                Ok(lhs_int - rhs_int)
            }

            // Handle multiplication cases
            Expr::Mul(lhs, rhs) => self.integrate_multiplication(lhs, rhs, var),

            // Handle division cases
            Expr::Div(lhs, rhs) => self.integrate_division(lhs, rhs, var),

            // ∫ x^n dx = x^(n+1)/(n+1) for n ≠ -1
            Expr::Pow(base, exp) => self.integrate_power(base, exp, var),

            // ∫ e^f dx - requires chain rule or substitution
            Expr::Exp(expr) => self.integrate_exponential(expr, var),

            // ∫ ln(f) dx - requires integration by parts
            Expr::Ln(expr) => self.integrate_logarithm(expr, var),

            // ∫ sin(f) dx
            Expr::sin(expr) => self.integrate_sin(expr, var),

            // ∫ cos(f) dx
            Expr::cos(expr) => self.integrate_cos(expr, var),

            // ∫ tg(f) dx
            Expr::tg(expr) => self.integrate_tan(expr, var),

            // ∫ ctg(f) dx
            Expr::ctg(expr) => self.integrate_cot(expr, var),

            // ∫ arcsin(f) dx
            Expr::arcsin(expr) => self.integrate_arcsin(expr, var),

            // ∫ arccos(f) dx
            Expr::arccos(expr) => self.integrate_arccos(expr, var),

            // ∫ arctg(f) dx
            Expr::arctg(expr) => self.integrate_arctan(expr, var),

            // ∫ arcctg(f) dx
            Expr::arcctg(expr) => self.integrate_arccot(expr, var),
        }
    }

    /// Enhanced multiplication integration that tries different strategies
    fn integrate_multiplication(&self, lhs: &Expr, rhs: &Expr, var: &str) -> Result<Expr, String> {
        // Check if one factor is constant
        if !lhs.contains_variable(var) {
            let rhs_int = rhs.integrate(var)?;
            return Ok(lhs.clone() * rhs_int);
        }

        if !rhs.contains_variable(var) {
            let lhs_int = lhs.integrate(var)?;
            return Ok(rhs.clone() * lhs_int);
        }

        // Try to recognize patterns for integration by parts
        // Pattern 1: polynomial * exponential
        if let Some(result) = self.integrate_polynomial_times_exponential(lhs, rhs, var) {
            return Ok(result);
        }
        if let Some(result) = self.integrate_polynomial_times_exponential(rhs, lhs, var) {
            return Ok(result);
        }

        // Pattern 2: polynomial * logarithm
        if let Some(result) = self.integrate_polynomial_times_logarithm(lhs, rhs, var) {
            return Ok(result);
        }
        if let Some(result) = self.integrate_polynomial_times_logarithm(rhs, lhs, var) {
            return Ok(result);
        }

        // Fall back to general integration by parts
        self.try_integration_by_parts(lhs, rhs, var)
    }

    /// Handle division in integration
    fn integrate_division(&self, lhs: &Expr, rhs: &Expr, var: &str) -> Result<Expr, String> {
        // If denominator is constant: ∫ f(x)/c dx = (1/c) * ∫ f(x) dx
        if !rhs.contains_variable(var) {
            let lhs_int = lhs.integrate(var)?;
            return Ok(lhs_int / rhs.clone());
        }

        // Special case: ∫ f'(x)/f(x) dx = ln|f(x)|
        if let Ok(derivative) = rhs.diff(var).simplify().to_string().parse::<String>() {
            if let Ok(numerator_str) = lhs.simplify().to_string().parse::<String>() {
                if derivative == numerator_str {
                    return Ok(Expr::Ln(Box::new(rhs.clone())));
                }
            }
        }

        // Special case: ∫ 1/x dx = ln|x|
        if let (Expr::Const(1.0), Expr::Var(x)) = (lhs, rhs) {
            if x == var {
                return Ok(Expr::Ln(Box::new(Expr::Var(var.to_string()))));
            }
        }

        Err(format!("Cannot integrate division: {} / {}", lhs, rhs))
    }

    /// Handle power integration
    fn integrate_power(&self, base: &Expr, exp: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ x^n dx where n is constant
        if let (Expr::Var(x), Expr::Const(n)) = (base, exp) {
            if x == var {
                if (*n - (-1.0)).abs() < f64::EPSILON {
                    // ∫ x^(-1) dx = ln|x|
                    return Ok(Expr::Ln(Box::new(Expr::Var(var.to_string()))));
                } else {
                    // ∫ x^n dx = x^(n+1)/(n+1)
                    let new_exp = Expr::Const(n + 1.0);
                    let integrated = Expr::Pow(
                        Box::new(Expr::Var(var.to_string())),
                        Box::new(new_exp.clone()),
                    ) / new_exp;
                    return Ok(integrated);
                }
            }
        }

        // Case 2: ∫ c^x dx = c^x / ln(c) where c is constant
        if let (Expr::Const(c), Expr::Var(x)) = (base, exp) {
            if x == var && *c > 0.0 && (*c - 1.0).abs() > f64::EPSILON {
                return Ok(Expr::Pow(
                    Box::new(Expr::Const(*c)),
                    Box::new(Expr::Var(var.to_string())),
                ) / Expr::Ln(Box::new(Expr::Const(*c))));
            }
        }

        // Case 3: ∫ (f(x))^n dx where n is constant - requires substitution
        if let Expr::Const(_) = exp {
            if !base.contains_variable(var) {
                // Base doesn't contain variable: ∫ c^n dx = c^n * x
                return Ok(self.clone() * Expr::Var(var.to_string()));
            }
        }

        Err(format!("Cannot integrate power: ({})^({})", base, exp))
    }

    /// Handle exponential integration
    fn integrate_exponential(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ e^x dx = e^x
        if let Expr::Var(x) = expr {
            if x == var {
                return Ok(Expr::Exp(Box::new(Expr::Var(var.to_string()))));
            }
        }

        // Case 2: ∫ e^(ax) dx = (1/a) * e^(ax) where a is constant
        if let Expr::Mul(lhs, rhs) = expr {
            if let (Expr::Const(a), Expr::Var(x)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    return Ok(Expr::Exp(Box::new(expr.clone())) / Expr::Const(*a));
                }
            }
            if let (Expr::Var(x), Expr::Const(a)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    return Ok(Expr::Exp(Box::new(expr.clone())) / Expr::Const(*a));
                }
            }
        }

        // Case 3: ∫ e^(ax + b) dx = (1/a) * e^(ax + b)
        if let Expr::Add(lhs, rhs) = expr {
            if let (Expr::Mul(coeff, var_expr), Expr::Const(_)) = (lhs.as_ref(), rhs.as_ref()) {
                if let (Expr::Const(a), Expr::Var(x)) = (coeff.as_ref(), var_expr.as_ref()) {
                    if x == var {
                        return Ok(Expr::Exp(Box::new(expr.clone())) / Expr::Const(*a));
                    }
                }
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate exponential: e^({})", expr))
    }

    /// Handle logarithm integration using integration by parts
    fn integrate_logarithm(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ ln(x) dx = x*ln(x) - x (integration by parts)
        if let Expr::Var(x) = expr {
            if x == var {
                let x_var = Expr::Var(var.to_string());
                return Ok(x_var.clone() * Expr::Ln(Box::new(x_var.clone())) - x_var);
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate logarithm: ln({})", expr))
    }

    /// Try integration by parts: ∫ u dv = uv - ∫ v du
    // Extended integration by parts to handle x^n * exp(ax) cases
    fn try_integration_by_parts(&self, u: &Expr, dv: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: x^n * e^(ax) where n is a positive integer
        if let Some(result) = self.integrate_polynomial_times_exponential(u, dv, var) {
            return Ok(result);
        }

        // Case 2: e^(ax) * x^n (order reversed)
        if let Some(result) = self.integrate_polynomial_times_exponential(dv, u, var) {
            return Ok(result);
        }

        // Case 3: x^n * ln(x) - integration by parts
        if let Some(result) = self.integrate_polynomial_times_logarithm(u, dv, var) {
            return Ok(result);
        }

        // Case 4: ln(x) * x^n (order reversed)
        if let Some(result) = self.integrate_polynomial_times_logarithm(dv, u, var) {
            return Ok(result);
        }

        // Original simple cases
        if let (Expr::Var(x), Expr::Exp(exp_inner)) = (u, dv) {
            if x == var {
                if let Expr::Var(exp_var) = exp_inner.as_ref() {
                    if exp_var == var {
                        let exp_x = Expr::Exp(Box::new(Expr::Var(var.to_string())));
                        let x_minus_1 = Expr::Var(var.to_string()) - Expr::Const(1.0);
                        return Ok(exp_x * x_minus_1);
                    }
                }
            }
        }

        // Try the other way around
        if let (Expr::Exp(exp_inner), Expr::Var(x)) = (u, dv) {
            if x == var {
                if let Expr::Var(exp_var) = exp_inner.as_ref() {
                    if exp_var == var {
                        let exp_x = Expr::Exp(Box::new(Expr::Var(var.to_string())));
                        let x_minus_1 = Expr::Var(var.to_string()) - Expr::Const(1.0);
                        return Ok(exp_x * x_minus_1);
                    }
                }
            }
        }

        Err(format!(
            "Integration by parts not implemented for: {} * {}",
            u, dv
        ))
    }

    /// Handle x^n * exp(ax) integration using recursive integration by parts
    fn integrate_polynomial_times_exponential(
        &self,
        poly: &Expr,
        exp: &Expr,
        var: &str,
    ) -> Option<Expr> {
        // Check if we have x^n * exp(ax) pattern
        if let Expr::Exp(exp_inner) = exp {
            if let Some((n, a)) = self.extract_power_and_exp_coefficient(poly, exp_inner, var) {
                return Some(self.integrate_xn_times_exp_ax(n, a, var));
            }
        }
        None
    }

    /// Extract n from x^n and a from exp(ax)
    fn extract_power_and_exp_coefficient(
        &self,
        poly: &Expr,
        exp_inner: &Expr,
        var: &str,
    ) -> Option<(i32, f64)> {
        let n = match poly {
            Expr::Var(x) if x == var => 1,
            Expr::Pow(base, exp) => {
                if let (Expr::Var(x), Expr::Const(power)) = (base.as_ref(), exp.as_ref()) {
                    if x == var && power.fract() == 0.0 && *power >= 0.0 {
                        *power as i32
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Const(_) => 0, // x^0 = 1
            _ => return None,
        };

        let a = match exp_inner {
            Expr::Var(x) if x == var => 1.0,
            Expr::Mul(lhs, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Const(coeff), Expr::Var(x)) if x == var => *coeff,
                (Expr::Var(x), Expr::Const(coeff)) if x == var => *coeff,
                _ => return None,
            },
            _ => return None,
        };

        Some((n, a))
    }

    /// Integrate x^n * exp(ax) using the recursive formula
    /// ∫ x^n * e^(ax) dx = (1/a) * x^n * e^(ax) - (n/a) * ∫ x^(n-1) * e^(ax) dx
    fn integrate_xn_times_exp_ax(&self, n: i32, a: f64, var: &str) -> Expr {
        if a == 0.0 {
            // If a = 0, then exp(ax) = 1, so we integrate x^n
            return self.integrate_power_simple(n, var);
        }

        if n == 0 {
            // ∫ e^(ax) dx = (1/a) * e^(ax)
            let exp_ax = if a == 1.0 {
                Expr::Exp(Box::new(Expr::Var(var.to_string())))
            } else {
                Expr::Exp(Box::new(Expr::Const(a) * Expr::Var(var.to_string())))
            };
            return exp_ax / Expr::Const(a);
        }

        if n == 1 {
            // ∫ x * e^(ax) dx = e^(ax) * (x/a - 1/a²)
            let x = Expr::Var(var.to_string());
            let exp_ax = if a == 1.0 {
                Expr::Exp(Box::new(x.clone()))
            } else {
                Expr::Exp(Box::new(Expr::Const(a) * x.clone()))
            };
            let term1 = x / Expr::Const(a);
            let term2 = Expr::Const(1.0) / Expr::Const(a * a);
            return exp_ax * (term1 - term2);
        }

        // For n > 1, use the recursive formula
        self.integrate_xn_times_exp_ax_recursive(n, a, var)
    }

    /// Recursive implementation of x^n * exp(ax) integration
    fn integrate_xn_times_exp_ax_recursive(&self, n: i32, a: f64, var: &str) -> Expr {
        if n == 0 {
            let exp_ax = if a == 1.0 {
                Expr::Exp(Box::new(Expr::Var(var.to_string())))
            } else {
                Expr::Exp(Box::new(Expr::Const(a) * Expr::Var(var.to_string())))
            };
            return exp_ax / Expr::Const(a);
        }

        let x = Expr::Var(var.to_string());
        let xn = if n == 1 {
            x.clone()
        } else {
            x.clone().pow(Expr::Const(n as f64))
        };

        let exp_ax = if a == 1.0 {
            Expr::Exp(Box::new(x.clone()))
        } else {
            Expr::Exp(Box::new(Expr::Const(a) * x.clone()))
        };

        // First term: (1/a) * x^n * e^(ax)
        let first_term = (xn * exp_ax.clone()) / Expr::Const(a);

        // Second term: -(n/a) * ∫ x^(n-1) * e^(ax) dx
        let second_term = (Expr::Const(n as f64) / Expr::Const(a))
            * self.integrate_xn_times_exp_ax_recursive(n - 1, a, var);

        first_term - second_term
    }

    /// Simple power integration x^n dx = x^(n+1)/(n+1)
    fn integrate_power_simple(&self, n: i32, var: &str) -> Expr {
        let x = Expr::Var(var.to_string());
        if n == -1 {
            x.ln()
        } else {
            let new_power = n + 1;
            x.pow(Expr::Const(new_power as f64)) / Expr::Const(new_power as f64)
        }
    }

    /// Handle x^n * ln(x) integration using integration by parts
    fn integrate_polynomial_times_logarithm(
        &self,
        poly: &Expr,
        ln_expr: &Expr,
        var: &str,
    ) -> Option<Expr> {
        if let Expr::Ln(ln_inner) = ln_expr {
            if let Expr::Var(x) = ln_inner.as_ref() {
                if x == var {
                    if let Some(n) = self.extract_power_from_polynomial(poly, var) {
                        return Some(self.integrate_xn_times_ln_x(n, var));
                    }
                }
            }
        }
        None
    }

    /// Extract power n from x^n
    fn extract_power_from_polynomial(&self, poly: &Expr, var: &str) -> Option<i32> {
        match poly {
            Expr::Var(x) if x == var => Some(1),
            Expr::Pow(base, exp) => {
                if let (Expr::Var(x), Expr::Const(power)) = (base.as_ref(), exp.as_ref()) {
                    if x == var && power.fract() == 0.0 && *power >= 0.0 {
                        Some(*power as i32)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Expr::Const(_) => Some(0),
            _ => None,
        }
    }

    /// Integrate x^n * ln(x) using integration by parts
    /// ∫ x^n * ln(x) dx = x^(n+1) * [ln(x)/(n+1) - 1/(n+1)²]
    fn integrate_xn_times_ln_x(&self, n: i32, var: &str) -> Expr {
        if n == -1 {
            // ∫ ln(x)/x dx = (ln(x))²/2
            let ln_x = Expr::Var(var.to_string()).ln();
            return ln_x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0);
        }

        let x = Expr::Var(var.to_string());
        let n_plus_1 = n + 1;
        let x_power = if n_plus_1 == 1 {
            x.clone()
        } else {
            x.clone().pow(Expr::Const(n_plus_1 as f64))
        };

        let ln_x = x.ln();
        let term1 = ln_x / Expr::Const(n_plus_1 as f64);
        let term2 = Expr::Const(1.0) / Expr::Const((n_plus_1 * n_plus_1) as f64);

        x_power * (term1 - term2)
    }

    /// Handle trigonometric function integration

    /// ∫ sin(f) dx integration
    fn integrate_sin(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ sin(x) dx = -cos(x)
        if let Expr::Var(x) = expr {
            if x == var {
                return Ok(-Expr::cos(Box::new(Expr::Var(var.to_string()))));
            }
        }

        // Case 2: ∫ sin(ax) dx = -(1/a)*cos(ax) where a is constant
        if let Expr::Mul(lhs, rhs) = expr {
            if let (Expr::Const(a), Expr::Var(x)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let cos_ax = Expr::cos(Box::new(expr.clone()));
                    return Ok(-cos_ax / Expr::Const(*a));
                }
            }
            if let (Expr::Var(x), Expr::Const(a)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let cos_ax = Expr::cos(Box::new(expr.clone()));
                    return Ok(-cos_ax / Expr::Const(*a));
                }
            }
        }

        // Case 3: ∫ sin(ax + b) dx = -(1/a)*cos(ax + b) where a, b are constants
        if let Expr::Add(lhs, rhs) = expr {
            if let (Expr::Mul(coeff, var_expr), Expr::Const(_)) = (lhs.as_ref(), rhs.as_ref()) {
                if let (Expr::Const(a), Expr::Var(x)) = (coeff.as_ref(), var_expr.as_ref()) {
                    if x == var {
                        let cos_expr = Expr::cos(Box::new(expr.clone()));
                        return Ok(-cos_expr / Expr::Const(*a));
                    }
                }
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate sin({})", expr))
    }

    /// ∫ cos(f) dx integration
    fn integrate_cos(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ cos(x) dx = sin(x)
        if let Expr::Var(x) = expr {
            if x == var {
                return Ok(Expr::sin(Box::new(Expr::Var(var.to_string()))));
            }
        }

        // Case 2: ∫ cos(ax) dx = (1/a)*sin(ax) where a is constant
        if let Expr::Mul(lhs, rhs) = expr {
            if let (Expr::Const(a), Expr::Var(x)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let sin_ax = Expr::sin(Box::new(expr.clone()));
                    return Ok(sin_ax / Expr::Const(*a));
                }
            }
            if let (Expr::Var(x), Expr::Const(a)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let sin_ax = Expr::sin(Box::new(expr.clone()));
                    return Ok(sin_ax / Expr::Const(*a));
                }
            }
        }

        // Case 3: ∫ cos(ax + b) dx = (1/a)*sin(ax + b) where a, b are constants
        if let Expr::Add(lhs, rhs) = expr {
            if let (Expr::Mul(coeff, var_expr), Expr::Const(_)) = (lhs.as_ref(), rhs.as_ref()) {
                if let (Expr::Const(a), Expr::Var(x)) = (coeff.as_ref(), var_expr.as_ref()) {
                    if x == var {
                        let sin_expr = Expr::sin(Box::new(expr.clone()));
                        return Ok(sin_expr / Expr::Const(*a));
                    }
                }
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate cos({})", expr))
    }

    /// ∫ tan(f) dx integration
    fn integrate_tan(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ tan(x) dx = -ln|cos(x)|
        if let Expr::Var(x) = expr {
            if x == var {
                let cos_x = Expr::cos(Box::new(Expr::Var(var.to_string())));
                return Ok(-cos_x.ln());
            }
        }

        // Case 2: ∫ tan(ax) dx = -(1/a)*ln|cos(ax)| where a is constant
        if let Expr::Mul(lhs, rhs) = expr {
            if let (Expr::Const(a), Expr::Var(x)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let cos_ax = Expr::cos(Box::new(expr.clone()));
                    return Ok(-cos_ax.ln() / Expr::Const(*a));
                }
            }
            if let (Expr::Var(x), Expr::Const(a)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let cos_ax = Expr::cos(Box::new(expr.clone()));
                    return Ok(-cos_ax.ln() / Expr::Const(*a));
                }
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate tan({})", expr))
    }

    /// ∫ cot(f) dx integration
    fn integrate_cot(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ cot(x) dx = ln|sin(x)|
        if let Expr::Var(x) = expr {
            if x == var {
                let sin_x = Expr::sin(Box::new(Expr::Var(var.to_string())));
                return Ok(sin_x.ln());
            }
        }

        // Case 2: ∫ cot(ax) dx = (1/a)*ln|sin(ax)| where a is constant
        if let Expr::Mul(lhs, rhs) = expr {
            if let (Expr::Const(a), Expr::Var(x)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let sin_ax = Expr::sin(Box::new(expr.clone()));
                    return Ok(sin_ax.ln() / Expr::Const(*a));
                }
            }
            if let (Expr::Var(x), Expr::Const(a)) = (lhs.as_ref(), rhs.as_ref()) {
                if x == var {
                    let sin_ax = Expr::sin(Box::new(expr.clone()));
                    return Ok(sin_ax.ln() / Expr::Const(*a));
                }
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate cot({})", expr))
    }

    /// ∫ arcsin(f) dx integration
    fn integrate_arcsin(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ arcsin(x) dx = x*arcsin(x) + sqrt(1-x²) (integration by parts)
        if let Expr::Var(x) = expr {
            if x == var {
                let x_var = Expr::Var(var.to_string());
                let arcsin_x = Expr::arcsin(Box::new(x_var.clone()));
                let one_minus_x_squared = Expr::Const(1.0) - x_var.clone().pow(Expr::Const(2.0));
                let sqrt_term = one_minus_x_squared.pow(Expr::Const(0.5));
                return Ok(x_var * arcsin_x + sqrt_term);
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate arcsin({})", expr))
    }

    /// ∫ arccos(f) dx integration
    fn integrate_arccos(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ arccos(x) dx = x*arccos(x) - sqrt(1-x²) (integration by parts)
        if let Expr::Var(x) = expr {
            if x == var {
                let x_var = Expr::Var(var.to_string());
                let arccos_x = Expr::arccos(Box::new(x_var.clone()));
                let one_minus_x_squared = Expr::Const(1.0) - x_var.clone().pow(Expr::Const(2.0));
                let sqrt_term = one_minus_x_squared.pow(Expr::Const(0.5));
                return Ok(x_var * arccos_x - sqrt_term);
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate arccos({})", expr))
    }

    /// ∫ arctan(f) dx integration
    fn integrate_arctan(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ arctan(x) dx = x*arctan(x) - (1/2)*ln(1+x²) (integration by parts)
        if let Expr::Var(x) = expr {
            if x == var {
                let x_var = Expr::Var(var.to_string());
                let arctan_x = Expr::arctg(Box::new(x_var.clone()));
                let one_plus_x_squared = Expr::Const(1.0) + x_var.clone().pow(Expr::Const(2.0));
                let ln_term = one_plus_x_squared.ln() / Expr::Const(2.0);
                return Ok(x_var * arctan_x - ln_term);
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate arctan({})", expr))
    }

    /// ∫ arccot(f) dx integration
    fn integrate_arccot(&self, expr: &Expr, var: &str) -> Result<Expr, String> {
        // Case 1: ∫ arccot(x) dx = x*arccot(x) + (1/2)*ln(1+x²) (integration by parts)
        if let Expr::Var(x) = expr {
            if x == var {
                let x_var = Expr::Var(var.to_string());
                let arccot_x = Expr::arcctg(Box::new(x_var.clone()));
                let one_plus_x_squared = Expr::Const(1.0) + x_var.clone().pow(Expr::Const(2.0));
                let ln_term = one_plus_x_squared.ln() / Expr::Const(2.0);
                return Ok(x_var * arccot_x + ln_term);
            }
        }

        // If expression doesn't contain the variable, treat as constant
        if !expr.contains_variable(var) {
            return Ok(self.clone() * Expr::Var(var.to_string()));
        }

        Err(format!("Cannot integrate arccot({})", expr))
    }

    /// Definite integration using the fundamental theorem of calculus
    pub fn definite_integrate(&self, var: &str, lower: f64, upper: f64) -> Result<f64, String> {
        let indefinite = self.integrate(var)?;
        let upper_val = indefinite.eval_expression(vec![var], &[upper]);
        let lower_val = indefinite.eval_expression(vec![var], &[lower]);
        Ok(upper_val - lower_val)
    }

    /// Numerical integration using Simpson's rule as fallback
    pub fn numerical_integrate(&self, lower: f64, upper: f64, n: usize) -> f64 {
        if n % 2 != 0 {
            panic!("n must be even for Simpson's rule");
        }

        let h = (upper - lower) / (n as f64);
        let f = self.lambdify1D();

        let mut sum = f(lower) + f(upper);

        for i in 1..n {
            let x = lower + (i as f64) * h;
            if i % 2 == 0 {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }

        sum * h / 3.0
    }

    /// Table of common integrals for pattern matching
    #[allow(dead_code)]
    fn get_integral_table() -> HashMap<String, String> {
        let mut table = HashMap::new();

        // Basic functions
        table.insert("1".to_string(), "x".to_string());
        table.insert("x".to_string(), "x^2/2".to_string());
        table.insert("x^2".to_string(), "x^3/3".to_string());
        table.insert("1/x".to_string(), "ln(x)".to_string());
        table.insert("exp(x)".to_string(), "exp(x)".to_string());
        table.insert("ln(x)".to_string(), "x*ln(x) - x".to_string());

        // Trigonometric functions
        table.insert("sin(x)".to_string(), "-cos(x)".to_string());
        table.insert("cos(x)".to_string(), "sin(x)".to_string());
        table.insert("tan(x)".to_string(), "-ln|cos(x)|".to_string());
        table.insert("cot(x)".to_string(), "ln|sin(x)|".to_string());
        table.insert("sec(x)".to_string(), "ln|sec(x) + tan(x)|".to_string());
        table.insert("csc(x)".to_string(), "-ln|csc(x) + cot(x)|".to_string());

        // Inverse trigonometric functions
        table.insert(
            "arcsin(x)".to_string(),
            "x*arcsin(x) + sqrt(1-x^2)".to_string(),
        );
        table.insert(
            "arccos(x)".to_string(),
            "x*arccos(x) - sqrt(1-x^2)".to_string(),
        );
        table.insert(
            "arctan(x)".to_string(),
            "x*arctan(x) - (1/2)*ln(1+x^2)".to_string(),
        );
        table.insert(
            "arccot(x)".to_string(),
            "x*arccot(x) + (1/2)*ln(1+x^2)".to_string(),
        );

        table
    }

    /// Numerical integration using Gaussian quadrature methods
    pub fn quad(
        &self,
        method: QuadMethod,
        degree: usize,
        lower: f64,
        upper: f64,
        alpha: Option<f64>,
    ) -> Result<f64, String> {
        // Get the function to integrate
        let f = self.lambdify1D();

        match method {
            QuadMethod::GaussLegendre => {
                let quad = GaussLegendre::new(degree)
                    .map_err(|e| format!("Failed to create Gauss-Legendre quadrature: {:?}", e))?;

                let result = quad.integrate(lower, upper, &f);
                Ok(result)
            }

            QuadMethod::GaussHermite => {
                let quad = GaussHermite::new(degree)
                    .map_err(|e| format!("Failed to create Gauss-Hermite quadrature: {:?}", e))?;

                // Gauss-Hermite is for integrals from -∞ to +∞ with weight e^(-x²)
                // The user needs to provide a function that already accounts for this
                if lower.is_finite() || upper.is_finite() {
                    return Err(
                        "Gauss-Hermite quadrature is for infinite intervals (-∞, +∞)".to_string(),
                    );
                }

                let result = quad.integrate(&f);
                Ok(result)
            }

            QuadMethod::GaussLaguerre => {
                let alpha = alpha.unwrap_or(0.5);
                let quad = GaussLaguerre::new(degree, alpha)
                    .map_err(|e| format!("Failed to create Gauss-Laguerre quadrature: {:?}", e))?;

                // Gauss-Laguerre is for integrals from 0 to +∞ with weight e^(-x)
                if lower != 0.0 || upper.is_finite() {
                    return Err("Gauss-Laguerre quadrature is for interval [0, +∞)".to_string());
                }

                let result = quad.integrate(&f);
                Ok(result)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuadMethod {
    /// Gauss-Legendre quadrature for finite intervals
    GaussLegendre,
    /// Gauss-Hermite quadrature for infinite intervals with weight e^(-x²)
    GaussHermite,
    /// Gauss-Laguerre quadrature for semi-infinite intervals [0,∞) with weight e^(-x)
    GaussLaguerre,
}

impl QuadMethod {
    /// Get a description of the quadrature method
    pub fn description(&self) -> &'static str {
        match self {
            QuadMethod::GaussLegendre => "Gauss-Legendre quadrature for finite intervals",
            QuadMethod::GaussHermite => {
                "Gauss-Hermite quadrature for infinite intervals with weight e^(-x²)"
            }
            QuadMethod::GaussLaguerre => {
                "Gauss-Laguerre quadrature for semi-infinite intervals [0,∞) with weight e^(-x)"
            }
        }
    }

    /// Check if the method is suitable for the given interval
    pub fn is_suitable_for_interval(&self, lower: f64, upper: f64) -> bool {
        match self {
            QuadMethod::GaussLegendre => lower.is_finite() && upper.is_finite(),
            QuadMethod::GaussHermite => lower.is_infinite() && upper.is_infinite(),
            QuadMethod::GaussLaguerre => lower == 0.0 && upper.is_infinite(),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
// tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod integration_tests {

    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;

    #[test]
    fn test_integrate_constant() {
        // ∫ 5 dx = 5x
        let expr = Expr::Const(5.0);
        let result = expr.integrate("x").unwrap();
        let expected = Expr::Const(5.0) * Expr::Var("x".to_string());
        assert_eq!(result.simplify(), expected.simplify());
    }

    #[test]
    fn test_integrate_variable() {
        // ∫ x dx = x²/2
        let expr = Expr::Var("x".to_string());
        let result = expr.integrate("x").unwrap();
        let expected = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        ) / Expr::Const(2.0);
        assert_eq!(result.simplify(), expected.simplify());
    }

    #[test]
    fn test_integrate_different_variable() {
        // ∫ y dx = y*x (y is treated as constant)
        let expr = Expr::Var("y".to_string());
        let result = expr.integrate("x").unwrap();
        let expected = Expr::Var("y".to_string()) * Expr::Var("x".to_string());
        assert_eq!(result.simplify(), expected.simplify());
    }

    #[test]
    fn test_integrate_addition() {
        // ∫ (x + 3) dx = x²/2 + 3x
        let expr = Expr::Var("x".to_string()) + Expr::Const(3.0);
        let result = expr.integrate("x").unwrap();

        let expected = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        ) / Expr::Const(2.0)
            + Expr::Const(3.0) * Expr::Var("x".to_string());

        // Test by evaluating at a point
        let x_val = 2.0; // 2^2/2 + 3*2 = 2 + 6 = 8
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        assert_relative_eq!(result_val, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_subtraction() {
        // ∫ (x² - x) dx = x³/3 - x²/2
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) - x.clone();
        let result = expr.integrate("x").unwrap();

        let expected = x.clone().pow(Expr::Const(3.0)) / Expr::Const(3.0)
            - x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0);

        // Test by evaluating at a point
        let x_val = 3.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_constant_multiplication() {
        // ∫ 3x dx = 3 * x²/2 = 3x²/2
        let expr = Expr::Const(3.0) * Expr::Var("x".to_string());
        let result = expr.integrate("x").unwrap();

        let expected = Expr::Const(3.0)
            * (Expr::Pow(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0)),
            ) / Expr::Const(2.0));

        // Test by evaluating at a point
        let x_val = 2.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_power_positive() {
        // ∫ x³ dx = x⁴/4
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0));
        let result = expr.integrate("x").unwrap();

        let expected = x.clone().pow(Expr::Const(4.0)) / Expr::Const(4.0);

        // Test by evaluating at a point
        let x_val = 2.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_power_negative_one() {
        // ∫ x⁻¹ dx = ∫ 1/x dx = ln|x|
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(-1.0));
        let result = expr.integrate("x").unwrap();

        let expected = Expr::Ln(Box::new(x.clone()));

        // Test by evaluating at a point (x > 0)
        let x_val = 2.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_division_by_constant() {
        // ∫ x/2 dx = (1/2) * x²/2 = x²/4
        let x = Expr::Var("x".to_string());
        let expr = x.clone() / Expr::Const(2.0);
        let result = expr.integrate("x").unwrap();

        let expected = x.clone().pow(Expr::Const(2.0)) / Expr::Const(4.0);

        // Test by evaluating at a point
        let x_val = 4.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_one_over_x() {
        // ∫ 1/x dx = ln|x|
        let expr = Expr::Const(1.0) / Expr::Var("x".to_string());
        let result = expr.integrate("x").unwrap();

        let expected = Expr::Ln(Box::new(Expr::Var("x".to_string())));

        // Test by evaluating at a point (x > 0)
        let x_val = std::f64::consts::E;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_exponential_simple() {
        // ∫ e^x dx = e^x
        let x = Expr::Var("x".to_string());
        let expr = x.clone().exp();
        let result = expr.integrate("x").unwrap();

        let expected = x.clone().exp();

        // Test by evaluating at a point
        let x_val = 1.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_exponential_with_coefficient() {
        // ∫ e^(2x) dx = (1/2) * e^(2x)
        let x = Expr::Var("x".to_string());
        let expr = (Expr::Const(2.0) * x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        let expected = expr.clone() / Expr::Const(2.0);

        // Test by evaluating at a point
        let x_val = 0.5;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_logarithm() {
        // ∫ ln(x) dx = x*ln(x) - x
        let x = Expr::Var("x".to_string());
        let expr = x.clone().ln();
        let result = expr.integrate("x").unwrap();

        let expected = x.clone() * x.clone().ln() - x.clone();

        // Test by evaluating at a point (x > 0)
        let x_val = 2.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫ (3x² + 2x + 1) dx = x³ + x² + x
        let x = Expr::Var("x".to_string());
        let expr = Expr::Const(3.0) * x.clone().pow(Expr::Const(2.0))
            + Expr::Const(2.0) * x.clone()
            + Expr::Const(1.0);
        let result = expr.integrate("x").unwrap();

        let expected =
            x.clone().pow(Expr::Const(3.0)) + x.clone().pow(Expr::Const(2.0)) + x.clone();

        // Test by evaluating at a point
        let x_val = 2.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_definite_integration() {
        // ∫₀² x dx = [x²/2]₀² = 4/2 - 0 = 2
        let x = Expr::Var("x".to_string());
        let result = x.definite_integrate("x", 0.0, 2.0).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_definite_integration_polynomial() {
        // ∫₁³ (x² + 1) dx = [x³/3 + x]₁³ = (27/3 + 3) - (1/3 + 1) = 9 + 3 - 1/3 - 1 = 11 - 1/3 = 32/3
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) + Expr::Const(1.0);
        let result = expr.definite_integrate("x", 1.0, 3.0).unwrap();
        let expected = 32.0 / 3.0;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_definite_integration_exponential() {
        // ∫₀¹ e^x dx = [e^x]₀¹ = e - 1
        let x = Expr::Var("x".to_string());
        let expr = x.clone().exp();
        let result = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_numerical_integration_simpson() {
        // Test Simpson's rule against known integral
        // ∫₀¹ x² dx = 1/3
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0));
        let result = expr.numerical_integrate(0.0, 1.0, 1000);
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_numerical_vs_analytical() {
        // Compare numerical and analytical integration for polynomial
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0)) + Expr::Const(2.0) * x.clone();

        let analytical = expr.definite_integrate("x", 0.0, 2.0).unwrap();
        let numerical = expr.numerical_integrate(0.0, 2.0, 1000);

        assert_relative_eq!(analytical, numerical, epsilon = 1e-6);
    }

    #[test]
    fn test_integration_by_parts_simple() {
        // ∫ x * e^x dx = e^x(x-1)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone().exp();
        let result = expr.integrate("x").unwrap();

        let expected = x.clone().exp() * (x.clone() - Expr::Const(1.0));

        // Test by evaluating at a point
        let x_val = 1.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    // Trigonometric integration tests
    #[test]
    fn test_integrate_sin() {
        // ∫ sin(x) dx = -cos(x)
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = -Expr::cos(Box::new(x.clone()));

        // Test by evaluating at a point
        let x_val = std::f64::consts::PI / 4.0; // π/4
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_cos() {
        // ∫ cos(x) dx = sin(x)
        let x = Expr::Var("x".to_string());
        let expr = Expr::cos(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = Expr::sin(Box::new(x.clone()));

        // Test by evaluating at a point
        let x_val = std::f64::consts::PI / 6.0; // π/6
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_sin_with_coefficient() {
        // ∫ sin(2x) dx = -(1/2)*cos(2x)
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(Expr::Const(2.0) * x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = -Expr::cos(Box::new(Expr::Const(2.0) * x.clone())) / Expr::Const(2.0);

        // Test by evaluating at a point
        let x_val = std::f64::consts::PI / 8.0; // π/8
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_cos_with_coefficient() {
        // ∫ cos(3x) dx = (1/3)*sin(3x)
        let x = Expr::Var("x".to_string());
        let expr = Expr::cos(Box::new(Expr::Const(3.0) * x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = Expr::sin(Box::new(Expr::Const(3.0) * x.clone())) / Expr::Const(3.0);

        // Test by evaluating at a point
        let x_val = std::f64::consts::PI / 9.0; // π/9
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_tan() {
        // ∫ tan(x) dx = -ln|cos(x)|
        let x = Expr::Var("x".to_string());
        let expr = Expr::tg(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = -Expr::cos(Box::new(x.clone())).ln();

        // Test by evaluating at a point (avoiding discontinuities)
        let x_val = std::f64::consts::PI / 6.0; // π/6
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_cot() {
        // ∫ cot(x) dx = ln|sin(x)|
        let x = Expr::Var("x".to_string());
        let expr = Expr::ctg(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected = Expr::sin(Box::new(x.clone())).ln();

        // Test by evaluating at a point (avoiding discontinuities)
        let x_val = std::f64::consts::PI / 3.0; // π/3
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_arcsin() {
        // ∫ arcsin(x) dx = x*arcsin(x) + sqrt(1-x²)
        let x = Expr::Var("x".to_string());
        let expr = Expr::arcsin(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let arcsin_x = Expr::arcsin(Box::new(x.clone()));
        let one_minus_x_squared = Expr::Const(1.0) - x.clone().pow(Expr::Const(2.0));
        let sqrt_term = one_minus_x_squared.pow(Expr::Const(0.5));
        let expected = x.clone() * arcsin_x + sqrt_term;

        // Test by evaluating at a point in domain
        let x_val = 0.5;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_arccos() {
        // ∫ arccos(x) dx = x*arccos(x) - sqrt(1-x²)
        let x = Expr::Var("x".to_string());
        let expr = Expr::arccos(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let arccos_x = Expr::arccos(Box::new(x.clone()));
        let one_minus_x_squared = Expr::Const(1.0) - x.clone().pow(Expr::Const(2.0));
        let sqrt_term = one_minus_x_squared.pow(Expr::Const(0.5));
        let expected = x.clone() * arccos_x - sqrt_term;

        // Test by evaluating at a point in domain
        let x_val = 0.5;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_integrate_arctan() {
        // ∫ arctan(x) dx = x*arctan(x) - (1/2)*ln(1+x²)
        let x = Expr::Var("x".to_string());
        let expr = Expr::arctg(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let arctan_x = Expr::arctg(Box::new(x.clone()));
        let one_plus_x_squared = Expr::Const(1.0) + x.clone().pow(Expr::Const(2.0));
        let ln_term = one_plus_x_squared.ln() / Expr::Const(2.0);
        let expected = x.clone() * arctan_x - ln_term;

        // Test by evaluating at a point
        let x_val = 1.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }

    #[test]
    fn test_definite_integration_trigonometric() {
        // ∫₀^(π/2) sin(x) dx = [-cos(x)]₀^(π/2) = -cos(π/2) + cos(0) = 0 + 1 = 1
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(x.clone()));
        let result = expr
            .definite_integrate("x", 0.0, std::f64::consts::PI / 2.0)
            .unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_definite_integration_cos() {
        // ∫₀^(π/2) cos(x) dx = [sin(x)]₀^(π/2) = sin(π/2) - sin(0) = 1 - 0 = 1
        let x = Expr::Var("x".to_string());
        let expr = Expr::cos(Box::new(x.clone()));
        let result = expr
            .definite_integrate("x", 0.0, std::f64::consts::PI / 2.0)
            .unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_trigonometric_polynomial() {
        // ∫ (x + sin(x)) dx = x²/2 - cos(x)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::sin(Box::new(x.clone()));
        let result = expr.integrate("x").unwrap();

        let expected =
            x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0) - Expr::cos(Box::new(x.clone()));

        // Test by evaluating at a point
        let x_val = std::f64::consts::PI / 4.0;
        let result_val = result.eval_expression(vec!["x"], &[x_val]);
        let expected_val = expected.eval_expression(vec!["x"], &[x_val]);
        assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod numerical_integration_tests {

    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;

    /// Test integration by parts for x^n * e^(ax) type functions
    #[test]
    fn test_integrate_x_exp_negative_ax() {
        // ∫ x * e^(-ax) dx = -e^(-ax)(x/a + 1/a²)
        // For a = 1: ∫ x * e^(-x) dx = -e^(-x)(x + 1)
        let x = Expr::Var("x".to_string());
        let a = Expr::Const(1.0);
        let expr = x.clone() * (-a.clone() * x.clone()).exp();

        // This is a complex case that requires multiple integration by parts
        // Let's test numerically first
        let numerical_result = expr.numerical_integrate(0.0, 2.0, 1000);

        // Analytical result: ∫₀² x*e^(-x) dx = [-e^(-x)(x+1)]₀² = -e^(-2)(3) - (-1) = 1 - 3e^(-2)
        let expected = 1.0 - 3.0 * (-2.0_f64).exp();
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_x_squared_exp_negative_x() {
        // ∫ x² * e^(-x) dx requires integration by parts twice
        // Result: -e^(-x)(x² + 2x + 2)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * (-x.clone()).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 1.0, 1000);

        // Analytical: ∫₀¹ x²e^(-x) dx = [-e^(-x)(x² + 2x + 2)]₀¹ = -e^(-1)(5) - (-2) = 2 - 5e^(-1)
        let expected = 2.0 - 5.0 * (-1.0_f64).exp();
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_x_cubed_exp_negative_ax() {
        // ∫ x³ * e^(-ax) dx - very complex integration by parts
        // For a = 1: ∫ x³ * e^(-x) dx = -e^(-x)(x³ + 3x² + 6x + 6)
        let x = Expr::Var("x".to_string());
        let a = Expr::Const(1.0);
        let expr = x.clone().pow(Expr::Const(3.0)) * (-a.clone() * x.clone()).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 1.0, 1000);

        // Analytical: ∫₀¹ x³e^(-x) dx = [-e^(-x)(x³ + 3x² + 6x + 6)]₀¹ = -e^(-1)(16) - (-6) = 6 - 16e^(-1)
        let expected = 6.0 - 16.0 * (-1.0_f64).exp();
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_x_exp_positive_ax() {
        // ∫ x * e^(ax) dx = e^(ax)(x/a - 1/a²)
        // For a = 2: ∫ x * e^(2x) dx = e^(2x)(x/2 - 1/4)
        let x = Expr::Var("x".to_string());
        let a = Expr::Const(2.0);
        let expr = x.clone() * (a.clone() * x.clone()).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 1.0, 1000);

        // Analytical: ∫₀¹ x*e^(2x) dx = [e^(2x)(x/2 - 1/4)]₀¹ = e²(1/2 - 1/4) - (0 - 1/4) = e²/4 + 1/4
        let expected = (2.0_f64).exp() / 4.0 + 0.25;
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_polynomial_times_exp() {
        // ∫ (x² + 2x + 1) * e^(-x) dx
        let x = Expr::Var("x".to_string());
        let poly =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(1.0);
        let expr = poly * (-x.clone()).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 2.0, 1000);

        // This is sum of individual integrals
        let expected = 5.0 - 17.0 / ((1.0_f64).exp()).powf(2.0);
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_integrate_x_sin_like_oscillatory() {
        // For oscillatory functions, we can test with exponentials that behave similarly
        // ∫ x * e^(ix) dx where i is imaginary (we'll use a small coefficient to simulate)
        let x = Expr::Var("x".to_string());
        let small_coeff = Expr::Const(0.1);
        let expr = x.clone() * (small_coeff * x.clone()).exp();

        // Test numerically over a period
        let numerical_result = expr.numerical_integrate(0.0, 10.0, 1000);

        // Should be finite and well-behaved
        assert!(numerical_result.is_finite());
        assert!(numerical_result > 0.0); // Since both x and e^(0.1x) are positive
    }

    #[test]
    fn test_integrate_rational_times_exp() {
        // ∫ (1/x) * e^x dx - this is a special function (exponential integral)
        // We'll test a simpler case: ∫ (1/(x+1)) * e^x dx
        let x = Expr::Var("x".to_string());
        let expr = (Expr::Const(1.0) / (x.clone() + Expr::Const(1.0))) * x.clone().exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 1.0, 1000);

        // This doesn't have a simple closed form, but should be finite
        assert!(numerical_result.is_finite());
        assert!(numerical_result > 0.0);
        assert_relative_eq!(numerical_result, 1.1253860830832697192, epsilon = 1e-6);
    }

    #[test]
    fn test_integrate_exp_times_ln() {
        // ∫ e^x * ln(x) dx - complex integration by parts
        let x = Expr::Var("x".to_string());
        let expr = x.clone().exp() * x.clone().ln();

        // Test numerically (avoiding x=0 where ln is undefined)
        let numerical_result = expr.numerical_integrate(1.0, 2.0, 1000);

        // Should be finite and positive in this range
        assert!(numerical_result.is_finite());
        assert!(numerical_result > 0.0);
        assert_relative_eq!(numerical_result, 2.06259, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_nested_exponentials() {
        // ∫ x * e^(e^x) dx - very complex
        let x = Expr::Var("x".to_string());
        let inner_exp = x.clone().exp();
        let expr = x.clone() * inner_exp.exp();

        // Test numerically over a small range
        let numerical_result = expr.numerical_integrate(0.0, 0.5, 10000);

        // Should be finite and positive
        assert!(numerical_result.is_finite());
        assert!(numerical_result > 0.0);
        assert_relative_eq!(numerical_result, 0.51593, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_power_times_ln() {
        // ∫ x^n * ln(x) dx = x^(n+1) * [ln(x)/(n+1) - 1/(n+1)²]
        // For n=2: ∫ x² * ln(x) dx = x³ * [ln(x)/3 - 1/9]
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * x.clone().ln();

        // Test numerically (avoiding x=0)
        let numerical_result = expr.numerical_integrate(1.0, 2.0, 1000);

        // Analytical result: ∫₁² x²ln(x) dx = [x³(ln(x)/3 - 1/9)]₁² = 8(ln(2)/3 - 1/9) - (0 - 1/9) = 8ln(2)/3 - 8/9 + 1/9 = 8ln(2)/3 - 7/9
        let expected = 8.0 * 2.0_f64.ln() / 3.0 - 7.0 / 9.0;
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_multiple_by_parts_chain() {
        // ∫ x³ * e^(2x) dx - requires integration by parts 3 times
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0)) * (Expr::Const(2.0) * x.clone()).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 1.0, 1000);

        // For ∫ x³e^(2x) dx = e^(2x)(x³/2 - 3x²/4 + 3x/4 - 3/8)
        // ∫₀¹ x³e^(2x) dx = [e^(2x)(x³/2 - 3x²/4 + 3x/4 - 3/8)]₀¹ = e²(1/2 - 3/4 + 3/4 - 3/8) - (0 - 3/8) = e²/8 + 3/8
        let expected = (2.0_f64).exp() / 8.0 + 3.0 / 8.0;
        assert_relative_eq!(numerical_result, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_alternating_signs() {
        // ∫ x * e^(-x) * cos-like behavior (using alternating exponentials)
        let x = Expr::Var("x".to_string());
        // Simulate oscillatory behavior with e^(-x) - e^(-2x)
        let expr = x.clone() * ((-x.clone()).exp() - (-Expr::Const(2.0) * x.clone()).exp());

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 3.0, 1000);

        // Should be finite
        assert!(numerical_result.is_finite());
    }

    #[test]
    fn test_integrate_fractional_powers_with_exp() {
        // ∫ x^(1/2) * e^(-x) dx - involves gamma functions
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(0.5)) * (-x.clone()).exp();

        // Test numerically (avoiding x=0)
        let numerical_result = expr.numerical_integrate(0.1, 2.0, 1000);

        // Should be finite and positive
        assert!(numerical_result.is_finite());
        assert!(numerical_result > 0.0);
        assert_relative_eq!(numerical_result, 0.634649, epsilon = 1e-4);
    }

    #[test]
    fn test_integrate_high_order_polynomial_exp() {
        // ∫ (x⁵ + 3x⁴ - 2x³ + x²) * e^(-x/2) dx
        let x = Expr::Var("x".to_string());
        let poly = x.clone().pow(Expr::Const(5.0))
            + Expr::Const(3.0) * x.clone().pow(Expr::Const(4.0))
            - Expr::Const(2.0) * x.clone().pow(Expr::Const(3.0))
            + x.clone().pow(Expr::Const(2.0));
        let expr = poly * (-x.clone() / Expr::Const(2.0)).exp();

        // Test numerically
        let numerical_result = expr.numerical_integrate(0.0, 4.0, 2000);

        // Should be finite
        assert!(numerical_result.is_finite());
    }

    #[test]
    fn test_compare_integration_methods() {
        // Compare different numerical integration methods for the same function
        // ∫ x² * e^(-x) dx from 0 to 2
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * (-x.clone()).exp();

        // Test with different step sizes
        let result_1000 = expr.numerical_integrate(0.0, 2.0, 1000);
        let result_2000 = expr.numerical_integrate(0.0, 2.0, 2000);

        // Results should be close (convergence test)
        assert_relative_eq!(result_1000, result_2000, epsilon = 1e-3);

        // Compare with analytical result: 2 - 8e^(-2)
        let analytical = expr.definite_integrate("x", 0.0, 2.0).unwrap();
        assert_relative_eq!(result_2000, analytical, epsilon = 1e-4);
    }
}

#[cfg(test)]
mod integration_by_parts_tests {

    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;

    #[test]
    fn test_x_times_exp_x_analytical() {
        // ∫ x * e^x dx = e^x(x - 1)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone().exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: e^x(x - 1)
        let expected = x.clone().exp() * (x.clone() - Expr::Const(1.0));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Also test definite integral ∫₀¹ x*e^x dx = 1
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        assert_relative_eq!(definite, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_x_times_exp_negative_x_analytical() {
        // ∫ x * e^(-x) dx = -e^(-x)(x + 1)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * (-x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: -e^(-x)(x + 1)
        let expected = -(-x.clone()).exp() * (x.clone() + Expr::Const(1.0));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 2.0, 0.5, 3.0];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₀¹ x*e^(-x) dx = 1 - 2e^(-1)
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected_definite = 1.0 - 2.0 * (-1.0_f64).exp();
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_squared_times_exp_x_analytical() {
        // ∫ x² * e^x dx = e^x(x² - 2x + 2)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * x.clone().exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: e^x(x² - 2x + 2)
        let expected = x.clone().exp()
            * (x.clone().pow(Expr::Const(2.0)) - Expr::Const(2.0) * x.clone() + Expr::Const(2.0));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₀¹ x²*e^x dx = e - 2
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected_definite = std::f64::consts::E - 2.0;
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_squared_times_exp_negative_x_analytical() {
        // ∫ x² * e^(-x) dx = -e^(-x)(x² + 2x + 2)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * (-x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: -e^(-x)(x² + 2x + 2)
        let expected = -(-x.clone()).exp()
            * (x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(2.0));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 2.0, 0.5, 3.0];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₀¹ x²*e^(-x) dx = 2 - 5e^(-1)
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected_definite = 2.0 - 5.0 * (-1.0_f64).exp();
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_cubed_times_exp_negative_x_analytical() {
        // ∫ x³ * e^(-x) dx = -e^(-x)(x³ + 3x² + 6x + 6)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0)) * (-x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: -e^(-x)(x³ + 3x² + 6x + 6)
        let expected = -(-x.clone()).exp()
            * (x.clone().pow(Expr::Const(3.0))
                + Expr::Const(3.0) * x.clone().pow(Expr::Const(2.0))
                + Expr::Const(6.0) * x.clone()
                + Expr::Const(6.0));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 2.0, 0.5];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₀¹ x³*e^(-x) dx = 6 - 16e^(-1)
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected_definite = 6.0 - 16.0 * (-1.0_f64).exp();
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_times_exp_2x_analytical() {
        // ∫ x * e^(2x) dx = e^(2x)(x/2 - 1/4)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * (Expr::Const(2.0) * x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: e^(2x)(x/2 - 1/4)
        let expected = (Expr::Const(2.0) * x.clone()).exp()
            * (x.clone() / Expr::Const(2.0) - Expr::Const(0.25));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 0.5, -0.5];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₀¹ x*e^(2x) dx = e²/4 + 1/4
        let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
        let expected_definite = (2.0_f64).exp() / 4.0 + 0.25;
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_squared_times_exp_2x_analytical() {
        // ∫ x² * e^(2x) dx = e^(2x)(x²/2 - x/2 + 1/4)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * (Expr::Const(2.0) * x.clone()).exp();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: e^(2x)(x²/2 - x/2 + 1/4)
        let expected = (Expr::Const(2.0) * x.clone()).exp()
            * (x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0) - x.clone() / Expr::Const(2.0)
                + Expr::Const(0.25));

        // Test by evaluating both at several points
        let test_points = vec![0.0, 1.0, 0.5, -0.5];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_x_times_ln_x_analytical() {
        // ∫ x * ln(x) dx = x²(ln(x)/2 - 1/4)
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone().ln();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: x²(ln(x)/2 - 1/4)
        let expected = x.clone().pow(Expr::Const(2.0))
            * (x.clone().ln() / Expr::Const(2.0) - Expr::Const(0.25));

        // Test by evaluating both at several points (x > 0)
        let test_points = vec![1.0, 2.0, 0.5, std::f64::consts::E];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₁² x*ln(x) dx = 2ln(2) - 3/4
        let definite = expr.definite_integrate("x", 1.0, 2.0).unwrap();
        let expected_definite = 2.0 * 2.0_f64.ln() - 0.75;
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }

    #[test]
    fn test_x_squared_times_ln_x_analytical() {
        // ∫ x² * ln(x) dx = x³(ln(x)/3 - 1/9)
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0)) * x.clone().ln();
        let result = expr.integrate("x").unwrap();

        // Expected analytical result: x³(ln(x)/3 - 1/9)
        let expected = x.clone().pow(Expr::Const(3.0))
            * (x.clone().ln() / Expr::Const(3.0) - Expr::Const(1.0 / 9.0));

        // Test by evaluating both at several points (x > 0)
        let test_points = vec![1.0, 2.0, 0.5, std::f64::consts::E];
        for point in test_points {
            let result_val = result.eval_expression(vec!["x"], &[point]);
            let expected_val = expected.eval_expression(vec!["x"], &[point]);
            assert_relative_eq!(result_val, expected_val, epsilon = 1e-10);
        }

        // Test definite integral ∫₁² x²*ln(x) dx = 8ln(2)/3 - 7/9
        let definite = expr.definite_integrate("x", 1.0, 2.0).unwrap();
        let expected_definite = 8.0 * 2.0_f64.ln() / 3.0 - 7.0 / 9.0;
        assert_relative_eq!(definite, expected_definite, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod quadrature_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gauss_legendre_quadrature() {
        // Test ∫₀¹ x² dx = 1/3
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0));

        let result = expr
            .quad(QuadMethod::GaussLegendre, 10, 0.0, 1.0, None)
            .unwrap();
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_legendre_polynomial() {
        // Test ∫₋₁¹ (x³ + 2x² + x + 1) dx = 10/3
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0))
            + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0))
            + x.clone()
            + Expr::Const(1.0);

        let result = expr
            .quad(QuadMethod::GaussLegendre, 15, -1.0, 1.0, None)
            .unwrap();
        let expected = 10.0 / 3.0;
        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_gauss_legendre_exponential() {
        // Test ∫₀¹ e^x dx = e - 1
        let x = Expr::Var("x".to_string());
        let expr = x.clone().exp();

        let result = expr
            .quad(QuadMethod::GaussLegendre, 20, 0.0, 1.0, None)
            .unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_hermite_error_for_finite_interval() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0));

        let result = expr.quad(QuadMethod::GaussHermite, 10, 0.0, 1.0, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("infinite intervals"));
    }

    #[test]
    fn test_gauss_laguerre_error_for_wrong_interval() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().exp();

        let result = expr.quad(QuadMethod::GaussLaguerre, 10, 1.0, 2.0, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("interval [0, +∞)"));
    }

    #[test]
    fn test_quad_method_description() {
        assert_eq!(
            QuadMethod::GaussLegendre.description(),
            "Gauss-Legendre quadrature for finite intervals"
        );
        assert_eq!(
            QuadMethod::GaussHermite.description(),
            "Gauss-Hermite quadrature for infinite intervals with weight e^(-x²)"
        );
        assert_eq!(
            QuadMethod::GaussLaguerre.description(),
            "Gauss-Laguerre quadrature for semi-infinite intervals [0,∞) with weight e^(-x)"
        );
    }

    #[test]
    fn test_quad_method_interval_suitability() {
        assert!(QuadMethod::GaussLegendre.is_suitable_for_interval(0.0, 1.0));
        assert!(!QuadMethod::GaussLegendre.is_suitable_for_interval(0.0, f64::INFINITY));

        assert!(
            QuadMethod::GaussHermite.is_suitable_for_interval(f64::NEG_INFINITY, f64::INFINITY)
        );
        assert!(!QuadMethod::GaussHermite.is_suitable_for_interval(0.0, 1.0));

        assert!(QuadMethod::GaussLaguerre.is_suitable_for_interval(0.0, f64::INFINITY));
        assert!(!QuadMethod::GaussLaguerre.is_suitable_for_interval(1.0, f64::INFINITY));
    }

    #[test]
    fn test_compare_quad_with_analytical() {
        // Compare quadrature with analytical integration
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(3.0)) + Expr::Const(2.0) * x.clone();

        let analytical = expr.definite_integrate("x", 0.0, 2.0).unwrap();
        let numerical = expr
            .quad(QuadMethod::GaussLegendre, 20, 0.0, 2.0, None)
            .unwrap();

        assert_relative_eq!(analytical, numerical, epsilon = 1e-12);
    }

    #[test]
    fn test_compare_quad_with_simpson() {
        // Compare Gauss-Legendre with Simpson's rule
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(4.0)) + x.clone().pow(Expr::Const(2.0));

        let gauss_result = expr
            .quad(QuadMethod::GaussLegendre, 25, 0.0, 1.0, None)
            .unwrap();
        let simpson_result = expr.numerical_integrate(0.0, 1.0, 1000);

        assert_relative_eq!(gauss_result, simpson_result, epsilon = 1e-6);
    }
}

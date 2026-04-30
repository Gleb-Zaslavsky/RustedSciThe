//! Lambdification: compile a symbolic [`Atom`] into a callable `f64` closure.
//!
//! The produced closure is `Send + Sync` and captures only `Arc`-shared data,
//! so it can be sent across threads or stored in a `static` without restriction.

use std::sync::Arc;

use super::{
    atom::Atom,
    evaluate::{FunctionMap, PreparedEvaluator},
    state::Symbol,
};
use crate::wrap_symbol;
impl Atom {
    /// Return a thread-safe closure that evaluates `self` numerically.
    ///
    /// `vars` is an ordered list of variable names.  The closure accepts a
    /// slice of the same length and maps position *i* to variable `vars[i]`.
    ///
    /// # Panics
    /// The closure panics if the slice length does not match `vars.len()`, or
    /// if the expression contains a symbol that is neither in `vars` nor a
    /// recognised builtin constant (`e`, `pi`).
    #[inline(always)]
    pub fn lambdify_borrowed_thread_safe(
        &self,
        vars: &[&str],
    ) -> Box<dyn Fn(&[f64]) -> f64 + Send + Sync> {
        // Intern every variable name once and store their compact symbols.
        let var_symbols: Vec<Symbol> = vars
            .iter()
            .map(|name| Symbol::new(wrap_symbol!(*name)))
            .collect();

        let prepared = Arc::new(
            PreparedEvaluator::new(self, &var_symbols, &FunctionMap::new())
                .expect("lambdify: failed to prepare evaluator"),
        );
        let n_vars = vars.len();

        Box::new(move |vals: &[f64]| {
            assert_eq!(
                vals.len(),
                n_vars,
                "lambdify: expected {} argument(s), got {}",
                n_vars,
                vals.len()
            );

            prepared
                .evaluate(vals)
                .expect("lambdify: evaluation failed")
        })
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::parse;

    /// Shorthand: lambdify, call, compare with tolerance.
    fn check(expr: &str, vars: &[&str], vals: &[f64], expected: f64) {
        let atom = parse!(expr).unwrap();
        let f = atom.lambdify_borrowed_thread_safe(vars);
        let got = f(vals);
        assert!(
            (got - expected).abs() < 1e-10,
            "{expr}({vals:?}) = {got}, expected {expected}"
        );
    }

    // ── basic arithmetic ──────────────────────────────────────────────────────

    #[test]
    fn constant_expression() {
        // No variables — the slice is empty.
        check("2+3", &[], &[], 5.0);
    }

    #[test]
    fn single_variable_identity() {
        check("x", &["x"], &[7.0], 7.0);
    }

    #[test]
    fn linear_polynomial() {
        // 3*x + 2  at x = 4  →  14
        check("3*x+2", &["x"], &[4.0], 14.0);
    }

    #[test]
    fn two_variable_sum() {
        check("x+y", &["x", "y"], &[3.0, 5.0], 8.0);
    }

    #[test]
    fn two_variable_product() {
        check("x*y", &["x", "y"], &[3.0, 5.0], 15.0);
    }

    #[test]
    fn subtraction_via_negation() {
        // x - y is stored as x + y*-1 internally
        check("x-y", &["x", "y"], &[10.0, 3.0], 7.0);
    }

    #[test]
    fn division() {
        check("x/y", &["x", "y"], &[9.0, 3.0], 3.0);
    }

    // ── powers ────────────────────────────────────────────────────────────────

    #[test]
    fn integer_power() {
        check("x^3", &["x"], &[2.0], 8.0);
    }

    #[test]
    fn fractional_power_as_sqrt() {
        // x^(1/2) at x = 9 → 3
        check("x^(1/2)", &["x"], &[9.0], 3.0);
    }

    #[test]
    fn negative_power() {
        // x^-1 at x = 4 → 0.25
        check("x^-1", &["x"], &[4.0], 0.25);
    }

    // ── builtin functions ─────────────────────────────────────────────────────

    #[test]
    fn exp_function() {
        let atom = parse!("exp(x)").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        assert!((f(&[1.0]) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_function() {
        let atom = parse!("log(x)").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        assert!((f(&[std::f64::consts::E]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sin_cos_identity() {
        // sin(x)^2 + cos(x)^2 = 1 for any x
        let atom = parse!("sin(x)^2+cos(x)^2").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        for x in [0.0, 0.5, 1.0, 2.0, std::f64::consts::PI] {
            assert!((f(&[x]) - 1.0).abs() < 1e-10, "failed at x={x}");
        }
    }

    #[test]
    fn trig_aliases() {
        // tan(x) = tg(x), cot(x) = ctg(x) — both parse to the same builtin
        check("tg(x)", &["x"], &[0.0], 0.0);
        check("ctg(x)", &["x"], &[std::f64::consts::FRAC_PI_4], 1.0);
    }

    #[test]
    fn inverse_trig() {
        check("arcsin(x)", &["x"], &[1.0], std::f64::consts::FRAC_PI_2);
        check("arccos(x)", &["x"], &[0.0], std::f64::consts::FRAC_PI_2);
        check("arctg(x)", &["x"], &[1.0], std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn sqrt_function() {
        check("sqrt(x)", &["x"], &[16.0], 4.0);
    }

    // ── builtin constants ─────────────────────────────────────────────────────

    #[test]
    fn builtin_pi_constant() {
        // sin(pi) ≈ 0
        let atom = parse!("sin(pi)").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&[]);
        assert!(f(&[]).abs() < 1e-10);
    }

    #[test]
    fn builtin_e_constant() {
        // log(e) = 1
        let atom = parse!("log(e)").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&[]);
        assert!((f(&[]) - 1.0).abs() < 1e-10);
    }

    // ── nested / compound expressions ─────────────────────────────────────────

    #[test]
    fn nested_expression() {
        // sin(x^2 + 1) at x = 0 → sin(1)
        let atom = parse!("sin(x^2+1)").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        assert!((f(&[0.0]) - 1.0_f64.sin()).abs() < 1e-10);
    }

    #[test]
    fn multivariate_polynomial() {
        // x^2 + 2*x*y + y^2 = (x+y)^2  at (3, 4) → 49
        check("x^2+2*x*y+y^2", &["x", "y"], &[3.0, 4.0], 49.0);
    }

    #[test]
    fn three_variables() {
        // a*x^2 + b*x + c  at a=1, x=2, b=3, c=4 → 4+6+4 = 14
        check(
            "a*x^2+b*x+c",
            &["a", "x", "b", "c"],
            &[1.0, 2.0, 3.0, 4.0],
            14.0,
        );
    }

    // ── Send + Sync: closure can be used across threads ───────────────────────

    #[test]
    fn closure_is_send_sync() {
        let atom = parse!("x^2+1").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);

        // Spawn a thread and move the closure into it.
        let handle = std::thread::spawn(move || f(&[3.0]));
        assert!((handle.join().unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn closure_shared_across_threads() {
        use std::sync::Arc;

        let atom = parse!("x^2").unwrap();
        let f = Arc::new(atom.lambdify_borrowed_thread_safe(&["x"]));

        let handles: Vec<_> = (1..=4)
            .map(|i| {
                let f = Arc::clone(&f);
                std::thread::spawn(move || f(&[i as f64]))
            })
            .collect();

        let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results, vec![1.0, 4.0, 9.0, 16.0]);
    }

    // ── error handling ────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "lambdify: expected 1 argument(s), got 2")]
    fn wrong_arity_panics() {
        let atom = parse!("x").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        f(&[1.0, 2.0]); // too many args
    }

    #[test]
    #[should_panic(expected = "lambdify: failed to prepare evaluator")]
    fn unbound_variable_panics() {
        // "y" is not in the var list, so preparation now fails eagerly.
        let atom = parse!("x+y").unwrap();
        let f = atom.lambdify_borrowed_thread_safe(&["x"]);
        f(&[1.0]);
    }
}

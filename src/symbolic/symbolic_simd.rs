use crate::symbolic::symbolic_engine::Expr;
use std::f64::consts::PI;
use wide::f64x4;

impl Expr {
    pub fn lambdify1D_simd_f64x4(&self) -> Box<dyn Fn(f64x4) -> f64x4> {
        match self {
            Expr::Var(_) => Box::new(|x| x),
            Expr::Const(val) => {
                let val = f64x4::splat(*val);
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_f64x4();
                let rhs_fn = rhs.lambdify1D_simd_f64x4();
                Box::new(move |x| lhs_fn(x) + rhs_fn(x))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_f64x4();
                let rhs_fn = rhs.lambdify1D_simd_f64x4();
                Box::new(move |x| lhs_fn(x) - rhs_fn(x))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_f64x4();
                let rhs_fn = rhs.lambdify1D_simd_f64x4();
                Box::new(move |x| lhs_fn(x) * rhs_fn(x))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_f64x4();
                let rhs_fn = rhs.lambdify1D_simd_f64x4();
                Box::new(move |x| lhs_fn(x) / rhs_fn(x))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify1D_simd_f64x4();
                let exp_fn = exp.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let b = base_fn(x);
                    let e = exp_fn(x);
                    let e = e.to_array()[0];
                    b.powf(e)
                })
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.exp()
                })
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.ln()
                })
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.sin()
                })
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.cos()
                })
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.tan()
                })
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    1.0 / a.tan()
                })
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.asin()
                })
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.acos()
                })
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    a.atan()
                })
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify1D_simd_f64x4();
                Box::new(move |x| {
                    let a = expr_fn(x);
                    (PI / 2.0 - a.atan()).into()
                })
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_f64x4_basic_ops() {
        let a = f64x4::from([1.0, 2.0, 3.0, 4.0]);
        let b = f64x4::from([5.0, 6.0, 7.0, 8.0]);

        let sum = a + b;
        assert_eq!(sum.to_array(), [6.0, 8.0, 10.0, 12.0]);

        let diff = b - a;
        assert_eq!(diff.to_array(), [4.0, 4.0, 4.0, 4.0]);

        let prod = a * b;
        assert_eq!(prod.to_array(), [5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_simd_lambdify_simple() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(2.0);
        let func = expr.lambdify1D_simd_f64x4();

        let input = f64x4::from([1.0, 2.0, 3.0, 4.0]);
        let result = func(input);
        assert_eq!(result.to_array(), [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_simd_lambdify_polynomial() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0);
        let func = expr.lambdify1D_simd_f64x4();

        let input = f64x4::from([1.0, 2.0, 3.0, 4.0]);
        let result = func(input);
        assert_eq!(result.to_array(), [4.0, 9.0, 16.0, 25.0]);
    }

    #[test]
    fn test_simd_vs_scalar_correctness() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone() + Expr::sin(Box::new(x.clone())) + Expr::Const(1.0);

        let scalar_func = expr.lambdify1D();
        let simd_func = expr.lambdify1D_simd_f64x4();

        let test_values = [1.0, 2.0, 3.0, 4.0];
        let simd_input = f64x4::from(test_values);
        let simd_result = simd_func(simd_input).to_array();

        for (i, &val) in test_values.iter().enumerate() {
            let scalar_result = scalar_func(val);
            assert!((scalar_result - simd_result[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_performance_comparison() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0);

        let scalar_func = expr.lambdify1D();
        let simd_func = expr.lambdify1D_simd_f64x4();

        const N: usize = 1_000_000;
        let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.001).collect();

        // Scalar benchmark
        let start = Instant::now();
        let mut scalar_results = Vec::with_capacity(N);
        for &val in &data {
            scalar_results.push(scalar_func(val));
        }
        let scalar_time = start.elapsed();

        // SIMD benchmark - optimized version
        let start = Instant::now();
        let mut simd_results = Vec::with_capacity(N);
        for chunk in data.chunks_exact(4) {
            let input = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = simd_func(input).to_array();
            simd_results.extend_from_slice(&result);
        }
        let simd_time = start.elapsed();

        println!("Scalar time: {:?}", scalar_time);
        println!("SIMD time: {:?}", simd_time);
        println!(
            "Speedup: {:.2}x",
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );

        // Verify correctness
        for i in 0..std::cmp::min(scalar_results.len(), simd_results.len()) {
            assert!((scalar_results[i] - simd_results[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_performance_complex_expr() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(x.clone())) * Expr::cos(Box::new(x.clone()))
            + x.clone() * x.clone() * x.clone()
            + Expr::Exp(Box::new(x.clone()));

        let scalar_func = expr.lambdify1D();
        let simd_func = expr.lambdify1D_simd_f64x4();

        const N: usize = 1_000_000;
        let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.001).collect();

        // Scalar benchmark
        let start = Instant::now();
        for &val in &data {
            scalar_func(val);
        }
        let scalar_time = start.elapsed();

        // SIMD benchmark
        let start = Instant::now();
        for chunk in data.chunks_exact(4) {
            let input = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            simd_func(input);
        }
        let simd_time = start.elapsed();

        println!("Complex expr - Scalar time: {:?}", scalar_time);
        println!("Complex expr - SIMD time: {:?}", simd_time);
        println!(
            "Complex expr - Speedup: {:.2}x",
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );
    }

    #[test]
    fn test_transcendental_functions() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::sin(Box::new(x.clone())) + Expr::cos(Box::new(x.clone()));

        let scalar_func = expr.lambdify1D();
        let simd_func = expr.lambdify1D_simd_f64x4();

        let test_values = [0.0, PI / 4.0, PI / 2.0, PI];
        let simd_input = f64x4::from(test_values);
        let simd_result = simd_func(simd_input).to_array();

        for (i, &val) in test_values.iter().enumerate() {
            let scalar_result = scalar_func(val);
            assert!((scalar_result - simd_result[i]).abs() < 1e-10);
        }
    }
}

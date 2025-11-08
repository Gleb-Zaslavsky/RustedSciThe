//! EXPERIMENTAL MODULE DON'T USE IN PROD
use crate::symbolic::symbolic_engine::Expr;
use std::arch::x86_64::*;
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
#[derive(Clone, Debug)]
pub enum Exprs {
    /// Symbolic variable with a name (e.g., "x", "y", "velocity")
    Var(String),
    /// Numerical constant value
    Const(__m256d),
    /// Addition operation: left + right
    Add(Box<Exprs>, Box<Exprs>),
    /// Subtraction operation: left - right
    Sub(Box<Exprs>, Box<Exprs>),
    /// Multiplication operation: left * right
    Mul(Box<Exprs>, Box<Exprs>),
    /// Division operation: left / right
    Div(Box<Exprs>, Box<Exprs>),
    /// Power operation: base ^ exponent
    Pow(Box<Exprs>, Box<Exprs>),
    /// Exponential function: e^x
    Exp(Box<Exprs>),
    /// Natural logarithm: ln(x)
    Ln(Box<Exprs>),
    /// Sine function: sin(x)
    sin(Box<Expr>),
    /// Cosine function: cos(x)
    cos(Box<Exprs>),
    /// Tangent function: tan(x) - uses mathematical notation 'tg'
    tg(Box<Expr>),
    /// Cotangent function: cot(x) - uses mathematical notation 'ctg'
    ctg(Box<Exprs>),
    /// Arcsine function: arcsin(x)
    arcsin(Box<Exprs>),
    /// Arccosine function: arccos(x)
    arccos(Box<Exprs>),
    /// Arctangent function: arctan(x) - uses mathematical notation 'arctg'
    arctg(Box<Exprs>),
    /// Arccotangent function: arccot(x) - uses mathematical notation 'arcctg'
    arcctg(Box<Exprs>),
}

impl Exprs {
    #[allow(dead_code)]
    fn lambdify1D_simd_easy(self) -> Box<dyn Fn(__m256d) -> __m256d> {
        match self {
            Exprs::Var(_) => Box::new(|x| x),
            Exprs::Const(val) => Box::new(move |_| val),
            Exprs::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| unsafe { _mm256_add_pd(lhs_fn(x), rhs_fn(x)) })
            }
            Exprs::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| unsafe { _mm256_sub_pd(lhs_fn(x), rhs_fn(x)) })
            }
            Exprs::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| unsafe { _mm256_mul_pd(lhs_fn(x), rhs_fn(x)) })
            }
            Exprs::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| unsafe { _mm256_div_pd(lhs_fn(x), rhs_fn(x)) })
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr64x {
    /// Symbolic variable with a name (e.g., "x", "y", "velocity")
    Var(String),
    /// Numerical constant value
    Const(f64x4),
    /// Addition operation: left + right
    Add(Box<Expr64x>, Box<Expr64x>),
    /// Subtraction operation: left - right
    Sub(Box<Expr64x>, Box<Expr64x>),
    /// Multiplication operation: left * right
    Mul(Box<Expr64x>, Box<Expr64x>),
    /// Division operation: left / right
    Div(Box<Expr64x>, Box<Expr64x>),
    /// Power operation: base ^ exponent
    Pow(Box<Expr64x>, Box<Expr64x>),
    /// Exponential function: e^x
    Exp(Box<Expr64x>),
    /// Natural logarithm: ln(x)
    Ln(Box<Expr64x>),
    /// Sine function: sin(x)
    sin(Box<Expr64x>),
    /// Cosine function: cos(x)
    cos(Box<Expr64x>),
    /// Tangent function: tan(x) - uses mathematical notation 'tg'
    tg(Box<Expr64x>),
    /// Cotangent function: cot(x) - uses mathematical notation 'ctg'
    ctg(Box<Expr64x>),
    /// Arcsine function: arcsin(x)
    arcsin(Box<Expr64x>),
    /// Arccosine function: arccos(x)
    arccos(Box<Expr64x>),
    /// Arctangent function: arctan(x) - uses mathematical notation 'arctg'
    arctg(Box<Expr64x>),
    /// Arccotangent function: arccot(x) - uses mathematical notation 'arcctg'
    arcctg(Box<Expr64x>),
}

impl Expr64x {
    #[allow(dead_code)]
    fn lambdify1D_simd_easy(self) -> Box<dyn Fn(f64x4) -> f64x4> {
        match self {
            Expr64x::Var(_) => Box::new(|x| x),
            Expr64x::Const(val) => Box::new(move |_| val),
            Expr64x::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| lhs_fn(x) + rhs_fn(x))
            }
            Expr64x::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| lhs_fn(x) - rhs_fn(x))
            }
            Expr64x::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| lhs_fn(x) * rhs_fn(x))
            }
            Expr64x::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D_simd_easy();
                let rhs_fn = rhs.lambdify1D_simd_easy();
                Box::new(move |x| lhs_fn(x) / rhs_fn(x))
            }
            _ => unimplemented!(),
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

    ////////////////////////safe f64x4 SIMD tests
    #[test]
    fn test_expr64x_basic() {
        // Test variable
        let x = Expr64x::Var("x".to_string());
        let func = x.lambdify1D_simd_easy();
        let input = f64x4::from([1.0, 2.0, 3.0, 4.0]);
        let result = func(input);
        assert_eq!(result.to_array(), [1.0, 2.0, 3.0, 4.0]);

        // Test constant
        let c = Expr64x::Const(f64x4::splat(5.0));
        let func = c.lambdify1D_simd_easy();
        let result = func(input);
        assert_eq!(result.to_array(), [5.0, 5.0, 5.0, 5.0]);

        // Test addition: x + 2
        let x = Expr64x::Var("x".to_string());
        let two = Expr64x::Const(f64x4::splat(2.0));
        let add_expr = Expr64x::Add(Box::new(x), Box::new(two));
        let func = add_expr.lambdify1D_simd_easy();
        let result = func(input);
        assert_eq!(result.to_array(), [3.0, 4.0, 5.0, 6.0]);

        // Test multiplication: x * 3
        let x = Expr64x::Var("x".to_string());
        let three = Expr64x::Const(f64x4::splat(3.0));
        let mul_expr = Expr64x::Mul(Box::new(x), Box::new(three));
        let func = mul_expr.lambdify1D_simd_easy();
        let result = func(input);
        assert_eq!(result.to_array(), [3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_expr64x_performance() {
        // Create equivalent expressions: x^2 + 2*x + 1
        let scalar_expr = {
            let x = Expr::Var("x".to_string());
            x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0)
        };

        let simd_expr = {
            let x = Expr64x::Var("x".to_string());
            let two = Expr64x::Const(f64x4::splat(2.0));
            let one = Expr64x::Const(f64x4::splat(1.0));
            Expr64x::Add(
                Box::new(Expr64x::Add(
                    Box::new(Expr64x::Mul(Box::new(x.clone()), Box::new(x))),
                    Box::new(Expr64x::Mul(
                        Box::new(Expr64x::Var("x".to_string())),
                        Box::new(two),
                    )),
                )),
                Box::new(one),
            )
        };

        let scalar_func = scalar_expr.lambdify1D();
        let simd_func = simd_expr.lambdify1D_simd_easy();

        const N: usize = 1_000_000;
        let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.001).collect();

        // Scalar benchmark
        let start = Instant::now();
        let mut scalar_results = Vec::with_capacity(N);
        for &val in &data {
            scalar_results.push(scalar_func(val));
        }
        let scalar_time = start.elapsed();

        // SIMD benchmark
        let start = Instant::now();
        let mut simd_results = Vec::with_capacity(N);
        for chunk in data.chunks_exact(4) {
            let input = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = simd_func(input).to_array();
            simd_results.extend_from_slice(&result);
        }
        let simd_time = start.elapsed();

        println!("Expr64x - Scalar time: {:?}", scalar_time);
        println!("Expr64x - SIMD time: {:?}", simd_time);
        println!(
            "Expr64x - Speedup: {:.2}x",
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );

        // Verify correctness
        for i in 0..std::cmp::min(16, std::cmp::min(scalar_results.len(), simd_results.len())) {
            assert!((scalar_results[i] - simd_results[i]).abs() < 1e-10);
        }
    }

    ////////////////////////unsafe raw SIMD tests
    #[test]
    fn lambdify1D_simd_easy_test() {
        // Test variable
        let x = Exprs::Var("x".to_string());
        let func = x.lambdify1D_simd_easy();
        let input = unsafe { _mm256_set_pd(4.0, 3.0, 2.0, 1.0) };
        let result = func(input);
        let expected = unsafe { _mm256_set_pd(4.0, 3.0, 2.0, 1.0) };
        unsafe {
            let result_arr = std::mem::transmute::<__m256d, [f64; 4]>(result);
            let expected_arr = std::mem::transmute::<__m256d, [f64; 4]>(expected);
            assert_eq!(result_arr, expected_arr);
        }

        // Test constant
        let c = Exprs::Const(unsafe { _mm256_set1_pd(5.0) });
        let func = c.lambdify1D_simd_easy();
        let result = func(input);
        unsafe {
            let result_arr = std::mem::transmute::<__m256d, [f64; 4]>(result);
            assert_eq!(result_arr, [5.0, 5.0, 5.0, 5.0]);
        }

        // Test addition: x + 2
        let x = Exprs::Var("x".to_string());
        let two = Exprs::Const(unsafe { _mm256_set1_pd(2.0) });
        let add_expr = Exprs::Add(Box::new(x), Box::new(two));
        let func = add_expr.lambdify1D_simd_easy();
        let result = func(input);
        unsafe {
            let result_arr = std::mem::transmute::<__m256d, [f64; 4]>(result);
            assert_eq!(result_arr, [3.0, 4.0, 5.0, 6.0]);
        }

        // Test multiplication: x * 3
        let x = Exprs::Var("x".to_string());
        let three = Exprs::Const(unsafe { _mm256_set1_pd(3.0) });
        let mul_expr = Exprs::Mul(Box::new(x), Box::new(three));
        let func = mul_expr.lambdify1D_simd_easy();
        let result = func(input);
        unsafe {
            let result_arr = std::mem::transmute::<__m256d, [f64; 4]>(result);
            assert_eq!(result_arr, [3.0, 6.0, 9.0, 12.0]);
        }
    }
    #[test]
    fn test_performance_simd_easy_vs_scalar() {
        // Create equivalent expressions: x^2 + 2*x + 1
        let scalar_expr = {
            let x = Expr::Var("x".to_string());
            x.clone() * x.clone() + x.clone() * Expr::Const(2.0) + Expr::Const(1.0)
        };

        let simd_expr = {
            let x = Exprs::Var("x".to_string());
            let two = Exprs::Const(unsafe { _mm256_set1_pd(2.0) });
            let one = Exprs::Const(unsafe { _mm256_set1_pd(1.0) });
            Exprs::Add(
                Box::new(Exprs::Add(
                    Box::new(Exprs::Mul(Box::new(x.clone()), Box::new(x))),
                    Box::new(Exprs::Mul(
                        Box::new(Exprs::Var("x".to_string())),
                        Box::new(two),
                    )),
                )),
                Box::new(one),
            )
        };

        let scalar_func = scalar_expr.lambdify1D();
        let simd_func = simd_expr.lambdify1D_simd_easy();

        const N: usize = 1_000_000;
        let data: Vec<f64> = (0..N).map(|i| i as f64 * 0.001).collect();

        // Scalar benchmark
        let start = Instant::now();
        let mut scalar_results = Vec::with_capacity(N);
        for &val in &data {
            scalar_results.push(scalar_func(val));
        }
        let scalar_time = start.elapsed();

        // SIMD benchmark
        let start = Instant::now();
        let mut simd_results = Vec::with_capacity(N);
        for chunk in data.chunks_exact(4) {
            let input = unsafe { _mm256_set_pd(chunk[3], chunk[2], chunk[1], chunk[0]) };
            let result = simd_func(input);
            unsafe {
                let result_arr = std::mem::transmute::<__m256d, [f64; 4]>(result);
                simd_results.extend_from_slice(&[
                    result_arr[0],
                    result_arr[1],
                    result_arr[2],
                    result_arr[3],
                ]);
            }
        }
        let simd_time = start.elapsed();

        println!("SIMD Easy - Scalar time: {:?}", scalar_time);
        println!("SIMD Easy - SIMD time: {:?}", simd_time);
        println!(
            "SIMD Easy - Speedup: {:.2}x",
            scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );

        // Verify correctness for first few values
        for i in 0..std::cmp::min(16, std::cmp::min(scalar_results.len(), simd_results.len())) {
            assert!((scalar_results[i] - simd_results[i]).abs() < 1e-10);
        }
    }
}

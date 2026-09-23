//! View-owned scalar Jacobian lowering stories.
//!
//! These tests intentionally stop before matrix assembly and solver control.
//! They compare the shared Expr differentiation path with AtomView-compatible
//! and Atom-native scalar closures. Real LSODE2/BVP Jacobians remain in their
//! numerical modules as integration consumers of this low-level evidence.

use std::hint::black_box;
use std::time::Instant;

use crate::symbolic::View::conversions::atom_to_expr;
use crate::symbolic::View::{PreparedSparseAtomSystem, inspect_atoms, inspect_exprs};
use crate::symbolic::symbolic_engine::Expr;

macro_rules! report_println {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn measure_scalar(
    evaluator: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    args: &[f64],
    repetitions: usize,
) -> f64 {
    for _ in 0..3 {
        black_box(evaluator(args));
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        black_box(evaluator(args));
    }
    started.elapsed().as_secs_f64() * 1_000_000_000.0 / repetitions as f64
}

#[test]
#[ignore = "release View scalar Jacobian lowering baseline"]
fn scalar_jacobian_lowering_story_preserves_values_and_reports_stages() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Symbolic_View",
        "symbolic::View::scalar_jacobian_lowering_story_preserves_values_and_reports_stages",
    );
    let repetitions = std::env::var("SYMBOLIC_VIEW_SCALAR_JACOBIAN_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(20_000);
    let names = ["x"];
    let args = [1.3_f64];
    let corpus = [
        (
            "exp_times_integer_power",
            "exp(0.7*x) * (1+x^2)^3 / (2+x)^0.5",
        ),
        (
            "fractional_power_times_exp",
            "(1+x^2)^0.5 * exp(-0.3*x) / (1+x)^2",
        ),
        (
            "nested_fractional_power",
            "exp(x^0.5) * (1+x)^3 / (2+x^2)^1.5",
        ),
        (
            "transcendental_rational",
            "sin(x^2) * exp(0.2*x) / (1+x^3)",
        ),
    ];

    report_println!(
        "[Symbolic View scalar Jacobian lowering] expressions={}; repetitions={repetitions}; x=1.3; matrix assembly excluded",
        corpus.len()
    );
    report_println!(
        "case | route | input_nodes | output_nodes | unique | repeated | depth | prepare_ms | closure_compile_ms | eval_ns/call | max_diff"
    );
    report_println!(
        "------------------------------------------------------------------------------------------------------------------"
    );

    for (case, source) in corpus {
        let equation = Expr::parse_expression(source);
        let input_shape = inspect_exprs(std::slice::from_ref(&equation));
        let legacy_started = Instant::now();
        let legacy_derivative = equation.diff("x").simplify();
        let legacy_prepare_ms = legacy_started.elapsed().as_secs_f64() * 1_000.0;
        let legacy_compile_started = Instant::now();
        let legacy_callback = Expr::lambdify_borrowed_thread_safe(&legacy_derivative, &names);
        let legacy_compile_ms = legacy_compile_started.elapsed().as_secs_f64() * 1_000.0;
        let legacy_value = legacy_callback(&args);
        let legacy_eval_ns = measure_scalar(legacy_callback.as_ref(), &args, repetitions);

        let atom_prepare_started = Instant::now();
        let prepared = PreparedSparseAtomSystem::from_exprs(
            std::slice::from_ref(&equation),
            &["x".to_string()],
            &[vec!["x".to_string()]],
        );
        let entries = prepared.calc_sparse_jacobian_with_bandwidth(None);
        assert_eq!(entries.len(), 1, "{case} should have one nonzero derivative");
        let atom_prepare_ms = atom_prepare_started.elapsed().as_secs_f64() * 1_000.0;
        let atom = entries[0].value.clone();
        let atom_shape = inspect_atoms(std::slice::from_ref(&atom));

        let compat_expr = atom_to_expr(&atom).simplify();
        let compat_compile_started = Instant::now();
        let compat_callback = Expr::lambdify_borrowed_thread_safe(&compat_expr, &names);
        let compat_compile_ms = compat_compile_started.elapsed().as_secs_f64() * 1_000.0;
        let compat_value = compat_callback(&args);
        let compat_eval_ns = measure_scalar(compat_callback.as_ref(), &args, repetitions);

        let native_compile_started = Instant::now();
        let native_callback = atom.lambdify_borrowed_thread_safe(&names);
        let native_compile_ms = native_compile_started.elapsed().as_secs_f64() * 1_000.0;
        let native_value = native_callback(&args);
        let native_eval_ns = measure_scalar(native_callback.as_ref(), &args, repetitions);

        let compat_diff = (compat_value - legacy_value).abs();
        let native_diff = (native_value - legacy_value).abs();
        assert!(compat_diff <= 1.0e-9, "{case}/AtomViewExprCompat drifted by {compat_diff:e}");
        assert!(native_diff <= 1.0e-9, "{case}/AtomNative drifted by {native_diff:e}");

        report_println!(
            "{case} | ExprLegacy | {:>11} | {:>12} | {:>6} | {:>8} | {:>5} | {:>10.3} | {:>18.3} | {:>12.3} | {:>9.3e}",
            input_shape.tree_nodes,
            input_shape.tree_nodes,
            input_shape.unique_subexpressions,
            input_shape.repeated_subexpressions,
            input_shape.max_depth,
            legacy_prepare_ms,
            legacy_compile_ms,
            legacy_eval_ns,
            0.0,
        );
        report_println!(
            "{case} | AtomViewExprCompat | {:>11} | {:>12} | {:>6} | {:>8} | {:>5} | {:>10.3} | {:>18.3} | {:>12.3} | {:>9.3e}",
            input_shape.tree_nodes,
            atom_shape.tree_nodes,
            atom_shape.unique_subexpressions,
            atom_shape.repeated_subexpressions,
            atom_shape.max_depth,
            atom_prepare_ms,
            compat_compile_ms,
            compat_eval_ns,
            compat_diff,
        );
        report_println!(
            "{case} | AtomNative | {:>11} | {:>12} | {:>6} | {:>8} | {:>5} | {:>10.3} | {:>18.3} | {:>12.3} | {:>9.3e}",
            input_shape.tree_nodes,
            atom_shape.tree_nodes,
            atom_shape.unique_subexpressions,
            atom_shape.repeated_subexpressions,
            atom_shape.max_depth,
            atom_prepare_ms,
            native_compile_ms,
            native_eval_ns,
            native_diff,
        );
    }
}

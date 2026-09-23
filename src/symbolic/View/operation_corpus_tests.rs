//! Structural and evaluator diagnostics for common Atom/Expr lowering forms.
//!
//! This corpus deliberately lives with `symbolic::View`, not with a numerical
//! solver. It is a debug-only attribution tool: it compares direct `Expr`
//! closures with an `Expr -> Atom -> Expr` round-trip without changing the
//! production lowering path. Real LSODE2 and BVP fixtures consume the same
//! low-level conclusions in their own integration stories.

use std::hint::black_box;
use std::time::Instant;

use crate::symbolic::View::conversions::{atom_to_expr, expr_to_atom};
use crate::symbolic::View::{ExpressionMetrics, inspect_atoms, inspect_exprs};
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

fn print_operation_fingerprint(case: &str, route: &str, metrics: &ExpressionMetrics) {
    report_println!(
        "  {case}/{route} operation_fingerprint: {}",
        metrics.operation_fingerprint()
    );
}

#[test]
#[ignore = "debug View operation corpus; release baseline is separate"]
fn operation_lowering_micro_corpus_preserves_values_and_reports_shape() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Symbolic_View",
        "symbolic::View::operation_lowering_micro_corpus_preserves_values_and_reports_shape",
    );
    let repetitions = std::env::var("SYMBOLIC_VIEW_OPERATION_MICRO_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(20_000);
    let names = ["x", "y", "p"];
    let args = [1.3_f64, 0.7_f64, 2.0_f64];
    let corpus = [
        ("subtraction", "x-y"),
        ("negative-coefficients", "x+(-2.0)*y"),
        ("subtraction-chain", "x-y+p-x"),
        ("explicit-division", "(1+x)/(2+x)"),
        ("division-by-constant", "x/3.0"),
        ("division-by-expression", "(x+y)/(2+p)"),
        ("negative-power-minus-one", "(1+x)^-1"),
        ("negative-power-minus-two", "(1+x)^-2"),
        ("power-two-thirds", "(1+x)^(2/3)"),
        ("power-half", "(1+x)^0.5"),
        ("integer-power", "(1+x)^3"),
        ("nested-power", "((1+x)^0.5)^2"),
        ("variable-power", "(1+x)^y"),
        ("function-exp-sin", "exp(0.2*x)+sin(x)"),
        ("function-log-cos", "log(1+x^2)+cos(exp(0.1*x))"),
        ("repeated-function", "exp(x)+exp(x)+sin(x)*sin(x)"),
        ("rational-coefficient", "x*(2/3)+y*(5/7)-p"),
        ("large-small-coefficients", "1e-12*x+1e12*y-p"),
        ("leaf-parameter-lookup", "x*p+y"),
        ("nary-add-shape", "x+y+p+1.0+2.0"),
        ("nary-mul-shape", "x*y*p*(1+x)"),
        ("repeated-subexpression", "(1+x^2)*(1+x^2)+(1+x^2)"),
    ];

    report_println!(
        "[Symbolic View operation lowering] expressions={}; repetitions={repetitions}; x=1.3,y=0.7,p=2.0; matrix assembly excluded",
        corpus.len()
    );
    report_println!(
        "case                 | route             | nodes | unique | repeated | div | pow | pow-1 | pow-frac | funcs | closure_compile_ms | eval_ns/call | max_diff"
    );
    report_println!(
        "-------------------------------------------------------------------------------------------------------------------------------------"
    );

    for (case, source) in corpus {
        let original = Expr::parse_expression(source);
        let atom = expr_to_atom(&original);
        let atom_roundtrip = atom_to_expr(&atom);
        let forms = [("Expr", original), ("Atom->Expr", atom_roundtrip)];
        let mut baseline_value = None;

        for (route, expression) in forms {
            let metrics = inspect_exprs(std::slice::from_ref(&expression));
            let compile_started = Instant::now();
            let evaluator = Expr::lambdify_borrowed_thread_safe(&expression, &names);
            let closure_compile_ms = compile_started.elapsed().as_secs_f64() * 1_000.0;
            let value = evaluator(&args);
            let eval_ns = measure_scalar(evaluator.as_ref(), &args, repetitions);
            let max_diff = baseline_value
                .map(|baseline: f64| (baseline - value).abs())
                .unwrap_or(0.0);
            if baseline_value.is_none() {
                baseline_value = Some(value);
            }
            assert!(
                max_diff <= 1.0e-12,
                "{case}/{route} operation lowering drifted by {max_diff:e}"
            );
            report_println!(
                "{case:<20} | {route:<17} | {:>5} | {:>6} | {:>8} | {:>3} | {:>3} | {:>5} | {:>8} | {:>5} | {:>18.3} | {:>12.3} | {:>9.3e}",
                metrics.tree_nodes,
                metrics.unique_subexpressions,
                metrics.repeated_subexpressions,
                metrics.operations.divisions,
                metrics.operations.powers,
                metrics.power_integer_negative,
                metrics.power_fractional,
                metrics.operations.functions,
                closure_compile_ms,
                eval_ns,
                max_diff,
            );
            print_operation_fingerprint(case, route, &metrics);
        }

        let metrics = inspect_atoms(std::slice::from_ref(&atom));
        let compile_started = Instant::now();
        let evaluator = atom.lambdify_borrowed_thread_safe(&names);
        let closure_compile_ms = compile_started.elapsed().as_secs_f64() * 1_000.0;
        let value = evaluator(&args);
        let eval_ns = measure_scalar(evaluator.as_ref(), &args, repetitions);
        let max_diff = baseline_value
            .map(|baseline: f64| (baseline - value).abs())
            .unwrap_or(0.0);
        assert!(
            max_diff <= 1.0e-12,
            "{case}/AtomNative operation lowering drifted by {max_diff:e}"
        );
        report_println!(
            "{case:<20} | {:<17} | {:>5} | {:>6} | {:>8} | {:>3} | {:>3} | {:>5} | {:>8} | {:>5} | {:>18.3} | {:>12.3} | {:>9.3e}",
            "AtomNative",
            metrics.tree_nodes,
            metrics.unique_subexpressions,
            metrics.repeated_subexpressions,
            metrics.operations.divisions,
            metrics.operations.powers,
            metrics.power_integer_negative,
            metrics.power_fractional,
            metrics.operations.functions,
            closure_compile_ms,
            eval_ns,
            max_diff,
        );
        print_operation_fingerprint(case, "AtomNative", &metrics);
    }
}

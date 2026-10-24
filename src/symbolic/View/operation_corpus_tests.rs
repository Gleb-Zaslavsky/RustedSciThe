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
use crate::symbolic::View::{
    FunctionMap, PreparedAtomSystem, PreparedAtomSystemScratch, PreparedEvaluationScratch,
    PreparedEvaluator, PreparedSparseAtomSystem, Symbol,
};
use crate::symbolic::symbolic_engine::Expr;

const OPERATION_CORPUS: [(&str, &str); 22] = [
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

const BUILTIN_UNARY_FUNCTIONS: [(&str, &str); 10] = [
    ("exp", "exp(x)"),
    ("log", "log(x)"),
    ("sin", "sin(x)"),
    ("cos", "cos(x)"),
    ("tan", "tan(x)"),
    ("cot", "cot(x)"),
    ("asin", "asin(x)"),
    ("acos", "acos(x)"),
    ("atan", "atan(x)"),
    ("acot", "acot(x)"),
];

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

fn measure_prepared_scalar(
    evaluator: &PreparedEvaluator,
    args: &[f64],
    scratch: &mut PreparedEvaluationScratch,
    repetitions: usize,
) -> f64 {
    for _ in 0..3 {
        black_box(
            evaluator
                .evaluate_with_scratch_typed(args, scratch)
                .expect("prepared evaluator should accept this finite corpus input"),
        );
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        black_box(
            evaluator
                .evaluate_with_scratch_typed(args, scratch)
                .expect("prepared evaluator should accept this finite corpus input"),
        );
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
    let corpus = OPERATION_CORPUS;

    report_println!(
        "[Symbolic View operation lowering] expressions={}; repetitions={repetitions}; x=1.3,y=0.7,p=2.0; matrix assembly excluded",
        corpus.len()
    );
    report_println!(
        "case                 | route             | nodes | unique | repeated | div | pow | pow-1 | pow-frac | funcs | compile_ms | eval_ns/call | max_diff"
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

        let compile_started = Instant::now();
        let variable_names = names
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        let sparse = PreparedSparseAtomSystem::from_atoms_discovering_dependencies(
            std::slice::from_ref(&atom),
            &variable_names,
        );
        let variables = sparse.variable_symbols();
        let function_map = FunctionMap::new();
        let prepared = PreparedEvaluator::new_typed(&atom, &variables, &function_map)
            .expect("prepared Atom evaluator should compile the corpus expression");
        let compile_ms = compile_started.elapsed().as_secs_f64() * 1_000.0;
        let mut scratch = PreparedEvaluationScratch::default();
        let value = prepared
            .evaluate_with_scratch_typed(&args, &mut scratch)
            .expect("prepared Atom evaluator should accept this finite corpus input");
        let max_diff = baseline_value
            .map(|baseline: f64| (baseline - value).abs())
            .unwrap_or(0.0);
        assert!(
            max_diff <= 1.0e-12,
            "{case}/PreparedEvaluator operation lowering drifted by {max_diff:e}"
        );
        let eval_ns = measure_prepared_scalar(&prepared, &args, &mut scratch, repetitions);
        report_println!(
            "{case:<20} | {:<17} | {:>5} | {:>6} | {:>8} | {:>3} | {:>3} | {:>5} | {:>8} | {:>5} | {:>10.3} | {:>12.3} | {:>9.3e}",
            "PreparedEvaluator",
            metrics.tree_nodes,
            metrics.unique_subexpressions,
            metrics.repeated_subexpressions,
            metrics.operations.divisions,
            metrics.operations.powers,
            metrics.power_integer_negative,
            metrics.power_fractional,
            metrics.operations.functions,
            compile_ms,
            eval_ns,
            max_diff,
        );
        print_operation_fingerprint(case, "PreparedEvaluator", &metrics);
    }
}

#[test]
fn operation_corpus_derivative_pipeline_preserves_componentwise_parity() {
    let names = ["x", "y", "p"];
    let input_names = names
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    let states = [
        [1.3_f64, 0.7_f64, 2.0_f64],
        [0.8, 1.2, 0.5],
        [2.1, 0.4, 3.0],
    ];

    for (case, source) in OPERATION_CORPUS {
        let expr = Expr::parse_expression(source);
        let atom = expr_to_atom(&expr);
        let atom_roundtrip = atom_to_expr(&atom);
        let prepared =
            PreparedAtomSystem::try_from_exprs(std::slice::from_ref(&expr), &input_names, None)
                .unwrap_or_else(|error| panic!("{case}: prepared Jacobian setup failed: {error}"));
        let prepared_from_atom = PreparedAtomSystem::try_from_atoms(
            std::slice::from_ref(&atom),
            &input_names,
            None,
            FunctionMap::default(),
        )
        .unwrap_or_else(|error| panic!("{case}: direct-Atom setup failed: {error}"));
        let entries = prepared.jacobian_entries().collect::<Vec<_>>();
        assert_eq!(
            entries,
            prepared_from_atom.jacobian_entries().collect::<Vec<_>>(),
            "{case}: Expr and direct-Atom preparation changed Jacobian ordering"
        );
        let mut scratch = PreparedAtomSystemScratch::default();
        let mut atom_scratch = PreparedAtomSystemScratch::default();
        let mut residual = [0.0];
        let mut jacobian = vec![0.0; prepared.jacobian_len()];
        let mut atom_residual = [0.0];
        let mut atom_jacobian = vec![0.0; prepared_from_atom.jacobian_len()];

        for (state_index, state) in states.iter().enumerate() {
            let expr_value = expr.lambdify_borrowed_thread_safe(&names)(state);
            let atom_value = atom.lambdify_borrowed_thread_safe(&names)(state);
            let roundtrip_value = atom_roundtrip.lambdify_borrowed_thread_safe(&names)(state);
            for (route, actual) in [
                ("AtomNative residual", atom_value),
                ("Atom-to-Expr residual", roundtrip_value),
            ] {
                assert!(
                    (actual - expr_value).abs() <= 1.0e-10 * expr_value.abs().max(1.0),
                    "{case}, state {state_index}, {route}: {actual} != Expr {expr_value}"
                );
            }

            prepared
                .residual_and_jacobian_into(state, &mut residual, &mut jacobian, &mut scratch)
                .unwrap_or_else(|error| panic!("{case}, state {state_index}: {error}"));
            prepared_from_atom
                .residual_and_jacobian_into(
                    state,
                    &mut atom_residual,
                    &mut atom_jacobian,
                    &mut atom_scratch,
                )
                .unwrap_or_else(|error| {
                    panic!("{case}, state {state_index}, direct Atom: {error}")
                });
            assert_eq!(
                residual, atom_residual,
                "{case}: prepared residual route drift"
            );
            assert_eq!(
                jacobian, atom_jacobian,
                "{case}: prepared Jacobian route drift"
            );
            assert!(
                (residual[0] - expr_value).abs() <= 1.0e-10 * expr_value.abs().max(1.0),
                "{case}, state {state_index}: prepared residual {} != Expr {expr_value}",
                residual[0]
            );

            for (column, name) in names.iter().enumerate() {
                let expr_derivative = expr.diff(name);
                let expr_derivative_value =
                    expr_derivative.lambdify_borrowed_thread_safe(&names)(state);
                let symbol = Symbol::new(crate::wrap_symbol!(name));
                let atom_derivative = atom
                    .try_derivative(symbol)
                    .unwrap_or_else(|error| panic!("{case}, d/d{name}: {error}"));
                let atom_derivative_value =
                    atom_derivative.lambdify_borrowed_thread_safe(&names)(state);
                let roundtrip_derivative = atom_to_expr(&atom_derivative);
                let roundtrip_derivative_value =
                    roundtrip_derivative.lambdify_borrowed_thread_safe(&names)(state);
                let prepared_derivative_value = entries
                    .iter()
                    .position(|entry| entry.row == 0 && entry.col == column)
                    .map(|index| jacobian[index])
                    .unwrap_or(0.0);
                let tolerance = 2.0e-10 * expr_derivative_value.abs().max(1.0);

                for (route, actual) in [
                    ("AtomNative derivative", atom_derivative_value),
                    ("Atom-to-Expr derivative", roundtrip_derivative_value),
                    ("PreparedAtomSystem Jacobian", prepared_derivative_value),
                ] {
                    assert!(
                        (actual - expr_derivative_value).abs() <= tolerance,
                        "{case}, state {state_index}, d/d{name}, {route}: {actual} != Expr derivative {expr_derivative_value}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_builtin_unary_function_preserves_expr_atom_derivative_parity() {
    let names = ["x"];
    let input_names = vec!["x".to_string()];
    // Inverse trigonometric functions stay strictly inside their real domain.
    let points = [0.2_f64, 0.5, 0.8];

    for (name, source) in BUILTIN_UNARY_FUNCTIONS {
        let expr = Expr::parse_expression(source);
        let atom = expr_to_atom(&expr);
        let roundtrip = atom_to_expr(&atom);
        let prepared =
            PreparedAtomSystem::try_from_exprs(std::slice::from_ref(&expr), &input_names, None)
                .unwrap_or_else(|error| panic!("{name}: prepared setup failed: {error}"));
        let entries = prepared.jacobian_entries().collect::<Vec<_>>();
        assert_eq!(
            entries.len(),
            1,
            "{name}: expected one structural derivative"
        );
        assert_eq!((entries[0].row, entries[0].col), (0, 0), "{name}");

        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual = [0.0];
        let mut jacobian = [0.0];

        for x in points {
            let args = [x];
            let expr_value = expr.lambdify_borrowed_thread_safe(&names)(&args);
            let atom_value = atom.lambdify_borrowed_thread_safe(&names)(&args);
            let roundtrip_value = roundtrip.lambdify_borrowed_thread_safe(&names)(&args);
            let expr_derivative = expr.diff("x");
            let expr_derivative_value =
                expr_derivative.lambdify_borrowed_thread_safe(&names)(&args);
            let atom_derivative = atom
                .try_derivative(Symbol::new(crate::wrap_symbol!("x")))
                .unwrap_or_else(|error| panic!("{name}, x={x}: derivative failed: {error}"));
            let atom_derivative_value =
                atom_derivative.lambdify_borrowed_thread_safe(&names)(&args);
            let roundtrip_derivative = atom_to_expr(&atom_derivative);
            let roundtrip_derivative_value =
                roundtrip_derivative.lambdify_borrowed_thread_safe(&names)(&args);
            prepared
                .residual_and_jacobian_into(&args, &mut residual, &mut jacobian, &mut scratch)
                .unwrap_or_else(|error| {
                    panic!("{name}, x={x}: prepared evaluation failed: {error}")
                });

            let value_tolerance = 2.0e-12 * expr_value.abs().max(1.0);
            for (route, actual) in [
                ("AtomNative value", atom_value),
                ("Atom-to-Expr value", roundtrip_value),
                ("Prepared residual", residual[0]),
            ] {
                assert!(
                    (actual - expr_value).abs() <= value_tolerance,
                    "{name}, x={x}, {route}: {actual} != Expr {expr_value}"
                );
            }

            let derivative_tolerance = 2.0e-11 * expr_derivative_value.abs().max(1.0);
            for (route, actual) in [
                ("AtomNative derivative", atom_derivative_value),
                ("Atom-to-Expr derivative", roundtrip_derivative_value),
                ("Prepared Jacobian", jacobian[0]),
            ] {
                assert!(
                    (actual - expr_derivative_value).abs() <= derivative_tolerance,
                    "{name}, x={x}, {route}: {actual} != Expr derivative {expr_derivative_value}"
                );
            }
        }
    }
}

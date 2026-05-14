//! Legacy compatibility example.
//!
//! This file preserves the old "numbered example" workflow that used to live
//! in `src/main.rs`, but now as a standalone example module.
//! It is intentionally lightweight and focuses on stable symbolic demos.
//!
//! Run:
//! `cargo run --example legacy_main_compat -- 0`

use RustedSciThe::numerical::BE::BE;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP as FrozenNRBVP;
use RustedSciThe::numerical::BVP_api::BVP;
use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::NR_for_ODE::NRODE;
use RustedSciThe::numerical::ODE_api::ODEsolver;
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

fn case_0_multivariable_symbolics() {
    let input = "exp(x)+log(y)";
    let parsed_expression = Expr::parse_expression(input);
    println!("parsed_expression = {}", parsed_expression);
    println!("pretty            = {}", parsed_expression.sym_to_str("x"));
    println!(
        "variables         = {:?}",
        parsed_expression.extract_variables()
    );
    println!("df/dx             = {}", parsed_expression.diff("x"));
    println!("df/dy             = {}", parsed_expression.diff("y"));

    let args = vec!["x", "y"];
    let f = parsed_expression.lambdify_borrowed_thread_safe(args.as_slice());
    println!("f(1,2)            = {}", f(&[1.0, 2.0]));
}

fn case_1_single_variable_symbolics() {
    let f = Expr::parse_expression("x+exp(x)");
    println!("f(1)              = {}", f.lambdify1D()(1.0));
    println!("df/dx             = {}", f.diff("x"));
    let (norm, ok) = f.compare_num1D("x", 0.0, 5.0, 64, 1e-6);
    println!("num-vs-analytical = norm={norm:.3e}, ok={ok}");
}

fn case_2_ode_lazy_from_strings() {
    // Legacy "lazy" IVP path: user provides RHS as plain strings.
    case_8_ode_solver_from_strings();
}

fn case_3_symbol_construction() {
    let symbols = Expr::Symbols("a, b, c");
    let (a, b, c) = (symbols[0].clone(), symbols[1].clone(), symbols[2].clone());
    let expr = a + Expr::exp(b * c);
    println!("expr              = {}", expr);
    println!("expr(a=1)         = {}", expr.set_variable("a", 1.0));
}

fn case_4_backward_euler_linear() {
    let eq1 = Expr::parse_expression("z+y");
    let eq2 = Expr::parse_expression("z");
    let equations = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let mut solver = BE::new();
    solver.set_initial(
        equations,
        values,
        "x".to_string(),
        1e-6,
        80,
        Some(1e-3),
        0.0,
        0.05,
        y0,
    );
    solver.solve();
    let (_t, y) = solver.get_result();
    let y = y.expect("Backward Euler legacy case should produce a solution matrix");
    println!("BE solved, rows    = {}", y.nrows());
    println!("BE solved, cols    = {}", y.ncols());
    let final_state: Vec<f64> = y.row(y.nrows() - 1).iter().copied().collect();
    println!("BE final state     = {:?}", final_state);
}

fn case_5_backward_euler_nonlinear() {
    let rhs = vec!["-z-exp(-y)", "y"];
    let equations = Expr::parse_vector_expression(rhs);
    let values = vec!["z".to_string(), "y".to_string()];
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let mut solver = BE::new();
    solver.set_initial(
        equations,
        values,
        "x".to_string(),
        1e-3,
        200,
        Some(1e-3),
        0.0,
        0.05,
        y0,
    );
    solver.solve();
    let (_t, y) = solver.get_result();
    let y = y.expect("Backward Euler nonlinear legacy case should produce a solution matrix");
    let final_state: Vec<f64> = y.row(y.nrows() - 1).iter().copied().collect();
    println!("BE(nonlinear) rows = {}", y.nrows());
    println!("BE(nonlinear) cols = {}", y.ncols());
    println!("BE(nonlinear) end  = {:?}", final_state);
}

fn case_6_indexed_variables() {
    let (indexed, names) = Expr::IndexedVars2D(1, 5, "x");
    println!("indexed vars      = {:?}", indexed);
    println!("names             = {:?}", names);
}

fn case_7_ode_solver_symbolic() {
    let y = Expr::Var("y".to_string());
    let z = Expr::Var("z".to_string());
    let eq1 = Expr::Const(-1.0) * z.clone() + (Expr::Const(-1.0) * y.clone()).exp();
    let eq2 = y;
    let equations = vec![eq1, eq2];

    let mut solver = ODEsolver::new_complex(
        equations,
        vec!["z".to_string(), "y".to_string()],
        "x".to_string(),
        "BDF".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 1.0]),
        0.1,
        1e-3,
        1e-5,
        1e-5,
        None,
        false,
        None,
    );
    solver.solve();
    let (_t, y) = solver.get_result();
    println!("ODE solved, rows   = {}", y.nrows());
    println!("ODE solved, cols   = {}", y.ncols());
    let final_state: Vec<f64> = y.row(y.nrows() - 1).iter().copied().collect();
    println!("ODE final state    = {:?}", final_state);
}

fn case_8_ode_solver_from_strings() {
    let rhs = vec!["-z-exp(-y)", "y"];
    let equations = Expr::parse_vector_expression(rhs);
    let mut solver = ODEsolver::new_complex(
        equations,
        vec!["z".to_string(), "y".to_string()],
        "x".to_string(),
        "BDF".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 1.0]),
        0.1,
        1e-3,
        1e-5,
        1e-5,
        None,
        false,
        None,
    );
    solver.solve();
    let (_t, y) = solver.get_result();
    println!("ODE(string) rows   = {}", y.nrows());
    println!("ODE(string) cols   = {}", y.ncols());
    let final_state: Vec<f64> = y.row(y.nrows() - 1).iter().copied().collect();
    println!("ODE(string) final  = {:?}", final_state);
}

fn case_9_parametric_nonlinear_system() {
    let eq1 = Expr::parse_expression("z^2+y^2-10.0*x");
    let eq2 = Expr::parse_expression("z-y-4.0*x");
    let eq_system = vec![eq1, eq2];
    let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-6;
    let max_iterations = 100;
    let max_error = 0.0;

    let mut solver = NRODE::new(
        eq_system,
        initial_guess,
        values,
        arg,
        tolerance,
        max_iterations,
        max_error,
    );
    solver.eq_generate();
    solver.set_t(1.0);
    let solution = solver
        .solve()
        .expect("Legacy NRODE case 9 should converge for t=1.0");
    println!("NRODE solved, dim   = {}", solution.len());
    println!("NRODE solution      = {:?}", solution);
}

fn case_10_lsode2_minimal_numerical() {
    let cfg = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-2.0*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-6,
        1e-8,
    )
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
    .with_analytical_callbacks(
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-2.0 * y[0]]),
        |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-2.0]),
    )
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(cfg);
    solver.solve();
    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);
    println!(
        "LSODE2(min) status = {}",
        solver.get_status().unwrap_or_default()
    );
    println!("LSODE2(min) final_t = {:.6}", final_t);
    println!("LSODE2(min) final_y = {:.8e}", final_y);
    println!("LSODE2(min) exact   = {:.8e}", (-2.0_f64).exp());
}

fn case_11_lsode2_minimal_lambdify() {
    let cfg = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-2.0*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-6,
        1e-8,
    )
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(cfg);
    solver.solve();
    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);
    println!(
        "LSODE2(lambdify) status = {}",
        solver.get_status().unwrap_or_default()
    );
    println!("LSODE2(lambdify) final_t = {:.6}", final_t);
    println!("LSODE2(lambdify) final_y = {:.8e}", final_y);
    println!("LSODE2(lambdify) exact   = {:.8e}", (-2.0_f64).exp());
}

fn case_15_bvp_frozen_sparse() {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^2");
    let eq_system = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-5;
    let max_iterations = 200;
    let t0 = 0.0;
    let t_end = 1.0;
    let n_steps = 80;
    let strategy = "Frozen".to_string();
    let strategy_params = Some(HashMap::from([("Frozen_naive".to_string(), None)]));
    let method = "Sparse".to_string();
    let linear_sys_method = None;
    let ones = vec![0.0; values.len() * n_steps];
    let initial_guess: DMatrix<f64> =
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
    let mut border_conditions = HashMap::new();
    border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

    let mut solver = FrozenNRBVP::new(
        eq_system,
        initial_guess,
        values,
        arg,
        border_conditions,
        t0,
        t_end,
        n_steps,
        strategy,
        strategy_params,
        linear_sys_method,
        method,
        tolerance,
        max_iterations,
    );
    let solution = solver
        .solve()
        .expect("Legacy frozen BVP case 15 should converge");
    println!("Frozen BVP solved   = {} values", solution.len());
}

fn case_19_bvp_solver_sparse() {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^3");
    let equations = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];
    let n_steps = 64;
    let initial_guess = DMatrix::from_column_slice(
        values.len(),
        n_steps,
        DVector::from_vec(vec![0.0; values.len() * n_steps]).as_slice(),
    );
    let mut bc = HashMap::new();
    bc.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    bc.insert("y".to_string(), vec![(1usize, 1.0f64)]);
    let rel_tol = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
    let bounds = HashMap::from([
        ("z".to_string(), (-10.0, 10.0)),
        ("y".to_string(), (-10.0, 10.0)),
    ]);
    let strategy_params = Some(HashMap::from([
        ("max_jac".to_string(), None),
        ("maxDampIter".to_string(), None),
        ("DampFacor".to_string(), None),
        ("adaptive".to_string(), None),
    ]));

    let mut solver = BVP::new(
        equations,
        initial_guess,
        values,
        "x".to_string(),
        bc,
        0.0,
        1.0,
        n_steps,
        "trapezoid".to_string(),
        "Damped".to_string(),
        strategy_params,
        None,
        "Sparse".to_string(),
        1e-5,
        8,
        Some(rel_tol),
        Some(bounds),
        Some("off".to_string()),
    );
    solver.solve();
    let result = solver.get_result().expect("BVP solution should exist");
    println!("BVP solved, rows   = {}", result.nrows());
    println!("BVP solved, cols   = {}", result.ncols());
}

fn print_help() {
    println!("legacy_main_compat");
    println!("Available numbered cases:");
    println!("  0  symbolic multivariable basics");
    println!("  1  symbolic single-variable basics");
    println!("  2  ODE from strings, shortest legacy path");
    println!("  3  direct symbolic construction");
    println!("  4  Backward Euler linear ODE (legacy path)");
    println!("  5  Backward Euler nonlinear ODE (legacy path)");
    println!("  6  indexed symbolic variables");
    println!("  7  ODE symbolic system (legacy BDF path)");
    println!("  8  ODE from string RHS list (legacy BDF path)");
    println!("  9  parametric nonlinear solve (legacy NRODE path)");
    println!("  10 LSODE2 minimal numerical solve (legacy-style entry)");
    println!("  11 LSODE2 minimal Lambdify solve (legacy-style entry)");
    println!("  15 BVP frozen sparse solve (legacy NRBVP path)");
    println!("  19 BVP sparse damped solve (legacy style)");
    println!();
    println!("Example:");
    println!("  cargo run --example legacy_main_compat -- 0");
}

fn main() {
    let arg = std::env::args().nth(1);
    let Some(case_str) = arg else {
        print_help();
        return;
    };
    let Ok(case_id) = case_str.parse::<u32>() else {
        print_help();
        return;
    };

    match case_id {
        0 => case_0_multivariable_symbolics(),
        1 => case_1_single_variable_symbolics(),
        2 => case_2_ode_lazy_from_strings(),
        3 => case_3_symbol_construction(),
        4 => case_4_backward_euler_linear(),
        5 => case_5_backward_euler_nonlinear(),
        6 => case_6_indexed_variables(),
        7 => case_7_ode_solver_symbolic(),
        8 => case_8_ode_solver_from_strings(),
        9 => case_9_parametric_nonlinear_system(),
        10 => case_10_lsode2_minimal_numerical(),
        11 => case_11_lsode2_minimal_lambdify(),
        15 => case_15_bvp_frozen_sparse(),
        19 => case_19_bvp_solver_sparse(),
        _ => {
            println!("Legacy case {} is not included in compat example.", case_id);
            println!("See modern guides in `examples/` for IVP/BVP/LSODE2 backends.");
        }
    }
}

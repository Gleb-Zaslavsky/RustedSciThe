//! LSODE2 Task-Shell Guide
//!
//! Goal:
//! run LSODE2 from a text task document (the same path used by CLI/document
//! workflows) instead of constructing solver config manually in Rust code.
//!
//! This guide demonstrates:
//! - `method: LSODE2` parser route
//! - symbolic Lambdify execution
//! - selecting Jacobian structure/linear solver policy from document fields
//! - postprocessing flags (`save_csv`, `csv_path`, `plot`)
//!
//! Tip:
//! parser also accepts `method: LSODE` and `method: LSODA` aliases.
//! `LSODE` is useful when you want manual family policy (`adams_only`/`bdf_only`);
//! `LSODA` maps to auto-switch family policy.
//! 
//! cargo run --example lsode2_task_shell_guide
//!

use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};

fn main() {
    // Document sections:
    // 1) `task` + solver method
    // 2) symbolic equations
    // 3) initial conditions
    // 4) LSODE2-specific solver options (prefix: `lsode2_`)
    let task = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y1: -10.0*y1 + 9.0*y2
y2: y1 - y2

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0, 0.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: ExprLegacy
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
lsode2_native_execution: faithful_bdf_solve

postprocessing
save_csv: true
csv_path: target/lsode2_task_shell_guide.csv
plot: false
"#;

    let spec = parse_ivp_task_from_str(task).expect("LSODE2 task should parse");
    println!("Parsed task method: {:?}", spec.solver.method);
    println!(
        "Parsed LSODE2 options present: {}",
        spec.solver_options.lsode2.is_some()
    );
    println!("Hint: change `method` to LSODE/LSODA to test manual vs automatic family control.");

    let result = run_ivp_task_from_str(task).expect("LSODE2 task should solve");
    println!("status = {:?}", result.status);
    if let Some(y) = result.y_result {
        let row = y.nrows() - 1;
        println!("final y = [{:.8e}, {:.8e}]", y[(row, 0)], y[(row, 1)]);
    }
}

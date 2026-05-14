use RustedSciThe::numerical::Radau::Radau_main::{Radau, RadauOrder, RadauSolverOptions};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn stiff_radau_options() -> RadauSolverOptions {
    let equations = vec![
        Expr::parse_expression("-1000.0*y1 + y2"),
        Expr::parse_expression("y1 - y2"),
    ];
    let values = vec!["y1".to_string(), "y2".to_string()];

    RadauSolverOptions::new(
        RadauOrder::Order5,
        equations,
        values,
        "t".to_string(),
        1e-6,
        100,
        Some(0.01),
        0.0,
        0.1,
        DVector::from_vec(vec![1.0, 1.0]),
    )
}

fn solve_and_print(label: &str, mut solver: Radau) {
    solver.set_console_logging(false);
    solver.disable_logging();
    solver.solve();

    let stats = solver.get_statistics();
    let (_, maybe_y) = solver.get_result();
    let final_y = maybe_y
        .and_then(|matrix| {
            if matrix.nrows() == 0 {
                None
            } else {
                Some(matrix.row(matrix.nrows() - 1).transpose().into_owned())
            }
        })
        .unwrap_or_else(|| DVector::zeros(0));

    println!("=== {} ===", label);
    println!("status = {}", solver.get_status());
    println!("final_y = {:?}", final_y);
    println!(
        "setup_ms = {:.3}, solve_ms = {:.3}, residual_ms(avg) = {:.6}, jacobian_ms(avg) = {:.6}",
        stats.backend_prepare_ms_total,
        stats.solve_ms_total,
        stats.avg_residual_ms().unwrap_or(0.0),
        stats.avg_jacobian_ms().unwrap_or(0.0)
    );
    println!(
        "steps = {}, newton_solves = {}, newton_iters(avg) = {:.3}",
        stats.step_calls,
        stats.newton_solve_calls,
        stats.avg_newton_iterations().unwrap_or(0.0)
    );
    println!();
}

fn main() {
    println!("Radau backend guide");
    println!();
    println!("Practical recommendations:");
    println!("- Small systems: Lambdify is often enough.");
    println!("- Medium/heavy stiff systems: start with AtomView + C-tcc.");
    println!("- If you care more about runtime throughput than setup cost: try C-gcc.");
    println!("- Zig can also be very strong on some stiff systems.");
    println!();
    println!("This example solves the same stiff 2x2 IVP with several backend choices.");
    println!();

    solve_and_print("Lambdify", Radau::new_with_options(stiff_radau_options()));

    solve_and_print(
        "AtomView + C-tcc",
        Radau::new_with_options(stiff_radau_options())
            .with_dense_generated_backend_c_tcc("target/generated-radau-guide"),
    );

    solve_and_print(
        "AtomView + C-gcc",
        Radau::new_with_options(stiff_radau_options())
            .with_dense_generated_backend_c_gcc("target/generated-radau-guide"),
    );

    solve_and_print(
        "AtomView + Zig",
        Radau::new_with_options(stiff_radau_options())
            .with_dense_generated_backend_zig("target/generated-radau-guide"),
    );

    println!("Tips:");
    println!(
        "- `with_dense_generated_backend_for_repeated_solves(...)` is the user-facing alias for the recommended compiled Radau path."
    );
    println!(
        "- `with_dense_generated_backend_c_tcc(...)` is the most practical compiled default for stiff Radau problems in this crate."
    );
    println!(
        "- `with_dense_generated_backend_c_gcc(...)` is better suited to long-lived generated artifacts."
    );
}

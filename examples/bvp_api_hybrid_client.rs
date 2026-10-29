//! Hybrid BVP client: the application owns the physical problem, while RST
//! parses only solver policy and routes the numerical solve.

use RustedSciThe::command_interpreter::task_parser::DocumentParser;
use RustedSciThe::command_interpreter::task_parser_bvp_hybrid::parse_hybrid_bvp_solver_settings_from_document;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolution, BvpSolver,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This document contains solver policy only. A physics package can produce
    // it without giving the BVP parser ownership of its model or mesh.
    let settings_text = r#"
solver_settings
solver_family: sci
execution: lambdify
assembly: atom_view_native
matrix_layout: banded_auto
tolerance: 0.0001
max_nodes: 256
telemetry: counters
"#;
    let mut parser = DocumentParser::new(settings_text.to_string());
    let settings_document = parser
        .parse_document()
        .map_err(std::io::Error::other)?
        .clone();
    let settings = parse_hybrid_bvp_solver_settings_from_document(&settings_document)?;

    // The application owns equations, state ordering, mesh, boundary values,
    // and the guess; only solver settings come from the parsed document.
    let mesh = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        mesh,
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?;
    let guess = DMatrix::from_row_slice(1, 5, &[1.0, 0.78, 0.61, 0.47, 0.37]);
    let config = settings.with_initial_guess(guess);
    let mut solver = BvpSolver::new(problem, config)?;

    if let BvpSolution::Sci(solution) = solver.solve()? {
        println!("y(1) = {:.8}", solution.y[4]);
    }
    println!("route telemetry: {:?}", solver.telemetry_snapshot());
    Ok(())
}

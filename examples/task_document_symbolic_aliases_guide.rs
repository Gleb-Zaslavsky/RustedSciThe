//! Named symbolic aliases in IVP/BVP task documents.
//!
//! Run with:
//! `cargo run --example task_document_symbolic_aliases_guide`
//!
//! `parameters` provides numeric constants. `where` and `substitute` define
//! named symbolic expressions that may depend on one another. Before a solver
//! is built, the parser applies numeric parameters, resolves the dependency
//! graph, and expands the aliases into solver-ready RHS expressions.

use RustedSciThe::command_interpreter::task_parser::DocumentParser;
use RustedSciThe::command_interpreter::task_parser_common::parse_symbolic_equation_system;

fn parse_document(text: &str) -> RustedSciThe::command_interpreter::task_parser::DocumentMap {
    let mut parser = DocumentParser::new(text.to_string());
    parser.parse_document().expect("task document should parse");
    parser
        .get_result()
        .expect("task document map should exist")
        .clone()
}

fn main() {
    let task = r#"
equations
arg: t
parameters: rate, offset
parameter_values: 2.0, 0.5
unknowns: y
rhs: drive - y

where
base: rate * t
gain: base + offset

substitute
drive: gain * y
"#;

    let document = parse_document(task);
    let equations = parse_symbolic_equation_system(&document, "t")
        .expect("the multi-level alias graph should resolve");

    println!("arg       = {}", equations.arg);
    println!("unknowns  = {:?}", equations.unknowns);
    println!("final rhs =\n{}", equations.rhs[0].pretty_print());

    let rhs = equations.rhs[0].lambdify_borrowed_thread_safe(&["t", "y"]);
    println!("rhs(t=3, y=4) = {}", rhs(&[3.0, 4.0]));

    // Solver-facing documents reject aliases that would replace declared
    // variables, parameters, or the independent argument.
    let invalid_task = r#"
equations
arg: t
unknowns: y
rhs: -y

where
y: 2 * t
"#;
    let invalid_document = parse_document(invalid_task);
    let error = parse_symbolic_equation_system(&invalid_document, "t")
        .expect_err("an alias must not shadow an unknown");
    println!("expected validation error: {error}");
}

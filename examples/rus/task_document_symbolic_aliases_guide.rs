//! Именованные символьные выражения в task documents для IVP и BVP.
//!
//! Запуск:
//! `cargo run --example rus_task_document_symbolic_aliases_guide`
//!
//! В разделе `parameters` задаются числовые константы. Разделы `where` и
//! `substitute` определяют именованные символьные выражения, которые могут
//! зависеть друг от друга. До построения solver-а parser подставляет числовые
//! параметры, разрешает граф зависимостей и раскрывает алиасы в готовых RHS.

use RustedSciThe::command_interpreter::task_parser::DocumentParser;
use RustedSciThe::command_interpreter::task_parser_common::parse_symbolic_equation_system;

fn parse_document(text: &str) -> RustedSciThe::command_interpreter::task_parser::DocumentMap {
    let mut parser = DocumentParser::new(text.to_string());
    parser
        .parse_document()
        .expect("документ задания должен распознаться");
    parser
        .get_result()
        .expect("должна быть построена карта документа")
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
        .expect("многоуровневый граф алиасов должен разрешиться");

    println!("аргумент       = {}", equations.arg);
    println!("неизвестные    = {:?}", equations.unknowns);
    println!("итоговая RHS =\n{}", equations.rhs[0].pretty_print());

    let rhs = equations.rhs[0].lambdify_borrowed_thread_safe(&["t", "y"]);
    println!("RHS(t=3, y=4) = {}", rhs(&[3.0, 4.0]));

    // В solver-facing документе алиас не может заменить объявленную
    // неизвестную, параметр или независимую переменную.
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
        .expect_err("алиас не может перекрывать неизвестную");
    println!("ожидаемая ошибка валидации: {error}");
}

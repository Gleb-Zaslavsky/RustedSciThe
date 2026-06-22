//! Гайд по LSODE2 через task-shell
//!
//! Цель:
//! запускать LSODE2 из текстового task-документа, то есть тем же путём,
//! который используют CLI и document workflows, вместо ручной сборки конфигурации
//! солвера в Rust-коде.
//!
//! Этот гайд показывает:
//! - ветку парсера `method: LSODE2`
//! - символическое исполнение через Lambdify
//! - выбор структуры якобиана и policy линейного солвера из полей документа
//! - флаги постпроцессинга (`save_csv`, `csv_path`, `plot`)
//!
//! Подсказка:
//! парсер также принимает алиасы `method: LSODE` и `method: LSODA`.
//! `LSODE` полезен, когда нужен ручной выбор семейства (`adams_only`/`bdf_only`);
//! `LSODA` соответствует авто-переключению семейств.
//!
//! cargo run --example lsode2_task_shell_guide
//!

use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};

fn main() {
    // Секции документа:
    // 1) `task` + метод солвера
    // 2) символьные уравнения
    // 3) начальные условия
    // 4) LSODE2-специфичные параметры солвера (префикс: `lsode2_`)
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

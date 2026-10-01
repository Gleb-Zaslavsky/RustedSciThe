//! IVP task-document preflight and solve for coupled kinetics and heat transfer.
//!
//! Run with:
//! `cargo run --example ivp_kinetics_symbolic_preflight_guide`
//!
//! The example has two deliberately separate phases:
//! 1. parse and validate the symbolic IVP system: the readable `parameters`
//!    section supplies numeric constants first, `where` definitions are
//!    resolved as a dependency graph, and the RHS retains only declared
//!    states;
//! 2. run the exact same text document through the IVP -> LSODE2 pipeline and
//!    stop when the oxidizer fraction reaches a small configured threshold.
//!
//! Normalizations made solely to turn the supplied draft into unambiguous
//! symbolic input:
//! - `eta_ox2` is treated as the declared state `eta_ox`;
//! - `G_ox2`/`G_b1` are consistently named `Gox2`/`Gb1`;
//! - `Z_s`/`L_s` are named `Z_ox2`/`L_ox2`, matching the `P_ox2` formula;
//! - powers originally written as parameter expressions (for example `10^11.9`)
//!   are pre-evaluated because numeric task parameters must be scalar values.
//!
//! For compact documents, the legacy `equations.parameters` plus
//! `equations.parameter_values` form remains supported. This guide uses the
//! dedicated `parameters` header because it is easier to audit in a kinetic
//! model with many constants.
//!
//! The initial oxidizer fraction is deliberately close to the stop threshold.
//! This keeps the guide fast and demonstrates solver-owned termination. Use
//! calibrated initial conditions and a physical domain length for a real run.

use std::collections::BTreeSet;
use std::path::Path;

use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};

const KINETICS_IVP_TASK: &str = r#"
# This whole document is parsed by the RustedSciThe task shell.
# Весь документ разбирается task-shell RustedSciThe.
# Lines beginning with '#' are comments and are ignored by the parser.
# Строки, начинающиеся с '#', являются комментариями и пропускаются.

task
# IVP selects an initial-value problem. LSODE2 selects the modern LSODE path.
# IVP задаёт задачу Коши, а LSODE2 выбирает современный путь LSODE.
solver: IVP
method: LSODE2

equations
# arg is the independent coordinate. unknowns and rhs have the same order:
# rhs[i] is d(unknowns[i])/d(arg).
# arg — независимая переменная. Порядок unknowns и rhs одинаков:
# rhs[i] есть d(unknowns[i])/d(arg).
arg: x
unknowns: T, q, eta_b1, eta_b2, eta_ox
rhs: q/Lambda, m*C*L*q/Lambda + Fv*L^2, -Gb1*L/m, -Gb2*L/m, -(Gox1+Gox2)*L/m

parameters
# Readable numeric-parameter form: one scalar value per line.
# Читаемый формат численных параметров: одно скалярное значение на строку.
# Do not combine this header with legacy equations.parameters /
# equations.parameter_values in the same task document.
# Не смешивайте этот заголовок с legacy-полями equations.parameters /
# equations.parameter_values в одном документе.
L: 0.001
R: 8.314
m: 0.3
P: 20.0
mu_ox2: 58.75
mu_ox1: 26.3
mu_b: 113.0
mu_bm: 68.0
g_b: 0.6
g_ox: 0.4
ro_b: 0.92
ro_ox: 1.98
R_ox: 0.001
Qox1: 1910.0
Qox2: -1780.0
Qb1: -2000.0
Qb2: 6600.0
Z_ox2: 47773447.0
L_ox2: 52216.0
Z_ox1: 7.94328234724282e11
E_ox1: 163176.0
k_b: 1.28e5
Z_b1: 6.30957344480193e13
E_b1: 193300.0
Lambda: 2.092e-4
C: 1.05

where
# Named algebraic definitions are expanded transitively before the solver is
# built. Each final RHS must retain only x and declared unknowns.
# Именованные алгебраические определения рекурсивно подставляются до создания
# солвера. В финальных RHS остаются только x и объявленные неизвестные.
Gox2: k_ox1*P_ox2*Z/(P - P_ox2 -P_b)*(1-eta_ox) 
Gb1: k_b1*(1-eta_b1-eta_b2) 
P_b: P_ox2*( g_b/g_ox)*((eta_b1+eta_b2)/eta_ox)*(mu_ox2/mu_b)
Z: mu_ox2/mu_ox1
Zb: mu_ox2/mu_b
P_ox2: Z_ox2*exp(-L_ox2/(R*T))
P_acid: 0.5*P_ox2
Gb2: (mu_bm/(delta_b*ro_b))*k_b*P_acid
delta_b: ((1-g_ox)/(3*g_ox))*(R_ox*ro_ox/ro_b)
Gox1: k_ox1*eta_ox
k_ox1: Z_ox1*exp(-E_ox1/(R*T))
k_b1: Z_b1*exp(-E_b1/(R*T))
Fv: Qox1*Gox1+Qox2*Gox2+Qb1*Gb1+Qb2*Gb2

initial_conditions
# y0 follows exactly the unknowns order: T, q, eta_b1, eta_b2, eta_ox.
# Do not put a trailing comma after the final value.
# y0 повторяет порядок unknowns: T, q, eta_b1, eta_b2, eta_ox.
# После последнего значения завершающая запятая не ставится.
t0: 0.0
t_end: 1000.0
y0: 300.0, 0.001, 0.001, 0.001, 0.001

solver_options
# Relative/absolute tolerances control local error; max_step caps one step.
# Относительный/абсолютный допуски задают локальную ошибку; max_step
# ограничивает длину одного шага.
rtol: 1e-6
atol: 1e-9
first_step: Some(1e-6)
max_step: 1.0
# bdf selects the faithful BDF-only LSODE2 controller. Other valid values:
# auto (LSODA-style Adams/BDF switching) and adams.
# bdf выбирает faithful BDF-only контроллер LSODE2. Другие допустимые
# значения: auto (LSODA-подобное переключение Adams/BDF) и adams.
lsode2_method_family: bdf
# AtomView builds symbolic derivatives; LambdifyExpr evaluates them without an
# external compiler. For AOT use AOT plus lsode2_aot_toolchain/profile.
# AtomView строит символьные производные; LambdifyExpr вычисляет их без
# внешнего компилятора. Для AOT используйте AOT и lsode2_aot_toolchain/profile.
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: LambdifyExpr
# Sparse selects sparse Jacobian storage. auto lets LSODE2 select the matching
# linear solver for that structure.
# Sparse выбирает разреженное хранение Якобиана. auto позволяет LSODE2
# выбрать подходящий линейный решатель для этой структуры.
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
# faithful_bdf_solve executes the native faithful BDF route rather than the
# historical bridge-only route.
# faithful_bdf_solve запускает нативный faithful BDF-маршрут, а не только
# исторический bridge-маршрут.
lsode2_native_execution: faithful_bdf_solve
# Stop predicates are evaluated after every accepted native step:
# ge: variable >= target; le: variable <= target;
# abs_distance: abs(variable - target) <= lsode2_stop_tolerance.
# Предикаты остановки проверяются после каждого принятого нативного шага:
# ge: variable >= target; le: variable <= target;
# abs_distance: abs(variable - target) <= lsode2_stop_tolerance.
# In the equations currently written below, eta_ox is consumed:
# d(eta_ox)/dx = (Gox1 + Gox2)*L/m. Therefore its matching stop predicate
# is le. If eta_ox is intended to mean conversion that grows towards one,
# its governing equation must be changed before using ge: 0.999.
# В записанных ниже уравнениях eta_ox расходуется:
# d(eta_ox)/dx = (Gox1 + Gox2)*L/m. Поэтому подходящий предикат — le.
# Если eta_ox должен означать конверсию, растущую к единице, прежде чем
# использовать ge: 0.999 необходимо изменить определяющее уравнение.
lsode2_stop_variable: eta_ox
lsode2_stop_comparator: le
lsode2_stop_target: 1e-3

postprocessing
# CSV contains the complete trajectory. gnuplot_png writes one PNG per state;
# if gnuplot is absent from PATH, this optional action is skipped.
# CSV содержит всю траекторию. gnuplot_png пишет один PNG на состояние; если
# gnuplot отсутствует в PATH, это необязательное действие пропускается.
save_csv: true
csv_path: target/ivp_kinetics_symbolic_preflight/solution.csv
gnuplot_png: true
gnuplot_dir: target/ivp_kinetics_symbolic_preflight/gnuplot
"#;

fn main() {
    // Phase 1: parser-only symbolic preflight.
    let spec = parse_ivp_task_from_str(KINETICS_IVP_TASK)
        .expect("kinetics IVP task document should parse and normalize");

    let expected_variables = BTreeSet::from([
        "T".to_string(),
        "q".to_string(),
        "eta_b1".to_string(),
        "eta_b2".to_string(),
        "eta_ox".to_string(),
    ]);

    println!("resolved states: {:?}", spec.equations.unknowns);
    println!(
        "numeric parameter count: {}",
        spec.equations.parameter_values.len()
    );
    for (state, rhs) in spec.equations.unknowns.iter().zip(&spec.equations.rhs) {
        let variables = rhs
            .all_arguments_are_variables()
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert!(
            variables.is_subset(&expected_variables),
            "RHS for `{state}` still contains non-state symbols: {variables:?}"
        );
        println!("d({state})/dx uses: {variables:?}");
    }

    println!("symbolic preflight passed: every RHS depends only on declared states.");
    let mut rhs_argument_names = vec![spec.equations.arg.as_str()];
    rhs_argument_names.extend(spec.equations.unknowns.iter().map(String::as_str));
    let mut initial_rhs_arguments = vec![spec.initial_conditions.t0];
    initial_rhs_arguments.extend_from_slice(&spec.initial_conditions.y0);
    for (state, rhs) in spec.equations.unknowns.iter().zip(&spec.equations.rhs) {
        let value = rhs.lambdify_borrowed_thread_safe(&rhs_argument_names)(&initial_rhs_arguments);
        assert!(
            value.is_finite(),
            "initial RHS for {} must be finite, got {value}",
            state
        );
        println!("initial d({state})/dx = {value:.6e}");
    }

    let eta_ox_index = spec
        .equations
        .unknowns
        .iter()
        .position(|name| name == "eta_ox")
        .expect("eta_ox must be declared by the task");
    let eta_stop = spec
        .solver_options
        .lsode2
        .as_ref()
        .expect("LSODE2 options must be present")
        .stop_conditions
        .iter()
        .find(|condition| condition.variable == "eta_ox")
        .expect("eta_ox stop condition must be configured");
    let initial_eta_ox = spec.initial_conditions.y0[eta_ox_index];
    let condition_holds_initially = stop_condition_holds(initial_eta_ox, eta_stop);
    println!(
        "stop condition: eta_ox {} {:.6e}; initial eta_ox={:.6e}; initially_satisfied={condition_holds_initially}",
        eta_stop.comparator.label(),
        eta_stop.target,
        initial_eta_ox,
    );

    // Phase 2: the same document enters the regular typed IVP/LSODE2 path.
    let result = run_ivp_task_from_str(KINETICS_IVP_TASK)
        .expect("kinetics IVP task should solve through LSODE2");
    let solution = result
        .y_result
        .expect("LSODE2 task should return a solution matrix");
    let final_eta_ox = solution[(solution.nrows() - 1, eta_ox_index)];

    println!("LSODE2 status: {:?}", result.status);
    println!("final eta_ox: {final_eta_ox:.6e}");
    let csv_path = Path::new("target/ivp_kinetics_symbolic_preflight/solution.csv");
    assert!(
        csv_path.exists(),
        "postprocessing should write {}",
        csv_path.display()
    );
    println!("CSV exported to {}", csv_path.display());
    println!("gnuplot PNG files are requested in target/ivp_kinetics_symbolic_preflight/gnuplot");
    assert!(
        stop_condition_holds(final_eta_ox, eta_stop),
        "expected LSODE2 stop condition eta_ox {} {}, got {final_eta_ox:e}",
        eta_stop.comparator.label(),
        eta_stop.target
    );
}

fn stop_condition_holds(
    value: f64,
    condition: &RustedSciThe::numerical::LSODE2::Lsode2StopCondition,
) -> bool {
    match condition.comparator {
        RustedSciThe::numerical::LSODE2::Lsode2StopComparator::GreaterEqual => {
            value >= condition.target
        }
        RustedSciThe::numerical::LSODE2::Lsode2StopComparator::LessEqual => {
            value <= condition.target
        }
        RustedSciThe::numerical::LSODE2::Lsode2StopComparator::AbsDistance => {
            (value - condition.target).abs() <= condition.tolerance
        }
    }
}

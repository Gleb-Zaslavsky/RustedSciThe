//! Гайд по краевым задачам для damped BVP-солвера.
//!
//! Этот пример задуман как единая пользовательская точка входа для текущей
//! истории бэкендов BVP:
//! - `Lambdify + ExprLegacy` - самый безопасный базовый путь для разового прогона.
//! - `AtomView + C-tcc` - практическая рекомендация для повторных решений одной и той же символической задачи.
//! - `AtomView + C-gcc` полезен, когда важнее пропускная способность нативных callback-ов, чем задержка старта.
//!
//! Какие вопросы закрывает этот пример:
//! 1. Как решить простую BVP через обычный lambdify-путь?
//! 2. Как переключить ту же задачу на AOT-путь?
//! 3. Где выбрать `ExprLegacy` вместо `AtomView`?
//! 4. Где выбрать нативный компилятор (`tcc`, `gcc`)?
//! 5. Где переопределить chunking или policy выполнения?
//!
//! Связанные запускаемые примеры:
//! - `examples/bvp_sci_numerical_guide.rs` показывает чисто числовой, безсимвольный API `BVP_sci` в режимах `FiniteDifference` и `AnalyticalPointwise`.
//! - `examples/bvp_sci_numerical_parameters_guide.rs` показывает, как тот же чисто числовой API решает задачу с неизвестными скалярными параметрами вместе с состоянием.
//!
//! Практические рекомендации по текущей диагностике BVP:
//! - используйте `Lambdify`, когда ожидается только один запуск задачи;
//! - используйте `AtomView + C-tcc`, когда одна и та же символическая задача будет решаться повторно с новыми начальными приближениями, значениями параметров или настройками солвера;
//! - используйте `AtomView + C-gcc`, когда задержка компиляции допустима, а нужен сильный ориентир по нативному runtime;
//! - на больших combustion-сетках картина меняется: на более мощной машине `AtomView + C-tcc/gcc` обгонял `Lambdify` примерно на 20-35% около `n_steps = 5000`, при этом `C-tcc` был чуть впереди. На практике это значит, что точка безубыточности для разового combustion-решения лежит где-то выше `n_steps = 2000` и ближе к режиму «десятков тысяч дискретных уравнений», после чего AOT продолжает уходить вперёд;
//! - Rust и Zig AOT-бэкенды существуют ниже по стеку, но наиболее отточенные варианты для BVP сейчас - `Lambdify`, `C-tcc` и `C-gcc`.
/*

|discretization,       |
| sym.jacobian stage   | turning into functions stage |	compilation and linking stage| loop stage|
|______________________|______________________________|______________________________|___________|
|                      |                              |				     |	     	 |
|		               | Classic Lambdification        |				     |		 |
| Classic Symbolic     |______________________________| not needed                   |		 |
|		               |			                      |			             |		 |
|                      | Classic AOT                  |				     |		 |
|______________________|______________________________|______________________________|___________|
|		       |	                      |				     |		 |
| AtomView Symbolic    |  AtomView Lambdification     | C (tcc or gcc),		     |		 |
|		       |______________________________| Zig			     |		 |
|		       |			      | Rust			     |		 |
|		       |  AtomView AOT                |				     |		 |
|		       |______________________________|______________________________|___________|

run: cargo run --example bvp_backends_guide


*/
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use nalgebra::{DMatrix, DVector};

fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
    DMatrix::from_column_slice(
        variable_count,
        n_steps,
        DVector::from_element(variable_count * n_steps, value).as_slice(),
    )
}

fn simple_linear_bvp(options: DampedSolverOptions, n_steps: usize) -> NRBVP {
    // Решаем y'' = 0 с условиями y(0) = 0 и y(1) = 1.
    // Первая форма:
    // y' = z
    // z' = 0
    let equations = vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")];
    let boundary_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0f64)]),
        ("z".to_string(), vec![(1usize, 1.0f64)]),
    ]);
    let initial_guess = uniform_initial_guess(2, n_steps, 0.5);

    let mut solver = NRBVP::new_with_options(
        equations,
        initial_guess,
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn base_options() -> DampedSolverOptions {
    let rel_tolerance = HashMap::from([("y".to_string(), 1e-7), ("z".to_string(), 1e-7)]);
    let bounds = HashMap::from([
        ("y".to_string(), (-10.0, 10.0)),
        ("z".to_string(), (-10.0, 10.0)),
    ]);
    let strategy_params = SolverParams {
        max_jac: Some(4),
        max_damp_iter: Some(5),
        damp_factor: Some(0.5),
        adaptive: None,
    };

    DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(strategy_params))
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(rel_tolerance)
        .with_bounds(bounds)
        .with_max_iterations(40)
        .with_loglevel(Some("error".to_string()))
}

fn lambdify_options() -> DampedSolverOptions {
    let config = GeneratedBackendConfig::default()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    base_options().with_generated_backend_config(config)
}

fn atomview_tcc_options() -> DampedSolverOptions {
    base_options().with_sparse_atomview_for_repeated_solves()
}

fn atomview_gcc_options() -> DampedSolverOptions {
    base_options().with_sparse_atomview_c_gcc()
}

fn manual_atomview_tcc_options() -> DampedSolverOptions {
    // Это вариант, где видны все ручки настройки.
    //
    // Меняйте эти части, когда нужно тонко настроить поведение:
    // - бэкенд символической сборки: ExprLegacy или AtomView
    // - нативный компилятор: tcc или gcc
    // - chunking: разбиение невязки / sparse якобиана
    // - policy выполнения: последовательный или параллельный runtime
    // - policy сборки: build-if-missing или rebuild-always
    let config = GeneratedBackendConfig::sparse_build_if_missing_release()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
        .with_aot_codegen_backend(
            RustedSciThe::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C,
        )
        .with_aot_c_compiler("tcc")
        .with_aot_execution_policy(AotExecutionPolicy::Parallel(
            ParallelExecutorConfig::default(),
        ))
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_aot_compile_dev_fastest()
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 16 }),
        ));

    base_options().with_generated_backend_config(config)
}

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn run_case(label: &str, options: DampedSolverOptions, n_steps: usize) {
    let started = Instant::now();
    let mut solver = simple_linear_bvp(options, n_steps);
    let solution = solver
        .try_solve()
        .unwrap_or_else(|err| panic!("{label} failed: {err:?}"))
        .expect("solver should converge on the simple linear BVP");
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let max_abs_solution = solution.iter().map(|value| value.abs()).fold(0.0, f64::max);

    println!("{label:24} total_ms = {total_ms:9.3}, max_abs_solution = {max_abs_solution:.6e}");
}

fn print_guide() {
    println!("BVP backend guide");
    println!("=================");
    println!("Recommended defaults:");
    println!("  - One-shot solve: Lambdify + ExprLegacy");
    println!("  - Repeated solves: AtomView + C-tcc");
    println!("  - Throughput-oriented native path: AtomView + C-gcc");
    println!("  - Large combustion-like grids: once you reach the tens-of-thousands");
    println!("    of discrete equations, AtomView + C-tcc/gcc can overtake one-shot");
    println!("    Lambdify end-to-end, with C-tcc currently the most practical choice.");
    println!();
    println!("Important knobs:");
    println!("  - Symbolic assembly: ExprLegacy vs AtomView");
    println!("  - Native compiler: tcc vs gcc");
    println!("  - Chunking: residual and sparse Jacobian chunk strategies");
    println!("  - Build policy: BuildIfMissing vs RebuildAlways");
    println!("  - Compile preset: Production / FastBuild / DevFastest");
    println!();
    println!("Where to look next:");
    println!("  - examples/bvp_sci_numerical_guide.rs:");
    println!("    pure numerical BVP_sci without symbolic APIs");
    println!("  - examples/bvp_sci_numerical_parameters_guide.rs:");
    println!("    pure numerical BVP_sci with solved parameters");
    println!();
    println!("Chunking tips:");
    println!("  - Start with backend defaults.");
    println!("  - Use coarse chunk counts for large problems when native runtime");
    println!("    parallelism is desirable.");
    println!("  - Whole-chunk mode is simplest when investigating correctness.");
    println!();
}

fn main() {
    let n_steps = 64;
    print_guide();

    println!("Demo problem: y'' = 0, y(0) = 0, y(1) = 1, n_steps = {n_steps}");
    println!();

    run_case("Lambdify / ExprLegacy", lambdify_options(), n_steps);

    if command_available("tcc") {
        run_case("AtomView / C-tcc", atomview_tcc_options(), n_steps);
    } else {
        println!("AtomView / C-tcc       skipped: `tcc` was not found in PATH");
    }

    if command_available("gcc") {
        run_case("AtomView / C-gcc", atomview_gcc_options(), n_steps);
    } else {
        println!("AtomView / C-gcc       skipped: `gcc` was not found in PATH");
    }

    println!();
    println!("Manual configuration example:");
    println!("  - same demo problem, but with explicit chunking/build/execution knobs exposed");
    if command_available("tcc") {
        run_case(
            "AtomView / manual C-tcc",
            manual_atomview_tcc_options(),
            n_steps,
        );
    } else {
        println!("AtomView / manual C-tcc skipped: `tcc` was not found in PATH");
    }
}
